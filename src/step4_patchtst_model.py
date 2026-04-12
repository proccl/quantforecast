import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle

print("=" * 60)
print("【步驟 4/8】PatchTST 模型實現")
print("=" * 60)

print("\n根據論文: A Time Series is Worth 64 Words (Nie et al., 2022)")
print("模型特點: Channel-Independent + Patching + Transformer Encoder")
print("-" * 60)

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        batch_size, n_features, n_patches, _ = patches.shape
        patches = patches.reshape(batch_size * n_features, n_patches, self.patch_len)
        patches = self.value_embedding(patches)
        patches = self.dropout(patches)
        return patches, n_features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PatchTST(nn.Module):
    def __init__(
        self,
        n_features=17,
        seq_len=20,
        pred_len=5,
        patch_len=5,
        stride=2,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        head_type='regression',
        use_revin=True
    ):
        super().__init__()
        
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.head_type = head_type
        self.use_revin = use_revin
        
        self.n_patches = (seq_len - patch_len) // stride + 1
        
        if use_revin:
            from step3_data_preprocessing import RevIN
            self.revin = RevIN(n_features)
        
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.flatten = nn.Flatten(start_dim=-2)
        
        if head_type == 'regression':
            self.head = nn.Linear(d_model * self.n_patches, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model * self.n_patches, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 2)
            )
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        patches, n_features = self.patch_embedding(x)
        patches = self.pos_encoder(patches)
        encoded = self.transformer_encoder(patches)
        
        batch_size = x.shape[0]
        encoded = encoded.reshape(batch_size, n_features, self.n_patches, self.d_model)
        encoded = self.flatten(encoded)
        encoded = encoded.mean(dim=1)
        
        output = self.head(encoded)
        return output

print("\n【4.1】載入配置")
print("-" * 40)

with open('data_config.pkl', 'rb') as f:
    config = pickle.load(f)

MODEL_CONFIG = {
    'n_features': config['n_features'],
    'seq_len': config['seq_len'],
    'pred_len': config['pred_len'],
    'patch_len': config['patch_len'],
    'stride': config['stride'],
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'd_ff': 128,
    'dropout': 0.1,
    'use_revin': True
}

n_patches = (config['seq_len'] - config['patch_len']) // config['stride'] + 1

print(f"✓ 輸入特徵數: {MODEL_CONFIG['n_features']}")
print(f"✓ 序列長度: {MODEL_CONFIG['seq_len']} 天")
print(f"✓ Patch 長度: {MODEL_CONFIG['patch_len']}")
print(f"✓ Patch 步長: {MODEL_CONFIG['stride']}")
print(f"✓ 計算得 Patch 數: {n_patches}")
print(f"✓ 模型維度 (d_model): {MODEL_CONFIG['d_model']}")
print(f"✓ 注意力頭數: {MODEL_CONFIG['n_heads']}")
print(f"✓ Transformer 層數: {MODEL_CONFIG['n_layers']}")
print(f"✓ Dropout: {MODEL_CONFIG['dropout']}")
print(f"✓ 使用 RevIN: {MODEL_CONFIG['use_revin']}")

print("\n【4.2】模型參數統計")
print("-" * 40)

model_reg = PatchTST(head_type='regression', **MODEL_CONFIG)
model_cls = PatchTST(head_type='classification', **MODEL_CONFIG)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✓ 回歸模型參數量: {count_params(model_reg):,}")
print(f"✓ 分類模型參數量: {count_params(model_cls):,}")

batch_size = 4
x_test = torch.randn(batch_size, MODEL_CONFIG['seq_len'], MODEL_CONFIG['n_features'])

with torch.no_grad():
    out_reg = model_reg(x_test)
    out_cls = model_cls(x_test)

print(f"\n【4.3】前向傳播測試")
print("-" * 40)
print(f"✓ 輸入形狀: {x_test.shape}")
print(f"✓ 回歸輸出形狀: {out_reg.shape} (預測收益率)")
print(f"✓ 分類輸出形狀: {out_cls.shape} (漲跌logits)")

print("\n【4.4】模型架構")
print("-" * 40)
print(model_reg)

print("\n" + "=" * 60)
print("【關鍵節點 4/8 完成】PatchTST 模型實現完成")
print("=" * 60)
