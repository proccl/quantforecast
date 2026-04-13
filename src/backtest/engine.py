"""
回測引擎
處理預測生成、交易模擬、風險指標計算
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from src.config import BacktestConfig
from src.models.patchtst import PatchTST

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回測結果數據類"""
    initial_capital: float
    final_capital: float
    total_return_pct: float
    buyhold_return_pct: float
    excess_return_pct: float
    max_drawdown_pct: float
    volatility_annual_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_trades: int
    win_count: int
    loss_count: int
    win_rate_pct: float
    test_accuracy: float
    test_samples: int
    date_range: str
    equity_curve: np.ndarray
    aligned_df: pd.DataFrame
    trades: List[Dict]


class BacktestEngine:
    """回測引擎"""
    
    def __init__(self, model: PatchTST, device: torch.device, config: BacktestConfig):
        self.model = model
        self.device = device
        self.config = config
    
    def run(
        self,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        seq_len: int,
        pred_len: int
    ) -> BacktestResult:
        """
        執行完整回測
        
        Returns:
            BacktestResult 對象
        """
        logger.info("開始回測...")
        
        # 1. 生成預測
        aligned_df = self._generate_predictions(
            test_df, feature_cols, seq_len, pred_len
        )
        
        # 2. 執行交易模擬
        equity_curve, trades, final_capital = self._simulate_trades(aligned_df)
        
        # 3. 計算風險指標
        metrics = self._calculate_metrics(equity_curve, aligned_df, final_capital, trades)
        
        # 4. 計算準確率
        test_accuracy = (
            aligned_df['pred_direction'] == aligned_df['actual_direction']
        ).mean()
        
        logger.info(f"✓ 回測完成，最終資金: {final_capital:,.0f} HKD")
        
        return BacktestResult(
            initial_capital=self.config.initial_capital,
            final_capital=final_capital,
            total_return_pct=metrics['total_return_pct'],
            buyhold_return_pct=metrics['buyhold_return_pct'],
            excess_return_pct=metrics['excess_return_pct'],
            max_drawdown_pct=metrics['max_drawdown_pct'],
            volatility_annual_pct=metrics['volatility_annual_pct'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            calmar_ratio=metrics['calmar_ratio'],
            total_trades=metrics['total_trades'],
            win_count=metrics['win_count'],
            loss_count=metrics['loss_count'],
            win_rate_pct=metrics['win_rate_pct'],
            test_accuracy=test_accuracy,
            test_samples=len(aligned_df),
            date_range=f"{aligned_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {aligned_df['date'].iloc[-1].strftime('%Y-%m-%d')}",
            equity_curve=equity_curve,
            aligned_df=aligned_df,
            trades=trades
        )
    
    def predict_future(
        self,
        latest_data: np.ndarray,
        latest_close: float,
        latest_date: pd.Timestamp,
        pred_len: int
    ) -> Dict:
        """
        預測未來價格
        
        Returns:
            預測結果字典
        """
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(latest_data).unsqueeze(0).to(self.device)
            pred = self.model(x)
            pred_values = pred.squeeze().cpu().numpy()
        
        # 處理預測值
        if isinstance(pred_values, np.ndarray):
            pred_scalar = float(pred_values.item()) if pred_values.size == 1 else float(pred_values[0])
        else:
            pred_scalar = float(pred_values)
        
        # 計算未來交易日
        future_dates = []
        current_date = latest_date
        while len(future_dates) < pred_len:
            current_date = current_date + pd.Timedelta(days=1)
            if current_date.weekday() < 5:
                future_dates.append(current_date)
        
        # 線性插值計算未來價格
        future_prices = []
        for i in range(pred_len):
            progress = (i + 1) / pred_len
            price = latest_close * (1 + pred_scalar * progress)
            future_prices.append(price)
        
        return {
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'latest_close': float(latest_close),
            'future_return_pct': float(pred_scalar * 100),
            'future_price_day5': float(future_prices[-1]),
            'future_prices_all': [float(p) for p in future_prices],
            'prediction_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'pred_direction': 'UP' if pred_scalar > 0 else 'DOWN'
        }
    
    def _generate_predictions(
        self,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        seq_len: int,
        pred_len: int
    ) -> pd.DataFrame:
        """生成歷史預測序列"""
        self.model.eval()
        
        predictions = []
        actuals = []
        test_dates = []
        test_dates_T5 = []
        test_close = []
        test_close_T5 = []
        test_directions = []
        
        with torch.no_grad():
            for i in range(len(test_df) - seq_len - pred_len + 1):
                seq_data = test_df[feature_cols].iloc[i:i + seq_len].values
                close_T = test_df['close'].iloc[i + seq_len]
                close_T5 = test_df['close'].iloc[i + seq_len:i + seq_len + pred_len].values[-1]
                target_return = close_T5 / close_T - 1
                
                x = torch.FloatTensor(seq_data).unsqueeze(0).to(self.device)
                pred = self.model(x)
                
                pred_return = pred.squeeze().cpu().numpy()
                if isinstance(pred_return, np.ndarray):
                    pred_return = float(pred_return.item()) if pred_return.size == 1 else float(pred_return[0])
                else:
                    pred_return = float(pred_return)
                
                predictions.append(pred_return)
                actuals.append(float(target_return))
                test_dates.append(test_df['date'].iloc[i + seq_len])
                test_dates_T5.append(test_df['date'].iloc[i + seq_len + pred_len - 1])
                test_close.append(close_T)
                test_close_T5.append(close_T5)
                test_directions.append(1 if target_return > 0 else 0)
        
        return pd.DataFrame({
            'date': test_dates,
            'date_T5': test_dates_T5,
            'close': test_close,
            'close_T5': test_close_T5,
            'pred_return': predictions,
            'actual_return': actuals,
            'pred_direction': [1 if p > 0 else 0 for p in predictions],
            'actual_direction': test_directions
        })
    
    def _simulate_trades(
        self,
        aligned_df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[Dict], float]:
        """執行交易模擬"""
        initial_capital = self.config.initial_capital
        capital = initial_capital
        position = 0.0
        trades = []
        equity_curve = []
        
        for i in range(len(aligned_df)):
            current_price = aligned_df['close'].iloc[i]
            pred_direction = aligned_df['pred_direction'].iloc[i]
            
            if pred_direction == 1 and position == 0:
                position = capital / current_price
                capital = 0
                trades.append({
                    'date': aligned_df['date'].iloc[i],
                    'action': 'BUY',
                    'price': current_price,
                    'shares': position
                })
            elif pred_direction == 0 and position > 0:
                capital = position * current_price
                trades.append({
                    'date': aligned_df['date'].iloc[i],
                    'action': 'SELL',
                    'price': current_price,
                    'proceeds': capital
                })
                position = 0
            
            current_equity = capital if position == 0 else position * current_price
            equity_curve.append(current_equity)
        
        equity_curve = np.array(equity_curve)
        final_capital = capital if position == 0 else position * aligned_df['close'].iloc[-1]
        
        return equity_curve, trades, final_capital
    
    def _calculate_metrics(
        self,
        equity_curve: np.ndarray,
        aligned_df: pd.DataFrame,
        final_capital: float,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """計算風險指標和交易統計"""
        initial_capital = self.config.initial_capital
        total_return_pct = (final_capital / initial_capital - 1) * 100
        
        # 買入持有基準
        buy_hold_shares = initial_capital / aligned_df['close'].iloc[0]
        buy_hold_final = buy_hold_shares * aligned_df['close'].iloc[-1]
        buyhold_return_pct = (buy_hold_final / initial_capital - 1) * 100
        excess_return_pct = total_return_pct - buyhold_return_pct
        
        # 日收益和風險指標
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        volatility_annual_pct = np.std(daily_returns) * np.sqrt(252) * 100
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown_pct = np.min(drawdown) * 100
        
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino_ratio = np.mean(daily_returns) / downside_std * np.sqrt(252)
        calmar_ratio = (total_return_pct / 100) / (abs(max_drawdown_pct) / 100 + 1e-8)
        
        # 交易統計
        trades_df = pd.DataFrame(trades)
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        win_count = 0
        loss_count = 0
        for i in range(len(sell_trades)):
            if i < len(buy_trades):
                buy_price = buy_trades.iloc[i]['price']
                sell_price = sell_trades.iloc[i]['price']
                if sell_price > buy_price:
                    win_count += 1
                else:
                    loss_count += 1
        
        win_rate_pct = win_count / max(win_count + loss_count, 1) * 100
        total_trades = len(buy_trades)
        
        return {
            'total_return_pct': total_return_pct,
            'buyhold_return_pct': buyhold_return_pct,
            'excess_return_pct': excess_return_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'volatility_annual_pct': volatility_annual_pct,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate_pct': win_rate_pct
        }
