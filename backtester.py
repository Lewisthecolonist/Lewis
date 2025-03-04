import pandas as pd
import numpy as np
import multiprocessing
import threading
import queue as queue
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
import random
from scipy import stats
from strategy import Strategy
from risk_manager import RiskManager
import strategy
from strategy_generator import StrategyGenerator
from strategy_selector import StrategySelector
from config import Config
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import zipfile
import io
from io import StringIO
import traceback
from strategy_optimizer import StrategyOptimizer
from collections import deque
from event import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent
import time
from market_simulator import MarketSimulator
historical_data = pd.read_csv('historical_data.csv.zip')
from strategy import TimeFrame
from strategy import (TrendFollowingStrategy, MeanReversionStrategy, MomentumStrategy, 
                      VolatilityStrategy, StatisticalArbitrageStrategy, 
                      SentimentAnalysisStrategy, BreakoutStrategy)
from api_call_manager import APICallManager
from strategy_manager import StrategyManager
import tracemalloc
tracemalloc.start()
import asyncio
import re

class TransactionCostModel:
    def __init__(self, config):
        self.config = config

    def calculate_costs(self, order_event, current_price):
        quantity = order_event.quantity
        notional_value = quantity * current_price

        base_fee = self.config.BASE_FEE * notional_value
        spread_cost = self.calculate_spread_cost(notional_value)
        market_impact = self.calculate_market_impact(quantity, current_price)
        slippage = self.calculate_slippage(notional_value)

        total_cost = base_fee + spread_cost + market_impact + slippage

        return {
            'base_fee': base_fee,
            'spread_cost': spread_cost,
            'market_impact': market_impact,
            'slippage': slippage,
            'total_cost': total_cost
        }

    def calculate_spread_cost(self, notional_value):
        return notional_value * self.config.SPREAD / 2

    def calculate_market_impact(self, quantity, current_price):
        # Simple square-root model
        market_cap = self.config.MARKET_CAP
        daily_volume = self.config.DAILY_VOLUME
        participation_rate = quantity / daily_volume
        return 0.1 * current_price * (quantity / daily_volume) ** 0.5

    def calculate_slippage(self, notional_value):
        return notional_value * self.config.SLIPPAGE


class BacktestPersistence:
    @staticmethod
    def save_backtest(backtester, filename='backtest.pkl'):
        # Create a dictionary of the backtester's important attributes
        backtest_data = {
            'config': backtester.config,
            'historical_data': backtester.historical_data,
            'portfolio': backtester.portfolio,
            'trades': backtester.trades,
            'performance_metrics': backtester.performance_metrics,
            'strategy_performance': backtester.strategy_performance,
            'portfolio_values': backtester.portfolio_values,
            'transaction_costs': backtester.transaction_costs
        }
        with open(filename, 'wb') as f:
            pickle.dump(backtest_data, f)
        print(f"Backtest saved: {filename}")

    @staticmethod
    def load_backtest(filename='backtest.pkl'):
        with open(filename, 'rb') as f:
            backtest_data = pickle.load(f)

        # Recreate the backtester object
        backtester = Backtester(backtest_data['config'], backtest_data['historical_data'])
        backtester.portfolio = backtest_data['portfolio']
        backtester.trades = backtest_data['trades']
        backtester.performance_metrics = backtest_data['performance_metrics']
        backtester.strategy_performance = backtest_data['strategy_performance']
        backtester.portfolio_values = backtest_data['portfolio_values']
        backtester.transaction_costs = backtest_data['transaction_costs']

        print(f"Backtest loaded: {filename}")
        return backtester

class BacktestReport:
    def __init__(self, backtester):
        self.backtester = backtester

    def generate_report(self, output_file='backtest_report.html'):
        report = f"""
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            <h2>Performance Metrics</h2>
            {self._generate_metrics_table()}
            <h2>Portfolio Value Over Time</h2>
            {self._generate_portfolio_chart()}
            <h2>Strategy Comparison</h2>
            {self._generate_strategy_comparison()}
            <h2>Transaction Cost Analysis</h2>
            {self._generate_transaction_cost_analysis()}
        </body>
        </html>
        """
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report generated: {output_file}")

    def _generate_metrics_table(self):
        results = self.backtester.get_results()
        table = "<table>"
        table += "<tr><th>Metric</th><th>Value</th></tr>"
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                table += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
        table += "</table>"
        return table

    def _generate_portfolio_chart(self):
        plt.figure(figsize=(10, 6))
        plt.plot([pv for _, pv in self.backtester.portfolio_values])
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        img_path = 'portfolio_chart.png'
        plt.savefig(img_path)
        plt.close()
        return f'<img src="{img_path}" alt="Portfolio Chart">'

    def _generate_strategy_comparison(self):
        strategy_performance = self.backtester.calculate_strategy_performance()
        strategy_returns = pd.DataFrame(strategy_performance).T
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=strategy_returns)
        plt.title('Strategy Returns Comparison')
        plt.xlabel('Strategy')
        plt.ylabel('Returns')
        img_path = 'strategy_comparison.png'
        plt.savefig(img_path)
        plt.close()
        return f'<img src="{img_path}" alt="Strategy Comparison">'

    def _generate_transaction_cost_analysis(self):
        costs = pd.DataFrame([trade['commission'] for trade in self.backtester.trades], columns=['Commission'])
        plt.figure(figsize=(10, 6))
        costs.plot(kind='bar')
        plt.title('Transaction Costs')
        plt.xlabel('Trade')
        plt.ylabel('Commission')
        img_path = 'transaction_costs.png'
        plt.savefig(img_path)
        plt.close()
        return f'<img src="{img_path}" alt="Transaction Costs">'
class Backtester(multiprocessing.Process):  # or threading.Thread
    def __init__(self, config, historical_data: pd.DataFrame, result_queue):
        super().__init__()
        self.max_concurrent_backtesters = config.BASE_PARAMS.get('MAX_CONCURRENT_BACKTESTERS')
        self.max_concurrent_strategies = config.BASE_PARAMS.get('MAX_CONCURRENT_STRATEGIES')
        self.progress_update_interval = config.BASE_PARAMS.get('PROGRESS_UPDATE_INTERVAL')  
        # Validate input data
        if historical_data.empty:
            raise ValueError("Historical data is empty")
            
        if not all(col in historical_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("Missing required columns in historical data")
            
        print(f"Initializing backtester with {len(historical_data)} data points")
        
        self.config = config
        print(f"Backtester config BASE_PARAMS: {hasattr(self.config, 'BASE_PARAMS')}")
        self.historical_data = historical_data        
        # Initialize strategies with proper strategy objects
        self.strategies = []
        for timeframe in TimeFrame:
            strategy = TrendFollowingStrategy(
                self.config,
                time.time(),
                timeframe,
                self.config.ADAPTIVE_PARAMS['TREND_FOLLOWING_PARAMS']
            )
            self.strategies.append(strategy)
        
        self.events = deque()
        self.current_position = 0
        self.cash = config.BASE_PARAMS['INITIAL_CAPITAL']
        self.portfolio_value = self.cash
        self.strategies = {
            TimeFrame.SHORT_TERM: {},
            TimeFrame.MID_TERM: {},
            TimeFrame.LONG_TERM: {},
            TimeFrame.SEASONAL_TERM: {}
        }
        self.total_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.impermanent_loss = 0.0
        self.strategy_types = [
            TrendFollowingStrategy,
            MeanReversionStrategy,
            MomentumStrategy,
            VolatilityStrategy,
            BreakoutStrategy,
            StatisticalArbitrageStrategy,
            SentimentAnalysisStrategy,
        ]
        for time_frame in TimeFrame:
            self.strategies[time_frame] = {}
        self.trades = []
        self.portfolio_values = []
        self.stop_event = multiprocessing.Event()
        self.strategy_generator = StrategyGenerator(config)
        self.strategy_selector = StrategySelector(config)
        self.market_simulator = MarketSimulator(config, strategy)
        self.risk_manager = RiskManager(config)
        self.strategy_optimizer = StrategyOptimizer(config, MarketSimulator, self.strategies)
        self.api_call_manager = APICallManager()
        self.strategy_manager = StrategyManager(config, use_ai_selection=False)
        self.current_strategy = None
        self.asyncio = asyncio
        # Progress tracking
        self.total_events = len(historical_data)
        self.processed_events = 0

    async def run(self):
        try:
            await self.initialize()
            
            for timestamp, data in self.historical_data.iterrows():
                self.processed_events += 1
                if self.processed_events % self.progress_update_interval == 0:
                    progress = (self.processed_events / self.total_events) * 100
                    print(f"Backtesting Progress: {progress:.2f}% ({self.processed_events}/{self.total_events})")
                
                market_event = self.create_market_event(timestamp, data)
                await self.process_event(market_event)
                
            # Calculate final results
            results = await self.get_results()
            self.results_queue.put(results)
            
            return results
        except Exception as e:
            print(f"Error during backtester run: {e}")
            raise
            
    async def initialize(self):
        """Initialize backtesting environment"""
        self.portfolio_value = self.config.BASE_PARAMS["INITIAL_CAPITAL"]
        self.positions = {}
        self.trades = []
        await self.strategy_manager.initialize_strategies(
            self.strategy_generator,
            self.historical_data
        )

    async def stop(self):
        self.stop_event.set()
        start_time = time.time()
        processed_rows = 0
    
        try:
            while not self.stop_event.is_set():
                for timestamp, row in self.historical_data.iterrows():
                    processed_rows += 1
                    if processed_rows % 100000 == 0:
                        print(f"Processed {processed_rows}/{len(self.historical_data)} rows")
            
                    if self.stop_event.is_set():
                        break
                
                market_event = MarketEvent(timestamp, row.to_dict())
                self.events.append(market_event)
                await self.process_events()  # Now this will work correctly
                self.update_portfolio_value(timestamp)
                self.portfolio_values.append((timestamp, self.portfolio_value))

        except Exception as e:
            print(f"Error during backtest: {e}")
            raise

        print(f"Backtest completed in {time.time() - start_time:.2f} seconds")
    def get_recent_data(self, timestamp_or_event, lookback_periods=100):
        """Get recent market data up to the specified timestamp"""
        # Handle both dictionary events and direct timestamps
        if isinstance(timestamp_or_event, dict):
            timestamp = timestamp_or_event['timestamp']
        else:
            timestamp = timestamp_or_event
        
        # Find the index of the current timestamp
        current_idx = self.historical_data.index.get_loc(timestamp)
    
        # Get data from lookback_periods ago up to current timestamp
        start_idx = max(0, current_idx - lookback_periods)
        recent_data = self.historical_data.iloc[start_idx:current_idx + 1]
    
        return recent_data
    
    def update_portfolio_value(self, timestamp):
        """Update the current portfolio value based on positions and current market price"""
        current_price = self.historical_data.loc[timestamp, 'close']
        self.portfolio_value = self.cash + (self.current_position * current_price)
        return self.portfolio_value

    async def execute_trade(self, signal, event):
        try:
            price = self.historical_data.loc[event.timestamp, 'close']
            commission = self.calculate_commission(quantity, price)
        
            if signal > 0:
                self.cash -= (price * quantity + commission)
                self.current_position += quantity
            elif signal < 0:
                self.cash += (price * quantity - commission)
                self.current_position -= quantity
            
            trade = {
                'timestamp': event.timestamp,
                'direction': 'BUY' if signal > 0 else 'SELL',
                'quantity': quantity,
                'price': price,
                'commission': commission
            }
            self.trades.append(trade)
            self.portfolio_values.append((event.timestamp, self.cash + self.current_position * price))
        
        except Exception as e:
            print(f"Error simulating trade: {e}")
    
    async def process_events(self):
        events_processed = 0
        while self.events:
            event = self.events.popleft()
            events_processed += 1
            
            if events_processed % 100 == 0:
                print(f"Processing event batch {events_processed}")
                
            if event.type == EventType.MARKET:
                await self.handle_market_event(event)
            elif event.type == EventType.SIGNAL:
                await self.handle_signal_event(event)
            elif event.type == EventType.ORDER:
                await self.handle_order_event(event)
            elif event.type == EventType.FILL:
                await self.handle_fill_event(event)

    async def handle_market_event(self, event):
        """Handle market update events"""
        recent_data = self.get_recent_data(event['timestamp'])
        param_mapping = {
            TrendFollowingStrategy: 'TREND_FOLLOWING_PARAMS',
            MeanReversionStrategy: 'MEAN_REVERSION_PARAMS',
            MomentumStrategy: 'MOMENTUM_PARAMS',
            VolatilityStrategy: 'VOLATILITY_CLUSTERING_PARAMS',
            StatisticalArbitrageStrategy: 'STATISTICAL_ARBITRAGE_PARAMS',
            SentimentAnalysisStrategy: 'SENTIMENT_ANALYSIS_PARAMS',
            BreakoutStrategy: 'BREAKOUT_PARAMS'
        }

    
        # Track active strategy count across all timeframes
        total_active_strategies = 0
    
        for strategy_type in self.strategy_types:
            for timeframe in TimeFrame:
                if total_active_strategies >= 4:  # One for each timeframe
                    break
                
                param_name = param_mapping[strategy_type]
            
                total_active_strategies += 1  # Increment before printing
            
                if self.processed_events % self.progress_update_interval == 0:
                    print(f"Strategy {strategy_type.__name__} on {timeframe} at {event['timestamp']}")
                    print(f"Active Strategies: {total_active_strategies}/4")
            
                strategy = strategy_type(
                    self.config,
                    time.time(),
                    timeframe,
                    self.config.ADAPTIVE_PARAMS[param_name]
                )
                signal = strategy.generate_signal(recent_data)
                if signal != 0:
                    signal_event = SignalEvent(
                        timestamp=event['timestamp'],
                        symbol=self.config.BASE_PARAMS['SYMBOL'],
                        signal=signal
                    )
                    self.events.append(signal_event)

    async def handle_signal_event(self, event: SignalEvent):
        order_type = 'MARKET'
        quantity, stop_loss, take_profit = await self.risk_manager.apply_risk_management(
            event.signal,
            self.portfolio_value,
            self.historical_data.loc[event.timestamp, 'close'],
            self.get_recent_data(event.timestamp)
        )
        if quantity != 0:
            self.events.append(OrderEvent(event.timestamp, event.symbol, order_type, quantity, 'BUY' if event.signal > 0 else 'SELL', stop_loss, take_profit))

    async def handle_order_event(self, event: OrderEvent):
        fill_cost = self.historical_data.loc[event.timestamp, 'close']
        commission = await self.calculate_commission(event.quantity, fill_cost)
        market_impact = await self.risk_manager.calculate_market_impact(event.quantity, fill_cost)
        total_cost = fill_cost + commission + market_impact
        self.events.append(FillEvent(event.timestamp, event.symbol, 'BACKTEST', event.quantity, event.direction, total_cost, commission))

    async def handle_fill_event(self, event: FillEvent):
        if event.direction == 'BUY':
            self.cash -= (event.fill_cost * event.quantity + event.commission)
            self.current_position += event.quantity
        else:  # SELL
            self.cash += (event.fill_cost * event.quantity - event.commission)
            self.current_position -= event.quantity

        await self.record_trade({
            'timestamp': event.timestamp,
            'direction': event.direction,
            'quantity': event.quantity,
            'price': event.fill_cost,
            'commission': event.commission
        })
    
    def create_market_event(self, timestamp, data):
        return {
            'timestamp': timestamp,
            'price': data['close'],
            'volume': data['volume'],
            'type': 'market_update'
        }
    async def process_event(self, event):
        """Process individual market events and generate signals"""
        if event['type'] == 'market_update':
            # Access dictionary using square bracket notation
            recent_data = self.get_recent_data(event['timestamp'])
            await self.handle_market_event(event)
        
            # Update portfolio using same access method
            self.update_portfolio_value(event['timestamp'])
            self.portfolio_values.append((event['timestamp'], self.portfolio_value))
        elif event.type == EventType.SIGNAL:
            await self.handle_signal_event(event)
        elif event.type == EventType.ORDER:
            await self.handle_order_event(event)
        elif event.type == EventType.FILL:
            await self.handle_fill_event(event)

    async def record_trade(self, trade_data):
        self.trades.append(trade_data)
        await self.update_strategy(trade_data['timestamp'])

    async def calculate_commission(self, quantity: int, price: float) -> float:
        return self.config.COMMISSION_RATE * quantity * price

    async def update_strategy(self, timestamp):
        if not await self.api_call_manager.can_make_call():
            wait_time = await self.api_call_manager.time_until_reset()
        
            if wait_time > self.config.BACKTEST_DURATION:
                self.calculate_performance_metrics()
                results = self.get_results()
                self.result_queue.put(results)
                self.stop_event.set()
                return
            
            print(f"API call limit reached. Waiting for {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)
            return await self.update_strategy(timestamp)

        for time_frame in TimeFrame:
            await self.api_call_manager.record_call()
            await self.strategy_manager.update_strategies(
                self.get_recent_data(timestamp), 
                time_frame, 
                self.strategy_generator
            )

        recent_data = self.get_recent_data(timestamp)




        self.current_strategy = await self.strategy_manager.select_best_strategies(recent_data, self.portfolio_value)














    def calculate_strategy_performance(self, time_frame: TimeFrame):
        return {name: strategy.calculate_performance(self.trades) for name, strategy in self.strategies[time_frame].items()}

    def calculate_performance_metrics(self):
        returns = pd.Series([pv for _, pv in self.portfolio_values]).pct_change()
        self.total_return = (self.portfolio_value - self.config.BASE_PARAMS['INITIAL_CAPITAL']) / self.config.BASE_PARAMS['INITIAL_CAPITAL']
        self.sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        self.max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        
        # Add check for empty trades list
        if len(self.trades) > 0:
            self.win_rate = sum(1 for trade in self.trades if trade['direction'] == 'SELL' and trade['price'] > trade['price']) / len(self.trades)
            self.profit_factor = sum(trade['price'] - trade['price'] for trade in self.trades if trade['direction'] == 'SELL' and trade['price'] > trade['price']) / abs(sum(trade['price'] - trade['price'] for trade in self.trades if trade['direction'] == 'SELL' and trade['price'] < trade['price']))
        else:
            self.win_rate = 0
            self.profit_factor = 0
        

        self.impermanent_loss = self.calculate_impermanent_loss()

    def calculate_impermanent_loss(self):
        # Simplified impermanent loss calculation
        if len(self.trades) < 2:
            return 0

        initial_price = self.trades[0]['price']
        final_price = self.trades[-1]['price']
        price_ratio = final_price / initial_price

        impermanent_loss = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
        return impermanent_loss

    async def monte_carlo_simulation(self, num_simulations: int = 1000) -> Dict[str, List[float]]:
        returns = pd.Series([pv for _, pv in self.portfolio_values]).pct_change().dropna()
    
        # Check if we have enough data points
        if len(returns) == 0:
            return {
                'final_portfolio_values': [self.config.BASE_PARAMS['INITIAL_CAPITAL']],
                'max_drawdowns': [0.0],
                'sharpe_ratios': [0.0]
            }

        simulation_results = {
            'final_portfolio_values': [],
            'max_drawdowns': [],
            'sharpe_ratios': []
        }

        for _ in range(num_simulations):
            simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
            cumulative_returns = (1 + simulated_returns).cumprod()
            portfolio_values = pd.Series(self.config.BASE_PARAMS['INITIAL_CAPITAL'] * cumulative_returns)

            if len(portfolio_values) > 0:
                final_value = portfolio_values.iloc[-1]
                max_drawdown = (portfolio_values / portfolio_values.cummax() - 1).min()
                sharpe_ratio = np.sqrt(252) * simulated_returns.mean() / simulated_returns.std()

                simulation_results['final_portfolio_values'].append(final_value)
                simulation_results['max_drawdowns'].append(max_drawdown)
                simulation_results['sharpe_ratios'].append(sharpe_ratio)

            await asyncio.sleep(0)  # Allow other coroutines to run

        return simulation_results    
        
    def run_mini_backtest(self, strategy):













        mini_trades = []
        for i in range(len(self.historical_data) - 1):
            signal = strategy.generate_signal(self.historical_data.iloc[i:i+2])
            if signal != 0:
                trade = {
                    'timestamp': self.historical_data.index[i+1],
                    'direction': 'BUY' if signal > 0 else 'SELL',
                    'quantity': 1,  # Simplified quantity
                    'price': self.historical_data.iloc[i+1]['close'],
                    'commission': self.calculate_commission(1, self.historical_data.iloc[i+1]['close'])
                }
                mini_trades.append(trade)
        return mini_trades

    def reset(self):
        self.current_position = 0
        self.cash = self.config.BASE_PARAMS['INITIAL_CAPITAL']
        self.portfolio_value = self.cash
        self.trades = []
        self.portfolio_values = []
        self.events.clear()

    def calculate_var(self, confidence_level=0.95):
        returns = pd.Series([pv for _, pv in self.portfolio_values]).pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        var = np.percentile(returns, 100 * (1 - confidence_level))
        return self.portfolio_value * var

    def calculate_cvar(self, confidence_level=0.95):
        returns = pd.Series([pv for _, pv in self.portfolio_values]).pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        var = np.percentile(returns, 100 * (1 - confidence_level))
        cvar = returns[returns <= var].mean()
        return self.portfolio_value * cvar

    async def get_results(self) -> Dict[str, Any]:
        mc_results = await self.monte_carlo_simulation()
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'final_portfolio_value': self.portfolio_value,
            'trades': self.trades,
            'var': self.calculate_var(),
            'cvar': self.calculate_cvar(),
            'impermanent_loss': self.impermanent_loss,
            'monte_carlo': {
                'mean_final_value': np.mean(mc_results['final_portfolio_values']),
                'median_final_value': np.median(mc_results['final_portfolio_values']),
                'mean_max_drawdown': np.mean(mc_results['max_drawdowns']),
                'mean_sharpe_ratio': np.mean(mc_results['sharpe_ratios'])
            }
        }

def print_interim_results(results):
    print("\nInterim Backtest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Current Portfolio Value: ${results['final_portfolio_value']:.2f}")

def run_backtest(config, historical_data: pd.DataFrame, result_queue: multiprocessing.Queue) -> None:
    backtester = Backtester(config, historical_data, result_queue)
    backtester.start()
    
    interim_results = []
    
    try:
        while True:
            time.sleep(config.BACKTEST_UPDATE_INTERVAL)
            if not backtester.is_alive():
                break
            
            # Retrieve and store interim results
            try:
                results = result_queue.get_nowait()
                interim_results.append(results)
                print_interim_results(results)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("Backtest interrupted by user.")
    finally:
        backtester.stop()
        backtester.join()
            

    print("Backtest completed.")
    
    # Get final results
    final_results = result_queue.get()
    
    # Generate the backtest report
    report = BacktestReport(backtester)
    report.generate_report()



