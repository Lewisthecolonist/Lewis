import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from backtester import Backtester
from market_maker import MarketMaker
from wallet import Wallet
import time
from config import Config
import zipfile
import ccxt.async_support as ccxt
import datetime
from decimal import Decimal
from datetime import timedelta
from queue import Queue
from api_call_manager import APICallManager
from inventory_manager import InventoryManager
from risk_manager import RiskManager
from strategy_manager import StrategyManager
from strategy_generator import StrategyGenerator
import logging
from inspect import isawaitable
import os
import json
import numpy as np
class TradingSystem:
    def __init__(self, config, historical_data):
        self.config = config
        print(f"Trading System config BASE_PARAMS: {hasattr(self.config, 'BASE_PARAMS')}")
        self.historical_data = historical_data
        self.results_queue = Queue()
        self.mode = self.select_mode()
        self.backtester = Backtester(self.config, self.historical_data, self.results_queue)
        self.api_call_manager = APICallManager()
        self.exchange = self._initialize_exchange()
        self.wallet = Wallet(self.exchange)
        self.inventory_manager = InventoryManager(self.config, self.exchange)
        self.risk_manager = RiskManager(self.config)
        self.strategy_manager = StrategyManager(self.config, use_ai_selection=True)
        self.market_maker = MarketMaker(self.config, strategy_config_path='strategies.json')
        
        # Performance monitoring
        self.performance_metrics = {}
        self.start_time = None
        self.is_running = False
    
    def _prepare_market_data(self, data):
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return data.reindex(columns=required_columns).ffill()
        
    def _initialize_exchange(self):
        return ccxt.kraken({
            'apiKey': self.config.BASE_PARAMS['KRAKEN_API_KEY'],
            'secret': self.config.BASE_PARAMS['KRAKEN_PRIVATE_KEY'],
            'enableRateLimit': True,
            'headers': {'User-Agent': 'Mozilla/5.0'},
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })

    async def start(self):
        try:
            if self.mode == 1:  # Backtesting only
                await self._run_backtest()
            
            elif self.mode == 2:  # Live trading only
                await self._start_live_trading()
            
            elif self.mode == 3:  # Both modes
                await self._run_backtest()
                await self._start_live_trading()
            
            self.is_running = True
            self.start_time = datetime.now()
        
        except Exception as e:
            logging.error(f"System startup failed: {e}")
            raise

    async def _run_backtest(self):
        backtester = Backtester(self.config, self.historical_data, self.results_queue)
        await backtester.run()
        results = await backtester.get_results()
        await self.process_backtest_results(results)

    async def _start_live_trading(self):
        await self.market_maker.initialize()
        await self.inventory_manager.update_balances()
        await self.strategy_manager.initialize_strategies(
            StrategyGenerator(self.config),
            self.historical_data
        )

    async def main_loop(self):
        while self.is_running:
            try:
                await self._process_market_cycle()
                await self._update_system_state()
                await asyncio.sleep(self.config.CYCLE_INTERVAL)
            except Exception as e:
                logging.error(f"Error in main loop: {e}")

    async def _process_market_cycle(self):
        market_data = await self._fetch_market_data()
        signals = await self.strategy_manager.generate_signals(market_data)
        risk_adjusted_signals = self.risk_manager.adjust_signals(signals)
        
        if await self.api_call_manager.can_make_call():
            await self.market_maker.process_signals(risk_adjusted_signals)

    async def _update_system_state(self):
        await self.inventory_manager.update_balances()
        await self.market_maker.monitor_system_health()
        self._update_performance_metrics()

    def _update_performance_metrics(self):
        current_metrics = {
            'portfolio_value': self.wallet.get_total_value(),
            'win_rate': self.calculate_win_rate(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown()
        }
        self.performance_metrics.update(current_metrics)

    def select_mode(self):
        print("\nSelect mode:")
        print("1. Backtesting only")
        print("2. Live trading only")
        print("3. Both backtesting and live trading")
        while True:
            try:
                mode = int(input("Enter mode (1-3): "))
                if 1 <= mode <= 3:
                    return mode
                print("Invalid mode. Please enter 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    async def run_backtester(self):
        print("Starting backtest...")
        # Single run
        self.backtest_results = await self.loop.run_in_executor(self.executor, self.backtester.run)
        print("Backtest completed, processing results...")
        self.results_queue.put(self.backtest_results)
        await self.process_backtest_results(self.backtest_results)
        print("Results processed")
        await self.stop()
    async def run_market_maker(self):
        end_time = self.start_time + self.market_maker_duration if self.market_maker_duration > 0 else None
        while self.is_running and (end_time is None or time.time() < end_time):
            try:
                market_data = await self.get_latest_market_data()
                self.market_maker.update(market_data)
                await self.market_maker.execute_trades(self.wallet)
                await asyncio.sleep(self.config.MARKET_MAKER_UPDATE_INTERVAL)
            except Exception as e:
                print(f"Error in market maker: {e}")
        await self.stop()
        print("Market maker completed")

    async def process_results_queue(self):
        while self.is_running:
            try:
                if not self.results_queue.empty():
                    result = self.results_queue.get_nowait()
                    await self.process_backtest_results(result)
                else:
                    await asyncio.sleep(0.1)  # Short sleep to prevent busy waiting
            except Exception as e:
                print(f"Error processing results queue: {e}")

    async def stop(self):
        self.is_running = False
        await self.wallet.close()
        self.executor.shutdown(wait=True)
        self.api_call_manager.save_state()

    async def process_backtest_results(self, results):
        if results is None:
            results = await self.backtester.get_results()
    
        if asyncio.iscoroutine(results):
            results_data = await results
        else:
            results_data = results
    
        print("\nDetailed Backtest Results:")
        print(f"Total Return: {results_data['total_return']:.2%}")
        print(f"Sharpe Ratio: {results_data['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results_data['max_drawdown']:.2%}")
        print(f"Win Rate: {results_data['win_rate']:.2%}")
        print(f"Profit Factor: {results_data['profit_factor']:.2f}")
        print(f"Value at Risk: ${abs(results_data['var']):,.2f}")
        print(f"Conditional VaR: ${abs(results_data['cvar']):,.2f}")
        print(f"Impermanent Loss: {results_data['impermanent_loss']:.2%}")
        print("\nMonte Carlo Simulation Results:")
        print(f"Mean Final Value: ${results_data['monte_carlo']['mean_final_value']:,.2f}")
        print(f"Median Final Value: ${results_data['monte_carlo']['median_final_value']:,.2f}")
        print(f"Mean Max Drawdown: {results_data['monte_carlo']['mean_max_drawdown']:.2%}")
        print(f"Mean Sharpe Ratio: {results_data['monte_carlo']['mean_sharpe_ratio']:.2f}")

        return results_data
    def plot_equity_curve(self, equity_curve):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.savefig('equity_curve.png')
        plt.close()
        print("Equity curve plot saved as 'equity_curve.png'")

    def save_results_to_file(self, results):
        import json
        with open('backtest_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("Backtest results saved to 'backtest_results.json'")

    async def get_latest_market_data(self):
        try:
            symbol = self.config.BASE_PARAMS['SYMBOL']
            
            # Fetch ticker data
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Fetch order book
            order_book = await self.exchange.fetch_order_book(symbol)
            
            # Fetch recent trades
            trades = await self.exchange.fetch_trades(symbol, limit=100)
            
            # Fetch OHLCV data for the last 24 hours
            since = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe='1h', since=since)
            
            # Calculate additional metrics
            vwap = sum(trade['price'] * trade['amount'] for trade in trades) / sum(trade['amount'] for trade in trades)
            volatility = self.calculate_volatility([candle[4] for candle in ohlcv])  # Using close prices
            
            market_data = {
                'symbol': symbol,
                'last': Decimal(str(ticker['last'])),
                'bid': Decimal(str(ticker['bid'])),
                'ask': Decimal(str(ticker['ask'])),
                'volume': Decimal(str(ticker['baseVolume'])),
                'timestamp': ticker['timestamp'],
                'vwap': Decimal(str(vwap)),
                'volatility': Decimal(str(volatility)),
                'order_book': {
                    'bids': [[Decimal(str(price)), Decimal(str(amount))] for price, amount in order_book['bids'][:5]],
                    'asks': [[Decimal(str(price)), Decimal(str(amount))] for price, amount in order_book['asks'][:5]]
                },
                'recent_trades': [
                    {
                        'price': Decimal(str(trade['price'])),
                        'amount': Decimal(str(trade['amount'])),
                        'side': trade['side'],
                        'timestamp': trade['timestamp']
                    } for trade in trades[:10]
                ],
                'ohlcv': [
                    {
                        'timestamp': candle[0],
                        'open': Decimal(str(candle[1])),
                        'high': Decimal(str(candle[2])),
                        'low': Decimal(str(candle[3])),
                        'close': Decimal(str(candle[4])),
                        'volume': Decimal(str(candle[5]))
                    } for candle in ohlcv
                ]
            }
            
            return market_data
        
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

    def calculate_volatility(self, prices):
        returns = pd.Series(prices).pct_change().dropna()
        return float(returns.std() * (252 ** 0.5))  # Annualized volatility

    async def initialize_api_call_manager(self):
        self.api_call_manager = APICallManager()
        await self.api_call_manager.load_state()

    async def initialize_strategies(self, market_data):
        try:
            # Ensure we await the strategy generation
            strategies = await self.strategy_generator.generate_strategies(market_data)
            for timeframe, strat_list in strategies.items():
                for strategy in strat_list:
                    self.strategy_manager.add_strategy(timeframe, strategy)
            return strategies
        except Exception as e:
            print.error(f"Error initializing strategies: {str(e)}")
            raise