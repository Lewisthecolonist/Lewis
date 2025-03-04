import json
import asyncio
from strategy_generator import StrategyGenerator
from strategy import Strategy
from typing import Dict, Callable
import pandas as pd
from strategy import TimeFrame
from config import Config
import aiofiles

class StrategyFactory:
    def __init__(self, config= Config, config_file_path= 'strategies.json'):
        self.config = config
        self.strategy_generator = StrategyGenerator(config)
        self.strategies = {}
        self.signal_methods = {}
        self.config_file_path = config_file_path
        self.load_strategy_config()
        asyncio.create_task(self.monitor_strategies())

    def load_strategy_config(self):
        try:
            with open(self.config_file_path, 'r') as f:
                content = f.read().strip()
                if content:
                    strategy_config = json.loads(content)
                    # Convert parameters to frozenset for hashing
                    parameters = frozenset(sorted(strategy_config.get('parameters', {}).items()))

                    return strategy_config
                return {}
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error loading strategy config from {self.config_file_path}. Creating new empty configuration.")
            return {}

    async def update_strategy(self, strategy_name: str, strategy: Strategy):
        # Create string key
        strategy_key = f"{strategy_name}_{strategy.time_frame}_{hash(frozenset(strategy.parameters.items()))}"
    
        # Store in memory
        self.strategies[strategy_key] = strategy
        self.signal_methods[strategy_key] = self.create_signal_method(strategy)
    
        # Load current strategies
        try:
            async with aiofiles.open(self.config_file_path, 'r') as f:
                content = await f.read()
                existing_strategies = json.loads(content) if content else {}
        except FileNotFoundError:
            existing_strategies = {}
    
        # Add new strategy
        existing_strategies[strategy_key] = {
            'name': strategy.name,
            'time_frame': strategy.time_frame.value,
            'parameters': dict(strategy.parameters),
            'favored_patterns': list(strategy.favored_patterns)
        }
    
        # Save all strategies
        async with aiofiles.open(self.config_file_path, 'w') as f:
            await f.write(json.dumps(existing_strategies, indent=4))

    def create_signal_method(self, strategy: Strategy) -> Callable:
        def signal_method(market_data: pd.DataFrame) -> float:
            return strategy.generate_signal(market_data)
        return signal_method

    def get_signal_method(self, strategy_name: str) -> Callable:
        return self.signal_methods.get(strategy_name, self.default_signal)

    @staticmethod
    def default_signal(market_data: pd.DataFrame) -> float:
        return 0

    async def monitor_strategies(self):
        while True:
            await asyncio.sleep(60)  # Check every minute
            current_strategies = set(self.strategies.keys())
            loaded_config = self.load_strategy_config()
            config_strategies = set(loaded_config.keys()) if loaded_config else set()

            if current_strategies != config_strategies:
                self.save_strategy_config()
                print("Strategy configuration updated.")
    def delete_strategy(self, strategy_name: str):
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            del self.signal_methods[strategy_name]
            self.save_strategy_config()
            print(f"Strategy {strategy_name} deleted.")

    def create_strategy(self, strategy_config: Dict) -> Strategy:
        parameters = frozenset(sorted(strategy_config.get('parameters', {}).items()))
        return Strategy(
            name=strategy_config['name'],
            parameters=parameters,
            favored_patterns=tuple(strategy_config['favored_patterns']),
            time_frame=TimeFrame(strategy_config['time_frame'])
        )
    def _get_timeframe_parameters(self, timeframe: TimeFrame):
        base_params = self.config.get_base_params()
        timeframe_specific = {
            TimeFrame.SHORT_TERM: {'interval': 'min', 'lookback': 60},
            TimeFrame.MID_TERM: {'interval': 'h', 'lookback': 504},
            TimeFrame.LONG_TERM: {'interval': 'D', 'lookback': 365},
            TimeFrame.SEASONAL_TERM: {'interval': 'ME', 'lookback': 1460}
        }
        return {**base_params, **timeframe_specific[timeframe]}
    
    async def save_strategy(self, strategy: Strategy):
        # Create unique string key
        strategy_key = f"{strategy.name}_{strategy.time_frame}_{hash(frozenset(strategy.parameters.items()))}"
    
        # Store strategy and its signal method
        self.strategies[strategy_key] = strategy
        self.signal_methods[strategy_key] = self.create_signal_method(strategy)
    
        # Convert to serializable format
        serializable_strategies = {
            key: {
                'name': strat.name,
                'time_frame': strat.time_frame.value,
                'parameters': dict(strat.parameters),
                'favored_patterns': list(strat.favored_patterns)
            }
            for key, strat in self.strategies.items()
        }
    
        # Save to file asynchronously
        async with aiofiles.open(self.config_file_path, 'w') as f:
            await f.write(json.dumps(serializable_strategies, indent=4))



