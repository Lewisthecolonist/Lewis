import google.generativeai as genai
import pandas as pd
import numpy as np
import os
import json
from decimal import Decimal
from functools import lru_cache
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from strategy import (
    Strategy,
    TimeFrame,
    TrendFollowingStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    VolatilityStrategy,
    StatisticalArbitrageStrategy,
    SentimentAnalysisStrategy
)
import logging
import asyncio
from api_call_manager import APICallManager
import time

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_generator.log'),
        logging.StreamHandler()
    ]
)

VALID_STRATEGY_PARAMETERS = {
    'trend_following': [
        'MOVING_AVERAGE_SHORT',
        'MOVING_AVERAGE_LONG',
        'TREND_STRENGTH_THRESHOLD',
        'TREND_CONFIRMATION_PERIOD',
        'MOMENTUM_FACTOR',
        'BREAKOUT_LEVEL',
        'TRAILING_STOP'
    ],
    'mean_reversion': [
        'MEAN_WINDOW',
        'STD_MULTIPLIER',
        'MEAN_REVERSION_THRESHOLD',
        'ENTRY_DEVIATION',
        'EXIT_DEVIATION',
        'BOLLINGER_PERIOD',
        'BOLLINGER_STD'
    ],
    'momentum': [
        'MOMENTUM_PERIOD',
        'MOMENTUM_THRESHOLD',
        'RSI_PERIOD',
        'RSI_OVERBOUGHT',
        'RSI_OVERSOLD',
        'ACCELERATION_FACTOR',
        'MAX_ACCELERATION',
        'MACD_FAST',
        'MACD_SLOW',
        'MACD_SIGNAL'
    ],
    'breakout': [
        'BREAKOUT_PERIOD',
        'BREAKOUT_THRESHOLD',
        'VOLUME_CONFIRMATION_MULT',
        'CONSOLIDATION_PERIOD',
        'SUPPORT_RESISTANCE_LOOKBACK',
        'BREAKOUT_CONFIRMATION_CANDLES',
        'ATR_PERIOD'
    ],
    'volatility_clustering': [
        'VOLATILITY_WINDOW',
        'HIGH_VOLATILITY_THRESHOLD',
        'LOW_VOLATILITY_THRESHOLD',
        'GARCH_LAG',
        'ATR_MULTIPLIER',
        'VOLATILITY_BREAKOUT_THRESHOLD',
        'VOLATILITY_MEAN_PERIOD'
    ],
    'statistical_arbitrage': [
        'LOOKBACK_PERIOD',
        'Z_SCORE_THRESHOLD',
        'CORRELATION_THRESHOLD',
        'HALF_LIFE',
        'HEDGE_RATIO',
        'ENTRY_THRESHOLD',
        'EXIT_THRESHOLD',
        'WINDOW_SIZE',
        'MIN_CORRELATION',
        'COINTEGRATION_THRESHOLD'
    ],
    'sentiment_analysis': [
        'POSITIVE_SENTIMENT_THRESHOLD',
        'NEGATIVE_SENTIMENT_THRESHOLD',
        'SENTIMENT_WINDOW',
        'SENTIMENT_IMPACT_WEIGHT',
        'NEWS_IMPACT_DECAY',
        'SENTIMENT_SMOOTHING_FACTOR',
        'SENTIMENT_VOLUME_THRESHOLD',
        'SENTIMENT_MOMENTUM_PERIOD'
    ]
}

PARAMETER_RANGES = {
    # Trend Following
    'MOVING_AVERAGE_SHORT': (2, 500),
    'MOVING_AVERAGE_LONG': (5, 1000),
    'TREND_STRENGTH_THRESHOLD': (0.001, 0.5),
    'TREND_CONFIRMATION_PERIOD': (1, 100),
    'MOMENTUM_FACTOR': (0.1, 5.0),
    'BREAKOUT_LEVEL': (0.01, 0.5),
    'TRAILING_STOP': (0.01, 0.3),

    # Mean Reversion
    'MEAN_WINDOW': (2, 500),
    'STD_MULTIPLIER': (0.1, 10.0),
    'MEAN_REVERSION_THRESHOLD': (0.01, 0.5),
    'ENTRY_DEVIATION': (0.01, 0.5),
    'EXIT_DEVIATION': (0.01, 0.5),
    'BOLLINGER_PERIOD': (5, 500),
    'BOLLINGER_STD': (0.5, 5.0),

    # Momentum
    'MOMENTUM_PERIOD': (1, 200),
    'MOMENTUM_THRESHOLD': (0.01, 0.5),
    'RSI_PERIOD': (2, 100),
    'RSI_OVERBOUGHT': (50, 90),
    'RSI_OVERSOLD': (10, 50),
    'ACCELERATION_FACTOR': (0.01, 0.5),
    'MAX_ACCELERATION': (0.1, 1.0),
    'MACD_FAST': (5, 50),
    'MACD_SLOW': (10, 200),
    'MACD_SIGNAL': (5, 50),

    # Breakout
    'BREAKOUT_PERIOD': (5, 500),
    'BREAKOUT_THRESHOLD': (0.01, 0.5),
    'VOLUME_CONFIRMATION_MULT': (1.0, 10.0),
    'CONSOLIDATION_PERIOD': (5, 100),
    'SUPPORT_RESISTANCE_LOOKBACK': (10, 500),
    'BREAKOUT_CONFIRMATION_CANDLES': (1, 20),
    'ATR_PERIOD': (5, 100),

    # Volatility Clustering
    'VOLATILITY_WINDOW': (5, 500),
    'HIGH_VOLATILITY_THRESHOLD': (0.1, 5.0),
    'LOW_VOLATILITY_THRESHOLD': (0.01, 1.0),
    'GARCH_LAG': (1, 20),
    'ATR_MULTIPLIER': (0.5, 5.0),
    'VOLATILITY_BREAKOUT_THRESHOLD': (0.1, 5.0),
    'VOLATILITY_MEAN_PERIOD': (5, 500),

    # Statistical Arbitrage
    'LOOKBACK_PERIOD': (10, 1000),
    'Z_SCORE_THRESHOLD': (0.1, 10.0),
    'CORRELATION_THRESHOLD': (0.1, 1.0),
    'HALF_LIFE': (1, 100),
    'HEDGE_RATIO': (0.1, 10.0),
    'ENTRY_THRESHOLD': (0.1, 5.0),
    'EXIT_THRESHOLD': (0.1, 5.0),
    'WINDOW_SIZE': (10, 1000),
    'MIN_CORRELATION': (0.1, 1.0),
    'COINTEGRATION_THRESHOLD': (0.01, 0.5),

    # Sentiment Analysis
    'POSITIVE_SENTIMENT_THRESHOLD': (0.5, 1.0),
    'NEGATIVE_SENTIMENT_THRESHOLD': (0.0, 0.5),
    'SENTIMENT_WINDOW': (1, 100),
    'SENTIMENT_IMPACT_WEIGHT': (0.0, 1.0),
    'NEWS_IMPACT_DECAY': (0.1, 1.0),
    'SENTIMENT_SMOOTHING_FACTOR': (0.1, 1.0),
    'SENTIMENT_VOLUME_THRESHOLD': (0.5, 5.0),
    'SENTIMENT_MOMENTUM_PERIOD': (1, 100)
}

@dataclass
class MarketMetrics:
    """Container for calculated market metrics"""
    rsi: float
    volatility: float
    trend_strength: float
    volume_profile: float
    moving_averages: Dict[str, float]
    support_resistance: Dict[str, float]

class StrategyValidationError(Exception):
    """Custom exception for strategy validation errors"""
    pass

class MarketState(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    CONSOLIDATING = "CONSOLIDATING"

class StrategyGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_logging()
        self._initialize_ai()
        self.api_call_manager = APICallManager()
        self.strategy_cache = {}
        self.market_metrics_cache = {}
        genai.configure(api_key=os.environ['GOOGLE_AI_API_KEY'])
        self.model = genai.GenerativeModel('gemini-pro')
        
    def _setup_logging(self):
        """Configure detailed logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
    def _initialize_ai(self):
        """Initialize AI model with error handling"""
        try:
            api_key = os.environ.get('GOOGLE_AI_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_AI_API_KEY not found in environment variables")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            self.logger.error(f"AI initialization failed: {e}")
            raise

    def calculate_market_metrics(self, data: pd.DataFrame) -> MarketMetrics:
        """Calculate comprehensive market metrics"""
        try:
            return MarketMetrics(
                rsi=self._calculate_rsi(data['close']),
                volatility=self._calculate_volatility(data),
                trend_strength=self._calculate_trend_strength(data),
                volume_profile=self._analyze_volume_profile(data),
                moving_averages=self._calculate_moving_averages(data),
                support_resistance=self._find_support_resistance(data)
            )
        except Exception as e:
            self.logger.error(f"Market metrics calculation failed: {e}")
            raise

    def _create_enhanced_prompt(self, market_data: pd.DataFrame, time_frame: TimeFrame) -> str:
        metrics = self.calculate_market_metrics(market_data)
        market_state = self._determine_market_state(metrics)
        volume_profile = self.analyze_volume_profile(market_data)
        support_resistance = self.identify_support_resistance_levels(market_data)
        market_efficiency = self.calculate_market_efficiency_ratio(market_data)
        liquidity_score = self.calculate_liquidity_score(market_data)
    
        prompt = f"""
Create innovative trading strategies optimized for these market conditions:

Market Analysis:
- State: {market_state.value}
- RSI: {metrics.rsi:.2f}
- Volatility: {metrics.volatility:.2f}
- Trend Strength: {metrics.trend_strength:.2f}
- Volume Profile: {volume_profile}
- Market Efficiency: {market_efficiency:.2f}
- Liquidity Score: {liquidity_score:.2f}
- Support Levels: {support_resistance['support_levels']}
- Resistance Levels: {support_resistance['resistance_levels']}

You have complete creative freedom to:
1. Combine multiple strategy patterns
2. Use unconventional parameter combinations
3. Create specialized strategies for current market conditions
4. Experiment with parameter values within technical limits
5. Design adaptive responses to changing market conditions

Response Format:
{{
    "name": "Strategy Name",
    "description": "Detailed strategy logic and adaptation rules",
    "patterns": ["primary_pattern", "secondary_pattern"],
    "parameters": {{
        "param1": "value1",
        "param2": "value2",
        ... // Add as many parameters as needed for your strategy
        "paramN": "valueN"  // Must have at least 2 parameters
    }},
    "timeframe": "{time_frame.value}",
    "market_conditions": {{
        "optimal_volatility": "range",
        "optimal_trend": "description",
        "optimal_liquidity": "range"
    }}
}}

"""
        return prompt

    async def generate_strategies(self, market_data: pd.DataFrame) -> Dict[TimeFrame, List[Strategy]]:
        strategies = {timeframe: [] for timeframe in TimeFrame}
    
        for timeframe in TimeFrame:
            prompt = self._create_enhanced_prompt(market_data, timeframe)
            response = await self.model.generate_content(prompt)
        
            try:    
                strategy_data = json.loads(response.text)
                if self._validate_strategy(strategy_data):
                    strategy = Strategy(
                        name=strategy_data['name'],
                        description=strategy_data['description'],
                        parameters=strategy_data['parameters'],
                        favored_patterns=strategy_data['patterns'],
                    time_frame=timeframe
                    )   
                    strategies[timeframe].append(strategy)
                
                    # Save to factory
                    strategy_key = f"{strategy.name}_{strategy.time_frame}_{hash(frozenset(strategy.parameters.items()))}"
                    await self.strategy_factory.update_strategy(strategy_key, strategy)
        
            except Exception as e:
                self.logger.error(f"Strategy generation error: {e}")
                continue
    
        return strategies

    def _validate_strategy(self, strategy_data: Dict) -> bool:
        try:
            # Check parameter ranges
            for param, value in strategy_data['parameters'].items():
                if param in PARAMETER_RANGES:
                    min_val, max_val = PARAMETER_RANGES[param]
                    if not min_val <= value <= max_val:
                        return False

            # Validate strategy patterns
            if not all(pattern in VALID_STRATEGY_PARAMETERS for pattern in strategy_data['patterns']):
                return False

            # Ensure parameters match patterns
            required_params = set()
            for pattern in strategy_data['patterns']:
                required_params.update(VALID_STRATEGY_PARAMETERS[pattern])
        
            if not all(param in strategy_data['parameters'] for param in required_params):
                return False

            return True
        except Exception as e:
            self.logger.error(f"Strategy validation error: {e}")
            return False

    def _create_fallback_strategy(self, timeframe: TimeFrame) -> Strategy:
        """Create a safe fallback strategy when generation fails"""
        fallback_configs = {
            TimeFrame.SHORT_TERM: {
                'type': MomentumStrategy,
                'params': {'MOMENTUM_PERIOD': 14, 'RSI_PERIOD': 14, 'MOMENTUM_THRESHOLD': 0.02}
            },
            TimeFrame.MID_TERM: {
                'type': MeanReversionStrategy,
                'params': {'MEAN_WINDOW': 20, 'STD_MULTIPLIER': 2.0, 'ENTRY_DEVIATION': 0.02}
            },
            TimeFrame.LONG_TERM: {
                'type': TrendFollowingStrategy,
                'params': {'MOVING_AVERAGE_SHORT': 50, 'MOVING_AVERAGE_LONG': 200, 'TREND_STRENGTH_THRESHOLD': 0.02}
            }
        }
        
        config = fallback_configs.get(timeframe, fallback_configs[TimeFrame.SHORT_TERM])
        return config['type'](self.config, time.time(), timeframe, config['params'])


    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        return data['close'].pct_change().std() * np.sqrt(252)

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        return abs(data['close'].pct_change().mean()) * 100

    def _analyze_volume_profile(self, data: pd.DataFrame) -> float:
        return (data['volume'] * data['close']).mean()

    def _calculate_moving_averages(self, data: pd.DataFrame) -> Dict[str, float]:
        return {
            'sma_20': data['close'].rolling(20).mean().iloc[-1],
            'sma_50': data['close'].rolling(50).mean().iloc[-1],
            'sma_200': data['close'].rolling(200).mean().iloc[-1]
        }

    def _find_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        return {
            'support': data['low'].tail(20).min(),
            'resistance': data['high'].tail(20).max()
        }

    def _determine_market_state(self, metrics: MarketMetrics) -> MarketState:
        if metrics.trend_strength > 0.5:
            return MarketState.TRENDING
        elif metrics.volatility > 0.2:
            return MarketState.VOLATILE
        elif metrics.volatility < 0.1:
            return MarketState.CONSOLIDATING
        return MarketState.RANGING

    def _validate_config(self):
        required_keys = ['API_KEY', 'ADAPTIVE_PARAMS', 'TIMEFRAMES']
        if not all(key in self.config for key in required_keys):
            raise ValueError(f"Missing required config keys: {required_keys}")

    def _resample_data(self, data: pd.DataFrame, timeframe: TimeFrame) -> pd.DataFrame:
        resample_rules = {
            TimeFrame.SHORT_TERM: '1h',
            TimeFrame.MID_TERM: '4h',
            TimeFrame.LONG_TERM: '1D',
            TimeFrame.SEASONAL_TERM: '1W'
        }
    
        return data.resample(resample_rules[timeframe]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    def _save_strategies(self, strategies: Dict[TimeFrame, List[Strategy]]) -> None:
        strategy_data = {
            tf.value: [
                {
                    "type": s.__class__.__name__,
                    "parameters": s.parameters,
                    "timeframe": s.time_frame.value
                } for s in strats
            ] for tf, strats in strategies.items()
        }   
    
        with open('strategies.json', 'w') as f:
            json.dump(strategy_data, f, indent=4)

    def _get_cached_strategies(self, timeframe: TimeFrame) -> List[Strategy]:
        if timeframe in self.strategy_cache:
            if time.time() - self.strategy_cache[timeframe]['timestamp'] < 3600:  # 1 hour cache
                return self.strategy_cache[timeframe]['strategies']
        return []

    def _cache_strategies(self, timeframe: TimeFrame, strategies: List[Strategy]) -> None:
        self.strategy_cache[timeframe] = {
            'timestamp': time.time(),
            'strategies': strategies
        }
    async def _generate_and_validate_strategies(self, market_data: pd.DataFrame, timeframe: TimeFrame) -> List[Strategy]:
        strategies = []
    
        # Create strategies synchronously since they don't need to be awaited
        trend_strategy = self._create_trend_following_strategy(market_data, timeframe)
        mean_rev_strategy = self._create_mean_reversion_strategy(market_data, timeframe)
        momentum_strategy = self._create_momentum_strategy(market_data, timeframe)
    
        strategies.extend(trend_strategy)
        strategies.extend(mean_rev_strategy)
        strategies.extend(momentum_strategy)
    
        return [s for s in strategies if self._validate_strategy_parameters(s)]
    def _create_trend_following_strategy(self, market_data: pd.DataFrame, timeframe: TimeFrame) -> List[Strategy]:
        return [TrendFollowingStrategy(
            self.config,
            time.time(),
            timeframe,
            self.config.ADAPTIVE_PARAMS['TREND_FOLLOWING_PARAMS']
        )]

    def _create_mean_reversion_strategy(self, market_data: pd.DataFrame, timeframe: TimeFrame) -> List[Strategy]:
        return [MeanReversionStrategy(
            self.config,
            time.time(),
            timeframe,
            self.config.ADAPTIVE_PARAMS['MEAN_REVERSION_PARAMS']
        )]

    def _create_momentum_strategy(self, market_data: pd.DataFrame, timeframe: TimeFrame) -> List[Strategy]:
        return [MomentumStrategy(
            self.config,
            time.time(),
            timeframe,
            self.config.ADAPTIVE_PARAMS['MOMENTUM_PARAMS']
        )]
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def analyze_volume_profile(self, market_data: pd.DataFrame) -> Dict[str, float]:
        volume = market_data['volume']
        price = market_data['close']
    
        # Calculate volume-weighted metrics
        vwap = (price * volume).cumsum() / volume.cumsum()
        volume_ma = volume.rolling(window=20).mean()
        relative_volume = volume / volume_ma
    
        # Identify key volume levels
        high_volume_threshold = volume_ma.mean() * 1.5
        low_volume_threshold = volume_ma.mean() * 0.5
    
        return {
            'vwap': vwap.iloc[-1],
            'relative_volume': relative_volume.iloc[-1],
            'high_volume_zones': (volume > high_volume_threshold).sum() / len(volume),
            'low_volume_zones': (volume < low_volume_threshold).sum() / len(volume),
            'volume_trend': (volume_ma.iloc[-1] / volume_ma.iloc[0]) - 1
        }

    def identify_support_resistance_levels(self, market_data: pd.DataFrame) -> Dict[str, List[float]]:
        prices = market_data['close']
        highs = market_data['high']
        lows = market_data['low']
        volumes = market_data['volume']
    
        # Calculate price clusters
        price_clusters = pd.concat([highs, lows])
        hist, bins = np.histogram(price_clusters, bins=50)
    
        # Find support and resistance levels
        support_levels = []
        resistance_levels = []
        for idx in range(1, len(prices)-1):
            if (lows.iloc[idx] < lows.iloc[idx-1] and lows.iloc[idx] < lows.iloc[idx+1]):
                support_levels.append(lows.iloc[idx])
            if (highs.iloc[idx] > highs.iloc[idx-1] and highs.iloc[idx] > highs.iloc[idx+1]):
                resistance_levels.append(highs.iloc[idx])
    
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }

    def calculate_market_efficiency_ratio(self, market_data: pd.DataFrame) -> float:
        price = market_data['close']
        directional_movement = abs(price.iloc[-1] - price.iloc[0])
        path_movement = abs(price.diff()).sum()
        return directional_movement / path_movement if path_movement != 0 else 0

    def calculate_liquidity_score(self, market_data: pd.DataFrame) -> float:
        volume_ma = market_data['volume'].rolling(20).mean()
        spread = (market_data['high'] - market_data['low']) / market_data['close']
        return (volume_ma.iloc[-1] / volume_ma.mean()) * (1 - spread.mean())
