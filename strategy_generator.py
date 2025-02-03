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
    PatternRecognitionStrategy,
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
        """Create a highly structured and specific prompt for the AI"""
        metrics = self.calculate_market_metrics(market_data)
        market_state = self._determine_market_state(metrics)
        
        prompt = f"""
STRICT INSTRUCTION: Generate exactly 2 trading strategies in valid JSON format.
Time Frame: {time_frame.value}

Market Context:
- State: {market_state.value}
- RSI: {metrics.rsi:.2f}
- Volatility: {metrics.volatility:.2f}
- Trend Strength: {metrics.trend_strength:.2f}

Required JSON Structure(remember this is simply an example although the notes are to be followed strictly):
[
    {{
        "name": "Strategy Name",
        "type": "<MUST BE ONE OF: trend_following, mean_reversion, momentum, breakout, volatility_clustering, statistical_arbitrage, sentiment_analysis>",
        "parameters": {{
            // EXACTLY 3-5 parameters from the valid parameter list
            // All parameter values must be numeric
            // No null or undefined values allowed
        }},
        "timeframe": "{time_frame.value}",
        "validation": {{
            "min_data_points": <integer>,
            "risk_level": <1-5>,
            "complexity": <1-5>
        }}
    }}
]

Parameter Constraints:
{json.dumps(VALID_STRATEGY_PARAMETERS, indent=2)}

CRITICAL REQUIREMENTS:
1. Response must be valid JSON
2. Parameters must match strategy type
3. All numeric values must be reasonable and within standard ranges
4. No missing or null values allowed
5. Strategies must be appropriate for the current market state
"""
        return prompt

    async def generate_strategies(self, market_data: pd.DataFrame) -> Dict[TimeFrame, List[Strategy]]:
        """Generate strategies with comprehensive validation and error handling"""
        strategies = {timeframe: [] for timeframe in TimeFrame}
        
        for timeframe in TimeFrame:
            try:
                resampled_data = self._resample_data(market_data, timeframe)
                if resampled_data.empty:
                    raise ValueError(f"No data available for {timeframe}")
                
                cached_strategies = self._get_cached_strategies(timeframe)
                if cached_strategies:
                    strategies[timeframe] = cached_strategies
                    continue
                
                new_strategies = await self._generate_and_validate_strategies(resampled_data, timeframe)
                strategies[timeframe] = new_strategies
                self._cache_strategies(timeframe, new_strategies)
                
            except Exception as e:
                self.logger.error(f"Strategy generation failed for {timeframe}: {e}")
                strategies[timeframe] = [self._create_fallback_strategy(timeframe)]
                
        self._save_strategies(strategies)
        return strategies

    def _validate_strategy_parameters(self, strategy: Strategy) -> bool:
        """Validate strategy parameters against defined constraints"""
        strategy_type = strategy.__class__.__name__.lower().replace('strategy', '')
        parameters = strategy.parameters
    
        valid_params = VALID_STRATEGY_PARAMETERS.get(strategy_type, [])
    
        # Parameter validation rules
        validation_rules = {
            'moving_average': lambda x: 1 <= x <= 500,
            'period': lambda x: 1 <= x <= 100,
            'threshold': lambda x: 0 < x < 1,
            'multiplier': lambda x: 0 < x <= 5,
            'window': lambda x: 1 <= x <= 500
        }
    
        try:
            # Validate parameter names
            if not all(param in valid_params for param in parameters):
                return False
            
            # Validate parameter values
            for param, value in parameters.items():
                if not isinstance(value, (int, float)):
                    return False
                
                # Apply relevant validation rule
                for rule_name, rule_func in validation_rules.items():
                    if rule_name in param.lower():
                        if not rule_func(value):
                            return False
                        
            return True
        except Exception as e:
            self.logger.error(f"Parameter validation failed: {e}")
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
