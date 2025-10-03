"""
ML Signal Enhancement Module
Integrates machine learning predictions with traditional technical analysis
to provide enhanced trading signals
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import pickle
import os

try:
    from ..models.lstm_predictor import LSTMPredictor
except ImportError:
    # Fallback if import fails
    class LSTMPredictor:
        def __init__(self, *args, **kwargs):
            pass

        async def predict_price(self, *args, **kwargs):
            return {'prediction': 0, 'confidence': 0, 'trend': 'HOLD'}

logger = logging.getLogger(__name__)


class MLSignalEnhancer:
    """
    Enhances traditional trading signals with machine learning predictions
    """

    def __init__(self, model_cache_dir: str = "ml_models"):
        self.model_cache_dir = model_cache_dir
        self.lstm_predictors = {}  # Cache for LSTM models per symbol
        self.prediction_cache = {}  # Cache for recent predictions
        self.cache_duration = 300  # 5 minutes cache duration

        # ML enhancement parameters
        self.ml_weight = 0.4  # Weight for ML predictions in final signal
        self.technical_weight = 0.6  # Weight for technical analysis
        self.min_ml_confidence = 0.3  # Minimum ML confidence to use prediction
        self.ensemble_threshold = 0.7  # Threshold for ensemble agreement

        # Performance tracking
        self.ml_accuracy_history = {}
        self.enhancement_stats = {
            'total_enhancements': 0,
            'ml_overrides': 0,
            'confidence_boosts': 0,
            'prediction_accuracy': {}
        }

        # Ensure model cache directory exists
        os.makedirs(model_cache_dir, exist_ok=True)

    async def enhance_signal(self, client, symbol: str, base_signal: Dict,
                           market_data: Optional[Dict] = None) -> Dict:
        """
        Enhance a base trading signal with ML predictions

        Args:
            client: Trading client for data fetching
            symbol: Trading pair symbol
            base_signal: Base signal from technical analysis
            market_data: Optional market data to avoid refetching

        Returns:
            Enhanced signal with ML insights
        """
        try:
            logger.info(f"Enhancing signal for {symbol} with ML predictions")

            # Check if we have cached prediction
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if cache_key in self.prediction_cache:
                age = (datetime.now() - self.prediction_cache[cache_key]['timestamp']).seconds
                if age < self.cache_duration:
                    ml_prediction = self.prediction_cache[cache_key]['prediction']
                    logger.info(f"Using cached ML prediction for {symbol}")
                else:
                    ml_prediction = await self._get_ml_prediction(client, symbol, market_data)
            else:
                ml_prediction = await self._get_ml_prediction(client, symbol, market_data)

            # Enhance the base signal
            enhanced_signal = await self._combine_signals(base_signal, ml_prediction, symbol)

            # Track enhancement statistics
            self._update_enhancement_stats(base_signal, enhanced_signal, ml_prediction)

            logger.info(f"Signal enhanced for {symbol}: {base_signal['signal']} -> "
                       f"{enhanced_signal['signal']} (confidence: {enhanced_signal['confidence']:.3f})")

            return enhanced_signal

        except Exception as e:
            logger.error(f"ML signal enhancement error for {symbol}: {e}")
            # Return original signal with enhancement metadata
            enhanced_signal = base_signal.copy()
            enhanced_signal.update({
                'ml_enhancement': {
                    'enabled': False,
                    'error': str(e),
                    'fallback_used': True
                },
                'enhanced': False
            })
            return enhanced_signal

    async def _get_ml_prediction(self, client, symbol: str,
                               market_data: Optional[Dict] = None) -> Dict:
        """Get ML prediction for the symbol"""
        try:
            # Get or create LSTM predictor for this symbol
            if symbol not in self.lstm_predictors:
                self.lstm_predictors[symbol] = LSTMPredictor(
                    symbol=symbol,
                    sequence_length=60,  # 60 periods for LSTM
                    features=['close', 'volume', 'high', 'low']
                )

            predictor = self.lstm_predictors[symbol]

            # Get market data if not provided
            if market_data is None:
                klines = await client.get_klines(symbol, '5m', limit=100)
                if not klines or len(klines) < 60:
                    return self._get_default_ml_prediction()

                # Prepare data for ML model
                market_data = {
                    'prices': [float(k[4]) for k in klines],  # Close prices
                    'volumes': [float(k[5]) for k in klines],  # Volumes
                    'highs': [float(k[2]) for k in klines],   # High prices
                    'lows': [float(k[3]) for k in klines],    # Low prices
                    'timestamps': [k[0] for k in klines]
                }

            # Get LSTM prediction
            lstm_result = await self._get_lstm_prediction(predictor, market_data)

            # Additional ML features
            ml_features = self._calculate_ml_features(market_data)

            # Combine LSTM with feature-based predictions
            combined_prediction = self._combine_ml_predictions(lstm_result, ml_features)

            # Cache the prediction
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.prediction_cache[cache_key] = {
                'prediction': combined_prediction,
                'timestamp': datetime.now()
            }

            return combined_prediction

        except Exception as e:
            logger.error(f"ML prediction error for {symbol}: {e}")
            return self._get_default_ml_prediction()

    async def _get_lstm_prediction(self, predictor: LSTMPredictor,
                                 market_data: Dict) -> Dict:
        """Get prediction from LSTM model"""
        try:
            # Prepare features for LSTM
            features = np.column_stack([
                market_data['prices'],
                market_data['volumes'],
                market_data['highs'],
                market_data['lows']
            ])

            # Get LSTM prediction
            if hasattr(predictor, 'predict_price'):
                prediction = await predictor.predict_price(features[-60:])  # Last 60 periods
            else:
                # Simplified prediction if LSTM not available
                recent_prices = market_data['prices'][-10:]
                price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

                prediction = {
                    'prediction': recent_prices[-1] * (1 + price_change * 0.1),
                    'confidence': min(abs(price_change) * 10, 0.8),
                    'trend': 'BUY' if price_change > 0.01 else 'SELL' if price_change < -0.01 else 'HOLD'
                }

            return {
                'lstm_prediction': prediction.get('prediction', market_data['prices'][-1]),
                'lstm_confidence': prediction.get('confidence', 0),
                'lstm_trend': prediction.get('trend', 'HOLD'),
                'lstm_available': True
            }

        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return {
                'lstm_prediction': market_data['prices'][-1],
                'lstm_confidence': 0,
                'lstm_trend': 'HOLD',
                'lstm_available': False,
                'error': str(e)
            }

    def _calculate_ml_features(self, market_data: Dict) -> Dict:
        """Calculate additional ML features"""
        try:
            prices = np.array(market_data['prices'])
            volumes = np.array(market_data['volumes'])

            # Price momentum features
            returns = np.diff(prices) / prices[:-1]
            momentum_5 = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            momentum_20 = np.mean(returns[-20:]) if len(returns) >= 20 else 0

            # Volatility features
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0

            # Volume features
            volume_trend = (volumes[-5:].mean() - volumes[-10:-5].mean()) / volumes[-10:-5].mean() \
                          if len(volumes) >= 10 else 0

            # Price pattern features
            higher_highs = len([i for i in range(1, min(10, len(prices)))
                               if prices[-i] > prices[-(i+1)]]) / min(10, len(prices)-1)

            # RSI-like feature
            gains = np.maximum(returns, 0)
            losses = np.maximum(-returns, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))

            # Generate ML signal based on features
            signal_score = (
                momentum_5 * 0.3 +
                momentum_20 * 0.2 +
                volume_trend * 0.2 +
                (higher_highs - 0.5) * 0.15 +
                (50 - abs(rsi - 50)) / 50 * 0.15  # RSI divergence from neutral
            )

            # Determine signal and confidence
            if signal_score > 0.1:
                ml_signal = 'BUY'
                confidence = min(signal_score * 2, 0.9)
            elif signal_score < -0.1:
                ml_signal = 'SELL'
                confidence = min(abs(signal_score) * 2, 0.9)
            else:
                ml_signal = 'HOLD'
                confidence = 0

            return {
                'ml_features': {
                    'momentum_5': momentum_5,
                    'momentum_20': momentum_20,
                    'volatility': volatility,
                    'volume_trend': volume_trend,
                    'higher_highs_ratio': higher_highs,
                    'rsi': rsi,
                    'signal_score': signal_score
                },
                'feature_signal': ml_signal,
                'feature_confidence': confidence,
                'features_available': True
            }

        except Exception as e:
            logger.error(f"ML features calculation error: {e}")
            return {
                'ml_features': {},
                'feature_signal': 'HOLD',
                'feature_confidence': 0,
                'features_available': False,
                'error': str(e)
            }

    def _combine_ml_predictions(self, lstm_result: Dict, ml_features: Dict) -> Dict:
        """Combine LSTM and feature-based predictions"""
        try:
            lstm_signal = lstm_result.get('lstm_trend', 'HOLD')
            lstm_confidence = lstm_result.get('lstm_confidence', 0)

            feature_signal = ml_features.get('feature_signal', 'HOLD')
            feature_confidence = ml_features.get('feature_confidence', 0)

            # Weight the predictions
            lstm_weight = 0.6 if lstm_result.get('lstm_available', False) else 0
            feature_weight = 0.4 if ml_features.get('features_available', False) else 0

            # Normalize weights if both are not available
            total_weight = lstm_weight + feature_weight
            if total_weight > 0:
                lstm_weight /= total_weight
                feature_weight /= total_weight
            else:
                lstm_weight = feature_weight = 0.5

            # Combine signals using ensemble approach
            signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}

            # Add LSTM vote
            if lstm_confidence >= self.min_ml_confidence:
                signal_scores[lstm_signal] += lstm_weight * lstm_confidence

            # Add feature-based vote
            if feature_confidence >= self.min_ml_confidence:
                signal_scores[feature_signal] += feature_weight * feature_confidence

            # Determine final ML signal
            final_signal = max(signal_scores, key=signal_scores.get)
            final_confidence = signal_scores[final_signal]

            # If confidence is too low, default to HOLD
            if final_confidence < self.min_ml_confidence:
                final_signal = 'HOLD'
                final_confidence = 0

            return {
                'ml_signal': final_signal,
                'ml_confidence': final_confidence,
                'lstm_component': {
                    'signal': lstm_signal,
                    'confidence': lstm_confidence,
                    'available': lstm_result.get('lstm_available', False)
                },
                'features_component': {
                    'signal': feature_signal,
                    'confidence': feature_confidence,
                    'available': ml_features.get('features_available', False),
                    'features': ml_features.get('ml_features', {})
                },
                'ensemble_scores': signal_scores,
                'ml_available': True
            }

        except Exception as e:
            logger.error(f"ML prediction combination error: {e}")
            return self._get_default_ml_prediction()

    async def _combine_signals(self, base_signal: Dict, ml_prediction: Dict, symbol: str) -> Dict:
        """Combine base technical signal with ML prediction"""
        try:
            base_sig = base_signal.get('signal', 'HOLD')
            base_conf = base_signal.get('confidence', 0)

            ml_sig = ml_prediction.get('ml_signal', 'HOLD')
            ml_conf = ml_prediction.get('ml_confidence', 0)

            # Enhanced signal logic
            enhanced_signal = base_sig
            enhanced_confidence = base_conf

            # Case 1: Both signals agree and ML confidence is high
            if base_sig == ml_sig and ml_conf >= self.ensemble_threshold:
                # Boost confidence when signals align with high ML confidence
                enhanced_confidence = min(base_conf * 1.3 + ml_conf * 0.2, 0.95)
                enhancement_type = 'ALIGNED_BOOST'
                self.enhancement_stats['confidence_boosts'] += 1

            # Case 2: Signals disagree but ML has high confidence
            elif base_sig != ml_sig and ml_conf >= self.ensemble_threshold and ml_conf > base_conf * 1.5:
                # Override with ML signal if much more confident
                enhanced_signal = ml_sig
                enhanced_confidence = ml_conf * 0.9  # Slight penalty for override
                enhancement_type = 'ML_OVERRIDE'
                self.enhancement_stats['ml_overrides'] += 1

            # Case 3: Base signal is HOLD but ML has strong opinion
            elif base_sig == 'HOLD' and ml_sig != 'HOLD' and ml_conf >= 0.6:
                enhanced_signal = ml_sig
                enhanced_confidence = ml_conf * 0.8  # Conservative confidence
                enhancement_type = 'ML_ACTIVATION'

            # Case 4: ML signal is HOLD - reduces confidence in base signal
            elif ml_sig == 'HOLD' and base_sig != 'HOLD' and ml_conf > 0.4:
                enhanced_confidence = base_conf * 0.8  # Reduce confidence
                enhancement_type = 'ML_DAMPENING'

            # Case 5: Default - weighted combination
            else:
                # Weight-based combination
                total_weight = self.technical_weight + (self.ml_weight if ml_conf >= self.min_ml_confidence else 0)
                technical_weight_norm = self.technical_weight / total_weight
                ml_weight_norm = (self.ml_weight if ml_conf >= self.min_ml_confidence else 0) / total_weight

                # Use signal with higher weighted confidence
                weighted_base = base_conf * technical_weight_norm
                weighted_ml = ml_conf * ml_weight_norm

                if weighted_ml > weighted_base and ml_sig != 'HOLD':
                    enhanced_signal = ml_sig
                    enhanced_confidence = weighted_ml + weighted_base * 0.2
                else:
                    enhanced_signal = base_sig
                    enhanced_confidence = weighted_base + weighted_ml * 0.2

                enhancement_type = 'WEIGHTED_COMBINATION'

            # Ensure confidence bounds
            enhanced_confidence = max(0, min(enhanced_confidence, 0.95))

            # Create enhanced signal dictionary
            enhanced_signal_dict = base_signal.copy()
            enhanced_signal_dict.update({
                'signal': enhanced_signal,
                'confidence': enhanced_confidence,
                'ml_enhancement': {
                    'enabled': True,
                    'enhancement_type': enhancement_type,
                    'ml_signal': ml_sig,
                    'ml_confidence': ml_conf,
                    'base_signal': base_sig,
                    'base_confidence': base_conf,
                    'agreement': base_sig == ml_sig,
                    'ml_prediction': ml_prediction
                },
                'enhanced': True
            })

            self.enhancement_stats['total_enhancements'] += 1
            return enhanced_signal_dict

        except Exception as e:
            logger.error(f"Signal combination error for {symbol}: {e}")
            enhanced_signal = base_signal.copy()
            enhanced_signal.update({
                'ml_enhancement': {
                    'enabled': False,
                    'error': str(e)
                },
                'enhanced': False
            })
            return enhanced_signal

    def _get_default_ml_prediction(self) -> Dict:
        """Get default ML prediction when ML is not available"""
        return {
            'ml_signal': 'HOLD',
            'ml_confidence': 0,
            'lstm_component': {
                'signal': 'HOLD',
                'confidence': 0,
                'available': False
            },
            'features_component': {
                'signal': 'HOLD',
                'confidence': 0,
                'available': False,
                'features': {}
            },
            'ensemble_scores': {'BUY': 0, 'SELL': 0, 'HOLD': 1},
            'ml_available': False
        }

    def _update_enhancement_stats(self, base_signal: Dict, enhanced_signal: Dict, ml_prediction: Dict):
        """Update enhancement performance statistics"""
        try:
            enhancement_type = enhanced_signal.get('ml_enhancement', {}).get('enhancement_type', 'NONE')

            # Track enhancement types
            if enhancement_type not in self.enhancement_stats:
                self.enhancement_stats[enhancement_type] = 0
            self.enhancement_stats[enhancement_type] += 1

            # Track signal changes
            if base_signal.get('signal') != enhanced_signal.get('signal'):
                self.enhancement_stats['signal_changes'] = self.enhancement_stats.get('signal_changes', 0) + 1

            # Track confidence changes
            conf_change = enhanced_signal.get('confidence', 0) - base_signal.get('confidence', 0)
            self.enhancement_stats['avg_confidence_change'] = (
                self.enhancement_stats.get('avg_confidence_change', 0) *
                (self.enhancement_stats['total_enhancements'] - 1) + conf_change
            ) / self.enhancement_stats['total_enhancements']

        except Exception as e:
            logger.error(f"Enhancement stats update error: {e}")

    def get_enhancement_statistics(self) -> Dict:
        """Get ML enhancement performance statistics"""
        return {
            'enhancement_stats': self.enhancement_stats.copy(),
            'active_models': len(self.lstm_predictors),
            'cached_predictions': len(self.prediction_cache),
            'ml_parameters': {
                'ml_weight': self.ml_weight,
                'technical_weight': self.technical_weight,
                'min_ml_confidence': self.min_ml_confidence,
                'ensemble_threshold': self.ensemble_threshold
            }
        }

    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
        logger.info("ML prediction cache cleared")

    def update_ml_parameters(self, **kwargs):
        """Update ML enhancement parameters"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                logger.info(f"ML parameter updated: {param} = {value}")
            else:
                logger.warning(f"Unknown ML parameter: {param}")

    async def train_symbol_model(self, client, symbol: str, days: int = 30):
        """Train or retrain ML model for specific symbol"""
        try:
            logger.info(f"Training ML model for {symbol}")

            # Get historical data for training
            klines = await client.get_klines(symbol, '5m', limit=days * 288)  # 288 5-min periods per day

            if len(klines) < 1000:  # Need sufficient data
                logger.warning(f"Insufficient data for training {symbol} model")
                return False

            # Initialize or get predictor
            if symbol not in self.lstm_predictors:
                self.lstm_predictors[symbol] = LSTMPredictor(
                    symbol=symbol,
                    sequence_length=60,
                    features=['close', 'volume', 'high', 'low']
                )

            predictor = self.lstm_predictors[symbol]

            # Prepare training data
            training_data = np.column_stack([
                [float(k[4]) for k in klines],  # Close
                [float(k[5]) for k in klines],  # Volume
                [float(k[2]) for k in klines],  # High
                [float(k[3]) for k in klines],  # Low
            ])

            # Train the model (if training method exists)
            if hasattr(predictor, 'train'):
                await predictor.train(training_data)
                logger.info(f"ML model training completed for {symbol}")
                return True
            else:
                logger.info(f"Training not available for {symbol} model")
                return False

        except Exception as e:
            logger.error(f"ML model training error for {symbol}: {e}")
            return False