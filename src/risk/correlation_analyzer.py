"""
Correlation Analyzer
Advanced correlation analysis for portfolio risk management and diversification
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyzes price correlations between trading pairs for portfolio optimization
    """

    def __init__(self, lookback_period: int = 30, min_correlation_threshold: float = 0.7):
        self.lookback_period = lookback_period  # Days of historical data
        self.min_correlation_threshold = min_correlation_threshold

        # Correlation data storage
        self.correlation_matrix = {}
        self.correlation_history = {}
        self.price_data_cache = {}

        # Analysis parameters
        self.update_frequency = 3600  # Update correlations every hour
        self.last_update = {}
        self.correlation_decay_factor = 0.95  # Exponential decay for older correlations

        # Risk thresholds
        self.high_correlation_threshold = 0.8
        self.extreme_correlation_threshold = 0.9
        self.diversification_target = 0.5  # Target max correlation for diversified portfolio

    async def analyze_portfolio_correlations(self, client, symbols: List[str]) -> Dict:
        """
        Analyze correlations across entire portfolio

        Args:
            client: Trading client for data fetching
            symbols: List of trading symbols to analyze

        Returns:
            Comprehensive correlation analysis
        """
        try:
            logger.info(f"Analyzing correlations for {len(symbols)} symbols")

            # Update price data for all symbols
            await self._update_price_data(client, symbols)

            # Calculate correlation matrix
            correlation_matrix = await self._calculate_correlation_matrix(symbols)

            # Analyze correlation patterns
            correlation_analysis = self._analyze_correlation_patterns(correlation_matrix, symbols)

            # Generate diversification recommendations
            diversification_advice = self._generate_diversification_recommendations(
                correlation_matrix, symbols
            )

            # Calculate portfolio correlation metrics
            portfolio_metrics = self._calculate_portfolio_correlation_metrics(
                correlation_matrix, symbols
            )

            # Identify correlation clusters
            correlation_clusters = self._identify_correlation_clusters(correlation_matrix, symbols)

            # Risk assessment
            risk_assessment = self._assess_correlation_risks(
                correlation_matrix, correlation_analysis, symbols
            )

            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'symbols_analyzed': symbols,
                'correlation_matrix': correlation_matrix,
                'portfolio_metrics': portfolio_metrics,
                'correlation_patterns': correlation_analysis,
                'correlation_clusters': correlation_clusters,
                'diversification_analysis': diversification_advice,
                'risk_assessment': risk_assessment,
                'high_correlation_pairs': self._get_high_correlation_pairs(correlation_matrix),
                'diversification_score': portfolio_metrics.get('diversification_score', 0)
            }

            logger.info(f"Correlation analysis complete. Diversification score: "
                       f"{portfolio_metrics.get('diversification_score', 0):.3f}")

            return analysis_result

        except Exception as e:
            logger.error(f"Portfolio correlation analysis error: {e}")
            return self._get_default_correlation_analysis(symbols)

    async def _update_price_data(self, client, symbols: List[str]):
        """Update price data cache for correlation calculations"""
        try:
            current_time = datetime.now()

            for symbol in symbols:
                # Check if update is needed
                last_update = self.last_update.get(symbol, datetime.min)
                if (current_time - last_update).seconds < self.update_frequency:
                    continue

                # Get historical klines
                limit = self.lookback_period * 288  # 288 5-min periods per day
                klines = await client.get_klines(symbol, '5m', limit=limit)

                if klines and len(klines) >= 100:  # Minimum data requirement
                    # Extract closing prices and timestamps
                    prices = [float(kline[4]) for kline in klines]
                    timestamps = [kline[0] for kline in klines]

                    # Calculate returns for correlation analysis
                    returns = []
                    for i in range(1, len(prices)):
                        return_val = (prices[i] - prices[i-1]) / prices[i-1]
                        returns.append(return_val)

                    self.price_data_cache[symbol] = {
                        'prices': prices,
                        'returns': returns,
                        'timestamps': timestamps,
                        'last_update': current_time
                    }

                    self.last_update[symbol] = current_time
                    logger.debug(f"Updated price data for {symbol}: {len(prices)} data points")

        except Exception as e:
            logger.error(f"Price data update error: {e}")

    async def _calculate_correlation_matrix(self, symbols: List[str]) -> Dict:
        """Calculate pairwise correlations between all symbols"""
        try:
            correlation_matrix = {}

            # Initialize matrix
            for symbol in symbols:
                correlation_matrix[symbol] = {}

            # Calculate pairwise correlations
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    elif symbol2 in correlation_matrix[symbol1]:
                        # Already calculated (symmetric matrix)
                        continue
                    else:
                        correlation = await self._calculate_pair_correlation(symbol1, symbol2)
                        correlation_matrix[symbol1][symbol2] = correlation
                        correlation_matrix[symbol2][symbol1] = correlation

            return correlation_matrix

        except Exception as e:
            logger.error(f"Correlation matrix calculation error: {e}")
            return {}

    async def _calculate_pair_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        try:
            # Get return data for both symbols
            data1 = self.price_data_cache.get(symbol1, {})
            data2 = self.price_data_cache.get(symbol2, {})

            returns1 = data1.get('returns', [])
            returns2 = data2.get('returns', [])

            if not returns1 or not returns2:
                return 0.0

            # Align data (use minimum length)
            min_length = min(len(returns1), len(returns2))
            if min_length < 50:  # Minimum data points for reliable correlation
                return 0.0

            returns1_aligned = returns1[-min_length:]
            returns2_aligned = returns2[-min_length:]

            # Calculate Pearson correlation
            correlation = np.corrcoef(returns1_aligned, returns2_aligned)[0, 1]

            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0.0

            return float(correlation)

        except Exception as e:
            logger.error(f"Pair correlation calculation error for {symbol1}-{symbol2}: {e}")
            return 0.0

    def _analyze_correlation_patterns(self, correlation_matrix: Dict, symbols: List[str]) -> Dict:
        """Analyze patterns in the correlation matrix"""
        try:
            if not correlation_matrix or not symbols:
                return {}

            correlations = []
            high_correlations = []
            negative_correlations = []

            # Extract all correlation values
            for symbol1 in symbols:
                for symbol2 in symbols:
                    if symbol1 != symbol2 and symbol1 in correlation_matrix:
                        corr = correlation_matrix[symbol1].get(symbol2, 0)
                        correlations.append(corr)

                        if corr >= self.high_correlation_threshold:
                            high_correlations.append((symbol1, symbol2, corr))
                        elif corr <= -0.3:  # Negative correlation threshold
                            negative_correlations.append((symbol1, symbol2, corr))

            if not correlations:
                return {}

            # Calculate statistics
            correlations = np.array(correlations)

            return {
                'average_correlation': float(np.mean(correlations)),
                'median_correlation': float(np.median(correlations)),
                'max_correlation': float(np.max(correlations)),
                'min_correlation': float(np.min(correlations)),
                'std_correlation': float(np.std(correlations)),
                'high_correlation_count': len(high_correlations),
                'negative_correlation_count': len(negative_correlations),
                'high_correlation_pairs': high_correlations[:10],  # Top 10
                'negative_correlation_pairs': negative_correlations[:5],  # Top 5
                'correlation_distribution': {
                    'very_high': len([c for c in correlations if c >= 0.9]),
                    'high': len([c for c in correlations if 0.7 <= c < 0.9]),
                    'moderate': len([c for c in correlations if 0.3 <= c < 0.7]),
                    'low': len([c for c in correlations if 0 <= c < 0.3]),
                    'negative': len([c for c in correlations if c < 0])
                }
            }

        except Exception as e:
            logger.error(f"Correlation pattern analysis error: {e}")
            return {}

    def _calculate_portfolio_correlation_metrics(self, correlation_matrix: Dict, symbols: List[str]) -> Dict:
        """Calculate portfolio-level correlation metrics"""
        try:
            if not correlation_matrix or len(symbols) < 2:
                return {}

            correlations = []
            for symbol1 in symbols:
                for symbol2 in symbols:
                    if (symbol1 != symbol2 and
                        symbol1 in correlation_matrix and
                        symbol2 in correlation_matrix[symbol1]):
                        correlations.append(correlation_matrix[symbol1][symbol2])

            if not correlations:
                return {}

            correlations = np.array(correlations)

            # Portfolio diversification score (inverse of average correlation)
            avg_correlation = np.mean(np.abs(correlations))
            diversification_score = max(0, 1 - avg_correlation)

            # Concentration risk (based on high correlations)
            high_corr_ratio = len([c for c in correlations if c >= self.high_correlation_threshold]) / len(correlations)

            return {
                'diversification_score': float(diversification_score),
                'average_correlation': float(np.mean(correlations)),
                'average_abs_correlation': float(avg_correlation),
                'max_correlation': float(np.max(correlations)),
                'correlation_concentration': float(high_corr_ratio),
                'portfolio_risk_level': self._assess_portfolio_risk_level(avg_correlation),
                'symbols_count': len(symbols),
                'total_pairs': len(correlations),
                'effective_diversification': float(1 / (1 + avg_correlation))  # Effective number of independent assets
            }

        except Exception as e:
            logger.error(f"Portfolio correlation metrics calculation error: {e}")
            return {}

    def _identify_correlation_clusters(self, correlation_matrix: Dict, symbols: List[str]) -> Dict:
        """Identify clusters of highly correlated symbols"""
        try:
            clusters = []
            processed_symbols = set()

            for symbol in symbols:
                if symbol in processed_symbols:
                    continue

                # Find highly correlated symbols
                cluster = [symbol]
                for other_symbol in symbols:
                    if (other_symbol != symbol and
                        other_symbol not in processed_symbols and
                        symbol in correlation_matrix and
                        other_symbol in correlation_matrix[symbol]):

                        correlation = correlation_matrix[symbol][other_symbol]
                        if correlation >= self.high_correlation_threshold:
                            cluster.append(other_symbol)

                # Add cluster if it has multiple symbols
                if len(cluster) > 1:
                    # Calculate cluster statistics
                    cluster_correlations = []
                    for i, sym1 in enumerate(cluster):
                        for j, sym2 in enumerate(cluster[i+1:], i+1):
                            if sym1 in correlation_matrix and sym2 in correlation_matrix[sym1]:
                                cluster_correlations.append(correlation_matrix[sym1][sym2])

                    clusters.append({
                        'symbols': cluster,
                        'size': len(cluster),
                        'avg_correlation': float(np.mean(cluster_correlations)) if cluster_correlations else 0,
                        'min_correlation': float(np.min(cluster_correlations)) if cluster_correlations else 0,
                        'max_correlation': float(np.max(cluster_correlations)) if cluster_correlations else 0
                    })

                    processed_symbols.update(cluster)

            # Sort clusters by size and correlation
            clusters.sort(key=lambda x: (x['size'], x['avg_correlation']), reverse=True)

            return {
                'clusters': clusters,
                'cluster_count': len(clusters),
                'clustered_symbols': len(processed_symbols),
                'unclustered_symbols': len(symbols) - len(processed_symbols),
                'largest_cluster_size': max([c['size'] for c in clusters]) if clusters else 0
            }

        except Exception as e:
            logger.error(f"Correlation cluster identification error: {e}")
            return {'clusters': [], 'cluster_count': 0}

    def _generate_diversification_recommendations(self, correlation_matrix: Dict, symbols: List[str]) -> Dict:
        """Generate recommendations for portfolio diversification"""
        try:
            recommendations = []

            # Analyze current diversification
            high_corr_pairs = self._get_high_correlation_pairs(correlation_matrix)

            if not high_corr_pairs:
                return {
                    'recommendations': ['Portfolio appears well diversified'],
                    'diversification_quality': 'GOOD',
                    'action_needed': False
                }

            # Generate specific recommendations
            symbol_correlation_counts = defaultdict(int)
            for pair in high_corr_pairs:
                symbol_correlation_counts[pair['symbol1']] += 1
                symbol_correlation_counts[pair['symbol2']] += 1

            # Most correlated symbols (candidates for reduction)
            most_correlated = sorted(symbol_correlation_counts.items(),
                                   key=lambda x: x[1], reverse=True)

            if most_correlated:
                top_symbol = most_correlated[0]
                recommendations.append(f"Consider reducing exposure to {top_symbol[0]} "
                                     f"(involved in {top_symbol[1]} high correlation pairs)")

            # Sector-specific recommendations
            sector_recommendations = self._generate_sector_diversification_advice(correlation_matrix, symbols)
            recommendations.extend(sector_recommendations)

            # General diversification advice
            if len(high_corr_pairs) > len(symbols) * 0.3:
                recommendations.append("Portfolio shows high correlation concentration - "
                                     "consider adding assets from different sectors")

            # Assess diversification quality
            avg_correlation = np.mean([pair['correlation'] for pair in high_corr_pairs])
            if avg_correlation >= 0.9:
                quality = 'POOR'
                action_needed = True
            elif avg_correlation >= 0.8:
                quality = 'FAIR'
                action_needed = True
            else:
                quality = 'GOOD'
                action_needed = False

            return {
                'recommendations': recommendations,
                'diversification_quality': quality,
                'action_needed': action_needed,
                'high_correlation_count': len(high_corr_pairs),
                'most_correlated_symbols': most_correlated[:3],
                'target_correlation': self.diversification_target
            }

        except Exception as e:
            logger.error(f"Diversification recommendations error: {e}")
            return {
                'recommendations': ['Error generating recommendations'],
                'diversification_quality': 'UNKNOWN',
                'action_needed': True
            }

    def _generate_sector_diversification_advice(self, correlation_matrix: Dict, symbols: List[str]) -> List[str]:
        """Generate sector-specific diversification advice"""
        recommendations = []

        # Simplified sector classification
        btc_symbols = [s for s in symbols if 'BTC' in s]
        eth_symbols = [s for s in symbols if 'ETH' in s]
        defi_symbols = [s for s in symbols if any(token in s for token in ['UNI', 'LINK', 'AAVE'])]

        if len(btc_symbols) > 2:
            recommendations.append(f"High BTC exposure ({len(btc_symbols)} symbols) - consider diversifying")

        if len(eth_symbols) > 2:
            recommendations.append(f"High ETH exposure ({len(eth_symbols)} symbols) - consider diversifying")

        if len(defi_symbols) > 3:
            recommendations.append(f"High DeFi exposure ({len(defi_symbols)} symbols) - consider diversifying")

        return recommendations

    def _get_high_correlation_pairs(self, correlation_matrix: Dict) -> List[Dict]:
        """Extract high correlation pairs from matrix"""
        high_corr_pairs = []

        for symbol1, correlations in correlation_matrix.items():
            for symbol2, correlation in correlations.items():
                if (symbol1 < symbol2 and  # Avoid duplicates
                    correlation >= self.high_correlation_threshold):
                    high_corr_pairs.append({
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': correlation,
                        'risk_level': 'EXTREME' if correlation >= self.extreme_correlation_threshold else 'HIGH'
                    })

        return sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)

    def _assess_correlation_risks(self, correlation_matrix: Dict,
                                correlation_analysis: Dict, symbols: List[str]) -> Dict:
        """Assess portfolio risks based on correlations"""
        try:
            high_corr_count = correlation_analysis.get('high_correlation_count', 0)
            avg_correlation = correlation_analysis.get('average_correlation', 0)

            # Risk level assessment
            if avg_correlation >= 0.8 or high_corr_count >= len(symbols):
                risk_level = 'HIGH'
                risk_description = 'Portfolio shows dangerous correlation concentration'
            elif avg_correlation >= 0.6 or high_corr_count >= len(symbols) * 0.5:
                risk_level = 'MEDIUM'
                risk_description = 'Portfolio moderately concentrated, diversification recommended'
            else:
                risk_level = 'LOW'
                risk_description = 'Portfolio appears well diversified'

            # Specific risk factors
            risk_factors = []
            if high_corr_count > 0:
                risk_factors.append(f"{high_corr_count} high correlation pairs detected")

            extreme_corr = len([p for p in self._get_high_correlation_pairs(correlation_matrix)
                               if p['correlation'] >= self.extreme_correlation_threshold])
            if extreme_corr > 0:
                risk_factors.append(f"{extreme_corr} extreme correlation pairs (>90%)")

            return {
                'risk_level': risk_level,
                'risk_description': risk_description,
                'risk_factors': risk_factors,
                'diversification_needed': risk_level in ['HIGH', 'MEDIUM'],
                'correlation_score': avg_correlation,
                'risk_score': min(avg_correlation * 2, 1.0)  # Normalized risk score
            }

        except Exception as e:
            logger.error(f"Correlation risk assessment error: {e}")
            return {
                'risk_level': 'UNKNOWN',
                'risk_description': 'Unable to assess correlation risks',
                'risk_factors': [],
                'diversification_needed': True
            }

    def _assess_portfolio_risk_level(self, avg_correlation: float) -> str:
        """Assess portfolio risk level based on average correlation"""
        if avg_correlation >= 0.8:
            return 'HIGH'
        elif avg_correlation >= 0.6:
            return 'MEDIUM'
        elif avg_correlation >= 0.4:
            return 'LOW'
        else:
            return 'MINIMAL'

    def _get_default_correlation_analysis(self, symbols: List[str]) -> Dict:
        """Return default analysis when calculation fails"""
        return {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': symbols,
            'correlation_matrix': {},
            'portfolio_metrics': {
                'diversification_score': 0,
                'symbols_count': len(symbols)
            },
            'correlation_patterns': {},
            'correlation_clusters': {'clusters': [], 'cluster_count': 0},
            'diversification_analysis': {
                'recommendations': ['Correlation analysis failed'],
                'diversification_quality': 'UNKNOWN',
                'action_needed': True
            },
            'risk_assessment': {
                'risk_level': 'UNKNOWN',
                'risk_description': 'Correlation analysis unavailable'
            },
            'high_correlation_pairs': [],
            'diversification_score': 0,
            'error': True
        }

    async def get_symbol_correlation_score(self, client, target_symbol: str,
                                         portfolio_symbols: List[str]) -> Dict:
        """Get correlation score for adding a new symbol to portfolio"""
        try:
            # Update price data including the new symbol
            all_symbols = portfolio_symbols + [target_symbol]
            await self._update_price_data(client, all_symbols)

            # Calculate correlations with existing portfolio
            correlations_with_portfolio = []
            for portfolio_symbol in portfolio_symbols:
                correlation = await self._calculate_pair_correlation(target_symbol, portfolio_symbol)
                correlations_with_portfolio.append(correlation)

            if not correlations_with_portfolio:
                return {'diversification_benefit': 0, 'recommendation': 'UNKNOWN'}

            avg_correlation = np.mean(np.abs(correlations_with_portfolio))
            max_correlation = max(correlations_with_portfolio)

            # Calculate diversification benefit
            diversification_benefit = max(0, 1 - avg_correlation)

            # Generate recommendation
            if max_correlation >= self.extreme_correlation_threshold:
                recommendation = 'AVOID'
                reason = f"Extremely high correlation ({max_correlation:.3f}) with existing positions"
            elif avg_correlation >= self.high_correlation_threshold:
                recommendation = 'CAUTION'
                reason = f"High average correlation ({avg_correlation:.3f}) with portfolio"
            elif avg_correlation <= self.diversification_target:
                recommendation = 'EXCELLENT'
                reason = f"Low correlation ({avg_correlation:.3f}) adds diversification"
            else:
                recommendation = 'GOOD'
                reason = f"Moderate correlation ({avg_correlation:.3f}) acceptable"

            return {
                'target_symbol': target_symbol,
                'portfolio_symbols': portfolio_symbols,
                'correlations': correlations_with_portfolio,
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'diversification_benefit': diversification_benefit,
                'recommendation': recommendation,
                'reason': reason,
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Symbol correlation score error for {target_symbol}: {e}")
            return {
                'target_symbol': target_symbol,
                'diversification_benefit': 0,
                'recommendation': 'ERROR',
                'error': str(e)
            }

    def get_correlation_statistics(self) -> Dict:
        """Get correlation analysis statistics"""
        return {
            'analysis_parameters': {
                'lookback_period': self.lookback_period,
                'min_correlation_threshold': self.min_correlation_threshold,
                'high_correlation_threshold': self.high_correlation_threshold,
                'extreme_correlation_threshold': self.extreme_correlation_threshold,
                'diversification_target': self.diversification_target
            },
            'cached_symbols': len(self.price_data_cache),
            'correlation_matrix_size': len(self.correlation_matrix),
            'update_frequency': self.update_frequency,
            'last_updates': len(self.last_update)
        }

    def update_correlation_parameters(self, **kwargs):
        """Update correlation analysis parameters"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                logger.info(f"Correlation parameter updated: {param} = {value}")
            else:
                logger.warning(f"Unknown correlation parameter: {param}")

    def clear_correlation_cache(self):
        """Clear correlation data cache"""
        self.correlation_matrix.clear()
        self.correlation_history.clear()
        self.price_data_cache.clear()
        self.last_update.clear()
        logger.info("Correlation analysis cache cleared")