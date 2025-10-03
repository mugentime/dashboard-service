"""
Portfolio Heat Manager
Advanced portfolio-wide risk management with heat mapping and correlation analysis
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class PortfolioHeatManager:
    """
    Manages portfolio-wide risk through heat mapping and correlation analysis
    """

    def __init__(self, max_portfolio_heat: float = 0.15, max_sector_heat: float = 0.08):
        self.max_portfolio_heat = max_portfolio_heat  # 15% max total portfolio risk
        self.max_sector_heat = max_sector_heat       # 8% max sector risk

        # Heat tracking
        self.position_heat = {}  # Heat contribution per position
        self.sector_heat = {}    # Heat per sector/category
        self.correlation_matrix = {}  # Symbol correlation matrix
        self.heat_history = []   # Historical heat data

        # Risk categories
        self.crypto_sectors = {
            'BTC': ['BTCUSDT', 'BTCBUSD'],
            'ETH': ['ETHUSDT', 'ETHBUSD'],
            'DeFi': ['UNIUSDT', 'LINKUSDT', 'AAVEUSDT', 'CRVUSDT'],
            'Layer1': ['ADAUSDT', 'DOTUSDT', 'SOLUSDT', 'AVAXUSDT'],
            'Layer2': ['MATICUSDT', 'ARBUSDT', 'OPUSDT'],
            'Meme': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT'],
            'Gaming': ['AXSUSDT', 'SANDUSDT', 'MANAUSDT'],
            'AI': ['FETUSDT', 'AGIXUSDT', 'OCEANUSDT']
        }

        # Heat calculation parameters
        self.volatility_weight = 0.3
        self.correlation_weight = 0.3
        self.position_size_weight = 0.2
        self.time_decay_weight = 0.2

        # Performance tracking
        self.heat_violations = []
        self.risk_adjustments = []

    async def calculate_portfolio_heat(self, positions: Dict, volatility_data: Dict) -> Dict:
        """
        Calculate comprehensive portfolio heat map

        Args:
            positions: Dictionary of current positions {symbol: position_data}
            volatility_data: Dictionary of volatility data {symbol: volatility_info}

        Returns:
            Comprehensive heat analysis
        """
        try:
            logger.info(f"Calculating portfolio heat for {len(positions)} positions")

            # Calculate individual position heat
            position_heat = {}
            total_portfolio_value = sum(abs(pos.get('notional', 0)) for pos in positions.values())

            for symbol, position in positions.items():
                heat = await self._calculate_position_heat(
                    symbol, position, volatility_data.get(symbol, {}), total_portfolio_value
                )
                position_heat[symbol] = heat

            # Calculate sector heat distribution
            sector_heat = self._calculate_sector_heat(position_heat, positions)

            # Calculate correlation-adjusted heat
            correlation_heat = await self._calculate_correlation_heat(position_heat, positions)

            # Overall portfolio metrics
            total_heat = sum(h['total_heat'] for h in position_heat.values())
            heat_concentration = self._calculate_heat_concentration(position_heat)
            diversification_score = self._calculate_diversification_score(sector_heat)

            # Risk warnings and recommendations
            warnings = self._generate_heat_warnings(total_heat, sector_heat, correlation_heat)
            recommendations = self._generate_heat_recommendations(
                position_heat, sector_heat, correlation_heat
            )

            # Update historical tracking
            self._update_heat_history(total_heat, sector_heat, position_heat)

            heat_analysis = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_metrics': {
                    'total_heat': total_heat,
                    'max_allowed_heat': self.max_portfolio_heat,
                    'heat_utilization': total_heat / self.max_portfolio_heat,
                    'heat_concentration': heat_concentration,
                    'diversification_score': diversification_score,
                    'total_positions': len(positions),
                    'total_portfolio_value': total_portfolio_value
                },
                'position_heat': position_heat,
                'sector_heat': sector_heat,
                'correlation_heat': correlation_heat,
                'risk_assessment': {
                    'risk_level': self._assess_risk_level(total_heat),
                    'warnings': warnings,
                    'recommendations': recommendations,
                    'safe_to_add_position': total_heat < self.max_portfolio_heat * 0.8
                },
                'heat_breakdown': {
                    'volatility_component': sum(h.get('volatility_heat', 0) for h in position_heat.values()),
                    'correlation_component': correlation_heat.get('total_correlation_heat', 0),
                    'concentration_component': heat_concentration * 0.1,
                    'time_decay_component': sum(h.get('time_decay_heat', 0) for h in position_heat.values())
                }
            }

            logger.info(f"Portfolio heat analysis complete: {total_heat:.3f} / {self.max_portfolio_heat:.3f} "
                       f"({total_heat/self.max_portfolio_heat:.1%} utilization)")

            return heat_analysis

        except Exception as e:
            logger.error(f"Portfolio heat calculation error: {e}")
            return self._get_default_heat_analysis()

    async def _calculate_position_heat(self, symbol: str, position: Dict,
                                     volatility_data: Dict, total_portfolio_value: float) -> Dict:
        """Calculate heat contribution from individual position"""
        try:
            position_size = abs(position.get('notional', 0))
            position_weight = position_size / total_portfolio_value if total_portfolio_value > 0 else 0

            # Volatility component
            atr_percent = volatility_data.get('atr_percent', 1.0)
            volatility_class = volatility_data.get('volatility_class', 'MEDIUM')
            volatility_multiplier = {
                'VERY_LOW': 0.5, 'LOW': 0.7, 'MEDIUM': 1.0, 'HIGH': 1.5, 'EXTREME': 2.0
            }.get(volatility_class, 1.0)

            volatility_heat = position_weight * (atr_percent / 100) * volatility_multiplier * self.volatility_weight

            # Position size component
            size_heat = position_weight * self.position_size_weight

            # Time decay component (positions held longer contribute less heat)
            entry_time = position.get('entry_time', datetime.now().isoformat())
            try:
                entry_datetime = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                time_held = (datetime.now() - entry_datetime).total_seconds() / 3600  # hours
                time_decay_factor = max(0.5, 1.0 - (time_held / 240))  # Decay over 10 days
            except:
                time_decay_factor = 1.0

            time_decay_heat = volatility_heat * time_decay_factor * self.time_decay_weight

            # Total heat for this position
            total_heat = volatility_heat + size_heat + time_decay_heat

            return {
                'symbol': symbol,
                'position_size': position_size,
                'position_weight': position_weight,
                'volatility_heat': volatility_heat,
                'size_heat': size_heat,
                'time_decay_heat': time_decay_heat,
                'total_heat': total_heat,
                'volatility_class': volatility_class,
                'atr_percent': atr_percent,
                'time_decay_factor': time_decay_factor,
                'heat_rank': 0  # Will be set later
            }

        except Exception as e:
            logger.error(f"Position heat calculation error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'total_heat': 0.01,  # Minimal default heat
                'error': str(e)
            }

    def _calculate_sector_heat(self, position_heat: Dict, positions: Dict) -> Dict:
        """Calculate heat distribution across crypto sectors"""
        sector_heat = defaultdict(lambda: {
            'total_heat': 0,
            'positions': [],
            'position_count': 0,
            'total_value': 0
        })

        # Classify positions by sector
        for symbol, heat_data in position_heat.items():
            sector = self._classify_symbol_sector(symbol)

            sector_heat[sector]['total_heat'] += heat_data['total_heat']
            sector_heat[sector]['positions'].append(symbol)
            sector_heat[sector]['position_count'] += 1
            sector_heat[sector]['total_value'] += heat_data.get('position_size', 0)

        # Check for sector concentration violations
        for sector, data in sector_heat.items():
            data['heat_violation'] = data['total_heat'] > self.max_sector_heat
            data['heat_utilization'] = data['total_heat'] / self.max_sector_heat
            data['risk_level'] = self._assess_risk_level(data['total_heat'])

        return dict(sector_heat)

    async def _calculate_correlation_heat(self, position_heat: Dict, positions: Dict) -> Dict:
        """Calculate correlation-adjusted portfolio heat"""
        try:
            symbols = list(position_heat.keys())
            if len(symbols) < 2:
                return {'total_correlation_heat': 0, 'correlation_pairs': {}}

            # Simplified correlation calculation (in production, use historical price correlation)
            correlation_pairs = {}
            total_correlation_heat = 0

            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    # Estimate correlation based on sector similarity
                    correlation = self._estimate_symbol_correlation(symbol1, symbol2)

                    if correlation > 0.5:  # High correlation
                        heat1 = position_heat[symbol1]['total_heat']
                        heat2 = position_heat[symbol2]['total_heat']

                        # Correlation increases combined heat
                        correlation_adjustment = heat1 * heat2 * correlation * self.correlation_weight
                        total_correlation_heat += correlation_adjustment

                        correlation_pairs[f"{symbol1}-{symbol2}"] = {
                            'correlation': correlation,
                            'heat_adjustment': correlation_adjustment,
                            'symbols': [symbol1, symbol2]
                        }

            return {
                'total_correlation_heat': total_correlation_heat,
                'correlation_pairs': correlation_pairs,
                'high_correlation_count': len([p for p in correlation_pairs.values() if p['correlation'] > 0.7])
            }

        except Exception as e:
            logger.error(f"Correlation heat calculation error: {e}")
            return {'total_correlation_heat': 0, 'correlation_pairs': {}, 'error': str(e)}

    def _classify_symbol_sector(self, symbol: str) -> str:
        """Classify symbol into crypto sector"""
        for sector, symbols in self.crypto_sectors.items():
            if symbol in symbols:
                return sector

        # Default classification based on symbol name
        if 'BTC' in symbol:
            return 'BTC'
        elif 'ETH' in symbol:
            return 'ETH'
        elif any(term in symbol for term in ['UNI', 'LINK', 'AAVE', 'CRV']):
            return 'DeFi'
        else:
            return 'Other'

    def _estimate_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Estimate correlation between two symbols"""
        sector1 = self._classify_symbol_sector(symbol1)
        sector2 = self._classify_symbol_sector(symbol2)

        # Same sector = high correlation
        if sector1 == sector2:
            return 0.8

        # Cross-sector correlations
        high_correlation_pairs = {
            ('BTC', 'ETH'): 0.7,
            ('DeFi', 'ETH'): 0.6,
            ('Layer1', 'ETH'): 0.5,
            ('Layer2', 'ETH'): 0.7
        }

        for (s1, s2), corr in high_correlation_pairs.items():
            if (sector1, sector2) == (s1, s2) or (sector1, sector2) == (s2, s1):
                return corr

        return 0.3  # Default moderate correlation

    def _calculate_heat_concentration(self, position_heat: Dict) -> float:
        """Calculate heat concentration (Herfindahl index for heat)"""
        if not position_heat:
            return 0

        total_heat = sum(h['total_heat'] for h in position_heat.values())
        if total_heat == 0:
            return 0

        # Calculate concentration index
        heat_shares = [h['total_heat'] / total_heat for h in position_heat.values()]
        concentration = sum(share ** 2 for share in heat_shares)

        return concentration

    def _calculate_diversification_score(self, sector_heat: Dict) -> float:
        """Calculate portfolio diversification score (0-1, higher is better)"""
        if not sector_heat:
            return 0

        total_heat = sum(s['total_heat'] for s in sector_heat.values())
        if total_heat == 0:
            return 0

        # Perfect diversification would be equal heat across all sectors
        num_sectors = len(sector_heat)
        ideal_heat_per_sector = total_heat / num_sectors

        # Calculate deviation from ideal diversification
        deviations = []
        for sector_data in sector_heat.values():
            deviation = abs(sector_data['total_heat'] - ideal_heat_per_sector) / ideal_heat_per_sector
            deviations.append(deviation)

        avg_deviation = sum(deviations) / len(deviations)
        diversification_score = max(0, 1 - avg_deviation)

        return diversification_score

    def _assess_risk_level(self, heat_value: float) -> str:
        """Assess risk level based on heat value"""
        if heat_value >= self.max_portfolio_heat:
            return 'CRITICAL'
        elif heat_value >= self.max_portfolio_heat * 0.8:
            return 'HIGH'
        elif heat_value >= self.max_portfolio_heat * 0.6:
            return 'MEDIUM'
        elif heat_value >= self.max_portfolio_heat * 0.3:
            return 'LOW'
        else:
            return 'MINIMAL'

    def _generate_heat_warnings(self, total_heat: float, sector_heat: Dict,
                               correlation_heat: Dict) -> List[str]:
        """Generate risk warnings based on heat analysis"""
        warnings = []

        # Portfolio-level warnings
        if total_heat >= self.max_portfolio_heat:
            warnings.append(f"CRITICAL: Portfolio heat ({total_heat:.3f}) exceeds maximum ({self.max_portfolio_heat})")
        elif total_heat >= self.max_portfolio_heat * 0.9:
            warnings.append(f"HIGH: Portfolio heat ({total_heat:.3f}) approaching maximum")

        # Sector concentration warnings
        for sector, data in sector_heat.items():
            if data['heat_violation']:
                warnings.append(f"Sector concentration risk: {sector} heat ({data['total_heat']:.3f}) "
                               f"exceeds limit ({self.max_sector_heat})")

        # Correlation warnings
        high_corr_count = correlation_heat.get('high_correlation_count', 0)
        if high_corr_count > 3:
            warnings.append(f"High correlation risk: {high_corr_count} highly correlated position pairs")

        return warnings

    def _generate_heat_recommendations(self, position_heat: Dict, sector_heat: Dict,
                                     correlation_heat: Dict) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        # Position-specific recommendations
        sorted_positions = sorted(position_heat.items(),
                                key=lambda x: x[1]['total_heat'], reverse=True)

        if len(sorted_positions) > 0:
            hottest_position = sorted_positions[0]
            if hottest_position[1]['total_heat'] > 0.05:  # 5% heat
                recommendations.append(f"Consider reducing {hottest_position[0]} position "
                                     f"(highest heat: {hottest_position[1]['total_heat']:.3f})")

        # Sector recommendations
        hottest_sector = max(sector_heat.items(), key=lambda x: x[1]['total_heat'])
        if hottest_sector[1]['total_heat'] > self.max_sector_heat * 0.8:
            recommendations.append(f"Reduce exposure to {hottest_sector[0]} sector "
                                 f"({hottest_sector[1]['total_heat']:.3f} heat)")

        # Correlation recommendations
        if correlation_heat.get('high_correlation_count', 0) > 2:
            recommendations.append("Consider diversifying across less correlated assets")

        # General recommendations
        concentration = self._calculate_heat_concentration(position_heat)
        if concentration > 0.3:
            recommendations.append("Portfolio is highly concentrated - consider diversification")

        return recommendations

    def _update_heat_history(self, total_heat: float, sector_heat: Dict, position_heat: Dict):
        """Update historical heat tracking"""
        try:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'total_heat': total_heat,
                'sector_heat': {sector: data['total_heat'] for sector, data in sector_heat.items()},
                'position_count': len(position_heat),
                'max_position_heat': max(h['total_heat'] for h in position_heat.values()) if position_heat else 0
            }

            self.heat_history.append(history_entry)

            # Keep only last 100 entries
            if len(self.heat_history) > 100:
                self.heat_history = self.heat_history[-100:]

        except Exception as e:
            logger.error(f"Heat history update error: {e}")

    def _get_default_heat_analysis(self) -> Dict:
        """Return default heat analysis when calculation fails"""
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_metrics': {
                'total_heat': 0,
                'max_allowed_heat': self.max_portfolio_heat,
                'heat_utilization': 0,
                'heat_concentration': 0,
                'diversification_score': 1.0,
                'total_positions': 0,
                'total_portfolio_value': 0
            },
            'position_heat': {},
            'sector_heat': {},
            'correlation_heat': {'total_correlation_heat': 0, 'correlation_pairs': {}},
            'risk_assessment': {
                'risk_level': 'UNKNOWN',
                'warnings': ['Heat calculation failed'],
                'recommendations': ['Review system configuration'],
                'safe_to_add_position': False
            },
            'error': True
        }

    async def check_new_position_heat(self, symbol: str, position_size: float,
                                    current_positions: Dict, volatility_data: Dict) -> Dict:
        """Check if adding a new position would exceed heat limits"""
        try:
            # Create temporary position for heat calculation
            temp_positions = current_positions.copy()
            temp_positions[symbol] = {
                'notional': position_size,
                'entry_time': datetime.now().isoformat()
            }

            # Calculate heat with new position
            heat_analysis = await self.calculate_portfolio_heat(temp_positions, volatility_data)

            # Check if limits would be exceeded
            new_total_heat = heat_analysis['portfolio_metrics']['total_heat']
            new_sector_heat = heat_analysis['sector_heat']

            sector = self._classify_symbol_sector(symbol)
            sector_heat_after = new_sector_heat.get(sector, {}).get('total_heat', 0)

            return {
                'symbol': symbol,
                'position_size': position_size,
                'heat_impact': {
                    'new_total_heat': new_total_heat,
                    'heat_increase': new_total_heat - sum(h.get('total_heat', 0)
                                                        for h in current_positions.values()),
                    'new_sector_heat': sector_heat_after,
                    'sector': sector
                },
                'limits_check': {
                    'portfolio_limit_exceeded': new_total_heat > self.max_portfolio_heat,
                    'sector_limit_exceeded': sector_heat_after > self.max_sector_heat,
                    'safe_to_add': (new_total_heat <= self.max_portfolio_heat and
                                   sector_heat_after <= self.max_sector_heat)
                },
                'recommendations': self._get_position_sizing_recommendations(
                    symbol, position_size, new_total_heat, sector_heat_after
                )
            }

        except Exception as e:
            logger.error(f"New position heat check error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'position_size': position_size,
                'limits_check': {'safe_to_add': False},
                'error': str(e)
            }

    def _get_position_sizing_recommendations(self, symbol: str, position_size: float,
                                           new_total_heat: float, sector_heat: float) -> List[str]:
        """Get position sizing recommendations"""
        recommendations = []

        if new_total_heat > self.max_portfolio_heat:
            reduction_needed = (new_total_heat - self.max_portfolio_heat) / new_total_heat
            new_size = position_size * (1 - reduction_needed * 1.1)  # 10% buffer
            recommendations.append(f"Reduce position size to {new_size:.2f} to stay within portfolio heat limit")

        if sector_heat > self.max_sector_heat:
            sector = self._classify_symbol_sector(symbol)
            reduction_needed = (sector_heat - self.max_sector_heat) / sector_heat
            new_size = position_size * (1 - reduction_needed * 1.1)  # 10% buffer
            recommendations.append(f"Reduce position size to {new_size:.2f} to stay within {sector} sector limit")

        if not recommendations:
            recommendations.append("Position size is within acceptable heat limits")

        return recommendations

    def get_heat_statistics(self) -> Dict:
        """Get portfolio heat management statistics"""
        return {
            'heat_parameters': {
                'max_portfolio_heat': self.max_portfolio_heat,
                'max_sector_heat': self.max_sector_heat,
                'volatility_weight': self.volatility_weight,
                'correlation_weight': self.correlation_weight,
                'position_size_weight': self.position_size_weight,
                'time_decay_weight': self.time_decay_weight
            },
            'heat_history_length': len(self.heat_history),
            'violations_count': len(self.heat_violations),
            'adjustments_count': len(self.risk_adjustments),
            'sectors_tracked': len(self.crypto_sectors),
            'last_analysis': self.heat_history[-1] if self.heat_history else None
        }

    def update_heat_parameters(self, **kwargs):
        """Update heat management parameters"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                logger.info(f"Heat parameter updated: {param} = {value}")
            else:
                logger.warning(f"Unknown heat parameter: {param}")

    def clear_heat_history(self):
        """Clear heat tracking history"""
        self.heat_history.clear()
        self.heat_violations.clear()
        self.risk_adjustments.clear()
        logger.info("Heat tracking history cleared")