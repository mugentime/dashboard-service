import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from config.config import config
from src.core.trading_engine import TradingEngine
from src.ml.strategy_generator import StrategyGenerator
from src.risk.risk_manager import RiskManager
from src.monitoring.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

class HiveMindCoordinator:
    """Coordinates the collective intelligence of trading agents"""

    def __init__(self):
        self.agents = {}
        self.consensus_votes = {}
        self.shared_memory = {}
        self.coordination_active = False
        self.swarm_performance = {}

    async def initialize_swarm(self):
        """Initialize the hive-mind swarm with specialized agents"""
        logger.info("Initializing Hive-Mind Trading Swarm...")

        # Initialize specialized agents
        self.agents = {
            'strategy_generator': StrategyGeneratorAgent(),
            'market_intelligence': MarketIntelligenceAgent(),
            'execution_engine': ExecutionEngineAgent(),
            'risk_guardian': RiskGuardianAgent(),
            'performance_analyzer': PerformanceAnalyzerAgent(),
            'self_evolution': SelfEvolutionAgent()
        }

        # Initialize shared memory
        self.shared_memory = {
            'market_conditions': {},
            'strategy_performance': {},
            'risk_metrics': {},
            'consensus_decisions': {},
            'learning_patterns': {}
        }

        # Start agent coordination
        for agent_name, agent in self.agents.items():
            await agent.initialize(self.shared_memory)

        self.coordination_active = True
        logger.info("Hive-Mind Swarm initialized with 6 specialized agents")

    async def coordinate_trading_decision(self, market_data: Dict) -> Dict:
        """Coordinate a collective trading decision"""
        if not self.coordination_active:
            return {'action': 'HOLD', 'consensus': False}

        # Step 1: Gather agent insights
        agent_insights = {}
        for agent_name, agent in self.agents.items():
            try:
                insight = await agent.analyze(market_data)
                agent_insights[agent_name] = insight
            except Exception as e:
                logger.error(f"Agent {agent_name} analysis failed: {e}")

        # Step 2: Build consensus
        consensus_decision = await self._build_consensus(agent_insights)

        # Step 3: Update shared memory
        await self._update_shared_memory(agent_insights, consensus_decision)

        # Step 4: Execute decision if consensus reached
        if consensus_decision['consensus_strength'] >= config.hive_mind.consensus_threshold:
            return consensus_decision
        else:
            return {'action': 'HOLD', 'consensus': False, 'reason': 'Insufficient consensus'}

    async def _build_consensus(self, agent_insights: Dict) -> Dict:
        """Build consensus from agent insights using weighted voting"""
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidence_sum = 0
        agent_weights = self._get_agent_weights()

        # Collect weighted votes
        for agent_name, insight in agent_insights.items():
            action = insight.get('action', 'HOLD')
            confidence = insight.get('confidence', 0.5)
            weight = agent_weights.get(agent_name, 1.0)

            weighted_vote = confidence * weight
            votes[action] += weighted_vote
            confidence_sum += weighted_vote

        # Determine consensus
        total_votes = sum(votes.values())
        if total_votes == 0:
            return {'action': 'HOLD', 'consensus_strength': 0}

        # Find winning action
        winning_action = max(votes, key=votes.get)
        consensus_strength = votes[winning_action] / total_votes

        # Calculate position sizing from agent recommendations
        position_sizes = [
            insight.get('position_size', 0) for insight in agent_insights.values()
            if insight.get('action') == winning_action
        ]
        avg_position_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0

        return {
            'action': winning_action,
            'consensus_strength': consensus_strength,
            'position_size': avg_position_size,
            'agent_votes': votes,
            'contributing_agents': [
                name for name, insight in agent_insights.items()
                if insight.get('action') == winning_action
            ]
        }

    def _get_agent_weights(self) -> Dict[str, float]:
        """Get performance-based weights for agent voting"""
        # Base weights
        weights = {
            'strategy_generator': 1.0,
            'market_intelligence': 1.0,
            'execution_engine': 0.8,
            'risk_guardian': 1.2,  # Higher weight for risk management
            'performance_analyzer': 1.0,
            'self_evolution': 0.9
        }

        # Adjust based on recent performance
        for agent_name in weights:
            recent_performance = self.swarm_performance.get(agent_name, {})
            accuracy = recent_performance.get('accuracy', 0.5)

            # Boost weight for high-performing agents
            if accuracy > 0.7:
                weights[agent_name] *= 1.2
            elif accuracy < 0.4:
                weights[agent_name] *= 0.8

        return weights

    async def _update_shared_memory(self, agent_insights: Dict, consensus_decision: Dict):
        """Update shared memory with latest insights and decisions"""
        timestamp = datetime.now()

        # Update agent insights
        self.shared_memory['latest_insights'] = {
            'timestamp': timestamp,
            'insights': agent_insights,
            'consensus': consensus_decision
        }

        # Update learning patterns
        if 'learning_patterns' not in self.shared_memory:
            self.shared_memory['learning_patterns'] = []

        self.shared_memory['learning_patterns'].append({
            'timestamp': timestamp,
            'market_conditions': agent_insights.get('market_intelligence', {}).get('conditions', {}),
            'strategy_signals': agent_insights.get('strategy_generator', {}).get('signals', {}),
            'risk_assessment': agent_insights.get('risk_guardian', {}).get('risk_level', 'medium'),
            'consensus_action': consensus_decision['action'],
            'consensus_strength': consensus_decision['consensus_strength']
        })

        # Keep only recent patterns (last 1000 entries)
        if len(self.shared_memory['learning_patterns']) > 1000:
            self.shared_memory['learning_patterns'] = self.shared_memory['learning_patterns'][-1000:]

    async def update_agent_performance(self, agent_name: str, trade_result: Dict):
        """Update agent performance tracking"""
        if agent_name not in self.swarm_performance:
            self.swarm_performance[agent_name] = {
                'trades': [],
                'accuracy': 0.5,
                'contribution_score': 1.0
            }

        performance = self.swarm_performance[agent_name]
        performance['trades'].append(trade_result)

        # Calculate accuracy (simplified)
        recent_trades = performance['trades'][-50:]  # Last 50 trades
        if recent_trades:
            successful_trades = sum(1 for trade in recent_trades if trade.get('return', 0) > 0)
            performance['accuracy'] = successful_trades / len(recent_trades)

        # Update contribution score based on consensus participation
        if trade_result.get('participated_in_consensus'):
            performance['contribution_score'] = min(performance['contribution_score'] * 1.01, 2.0)

    async def evolve_swarm(self):
        """Trigger swarm evolution and self-improvement"""
        logger.info("Initiating swarm evolution...")

        # Analyze collective performance
        collective_metrics = await self._analyze_collective_performance()

        # Identify underperforming agents
        underperforming_agents = [
            name for name, perf in self.swarm_performance.items()
            if perf['accuracy'] < 0.4
        ]

        # Trigger agent evolution
        for agent_name in underperforming_agents:
            agent = self.agents.get(agent_name)
            if agent and hasattr(agent, 'evolve'):
                await agent.evolve(self.shared_memory['learning_patterns'])

        # Update swarm configuration if needed
        if collective_metrics['overall_accuracy'] < 0.5:
            await self._reconfigure_swarm()

        logger.info("Swarm evolution completed")

    async def _analyze_collective_performance(self) -> Dict:
        """Analyze overall swarm performance"""
        total_accuracy = 0
        total_agents = 0

        for agent_name, performance in self.swarm_performance.items():
            total_accuracy += performance['accuracy']
            total_agents += 1

        overall_accuracy = total_accuracy / total_agents if total_agents > 0 else 0.5

        return {
            'overall_accuracy': overall_accuracy,
            'agent_count': total_agents,
            'consensus_success_rate': self._calculate_consensus_success_rate(),
            'collective_sharpe': self._calculate_collective_sharpe()
        }

    def _calculate_consensus_success_rate(self) -> float:
        """Calculate how often consensus decisions were profitable"""
        patterns = self.shared_memory.get('learning_patterns', [])
        if not patterns:
            return 0.5

        recent_patterns = patterns[-100:]  # Last 100 decisions
        successful_consensus = sum(
            1 for pattern in recent_patterns
            if pattern.get('consensus_strength', 0) > 0.6 and pattern.get('outcome', 'neutral') == 'positive'
        )

        return successful_consensus / len(recent_patterns) if recent_patterns else 0.5

    def _calculate_collective_sharpe(self) -> float:
        """Calculate collective Sharpe ratio of swarm decisions"""
        # This would calculate based on actual trading results
        # For now, return a placeholder
        return 1.5

    async def _reconfigure_swarm(self):
        """Reconfigure swarm parameters for better performance"""
        logger.info("Reconfiguring swarm for improved performance...")

        # Adjust consensus threshold
        current_threshold = config.hive_mind.consensus_threshold
        if self._calculate_consensus_success_rate() < 0.5:
            new_threshold = min(current_threshold + 0.1, 0.8)
            config.hive_mind.consensus_threshold = new_threshold
            logger.info(f"Increased consensus threshold to {new_threshold}")

        # Rebalance agent weights
        self._rebalance_agent_weights()

    def _rebalance_agent_weights(self):
        """Rebalance agent weights based on performance"""
        for agent_name, performance in self.swarm_performance.items():
            accuracy = performance['accuracy']

            # Boost high performers, reduce low performers
            if accuracy > 0.7:
                performance['contribution_score'] = min(performance['contribution_score'] * 1.1, 2.0)
            elif accuracy < 0.3:
                performance['contribution_score'] = max(performance['contribution_score'] * 0.9, 0.5)

    async def get_swarm_status(self) -> Dict:
        """Get current swarm status and metrics"""
        collective_metrics = await self._analyze_collective_performance()

        return {
            'active': self.coordination_active,
            'agent_count': len(self.agents),
            'collective_performance': collective_metrics,
            'agent_performance': self.swarm_performance,
            'consensus_threshold': config.hive_mind.consensus_threshold,
            'shared_memory_size': len(self.shared_memory.get('learning_patterns', [])),
            'last_decision': self.shared_memory.get('latest_insights', {}).get('consensus', {}),
            'uptime': datetime.now().isoformat()
        }

# Specialized Agent Classes
class StrategyGeneratorAgent:
    """Agent responsible for ML strategy generation"""

    def __init__(self):
        self.strategy_generator = StrategyGenerator()
        self.shared_memory = None

    async def initialize(self, shared_memory: Dict):
        self.shared_memory = shared_memory

    async def analyze(self, market_data: Dict) -> Dict:
        """Analyze market data and generate strategy signals"""
        # This would use the actual strategy generator
        return {
            'action': 'BUY',  # Placeholder
            'confidence': 0.75,
            'position_size': 0.1,
            'reasoning': 'ML model predicts upward movement'
        }

class MarketIntelligenceAgent:
    """Agent for market analysis and intelligence"""

    async def initialize(self, shared_memory: Dict):
        self.shared_memory = shared_memory

    async def analyze(self, market_data: Dict) -> Dict:
        return {
            'action': 'BUY',
            'confidence': 0.6,
            'conditions': {'trend': 'bullish', 'volatility': 'medium'},
            'reasoning': 'Market showing bullish sentiment'
        }

class ExecutionEngineAgent:
    """Agent for trade execution optimization"""

    async def initialize(self, shared_memory: Dict):
        self.shared_memory = shared_memory

    async def analyze(self, market_data: Dict) -> Dict:
        return {
            'action': 'BUY',
            'confidence': 0.8,
            'optimal_timing': 'immediate',
            'reasoning': 'Optimal execution conditions'
        }

class RiskGuardianAgent:
    """Agent for risk management and capital protection"""

    def __init__(self):
        self.risk_manager = RiskManager()

    async def initialize(self, shared_memory: Dict):
        self.shared_memory = shared_memory

    async def analyze(self, market_data: Dict) -> Dict:
        return {
            'action': 'HOLD',  # Conservative by default
            'confidence': 0.9,
            'risk_level': 'low',
            'max_position_size': 0.05,
            'reasoning': 'Risk within acceptable parameters'
        }

class PerformanceAnalyzerAgent:
    """Agent for performance analysis and optimization"""

    def __init__(self):
        self.performance_tracker = PerformanceTracker()

    async def initialize(self, shared_memory: Dict):
        self.shared_memory = shared_memory

    async def analyze(self, market_data: Dict) -> Dict:
        return {
            'action': 'BUY',
            'confidence': 0.7,
            'performance_outlook': 'positive',
            'reasoning': 'Performance metrics support trade'
        }

class SelfEvolutionAgent:
    """Agent for continuous self-improvement"""

    async def initialize(self, shared_memory: Dict):
        self.shared_memory = shared_memory

    async def analyze(self, market_data: Dict) -> Dict:
        return {
            'action': 'BUY',
            'confidence': 0.65,
            'evolution_recommendation': 'maintain_current_strategy',
            'reasoning': 'Current strategies performing adequately'
        }

    async def evolve(self, learning_patterns: List[Dict]):
        """Evolve agent behavior based on learning patterns"""
        logger.info("Self-evolution agent analyzing patterns for improvement...")
        # Implementation would analyze patterns and update agent behaviors