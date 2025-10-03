import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from src.agents.hive_mind_coordinator import HiveMindCoordinator

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Structured test result data"""
    test_name: str
    status: str  # 'passed', 'failed', 'skipped'
    duration: float
    error_message: Optional[str]
    performance_metrics: Dict[str, Any]
    screenshot_path: Optional[str]
    category: str  # 'responsiveness', 'performance', 'accessibility'

@dataclass
class UIOptimization:
    """UI optimization recommendation"""
    priority: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'css', 'javascript', 'architecture', 'api'
    description: str
    fix_code: str
    file_path: str
    confidence: float
    impact_score: float

class HiveMindTestAnalyzer:
    """Analyzes Playwright test results using hive-mind collective intelligence"""

    def __init__(self, hive_mind: HiveMindCoordinator):
        self.hive_mind = hive_mind
        self.test_results: List[TestResult] = []
        self.analysis_history: List[Dict] = []
        self.optimization_cache: Dict[str, List[UIOptimization]] = {}

    async def analyze_test_results(self, test_results_path: str) -> Dict[str, Any]:
        """Analyze test results using collective intelligence"""
        logger.info("🧠 Starting hive-mind analysis of test results...")

        try:
            # Load test results
            test_data = await self._load_test_results(test_results_path)

            # Parse into structured format
            self.test_results = await self._parse_test_results(test_data)

            # Coordinate hive-mind analysis
            analysis_result = await self._coordinate_collective_analysis()

            # Generate optimization recommendations
            optimizations = await self._generate_optimizations(analysis_result)

            # Build consensus on fixes
            consensus_fixes = await self._build_consensus_on_fixes(optimizations)

            # Create comprehensive analysis report
            analysis_report = {
                'timestamp': datetime.now().isoformat(),
                'test_summary': self._create_test_summary(),
                'hive_mind_analysis': analysis_result,
                'optimization_recommendations': consensus_fixes,
                'performance_insights': self._extract_performance_insights(),
                'prioritized_fixes': self._prioritize_fixes(consensus_fixes),
                'next_actions': self._generate_next_actions(consensus_fixes)
            }

            # Store analysis for learning
            self.analysis_history.append(analysis_report)

            logger.info("✅ Hive-mind test analysis completed")
            return analysis_report

        except Exception as e:
            logger.error(f"❌ Hive-mind test analysis failed: {e}")
            raise

    async def _load_test_results(self, results_path: str) -> Dict:
        """Load test results from file"""
        try:
            with open(results_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Test results file not found: {results_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in test results: {e}")
            raise

    async def _parse_test_results(self, test_data: Dict) -> List[TestResult]:
        """Parse raw test data into structured format"""
        results = []

        for suite in test_data.get('suites', []):
            suite_name = suite.get('title', 'Unknown Suite')
            category = self._determine_test_category(suite_name)

            for spec in suite.get('specs', []):
                for test in spec.get('tests', []):
                    result = TestResult(
                        test_name=f"{suite_name} - {test.get('title', 'Unknown Test')}",
                        status=test.get('status', 'unknown'),
                        duration=test.get('duration', 0),
                        error_message=self._extract_error_message(test),
                        performance_metrics=self._extract_performance_metrics(test),
                        screenshot_path=self._extract_screenshot_path(test),
                        category=category
                    )
                    results.append(result)

        return results

    def _determine_test_category(self, suite_name: str) -> str:
        """Determine test category from suite name"""
        suite_lower = suite_name.lower()

        if 'performance' in suite_lower or 'benchmark' in suite_lower:
            return 'performance'
        elif 'accessibility' in suite_lower or 'a11y' in suite_lower:
            return 'accessibility'
        elif 'responsive' in suite_lower or 'mobile' in suite_lower:
            return 'responsiveness'
        else:
            return 'functional'

    def _extract_error_message(self, test: Dict) -> Optional[str]:
        """Extract error message from test result"""
        for result in test.get('results', []):
            if result.get('status') == 'failed':
                error = result.get('error', {})
                return error.get('message', 'Unknown error')
        return None

    def _extract_performance_metrics(self, test: Dict) -> Dict[str, Any]:
        """Extract performance metrics from test"""
        metrics = {}

        # Look for performance data in test annotations or attachments
        for result in test.get('results', []):
            attachments = result.get('attachments', [])
            for attachment in attachments:
                if attachment.get('name', '').startswith('performance-'):
                    try:
                        # Parse performance data if available
                        if attachment.get('contentType') == 'application/json':
                            perf_data = json.loads(attachment.get('body', '{}'))
                            metrics.update(perf_data)
                    except:
                        pass

        return metrics

    def _extract_screenshot_path(self, test: Dict) -> Optional[str]:
        """Extract screenshot path from test result"""
        for result in test.get('results', []):
            attachments = result.get('attachments', [])
            for attachment in attachments:
                if attachment.get('contentType', '').startswith('image/'):
                    return attachment.get('path')
        return None

    async def _coordinate_collective_analysis(self) -> Dict[str, Any]:
        """Coordinate analysis across all hive-mind agents"""
        logger.info("🔄 Coordinating collective intelligence analysis...")

        # Prepare analysis data for agents
        analysis_data = {
            'test_results': [
                {
                    'name': result.test_name,
                    'status': result.status,
                    'duration': result.duration,
                    'category': result.category,
                    'performance_metrics': result.performance_metrics,
                    'error': result.error_message
                }
                for result in self.test_results
            ],
            'summary_stats': self._calculate_summary_stats(),
            'failure_patterns': self._identify_failure_patterns(),
            'performance_trends': self._analyze_performance_trends()
        }

        # Get insights from each agent
        agent_insights = {}

        for agent_name, agent in self.hive_mind.agents.items():
            try:
                insight = await self._get_agent_test_analysis(agent_name, agent, analysis_data)
                agent_insights[agent_name] = insight
            except Exception as e:
                logger.error(f"Agent {agent_name} analysis failed: {e}")
                agent_insights[agent_name] = {'error': str(e)}

        # Build collective consensus
        consensus = await self._build_analysis_consensus(agent_insights)

        return {
            'individual_insights': agent_insights,
            'collective_consensus': consensus,
            'confidence_score': self._calculate_consensus_confidence(agent_insights),
            'analysis_timestamp': datetime.now().isoformat()
        }

    async def _get_agent_test_analysis(self, agent_name: str, agent: Any, data: Dict) -> Dict:
        """Get test analysis from specific agent"""

        if agent_name == 'strategy_generator':
            return await self._analyze_strategy_perspective(data)
        elif agent_name == 'market_intelligence':
            return await self._analyze_intelligence_perspective(data)
        elif agent_name == 'execution_engine':
            return await self._analyze_execution_perspective(data)
        elif agent_name == 'risk_guardian':
            return await self._analyze_risk_perspective(data)
        elif agent_name == 'performance_analyzer':
            return await self._analyze_performance_perspective(data)
        elif agent_name == 'self_evolution':
            return await self._analyze_evolution_perspective(data)
        else:
            return {'analysis': 'Generic analysis', 'recommendations': []}

    async def _analyze_strategy_perspective(self, data: Dict) -> Dict:
        """Analyze from strategy generation perspective"""
        failed_tests = [t for t in data['test_results'] if t['status'] == 'failed']

        recommendations = []

        # Focus on UI strategy for better user experience
        if any('responsiveness' in t['category'] for t in failed_tests):
            recommendations.append({
                'type': 'ui_strategy',
                'priority': 'high',
                'description': 'Implement adaptive UI strategy based on device capabilities',
                'confidence': 0.8
            })

        return {
            'perspective': 'strategy_generation',
            'key_insights': [
                'UI responsiveness directly impacts user engagement',
                'Failed tests indicate need for adaptive strategy'
            ],
            'recommendations': recommendations,
            'confidence': 0.75
        }

    async def _analyze_intelligence_perspective(self, data: Dict) -> Dict:
        """Analyze from market intelligence perspective"""
        performance_issues = [
            t for t in data['test_results']
            if t['category'] == 'performance' and t['status'] == 'failed'
        ]

        return {
            'perspective': 'market_intelligence',
            'key_insights': [
                'Performance bottlenecks affect real-time data processing',
                'Network latency impacts trading decision speed'
            ],
            'recommendations': [
                {
                    'type': 'data_processing',
                    'priority': 'critical',
                    'description': 'Optimize real-time data pipeline for better performance',
                    'confidence': 0.9
                }
            ],
            'confidence': 0.85
        }

    async def _analyze_execution_perspective(self, data: Dict) -> Dict:
        """Analyze from execution engine perspective"""
        interaction_failures = [
            t for t in data['test_results']
            if 'interaction' in t['name'].lower() and t['status'] == 'failed'
        ]

        return {
            'perspective': 'execution_engine',
            'key_insights': [
                'Button response times critical for trade execution',
                'UI lag can delay critical trading actions'
            ],
            'recommendations': [
                {
                    'type': 'interaction_optimization',
                    'priority': 'critical',
                    'description': 'Reduce button click response time to <50ms',
                    'confidence': 0.95
                }
            ],
            'confidence': 0.9
        }

    async def _analyze_risk_perspective(self, data: Dict) -> Dict:
        """Analyze from risk management perspective"""
        return {
            'perspective': 'risk_guardian',
            'key_insights': [
                'UI failures could lead to missed risk alerts',
                'Mobile responsiveness critical for monitoring on-the-go'
            ],
            'recommendations': [
                {
                    'type': 'risk_ui',
                    'priority': 'high',
                    'description': 'Ensure critical alerts work on all device sizes',
                    'confidence': 0.8
                }
            ],
            'confidence': 0.8
        }

    async def _analyze_performance_perspective(self, data: Dict) -> Dict:
        """Analyze from performance analysis perspective"""
        performance_metrics = data.get('performance_trends', {})

        return {
            'perspective': 'performance_analyzer',
            'key_insights': [
                f"Average load time: {performance_metrics.get('avg_load_time', 'unknown')}",
                'Memory usage patterns need optimization'
            ],
            'recommendations': [
                {
                    'type': 'performance_optimization',
                    'priority': 'high',
                    'description': 'Implement code splitting and lazy loading',
                    'confidence': 0.85
                }
            ],
            'confidence': 0.9
        }

    async def _analyze_evolution_perspective(self, data: Dict) -> Dict:
        """Analyze from self-evolution perspective"""
        failure_patterns = data.get('failure_patterns', [])

        return {
            'perspective': 'self_evolution',
            'key_insights': [
                'Test failures show areas for autonomous improvement',
                'Pattern analysis reveals systematic issues'
            ],
            'recommendations': [
                {
                    'type': 'automated_optimization',
                    'priority': 'medium',
                    'description': 'Implement self-healing UI components',
                    'confidence': 0.7
                }
            ],
            'confidence': 0.75
        }

    async def _build_analysis_consensus(self, agent_insights: Dict) -> Dict:
        """Build consensus from agent insights"""
        all_recommendations = []

        for agent_name, insight in agent_insights.items():
            if 'recommendations' in insight:
                for rec in insight['recommendations']:
                    rec['source_agent'] = agent_name
                    all_recommendations.append(rec)

        # Group recommendations by type
        recommendation_groups = {}
        for rec in all_recommendations:
            rec_type = rec.get('type', 'other')
            if rec_type not in recommendation_groups:
                recommendation_groups[rec_type] = []
            recommendation_groups[rec_type].append(rec)

        # Build consensus on each type
        consensus_recommendations = []
        for rec_type, recs in recommendation_groups.items():
            if len(recs) >= 2:  # At least 2 agents agree
                consensus_rec = {
                    'type': rec_type,
                    'consensus_strength': len(recs) / len(self.hive_mind.agents),
                    'priority': self._determine_consensus_priority(recs),
                    'description': self._merge_descriptions(recs),
                    'confidence': np.mean([r.get('confidence', 0.5) for r in recs]),
                    'supporting_agents': [r['source_agent'] for r in recs]
                }
                consensus_recommendations.append(consensus_rec)

        return {
            'recommendations': consensus_recommendations,
            'consensus_threshold_met': len(consensus_recommendations) > 0,
            'total_recommendations': len(all_recommendations),
            'consensus_count': len(consensus_recommendations)
        }

    def _determine_consensus_priority(self, recommendations: List[Dict]) -> str:
        """Determine consensus priority from multiple recommendations"""
        priorities = [r.get('priority', 'low') for r in recommendations]
        priority_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}

        avg_weight = np.mean([priority_weights.get(p, 1) for p in priorities])

        if avg_weight >= 3.5:
            return 'critical'
        elif avg_weight >= 2.5:
            return 'high'
        elif avg_weight >= 1.5:
            return 'medium'
        else:
            return 'low'

    def _merge_descriptions(self, recommendations: List[Dict]) -> str:
        """Merge recommendation descriptions"""
        descriptions = [r.get('description', '') for r in recommendations]
        # For now, return the most detailed description
        return max(descriptions, key=len) if descriptions else ''

    async def _generate_optimizations(self, analysis_result: Dict) -> List[UIOptimization]:
        """Generate specific UI optimizations based on analysis"""
        optimizations = []

        consensus = analysis_result.get('collective_consensus', {})
        recommendations = consensus.get('recommendations', [])

        for rec in recommendations:
            if rec['consensus_strength'] >= 0.6:  # 60% consensus threshold
                optimization = await self._create_optimization_from_recommendation(rec)
                if optimization:
                    optimizations.append(optimization)

        return optimizations

    async def _create_optimization_from_recommendation(self, rec: Dict) -> Optional[UIOptimization]:
        """Create specific optimization from recommendation"""
        rec_type = rec.get('type', '')

        if rec_type == 'interaction_optimization':
            return UIOptimization(
                priority=rec.get('priority', 'medium'),
                category='javascript',
                description='Optimize button click response time',
                fix_code=self._generate_interaction_fix_code(),
                file_path='src/api/web_dashboard.py',
                confidence=rec.get('confidence', 0.5),
                impact_score=0.8
            )
        elif rec_type == 'performance_optimization':
            return UIOptimization(
                priority=rec.get('priority', 'medium'),
                category='javascript',
                description='Implement code splitting and lazy loading',
                fix_code=self._generate_performance_fix_code(),
                file_path='src/api/web_dashboard.py',
                confidence=rec.get('confidence', 0.5),
                impact_score=0.9
            )
        # Add more optimization types as needed

        return None

    def _generate_interaction_fix_code(self) -> str:
        """Generate code to fix interaction responsiveness"""
        return """
// Optimize button interactions with debouncing and immediate feedback
function optimizeButtonInteractions() {
    const buttons = document.querySelectorAll('.btn');

    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Immediate visual feedback
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = 'scale(1)';
            }, 100);

            // Debounce rapid clicks
            if (this.dataset.clicking) return;
            this.dataset.clicking = 'true';
            setTimeout(() => delete this.dataset.clicking, 500);
        });
    });
}

// Call on page load
document.addEventListener('DOMContentLoaded', optimizeButtonInteractions);
        """

    def _generate_performance_fix_code(self) -> str:
        """Generate code for performance optimization"""
        return """
// Implement lazy loading for chart components
const chartObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            loadChart(entry.target);
            chartObserver.unobserve(entry.target);
        }
    });
});

// Optimize WebSocket message handling
let messageQueue = [];
let processingQueue = false;

function optimizeWebSocketMessages(data) {
    messageQueue.push(data);

    if (!processingQueue) {
        processingQueue = true;
        requestAnimationFrame(processMessageQueue);
    }
}

function processMessageQueue() {
    const batch = messageQueue.splice(0, 10); // Process in batches
    batch.forEach(updateDashboard);

    if (messageQueue.length > 0) {
        requestAnimationFrame(processMessageQueue);
    } else {
        processingQueue = false;
    }
}
        """

    async def _build_consensus_on_fixes(self, optimizations: List[UIOptimization]) -> List[UIOptimization]:
        """Build consensus on which fixes to implement"""
        # Sort by impact score and confidence
        scored_optimizations = []

        for opt in optimizations:
            score = (opt.impact_score * 0.6) + (opt.confidence * 0.4)
            if opt.priority == 'critical':
                score *= 1.5
            elif opt.priority == 'high':
                score *= 1.2

            scored_optimizations.append((score, opt))

        # Return top optimizations that meet consensus threshold
        sorted_opts = sorted(scored_optimizations, key=lambda x: x[0], reverse=True)

        return [opt for score, opt in sorted_opts if score >= 0.6]

    def _calculate_summary_stats(self) -> Dict:
        """Calculate summary statistics from test results"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.status == 'passed')
        failed_tests = sum(1 for r in self.test_results if r.status == 'failed')

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'average_duration': np.mean([r.duration for r in self.test_results]) if self.test_results else 0
        }

    def _identify_failure_patterns(self) -> List[Dict]:
        """Identify patterns in test failures"""
        failed_tests = [r for r in self.test_results if r.status == 'failed']

        patterns = []

        # Group by category
        category_failures = {}
        for test in failed_tests:
            cat = test.category
            if cat not in category_failures:
                category_failures[cat] = []
            category_failures[cat].append(test)

        for category, failures in category_failures.items():
            if len(failures) >= 2:  # Pattern if 2+ failures in category
                patterns.append({
                    'type': 'category_failure',
                    'category': category,
                    'count': len(failures),
                    'tests': [f.test_name for f in failures]
                })

        return patterns

    def _analyze_performance_trends(self) -> Dict:
        """Analyze performance trends from test results"""
        performance_tests = [
            r for r in self.test_results
            if r.category == 'performance' and r.performance_metrics
        ]

        if not performance_tests:
            return {}

        durations = [r.duration for r in performance_tests]

        return {
            'avg_load_time': np.mean(durations),
            'max_load_time': np.max(durations),
            'min_load_time': np.min(durations),
            'performance_variance': np.var(durations)
        }

    def _calculate_consensus_confidence(self, agent_insights: Dict) -> float:
        """Calculate confidence in consensus analysis"""
        confidences = [
            insight.get('confidence', 0.5)
            for insight in agent_insights.values()
            if 'confidence' in insight
        ]

        return np.mean(confidences) if confidences else 0.5

    def _create_test_summary(self) -> Dict:
        """Create summary of test results"""
        return {
            'total_tests': len(self.test_results),
            'passed': sum(1 for r in self.test_results if r.status == 'passed'),
            'failed': sum(1 for r in self.test_results if r.status == 'failed'),
            'skipped': sum(1 for r in self.test_results if r.status == 'skipped'),
            'categories': {
                'performance': sum(1 for r in self.test_results if r.category == 'performance'),
                'responsiveness': sum(1 for r in self.test_results if r.category == 'responsiveness'),
                'accessibility': sum(1 for r in self.test_results if r.category == 'accessibility'),
                'functional': sum(1 for r in self.test_results if r.category == 'functional')
            }
        }

    def _extract_performance_insights(self) -> Dict:
        """Extract performance insights from analysis"""
        performance_tests = [r for r in self.test_results if r.category == 'performance']

        insights = {
            'load_time_issues': [],
            'memory_issues': [],
            'interaction_delays': [],
            'api_performance': []
        }

        for test in performance_tests:
            if test.status == 'failed':
                if 'load' in test.test_name.lower():
                    insights['load_time_issues'].append(test.test_name)
                elif 'memory' in test.test_name.lower():
                    insights['memory_issues'].append(test.test_name)
                elif 'interaction' in test.test_name.lower():
                    insights['interaction_delays'].append(test.test_name)
                elif 'api' in test.test_name.lower():
                    insights['api_performance'].append(test.test_name)

        return insights

    def _prioritize_fixes(self, consensus_fixes: List[UIOptimization]) -> List[UIOptimization]:
        """Prioritize fixes based on impact and feasibility"""
        priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}

        return sorted(
            consensus_fixes,
            key=lambda x: (
                priority_order.get(x.priority, 1),
                x.impact_score,
                x.confidence
            ),
            reverse=True
        )

    def _generate_next_actions(self, consensus_fixes: List[UIOptimization]) -> List[str]:
        """Generate list of next actions based on analysis"""
        actions = []

        if not consensus_fixes:
            actions.append("No critical issues found - maintain current performance")
            return actions

        critical_fixes = [f for f in consensus_fixes if f.priority == 'critical']
        if critical_fixes:
            actions.append(f"Immediately implement {len(critical_fixes)} critical fixes")

        high_fixes = [f for f in consensus_fixes if f.priority == 'high']
        if high_fixes:
            actions.append(f"Schedule {len(high_fixes)} high-priority optimizations")

        actions.append("Run validation tests after implementing fixes")
        actions.append("Monitor performance metrics for improvements")

        return actions