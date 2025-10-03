import asyncio
import logging
import subprocess
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from src.testing.hive_mind_test_analyzer import HiveMindTestAnalyzer
from src.agents.ui_optimization_agent import UIOptimizationAgent
from src.agents.hive_mind_coordinator import HiveMindCoordinator

logger = logging.getLogger(__name__)

class AutomatedTestingLoop:
    """Automated testing and optimization loop using Playwright MCP and Hive-Mind"""

    def __init__(self, hive_mind: HiveMindCoordinator):
        self.hive_mind = hive_mind
        self.test_analyzer = HiveMindTestAnalyzer(hive_mind)
        self.ui_optimizer = UIOptimizationAgent(self.test_analyzer)
        self.iteration_count = 0
        self.max_iterations = 5
        self.target_success_rate = 0.95
        self.improvement_threshold = 0.05  # 5% improvement minimum

    async def run_complete_testing_loop(self) -> Dict[str, Any]:
        """Run the complete testing and optimization loop"""
        logger.info("🚀 Starting automated testing and optimization loop...")

        loop_results = {
            'start_time': datetime.now().isoformat(),
            'iterations': [],
            'final_status': 'pending',
            'total_improvements': {},
            'recommendations': []
        }

        try:
            # Initial baseline test
            baseline_results = await self._run_baseline_tests()
            loop_results['baseline'] = baseline_results

            # Main optimization loop
            while (self.iteration_count < self.max_iterations and
                   not self._has_reached_target(baseline_results)):

                self.iteration_count += 1
                logger.info(f"🔄 Starting optimization iteration {self.iteration_count}/{self.max_iterations}")

                iteration_result = await self._run_single_iteration()
                loop_results['iterations'].append(iteration_result)

                # Check if we should continue
                if iteration_result.get('success_rate', 0) >= self.target_success_rate:
                    logger.info("🎯 Target success rate achieved!")
                    break

                if iteration_result.get('improvement_score', 0) < self.improvement_threshold:
                    logger.warning("⚠️ Minimal improvement detected, may need different approach")

            # Final results
            loop_results['end_time'] = datetime.now().isoformat()
            loop_results['final_status'] = await self._determine_final_status(loop_results)
            loop_results['total_improvements'] = await self._calculate_total_improvements(loop_results)
            loop_results['recommendations'] = await self._generate_final_recommendations(loop_results)

            logger.info(f"✅ Testing loop completed after {self.iteration_count} iterations")
            return loop_results

        except Exception as e:
            logger.error(f"❌ Testing loop failed: {e}")
            loop_results['final_status'] = 'failed'
            loop_results['error'] = str(e)
            return loop_results

    async def _run_baseline_tests(self) -> Dict[str, Any]:
        """Run initial baseline tests"""
        logger.info("📊 Running baseline tests...")

        try:
            # Run Playwright tests
            test_results = await self._execute_playwright_tests()

            # Analyze baseline performance
            baseline_analysis = {
                'timestamp': datetime.now().isoformat(),
                'test_results': test_results,
                'performance_metrics': await self._extract_performance_metrics(test_results),
                'success_rate': self._calculate_success_rate(test_results),
                'baseline_established': True
            }

            return baseline_analysis

        except Exception as e:
            logger.error(f"❌ Baseline test failed: {e}")
            return {'error': str(e), 'baseline_established': False}

    async def _run_single_iteration(self) -> Dict[str, Any]:
        """Run a single iteration of test -> analyze -> optimize -> retest"""
        iteration_start = datetime.now()

        iteration_result = {
            'iteration_number': self.iteration_count,
            'start_time': iteration_start.isoformat(),
            'phases': {}
        }

        try:
            # Phase 1: Run tests
            logger.info("📝 Phase 1: Running Playwright tests...")
            test_results = await self._execute_playwright_tests()
            iteration_result['phases']['testing'] = {
                'status': 'completed',
                'test_results': test_results,
                'duration': (datetime.now() - iteration_start).total_seconds()
            }

            # Phase 2: Hive-mind analysis
            logger.info("🧠 Phase 2: Running hive-mind analysis...")
            analysis_start = datetime.now()

            # Save test results to file for analysis
            results_file = f"test-results/iteration-{self.iteration_count}-results.json"
            await self._save_test_results(test_results, results_file)

            # Analyze with hive-mind
            analysis_result = await self.test_analyzer.analyze_test_results(results_file)

            iteration_result['phases']['analysis'] = {
                'status': 'completed',
                'analysis_result': analysis_result,
                'duration': (datetime.now() - analysis_start).total_seconds()
            }

            # Phase 3: Apply optimizations
            logger.info("🔧 Phase 3: Applying optimizations...")
            optimization_start = datetime.now()

            optimization_result = await self.ui_optimizer.process_test_results_and_optimize(results_file)

            iteration_result['phases']['optimization'] = {
                'status': 'completed',
                'optimization_result': optimization_result,
                'duration': (datetime.now() - optimization_start).total_seconds()
            }

            # Phase 4: Validation tests
            logger.info("✅ Phase 4: Running validation tests...")
            validation_start = datetime.now()

            validation_results = await self._execute_playwright_tests()

            iteration_result['phases']['validation'] = {
                'status': 'completed',
                'validation_results': validation_results,
                'duration': (datetime.now() - validation_start).total_seconds()
            }

            # Calculate iteration metrics
            iteration_result.update(await self._calculate_iteration_metrics(
                test_results, validation_results, analysis_result, optimization_result
            ))

            iteration_result['total_duration'] = (datetime.now() - iteration_start).total_seconds()
            iteration_result['status'] = 'completed'

            return iteration_result

        except Exception as e:
            logger.error(f"❌ Iteration {self.iteration_count} failed: {e}")
            iteration_result['status'] = 'failed'
            iteration_result['error'] = str(e)
            return iteration_result

    async def _execute_playwright_tests(self) -> Dict[str, Any]:
        """Execute Playwright tests and return results"""
        logger.info("🎭 Executing Playwright tests...")

        try:
            # Ensure test results directory exists
            results_dir = Path("test-results")
            results_dir.mkdir(exist_ok=True)

            # Run Playwright tests
            cmd = [
                "npx", "playwright", "test",
                "--reporter=json",
                f"--output-dir=test-results"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )

            stdout, stderr = await process.communicate()

            # Parse test results
            if process.returncode == 0 or process.returncode == 1:  # 1 = tests failed but ran
                try:
                    # Try to read from results.json if it exists
                    results_file = results_dir / "results.json"
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            test_results = json.load(f)
                    else:
                        # Parse from stdout
                        test_results = json.loads(stdout.decode())

                    return test_results

                except json.JSONDecodeError:
                    logger.warning("Could not parse test results as JSON")
                    return {
                        'error': 'JSON parse error',
                        'stdout': stdout.decode(),
                        'stderr': stderr.decode()
                    }
            else:
                logger.error(f"Playwright tests failed with exit code {process.returncode}")
                return {
                    'error': f'Test execution failed: exit code {process.returncode}',
                    'stderr': stderr.decode()
                }

        except Exception as e:
            logger.error(f"Error executing Playwright tests: {e}")
            return {'error': str(e)}

    async def _save_test_results(self, test_results: Dict, filename: str):
        """Save test results to file"""
        results_dir = Path("test-results")
        results_dir.mkdir(exist_ok=True)

        file_path = results_dir / filename
        with open(file_path, 'w') as f:
            json.dump(test_results, f, indent=2)

    async def _extract_performance_metrics(self, test_results: Dict) -> Dict[str, Any]:
        """Extract performance metrics from test results"""
        metrics = {
            'load_times': [],
            'interaction_times': [],
            'memory_usage': [],
            'api_response_times': []
        }

        # Parse test results for performance data
        for suite in test_results.get('suites', []):
            if 'performance' in suite.get('title', '').lower():
                for spec in suite.get('specs', []):
                    for test in spec.get('tests', []):
                        # Extract metrics from test results if available
                        duration = test.get('duration', 0)
                        if 'load' in test.get('title', '').lower():
                            metrics['load_times'].append(duration)
                        elif 'interaction' in test.get('title', '').lower():
                            metrics['interaction_times'].append(duration)

        return metrics

    def _calculate_success_rate(self, test_results: Dict) -> float:
        """Calculate test success rate"""
        stats = test_results.get('stats', {})
        total = stats.get('total', 0)
        passed = stats.get('passed', 0)

        return passed / total if total > 0 else 0.0

    def _has_reached_target(self, test_results: Dict) -> bool:
        """Check if target success rate has been reached"""
        success_rate = self._calculate_success_rate(test_results)
        return success_rate >= self.target_success_rate

    async def _calculate_iteration_metrics(self, initial_results: Dict, validation_results: Dict,
                                         analysis_result: Dict, optimization_result: Dict) -> Dict:
        """Calculate metrics for this iteration"""
        initial_success_rate = self._calculate_success_rate(initial_results)
        validation_success_rate = self._calculate_success_rate(validation_results)

        improvement_score = validation_success_rate - initial_success_rate

        return {
            'initial_success_rate': initial_success_rate,
            'validation_success_rate': validation_success_rate,
            'improvement_score': improvement_score,
            'optimizations_applied': optimization_result.get('optimizations_implemented', 0),
            'hive_mind_confidence': analysis_result.get('hive_mind_analysis', {}).get('confidence_score', 0),
            'performance_improved': improvement_score > 0
        }

    async def _determine_final_status(self, loop_results: Dict) -> str:
        """Determine the final status of the testing loop"""
        iterations = loop_results.get('iterations', [])

        if not iterations:
            return 'no_iterations_completed'

        final_iteration = iterations[-1]
        final_success_rate = final_iteration.get('validation_success_rate', 0)

        if final_success_rate >= self.target_success_rate:
            return 'target_achieved'
        elif final_success_rate >= 0.8:
            return 'substantial_improvement'
        elif final_success_rate >= 0.6:
            return 'moderate_improvement'
        else:
            return 'minimal_improvement'

    async def _calculate_total_improvements(self, loop_results: Dict) -> Dict[str, Any]:
        """Calculate total improvements across all iterations"""
        baseline = loop_results.get('baseline', {})
        iterations = loop_results.get('iterations', [])

        if not iterations or not baseline:
            return {}

        baseline_success_rate = baseline.get('success_rate', 0)
        final_success_rate = iterations[-1].get('validation_success_rate', 0)

        total_improvement = final_success_rate - baseline_success_rate

        return {
            'baseline_success_rate': baseline_success_rate,
            'final_success_rate': final_success_rate,
            'total_improvement': total_improvement,
            'improvement_percentage': (total_improvement / baseline_success_rate * 100) if baseline_success_rate > 0 else 0,
            'iterations_completed': len(iterations),
            'target_achieved': final_success_rate >= self.target_success_rate
        }

    async def _generate_final_recommendations(self, loop_results: Dict) -> List[str]:
        """Generate final recommendations based on loop results"""
        recommendations = []

        final_status = loop_results.get('final_status', '')
        total_improvements = loop_results.get('total_improvements', {})

        if final_status == 'target_achieved':
            recommendations.extend([
                "🎯 Excellent! Target success rate achieved",
                "📊 Monitor performance metrics to maintain improvements",
                "🔄 Consider increasing target thresholds for further optimization"
            ])
        elif final_status == 'substantial_improvement':
            recommendations.extend([
                "✅ Substantial improvements made",
                "🎯 Continue optimizing to reach target success rate",
                "🔍 Analyze remaining failure patterns for additional fixes"
            ])
        elif final_status == 'moderate_improvement':
            recommendations.extend([
                "📈 Moderate improvements achieved",
                "🧠 Consider alternative optimization strategies",
                "🔄 May need additional hive-mind analysis iterations"
            ])
        else:
            recommendations.extend([
                "⚠️ Limited improvements detected",
                "🔄 Review optimization strategy",
                "🧠 Consider manual intervention or different approaches",
                "📊 Analyze test failures for systemic issues"
            ])

        # Add specific technical recommendations
        iterations = loop_results.get('iterations', [])
        if iterations:
            last_iteration = iterations[-1]

            if last_iteration.get('optimizations_applied', 0) == 0:
                recommendations.append("🔧 No optimizations were applied - review hive-mind consensus threshold")

            if last_iteration.get('hive_mind_confidence', 0) < 0.6:
                recommendations.append("🧠 Low hive-mind confidence - may need additional training data")

        return recommendations

    async def force_stop_loop(self):
        """Force stop the testing loop"""
        logger.info("🛑 Forcing stop of testing loop...")
        self.iteration_count = self.max_iterations

    async def get_loop_status(self) -> Dict[str, Any]:
        """Get current status of the testing loop"""
        return {
            'iteration_count': self.iteration_count,
            'max_iterations': self.max_iterations,
            'target_success_rate': self.target_success_rate,
            'improvement_threshold': self.improvement_threshold,
            'is_running': self.iteration_count < self.max_iterations,
            'progress_percentage': (self.iteration_count / self.max_iterations) * 100
        }