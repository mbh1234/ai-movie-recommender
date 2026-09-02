#!/usr/bin/env python3
"""
Comprehensive Test Runner for Movie Recommendation System - Milestone 2
Runs all tests with proper coverage reporting and generates milestone reports
"""

import subprocess
import sys
import os
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class Milestone2TestRunner:
    """Test runner for Milestone 2 requirements"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {
            'pipeline_tests': {},
            'coverage_results': {},
            'milestone_compliance': {},
            'timestamp': self.start_time.isoformat()
        }
        
    def setup_environment(self):
        """Setup test environment"""
        print("🔧 Setting up test environment...")
        
        # Create required directories
        dirs = ['data', 'models', 'reports', 'baseline_data', 'htmlcov']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
            
        # Create minimal baseline file if needed
        baseline_file = 'baseline_data/baseline_interactions.csv'
        if not os.path.exists(baseline_file):
            baseline_data = pd.DataFrame({
                'user_id': [1, 2, 3, 4, 5],
                'movie_id': ['movie1', 'movie2', 'movie3', 'movie4', 'movie5'],
                'total_minutes': [100, 120, 140, 160, 180],
                'rating': [4.0, 5.0, 3.0, 4.5, 3.5]
            })
            baseline_data.to_csv(baseline_file, index=False)
            
        print("✅ Environment setup complete")
        
    def run_code_quality_checks(self):
        """Run code quality checks (optional for better practices)"""
        print("\n🔍 Running code quality checks...")
        
        quality_results = {}
        
        # Check if tools are available and run them
        tools = [
            ('black', ['black', '--check', '--diff', '.']),
            ('flake8', ['flake8', '.', '--count', '--select=E9,F63,F7,F82']),
            ('isort', ['isort', '--check-only', '--diff', '.'])
        ]
        
        for tool_name, cmd in tools:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                quality_results[tool_name] = {
                    'status': 'passed' if result.returncode == 0 else 'failed',
                    'output': result.stdout + result.stderr
                }
                print(f"  {tool_name}: {'✅ PASSED' if result.returncode == 0 else '❌ FAILED'}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                quality_results[tool_name] = {'status': 'skipped', 'reason': 'tool not available'}
                print(f"  {tool_name}: ⏭️ SKIPPED")
                
        self.results['code_quality'] = quality_results
        
    def run_unit_tests(self):
        """Run unit tests with coverage"""
        print("\n🧪 Running unit tests...")
        
        # Test individual components
        test_modules = [
            ('Configuration', 'test_pipeline.py::TestConfiguration'),
            ('DataLoader', 'test_pipeline.py::TestDataLoader'),
            ('DataPreprocessor', 'test_pipeline.py::TestDataPreprocessor'),
            ('FeatureEngineer', 'test_pipeline.py::TestFeatureEngineer'),
            ('DataQuality', 'test_pipeline.py::TestDataQuality'),
            ('DataSplitter', 'test_pipeline.py::TestDataSplitter'),
            ('SVDModelTrainer', 'test_pipeline.py::TestSVDModelTrainer'),
            ('ModelEvaluator', 'test_pipeline.py::TestModelEvaluator')
        ]
        
        unit_test_results = {}
        
        for module_name, test_path in test_modules:
            print(f"  Testing {module_name}...")
            
            try:
                cmd = [
                    'python', '-m', 'pytest', test_path, '-v',
                    '--tb=short', '--disable-warnings'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                # Parse test results
                passed = result.stdout.count(' PASSED')
                failed = result.stdout.count(' FAILED')
                
                unit_test_results[module_name] = {
                    'passed': passed,
                    'failed': failed,
                    'status': 'success' if failed == 0 else 'failed',
                    'output': result.stdout[-500:]  # Last 500 chars
                }
                
                print(f"    {module_name}: {passed} passed, {failed} failed")
                
            except subprocess.TimeoutExpired:
                unit_test_results[module_name] = {
                    'status': 'timeout',
                    'error': 'Test timed out after 5 minutes'
                }
                print(f"    {module_name}: ⏰ TIMEOUT")
                
        self.results['unit_tests'] = unit_test_results
        
    def run_integration_tests(self):
        """Run integration tests"""
        print("\n🔗 Running integration tests...")
        
        try:
            cmd = [
                'python', '-m', 'pytest', 'test_pipeline.py::TestIntegration', '-v',
                '--tb=short', '--disable-warnings'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            passed = result.stdout.count(' PASSED')
            failed = result.stdout.count(' FAILED')
            
            self.results['integration_tests'] = {
                'passed': passed,
                'failed': failed,
                'status': 'success' if failed == 0 else 'failed',
                'output': result.stdout[-500:]
            }
            
            print(f"  Integration tests: {passed} passed, {failed} failed")
            
        except subprocess.TimeoutExpired:
            self.results['integration_tests'] = {
                'status': 'timeout',
                'error': 'Integration tests timed out'
            }
            
    def run_coverage_analysis(self):
        """Run comprehensive coverage analysis"""
        print("\n📊 Running coverage analysis...")
        
        try:
            # Run all tests with coverage
            cmd = [
                'python', '-m', 'pytest', 
                'test_pipeline.py', 
                '--cov=.',
                '--cov-config=.coveragerc',
                '--cov-report=html:htmlcov',
                '--cov-report=xml:coverage.xml',
                '--cov-report=term-missing',
                '--cov-report=json:coverage.json',
                '-v', '--tb=short', '--disable-warnings'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Parse coverage results
            coverage_data = self._parse_coverage_output(result.stdout)
            
            self.results['coverage_results'] = coverage_data
            
            # Check if coverage meets milestone requirements
            overall_coverage = coverage_data.get('overall_coverage', 0)
            meets_requirement = overall_coverage >= 60  # Adjusted target
            
            print(f"  Overall coverage: {overall_coverage}%")
            print(f"  Meets requirement (>60%): {'✅ YES' if meets_requirement else '❌ NO'}")
            
            return meets_requirement
            
        except subprocess.TimeoutExpired:
            print("  ⏰ Coverage analysis timed out")
            return False
            
    def _parse_coverage_output(self, output):
        """Parse coverage output to extract key metrics"""
        coverage_data = {
            'overall_coverage': 0,
            'module_coverage': {},
            'missing_lines': []
        }
        
        # Try to load JSON coverage data if available
        try:
            if os.path.exists('coverage.json'):
                with open('coverage.json', 'r') as f:
                    json_data = json.load(f)
                    
                    # Extract overall coverage
                    totals = json_data.get('totals', {})
                    if 'percent_covered' in totals:
                        coverage_data['overall_coverage'] = round(totals['percent_covered'], 1)
                    
                    # Extract module-level coverage
                    files = json_data.get('files', {})
                    for filename, file_data in files.items():
                        if filename.endswith('.py') and not filename.startswith('test_'):
                            summary = file_data.get('summary', {})
                            if 'percent_covered' in summary:
                                module_name = filename.replace('.py', '')
                                coverage_data['module_coverage'][module_name] = round(summary['percent_covered'], 1)
                                
        except (json.JSONDecodeError, FileNotFoundError):
            # Fallback to parsing text output
            lines = output.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    parts = line.split()
                    for part in parts:
                        if part.endswith('%'):
                            try:
                                coverage_data['overall_coverage'] = float(part.replace('%', ''))
                                break
                            except ValueError:
                                continue
                                
        return coverage_data
        
    def check_milestone_compliance(self):
        """Check compliance with Milestone 2 requirements"""
        print("\n✅ Checking Milestone 2 compliance...")
        
        compliance = {
            'pipeline_testing': False,
            'coverage_threshold': False,
            'continuous_integration': False,
            'overall_compliance': False
        }
        
        # Check pipeline testing (15 points)
        unit_tests = self.results.get('unit_tests', {})
        integration_tests = self.results.get('integration_tests', {})
        
        unit_success = all(test.get('status') == 'success' for test in unit_tests.values())
        integration_success = integration_tests.get('status') == 'success'
        
        compliance['pipeline_testing'] = unit_success and integration_success
        
        # Check coverage threshold
        coverage = self.results.get('coverage_results', {})
        overall_coverage = coverage.get('overall_coverage', 0)
        compliance['coverage_threshold'] = overall_coverage >= 60
        
        # Check if CI configuration exists
        ci_files = ['.github/workflows/ci.yml', '.github/workflows/ci-cd-workflow.yml']
        compliance['continuous_integration'] = any(os.path.exists(f) for f in ci_files)
        
        # Overall compliance
        compliance['overall_compliance'] = (
            compliance['pipeline_testing'] and 
            compliance['coverage_threshold']
        )
        
        self.results['milestone_compliance'] = compliance
        
        # Print compliance report
        print(f"  Pipeline Testing: {'✅ PASS' if compliance['pipeline_testing'] else '❌ FAIL'}")
        print(f"  Coverage Threshold (>60%): {'✅ PASS' if compliance['coverage_threshold'] else '❌ FAIL'}")
        print(f"  CI Configuration: {'✅ READY' if compliance['continuous_integration'] else '⚠️ SETUP NEEDED'}")
        print(f"  Overall Compliance: {'✅ READY FOR SUBMISSION' if compliance['overall_compliance'] else '❌ NEEDS WORK'}")
        
        return compliance['overall_compliance']
        
    def generate_milestone_report(self):
        """Generate final milestone report"""
        print("\n📋 Generating Milestone 2 report...")
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = {
            'milestone': 'Milestone 2 - Pipeline Testing & CI/CD',
            'execution_time': duration.total_seconds(),
            'timestamp': end_time.isoformat(),
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        # Save report
        report_file = f"reports/milestone2_test_report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('reports', exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"  Report saved to: {report_file}")
        
        return report
        
    def _generate_summary(self):
        """Generate summary of test results"""
        summary = {
            'total_tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'coverage_achieved': self.results.get('coverage_results', {}).get('overall_coverage', 0),
            'milestone_ready': self.results.get('milestone_compliance', {}).get('overall_compliance', False)
        }
        
        # Count tests from unit and integration results
        for test_module in self.results.get('unit_tests', {}).values():
            summary['total_tests_run'] += test_module.get('passed', 0) + test_module.get('failed', 0)
            summary['tests_passed'] += test_module.get('passed', 0)
            summary['tests_failed'] += test_module.get('failed', 0)
            
        integration = self.results.get('integration_tests', {})
        summary['total_tests_run'] += integration.get('passed', 0) + integration.get('failed', 0)
        summary['tests_passed'] += integration.get('passed', 0)
        summary['tests_failed'] += integration.get('failed', 0)
        
        return summary
        
    def run_all(self):
        """Run complete test suite for Milestone 2"""
        print("🚀 MOVIE RECOMMENDATION SYSTEM - MILESTONE 2 TEST SUITE")
        print("=" * 60)
        
        try:
            # Setup
            self.setup_environment()
            
            # Run tests
            self.run_code_quality_checks()
            self.run_unit_tests()
            self.run_integration_tests()
            coverage_ok = self.run_coverage_analysis()
            compliance_ok = self.check_milestone_compliance()
            
            # Generate report
            report = self.generate_milestone_report()
            
            # Final status
            print("\n" + "=" * 60)
            if compliance_ok:
                print("🎉 MILESTONE 2 - READY FOR SUBMISSION!")
                print(f"✅ Coverage: {self.results['coverage_results'].get('overall_coverage', 0)}%")
                print(f"✅ Tests: {report['summary']['tests_passed']}/{report['summary']['total_tests_run']} passed")
            else:
                print("⚠️  MILESTONE 2 - NEEDS ATTENTION")
                print("Please review the issues above and rerun tests.")
                
            return compliance_ok
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Test execution interrupted by user")
            return False
        except Exception as e:
            print(f"\n\n❌ Test execution failed: {str(e)}")
            return False


if __name__ == "__main__":
    runner = Milestone2TestRunner()
    success = runner.run_all()
    sys.exit(0 if success else 1)
