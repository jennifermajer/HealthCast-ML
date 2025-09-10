#!/usr/bin/env python3

try:
    import dotenv
except ImportError:
    import subprocess
    import sys
    print("Installing python-dotenv...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    import dotenv

#pip install -r requirements.txt
"""
Complete Climate-Health Analysis Pipeline

This script runs the complete machine learning analysis pipeline for predicting
health consultations based on climate variables. It handles both synthetic data
(for public use) and real data (for internal IMC use) seamlessly.

Usage:
    python run_analysis.py                    # Run with default config
    python run_analysis.py --synthetic       # Force synthetic data mode
    python run_analysis.py --config custom.yaml  # Use custom config
    python run_analysis.py --quick          # Quick analysis with reduced models
    python run_analysis.py --help           # Show help message

Environment Variables:
    USE_SYNTHETIC: Set to 'true' for synthetic data, 'false' for real data
    LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)
    RANDOM_SEED: Set random seed for reproducibility
"""

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from dotenv import load_dotenv
    from data_processing import load_and_merge_data, get_data_summary
    from feature_engineering import create_features
    from models import train_all_models
    from evaluation import evaluate_models
    from utils import (
        setup_logging, create_climate_health_plots, validate_data_quality,
        save_data_quality_report, format_results_summary, print_analysis_summary,
        export_results_to_excel, create_model_comparison_report, create_interactive_dashboard
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed all required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)

class ClimatehealthAnalysis:
    """Main analysis pipeline coordinator"""
    
    def __init__(self, config_path: str = 'config.yaml', args: Optional[argparse.Namespace] = None):
        self.config_path = config_path
        self.args = args or argparse.Namespace()
        self.start_time = time.time()
        self.logger = None
        self.results = {}
        
        # Initialize logging and environment
        self._setup_environment()
        self._setup_logging()
        
    def _setup_environment(self):
        """Setup environment variables and configuration"""
        # Load environment variables
        if Path('.env').exists():
            load_dotenv()
        
        # Override environment variables from command line arguments
        if hasattr(self.args, 'synthetic') and self.args.synthetic:
            os.environ['USE_SYNTHETIC'] = 'true'
        
        if hasattr(self.args, 'log_level') and self.args.log_level:
            os.environ['LOG_LEVEL'] = self.args.log_level.upper()
        
        # Set random seed for reproducibility
        random_seed = os.getenv('RANDOM_SEED', '42')
        os.environ['PYTHONHASHSEED'] = random_seed
        
        import numpy as np
        import random
        np.random.seed(int(random_seed))
        random.seed(int(random_seed))
        
        # Try to set TensorFlow seed if available
        try:
            import tensorflow as tf
            tf.random.set_seed(int(random_seed))
        except ImportError:
            pass  # TensorFlow not available
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        log_file = 'logs/analysis.log'
        
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        self.logger = setup_logging(log_level, log_file)
        self.logger.info("=" * 80)
        self.logger.info("🌡️ CLIMATE-HEALTH ANALYSIS PIPELINE STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"📅 Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"🔧 Configuration file: {self.config_path}")
        self.logger.info(f"🎯 Data mode: {'SYNTHETIC' if os.getenv('USE_SYNTHETIC', 'true').lower() == 'true' else 'PRIVATE'}")
        self.logger.info(f"📝 Log level: {log_level}")
        self.logger.info(f"🎲 Random seed: {os.getenv('RANDOM_SEED', '42')}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline
        
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Step 1: Load and merge data
            self.logger.info("\n" + "="*50)
            self.logger.info("📊 STEP 1: DATA LOADING AND MERGING")
            self.logger.info("="*50)
            
            merged_data = self._load_and_validate_data()
            
            # Step 2: Feature engineering
            self.logger.info("\n" + "="*50)
            self.logger.info("🔧 STEP 2: FEATURE ENGINEERING")
            self.logger.info("="*50)
            
            feature_data = self._engineer_features(merged_data)
            
            # Step 3: Model training
            self.logger.info("\n" + "="*50)
            self.logger.info("🤖 STEP 3: MODEL TRAINING")
            self.logger.info("="*50)
            
            models = self._train_models(feature_data)
            
            # Step 4: Model evaluation
            self.logger.info("\n" + "="*50)
            self.logger.info("📈 STEP 4: MODEL EVALUATION")
            self.logger.info("="*50)
            
            evaluation_results = self._evaluate_models(models, feature_data)
            
            # Step 5: Visualization and reporting
            self.logger.info("\n" + "="*50)
            self.logger.info("📊 STEP 5: VISUALIZATION AND REPORTING")
            self.logger.info("="*50)
            
            self._generate_reports_and_visualizations(merged_data, models, evaluation_results)
            
            # Compile final results
            self.results = {
                'data_summary': get_data_summary(merged_data),
                'feature_data_shape': feature_data.shape,
                'trained_models': list(models.keys()),
                'evaluation_results': evaluation_results,
                'analysis_duration': time.time() - self.start_time
            }
            
            self._log_completion_summary()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {str(e)}")
            self.logger.error(f"📍 Error traceback:\n{traceback.format_exc()}")
            raise
    
    def _load_and_validate_data(self) -> Any:
        """Load and validate data"""
        self.logger.info("📥 Loading and merging health and climate data...")
        
        try:
            # Load and merge data
            merged_data = load_and_merge_data(self.config_path)
            
            # Generate data summary
            data_summary = get_data_summary(merged_data)
            self.logger.info(f"✅ Data loaded successfully:")
            self.logger.info(f"   • Total records: {data_summary['total_records']:,}")
            self.logger.info(f"   • Date range: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}")
            self.logger.info(f"   • Geographic regions: {data_summary['geographic_coverage']['admin1_regions']}")
            self.logger.info(f"   • Days of data: {data_summary['date_range']['days']}")
            
            # Validate data quality
            self.logger.info("🔍 Performing data quality validation...")
            quality_report = validate_data_quality(merged_data)
            
            # Save quality report
            save_data_quality_report(quality_report, 'results/reports/data_quality_report.json')
            
            # Log quality issues
            if quality_report['data_quality_issues']:
                self.logger.warning(f"⚠️ Data quality issues identified:")
                for issue in quality_report['data_quality_issues']:
                    self.logger.warning(f"   • {issue}")
            else:
                self.logger.info("✅ No significant data quality issues detected")
            
            return merged_data
            
        except Exception as e:
            self.logger.error(f"❌ Data loading failed: {str(e)}")
            raise
    
    def _engineer_features(self, merged_data: Any) -> Any:
        """Engineer features for machine learning"""
        self.logger.info("🔧 Engineering features for machine learning...")
        
        try:
            feature_data = create_features(merged_data)
            
            self.logger.info(f"✅ Feature engineering completed:")
            self.logger.info(f"   • Final dataset shape: {feature_data.shape}")
            self.logger.info(f"   • Features created: {feature_data.shape[1] - 3} (excluding date, admin1, morbidity_category)")  # Rough estimate
            self.logger.info(f"   • Target variable range: {feature_data['consultation_count'].min()} to {feature_data['consultation_count'].max()}")
            
            return feature_data
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {str(e)}")
            raise
    
    def _train_models(self, feature_data: Any) -> Dict:
        """Train machine learning models"""
        self.logger.info("🤖 Training machine learning models...")
        
        try:
            # Load configuration for model training
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Modify config for quick analysis if requested
            if hasattr(self.args, 'quick') and self.args.quick:
                self.logger.info("⚡ Quick analysis mode - using reduced model parameters")
                config = self._apply_quick_config(config)
            
            # Train models
            models = train_all_models(feature_data, config)
            
            self.logger.info(f"✅ Model training completed:")
            self.logger.info(f"   • Models trained: {len(models)}")
            for model_name, model in models.items():
                self.logger.info(f"     - {model.name}")
            
            return models
            
        except Exception as e:
            self.logger.error(f"❌ Model training failed: {str(e)}")
            raise
    
    def _apply_quick_config(self, config: Dict) -> Dict:
        """Apply quick analysis configuration for faster training"""
        quick_config = config.copy()
        
        # Reduce model complexity for quick analysis
        if 'random_forest' in quick_config:
            quick_config['random_forest'] = {'n_estimators': 20, 'max_depth': 5}
        
        if 'xgboost' in quick_config:
            quick_config['xgboost'] = {'n_estimators': 20, 'max_depth': 3, 'learning_rate': 0.1}
        
        if 'lightgbm' in quick_config:
            quick_config['lightgbm'] = {'n_estimators': 20, 'max_depth': 3}
        
        # Reduce deep learning complexity
        if 'lstm' in quick_config:
            quick_config['lstm'] = {
                'sequence_length': 7, 'lstm_units': 16, 'epochs': 10, 'batch_size': 32
            }
        
        if 'gru' in quick_config:
            quick_config['gru'] = {
                'sequence_length': 7, 'gru_units': 16, 'epochs': 10, 'batch_size': 32
            }
        
        # Reduce cross-validation folds
        if 'evaluation' in quick_config:
            quick_config['evaluation']['cv_folds'] = 3
        
        return quick_config
    
    def _evaluate_models(self, models: Dict, feature_data: Any) -> Dict:
        """Evaluate trained models"""
        self.logger.info("📈 Evaluating model performance...")
        
        try:
            # Load configuration for evaluation
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Run comprehensive evaluation
            evaluation_results = evaluate_models(models, feature_data, config)
            
            self.logger.info("✅ Model evaluation completed:")
            
            # Log key results
            if 'model_rankings' in evaluation_results and 'overall' in evaluation_results['model_rankings']:
                rankings = evaluation_results['model_rankings']['overall']
                best_model = min(rankings.items(), key=lambda x: x[1])[0]
                self.logger.info(f"   • Best performing model: {best_model.replace('_', ' ').title()}")
            
            if 'time_series_cv' in evaluation_results:
                cv_results = evaluation_results['time_series_cv']
                successful_models = [name for name, metrics in cv_results.items() if 'error' not in metrics]
                self.logger.info(f"   • Models successfully evaluated: {len(successful_models)}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {str(e)}")
            raise
    
    def _generate_reports_and_visualizations(self, merged_data: Any, models: Dict, evaluation_results: Dict):
        """Generate comprehensive reports and visualizations"""
        self.logger.info("📊 Generating reports and visualizations...")
        
        try:
            # Create output directories
            Path('results/figures').mkdir(parents=True, exist_ok=True)
            Path('results/reports').mkdir(parents=True, exist_ok=True)
            Path('results/models').mkdir(parents=True, exist_ok=True)
            
            # Generate visualizations
            self.logger.info("📈 Creating visualization suite...")
            create_climate_health_plots(merged_data, 'results/figures')
            
            # Create interactive dashboard
            self.logger.info("🖥️ Creating interactive dashboard...")
            create_interactive_dashboard(merged_data, 'results/figures')
            
            # Generate reports
            self.logger.info("📝 Generating comprehensive reports...")
            
            # Excel report
            export_results_to_excel(evaluation_results, 'results/reports/analysis_results.xlsx')
            
            # HTML model comparison report
            create_model_comparison_report(models, evaluation_results, 'results/reports/model_comparison.html')
            
            # Console summary
            print_analysis_summary(evaluation_results)
            
            self.logger.info("✅ All reports and visualizations generated successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {str(e)}")
            # Don't raise here - analysis results are still valuable even without perfect reports
            self.logger.warning("⚠️ Continuing despite report generation issues...")
    
    def _log_completion_summary(self):
        """Log completion summary"""
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        self.logger.info("="*80)
        self.logger.info(f"⏱️ Total duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        self.logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'trained_models' in self.results:
            self.logger.info(f"🤖 Models trained: {len(self.results['trained_models'])}")
        
        self.logger.info(f"📊 Results saved to: results/")
        self.logger.info(f"📝 Detailed logs: logs/analysis.log")
        
        # Quick access to key files
        key_outputs = [
            "results/figures/interactive_dashboard.html",
            "results/reports/model_comparison.html", 
            "results/reports/analysis_results.xlsx"
        ]
        
        self.logger.info("🔗 Key output files:")
        for output in key_outputs:
            if Path(output).exists():
                self.logger.info(f"   • {output}")
        
        self.logger.info("="*80)

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Climate-Health Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py                    # Run with default settings
  python run_analysis.py --synthetic       # Force synthetic data mode
  python run_analysis.py --quick           # Quick analysis (faster)
  python run_analysis.py --config custom.yaml  # Custom configuration
  
Environment Variables:
  USE_SYNTHETIC=true/false    # Data source selection
  LOG_LEVEL=INFO/DEBUG        # Logging verbosity  
  RANDOM_SEED=42              # Reproducibility seed
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--synthetic', '-s',
        action='store_true',
        help='Force use of synthetic data (overrides environment variable)'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick analysis with reduced model complexity'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set logging level (overrides environment variable)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and data without running full analysis'
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['poisson', 'negative_binomial', 'random_forest', 'xgboost', 'lightgbm', 'lstm', 'gru', 'ensemble'],
        help='Specify which models to train (default: all available)'
    )
    
    return parser

def validate_setup() -> bool:
    """Validate that the environment is properly set up"""
    print("🔧 Validating environment setup...")
    
    # Check required directories
    required_dirs = ['src', 'data', 'results', 'logs']
    for directory in required_dirs:
        if not Path(directory).exists():
            print(f"⚠️ Creating missing directory: {directory}")
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Check configuration file
    if not Path('config.yaml').exists():
        print("❌ Configuration file 'config.yaml' not found!")
        print("Please create a configuration file. See README.md for details.")
        return False
    
    # Check for data files if not using synthetic data
    use_synthetic = os.getenv('USE_SYNTHETIC', 'true').lower() == 'true'
    if not use_synthetic:
        data_dir = Path('data/raw')
        if not data_dir.exists() or not any(data_dir.glob('*.csv')):
            print("⚠️ No data files found in data/raw/ directory.")
            print("Either place your data files there or use synthetic data mode.")
            print("Set USE_SYNTHETIC=true or use --synthetic flag.")
    
    print("✅ Environment validation completed")
    return True

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("🌡️  CLIMATE-HEALTH MACHINE LEARNING ANALYSIS PIPELINE")
    print("    Predicting Health Consultations from Climate Variables")
    print("="*80)
    
    try:
        # Validate environment setup
        if not validate_setup():
            sys.exit(1)
        
        # Validate configuration file exists
        if not Path(args.config).exists():
            print(f"❌ Configuration file not found: {args.config}")
            sys.exit(1)
        
        # Handle dry run
        if args.dry_run:
            print("🧪 Dry run mode - validating configuration only...")
            try:
                # Test data loading
                analysis = ClimatehealthAnalysis(args.config, args)
                analysis.logger.info("🧪 Dry run: Testing data loading...")
                merged_data = analysis._load_and_validate_data()
                print(f"✅ Dry run successful - found {len(merged_data):,} records")
                return
            except Exception as e:
                print(f"❌ Dry run failed: {e}")
                sys.exit(1)
        
        # Run complete analysis
        print("🚀 Starting complete analysis pipeline...\n")
        
        analysis = ClimatehealthAnalysis(args.config, args)
        results = analysis.run_complete_analysis()
        
        # Success message
        print("\n🎉 Analysis completed successfully!")
        print("📊 Check the results/ directory for detailed outputs")
        print("📝 Check logs/analysis.log for detailed execution logs")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
        print("📝 Check logs/analysis.log for detailed error information")
        return 1

if __name__ == '__main__':
    sys.exit(main())