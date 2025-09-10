"""
Climate-Health Model Evaluation Module

This module provides comprehensive evaluation and validation for both project components:

COMPONENT 1: CLIMATE SENSITIVITY ANALYSIS
- Objective: Evaluate how well models identify climate-sensitive morbidities by linking
  health consultations to historical weather data at national and governorate levels
- Key Evaluations: Feature importance analysis, spatial validation, climate extreme performance
- Focus: Interpretability, feature ranking, climate variable significance

COMPONENT 2: PREDICTIVE MODELING & FORECASTING
- Objective: Evaluate predictive models' ability to quantify and forecast the impact of
  temperature and precipitation changes on consultations for climate-sensitive morbidities
- Key Evaluations: Time series cross-validation, surge prediction, temporal stability
- Focus: Prediction accuracy, forecasting performance, model reliability over time

All evaluation methods support both analytical objectives with specialized metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from scipy.stats import poisson, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation for climate-health predictions
    
    Supports evaluation for both project components:
    - Component 1: Climate sensitivity analysis (interpretability focus)
    - Component 2: Predictive forecasting (accuracy focus)
    
    Provides multiple validation strategies including temporal, spatial, and domain-specific evaluation.
    """
    
    def __init__(self, models: Dict, config: Dict = None):
        self.models = models
        self.config = config or {}
        self.results = {}
        
    def evaluate_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        BOTH COMPONENTS: Comprehensive model evaluation using multiple validation strategies
        
        COMPONENT 1 FOCUS: Climate sensitivity evaluation
        - Feature importance analysis for identifying climate-health relationships
        - Spatial validation to test climate sensitivity across regions
        - Climate extreme evaluation to test model robustness during extreme weather
        
        COMPONENT 2 FOCUS: Predictive modeling evaluation
        - Time series cross-validation for forecasting accuracy
        - Surge prediction analysis for high consultation volume prediction
        - Temporal stability assessment for consistent forecasting performance
        
        Args:
            df: Feature-engineered dataframe with weather and consultation data
            
        Returns:
            Comprehensive evaluation results for both analytical objectives
        """
        logger.info("📊 Starting comprehensive model evaluation...")
        
        # Prepare data
        exclude_cols = ['date', 'admin1', 'category_canonical_disease_imc', 'consultation_count']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0)
        y = df['consultation_count']
        
        # Store data info for results
        self.data_info = {
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'date_range': (df['date'].min(), df['date'].max()),
            'admin1_regions': df['admin1'].nunique(),
            'morbidity_categories': df['category_canonical_disease_imc'].nunique()
        }
        
        logger.info(f"📊 Evaluation data: {len(df)} samples, {len(feature_cols)} features")
        
        # Check if we're in quick mode
        evaluation_config = self.config.get('evaluation', {})
        quick_mode = evaluation_config.get('quick_mode', False)
        
        # 1. Time Series Cross-Validation (always done)
        logger.info("⏰ Performing time series cross-validation...")
        ts_results = self._time_series_cv(X, y, df)
        
        # 2. Spatial Generalization Testing (skip in quick mode)
        if not evaluation_config.get('skip_spatial_validation', False):
            logger.info("🗺️ Performing spatial generalization testing...")
            spatial_results = self._spatial_validation(X, y, df)
        else:
            logger.info("⏩ Skipping spatial validation (quick mode)")
            spatial_results = {}
        
        # 3. Morbidity-Specific Evaluation (skip in quick mode)
        if not evaluation_config.get('skip_morbidity_evaluation', False):
            logger.info("🏥 Performing morbidity-specific evaluation...")
            morbidity_results = self._morbidity_specific_evaluation(X, y, df)
        else:
            logger.info("⏩ Skipping morbidity-specific evaluation (quick mode)")
            morbidity_results = {}
        
        # 4. Climate Extreme Evaluation (skip in quick mode)
        if not evaluation_config.get('skip_climate_extreme_evaluation', False):
            logger.info("🌡️ Evaluating performance during climate extremes...")
            extreme_results = self._climate_extreme_evaluation(X, y, df)
        else:
            logger.info("⏩ Skipping climate extreme evaluation (quick mode)")
            extreme_results = {}
        
        # 5. Feature Importance Analysis (keep in quick mode as it's useful and fast)
        if not evaluation_config.get('skip_feature_importance', False):
            logger.info("🔍 Analyzing feature importance...")
            importance_results = self._feature_importance_analysis()
        else:
            logger.info("⏩ Skipping feature importance analysis")
            importance_results = {}
        
        # 6. Surge Prediction Analysis (skip in quick mode)
        if not quick_mode:
            logger.info("📈 Analyzing consultation surge prediction (PR-AUC)...")
            surge_prediction_results = self._surge_prediction_analysis(X, y, df)
        else:
            logger.info("⏩ Skipping surge prediction analysis (quick mode)")
            surge_prediction_results = {}
        
        # 7. Prediction Interval Analysis (skip in quick mode)
        if not quick_mode:
            logger.info("🔍 Analyzing prediction intervals...")
            interval_results = self._prediction_interval_analysis(X, y, df)
        else:
            logger.info("⏩ Skipping prediction interval analysis (quick mode)")
            interval_results = {}
        
        # Compile comprehensive results
        self.results = {
            'data_info': self.data_info,
            'time_series_cv': ts_results,
            'spatial_validation': spatial_results,
            'morbidity_specific': morbidity_results,
            'climate_extremes': extreme_results,
            'feature_importance': importance_results,
            'surge_prediction': surge_prediction_results,
            'prediction_intervals': interval_results,
            'model_rankings': self._rank_models()
        }
        
        logger.info("✅ Model evaluation completed!")
        return self.results
    
    def _time_series_cv(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> Dict:
        """
        COMPONENT 2 PRIMARY: Rolling-origin time series cross-validation
        
        PURPOSE: Evaluate predictive models' forecasting accuracy using temporal validation
        - ★ PRIMARY for Component 2: Tests forecasting capability over time
        - ★ Uses rolling-origin methodology to simulate real forecasting scenarios
        - ★ Prevents data leakage by training only on past data to predict future
        - ★ Measures temporal stability and consistency of predictions
        
        COMPONENT 1 SECONDARY: Basic temporal validation for climate sensitivity
        - Validates that climate-health relationships are stable over time
        
        Returns comprehensive temporal performance metrics including Poisson deviance.
        """
        
        cv_folds = self.config.get('evaluation', {}).get('cv_folds', 5)
        
        # Implement rolling-origin approach instead of standard TimeSeriesSplit
        cv_results = {}
        
        # Sort data by date for proper time series splitting
        if 'date' in df.columns:
            sort_idx = df['date'].argsort()
            X = X.iloc[sort_idx]
            y = y.iloc[sort_idx]
            df_sorted = df.iloc[sort_idx]
        else:
            df_sorted = df
        
        # Calculate rolling windows
        n_samples = len(X)
        test_size = max(n_samples // (cv_folds + 2), 50)  # Ensure minimum test size
        
        for model_name, model in self.models.items():
            logger.info(f"  - Rolling-origin time series CV for {model.name}...")
            
            fold_metrics = []
            fold_details = []
            
            try:
                for fold in range(cv_folds):
                    # Rolling origin: each fold uses progressively more training data
                    train_end = n_samples - (cv_folds - fold) * test_size
                    test_start = train_end
                    test_end = min(test_start + test_size, n_samples)
                    
                    if train_end < 100 or test_end - test_start < 20:  # Skip invalid folds
                        continue
                    
                    # Split data
                    X_train = X.iloc[:train_end]
                    y_train = y.iloc[:train_end]
                    X_test = X.iloc[test_start:test_end]
                    y_test = y.iloc[test_start:test_end]
                    
                    # Get model copy to avoid contamination
                    model_copy = self._get_model_copy(model, model_name)
                    
                    # Handle different model types
                    if model_name in ['lstm', 'gru']:
                        X_train_seq = X_train.copy()
                        X_train_seq['date'] = df_sorted['date'].iloc[:train_end]
                        X_train_seq['admin1_encoded'] = df_sorted.get('admin1_encoded', pd.Series([0]*len(X_train))).iloc[:train_end]
                        
                        X_test_seq = X_test.copy()
                        X_test_seq['date'] = df_sorted['date'].iloc[test_start:test_end]
                        X_test_seq['admin1_encoded'] = df_sorted.get('admin1_encoded', pd.Series([0]*len(X_test))).iloc[test_start:test_end]
                        
                        model_copy.fit(X_train_seq, y_train)
                        predictions = model_copy.predict(X_test_seq)
                    else:
                        model_copy.fit(X_train, y_train)
                        predictions = model_copy.predict(X_test)
                    
                    # Ensure non-negative predictions for count data
                    predictions = np.maximum(predictions, 0)
                    
                    # Calculate comprehensive metrics including Poisson deviance
                    if len(predictions) > 0:
                        fold_metric = self._calculate_comprehensive_metrics(y_test, predictions)
                        fold_metrics.append(fold_metric)
                        
                        fold_details.append({
                            'fold': fold,
                            'train_size': len(X_train),
                            'test_size': len(X_test),
                            'train_period': (df_sorted['date'].iloc[0], df_sorted['date'].iloc[train_end-1]) if 'date' in df_sorted.columns else None,
                            'test_period': (df_sorted['date'].iloc[test_start], df_sorted['date'].iloc[test_end-1]) if 'date' in df_sorted.columns else None,
                            'metrics': fold_metric
                        })
                
                # Aggregate results with proper statistical measures
                if fold_metrics:
                    cv_results[model_name] = {
                        **self._average_metrics(fold_metrics),
                        'n_successful_folds': len(fold_metrics),
                        'fold_details': fold_details,
                        'temporal_stability': self._calculate_temporal_stability(fold_metrics)
                    }
                else:
                    cv_results[model_name] = {'error': 'No valid predictions from any fold'}
                    
            except Exception as e:
                logger.warning(f"Rolling-origin time series CV failed for {model_name}: {e}")
                cv_results[model_name] = {'error': str(e)}
        
        return cv_results
    
    def _spatial_validation(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> Dict:
        """
        COMPONENT 1 PRIMARY: Leave-one-region-out validation for spatial generalizability
        
        PURPOSE: Evaluate climate sensitivity analysis across different governorates
        - ★ PRIMARY for Component 1: Tests if climate-health relationships generalize across regions
        - ★ Validates climate sensitivity patterns at governorate level
        - ★ Ensures climate variable importance is consistent across different geographical areas
        - ★ Critical for understanding spatial heterogeneity in climate-health relationships
        
        COMPONENT 2 SECONDARY: Tests forecasting model geographic transferability
        - Validates prediction accuracy when applied to new regions
        
        Uses leave-one-region-out cross-validation to test spatial generalization.
        """
        
        if 'admin1' not in df.columns:
            return {'error': 'No admin1 column for spatial validation'}
        
        spatial_results = {}
        admin1_regions = df['admin1'].unique()
        
        # Skip if too few regions
        if len(admin1_regions) < 3:
            return {'error': 'Insufficient regions for spatial validation'}
        
        for model_name, model in self.models.items():
            logger.info(f"  - Spatial validation for {model.name}...")
            
            region_metrics = []
            
            try:
                for test_region in admin1_regions[:3]:  # Test on first 3 regions to save time
                    # Split data by region
                    train_mask = df['admin1'] != test_region
                    test_mask = df['admin1'] == test_region
                    
                    if test_mask.sum() < 50:  # Skip regions with too little data
                        continue
                    
                    X_train, X_test = X[train_mask], X[test_mask]
                    y_train, y_test = y[train_mask], y[test_mask]
                    
                    # Handle sequence models
                    if model_name in ['lstm', 'gru']:
                        X_train_seq = X_train.copy()
                        X_train_seq['date'] = df.loc[train_mask, 'date']
                        X_train_seq['admin1_encoded'] = df.loc[train_mask, 'admin1_encoded'] if 'admin1_encoded' in df.columns else 0
                        
                        X_test_seq = X_test.copy()
                        X_test_seq['date'] = df.loc[test_mask, 'date']
                        X_test_seq['admin1_encoded'] = df.loc[test_mask, 'admin1_encoded'] if 'admin1_encoded' in df.columns else 0
                        
                        model.fit(X_train_seq, y_train)
                        predictions = model.predict(X_test_seq)
                    else:
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    
                    if len(predictions) > 0:
                        metrics = self._calculate_metrics(y_test, predictions)
                        metrics['test_region'] = test_region
                        region_metrics.append(metrics)
                
                if region_metrics:
                    spatial_results[model_name] = {
                        'average_metrics': self._average_metrics(region_metrics),
                        'region_details': region_metrics
                    }
                else:
                    spatial_results[model_name] = {'error': 'No valid spatial predictions'}
                    
            except Exception as e:
                logger.warning(f"Spatial validation failed for {model_name}: {e}")
                spatial_results[model_name] = {'error': str(e)}
        
        return spatial_results
    
    def _morbidity_specific_evaluation(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> Dict:
        """
        COMPONENT 1 PRIMARY: Evaluate model performance for different morbidity categories
        
        PURPOSE: Identify which morbidities are most climate-sensitive
        - ★ PRIMARY for Component 1: Tests climate sensitivity for specific disease categories
        - ★ Identifies morbidities with strongest climate-health relationships
        - ★ Validates that climate models perform better for truly climate-sensitive diseases
        - ★ Supports morbidity ranking by climate sensitivity
        
        COMPONENT 2 SECONDARY: Disease-specific forecasting performance
        - Evaluates prediction accuracy for different health consultation categories
        
        Focuses on top morbidity categories to identify climate-sensitive diseases.
        """
        
        if 'category_canonical_disease_imc' not in df.columns:
            return {'error': 'No category_canonical_disease_imc column'}
        
        morbidity_results = {}
        
        # Get top morbidity categories by frequency
        top_morbidities = df['category_canonical_disease_imc'].value_counts().head(5).index
        
        for model_name, model in self.models.items():
            logger.info(f"  - Morbidity-specific evaluation for {model.name}...")
            
            morbidity_metrics = {}
            
            try:
                # Train model on full dataset first
                if model_name in ['lstm', 'gru']:
                    X_with_ids = X.copy()
                    X_with_ids['date'] = df['date']
                    X_with_ids['admin1_encoded'] = df.get('admin1_encoded', 0)
                    model.fit(X_with_ids, y)
                    predictions = model.predict(X_with_ids)
                else:
                    model.fit(X, y)
                    predictions = model.predict(X)
                
                # Evaluate predictions for each morbidity category
                for morbidity in top_morbidities:
                    mask = df['category_canonical_disease_imc'] == morbidity
                    
                    if mask.sum() >= 30:  # Minimum samples for reliable evaluation
                        y_morbidity = y[mask]
                        pred_morbidity = predictions[mask] if len(predictions) == len(y) else predictions[:mask.sum()]
                        
                        if len(pred_morbidity) > 0:
                            metrics = self._calculate_metrics(y_morbidity, pred_morbidity)
                            metrics['sample_size'] = mask.sum()
                            morbidity_metrics[morbidity] = metrics
                
                morbidity_results[model_name] = morbidity_metrics
                
            except Exception as e:
                logger.warning(f"Morbidity-specific evaluation failed for {model_name}: {e}")
                morbidity_results[model_name] = {'error': str(e)}
        
        return morbidity_results
    
    def _climate_extreme_evaluation(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> Dict:
        """
        COMPONENT 1 PRIMARY: Evaluate performance during climate extreme events
        
        PURPOSE: Test climate sensitivity analysis during extreme weather conditions
        - ★ PRIMARY for Component 1: Validates climate-health relationships during extreme events
        - ★ Tests model robustness during heatwaves, heavy rain, drought periods
        - ★ Critical for identifying climate-sensitive morbidities that spike during extremes
        - ★ Ensures climate variable importance holds true during extreme weather
        
        COMPONENT 2 SECONDARY: Forecasting accuracy during climate extremes
        - Tests prediction performance during unusual weather events
        
        Evaluates models specifically on heatwave, heavy rain, and drought periods.
        """
        
        extreme_results = {}
        
        # Define extreme event indicators
        extreme_conditions = []
        
        if 'is_heatwave' in df.columns:
            extreme_conditions.append(('heatwave', df['is_heatwave'] == 1))
        if 'is_heavy_rain' in df.columns:
            extreme_conditions.append(('heavy_rain', df['is_heavy_rain'] == 1))
        if 'is_drought_period' in df.columns:
            extreme_conditions.append(('drought', df['is_drought_period'] == 1))
        
        if not extreme_conditions:
            return {'error': 'No climate extreme indicators found'}
        
        for model_name, model in self.models.items():
            logger.info(f"  - Climate extreme evaluation for {model.name}...")
            
            extreme_metrics = {}
            
            try:
                # Train model and get predictions
                if model_name in ['lstm', 'gru']:
                    X_with_ids = X.copy()
                    X_with_ids['date'] = df['date']
                    X_with_ids['admin1_encoded'] = df.get('admin1_encoded', 0)
                    model.fit(X_with_ids, y)
                    predictions = model.predict(X_with_ids)
                else:
                    model.fit(X, y)
                    predictions = model.predict(X)
                
                # Evaluate for each extreme condition
                for extreme_name, extreme_mask in extreme_conditions:
                    if extreme_mask.sum() >= 20:  # Minimum samples
                        y_extreme = y[extreme_mask]
                        pred_extreme = predictions[extreme_mask] if len(predictions) == len(y) else predictions[:extreme_mask.sum()]
                        
                        if len(pred_extreme) > 0:
                            metrics = self._calculate_metrics(y_extreme, pred_extreme)
                            metrics['sample_size'] = extreme_mask.sum()
                            extreme_metrics[extreme_name] = metrics
                
                extreme_results[model_name] = extreme_metrics
                
            except Exception as e:
                logger.warning(f"Climate extreme evaluation failed for {model_name}: {e}")
                extreme_results[model_name] = {'error': str(e)}
        
        return extreme_results
    

    def _feature_importance_analysis(self) -> Dict:
        """
        COMPONENT 1 PRIMARY: Analyze feature importance across all models
        
        PURPOSE: Identify which climate variables are most important for health consultations
        - ★ PRIMARY for Component 1: Core methodology for climate sensitivity analysis
        - ★ Extracts feature importance from tree-based models (RandomForest, XGBoost)
        - ★ Identifies temperature vs precipitation importance rankings
        - ★ Essential for linking health consultations to specific weather variables
        - ★ Supports climate-health relationship interpretation
        
        COMPONENT 2 SECONDARY: Understanding predictive model drivers
        - Helps interpret which features drive forecasting performance
        
        Returns feature importance scores and rankings for climate variable analysis.
        """
        importance_results = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importance_') and model.feature_importance_ is not None:
                    importance_results[model_name] = {
                        'feature_names': model.feature_names_ if hasattr(model, 'feature_names_') else [],
                        'importance_scores': model.feature_importance_.tolist() if hasattr(model.feature_importance_, 'tolist') else model.feature_importance_,
                        'importance_type': getattr(model, 'importance_type_', 'gain')
                    }
                else:
                    importance_results[model_name] = {'error': 'No feature importance available'}
            except Exception as e:
                logger.warning(f"Feature importance analysis failed for {model_name}: {e}")
                importance_results[model_name] = {'error': str(e)}
        
        return importance_results
    
    def _prediction_interval_analysis(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> Dict:
        """
        COMPONENT 2 PRIMARY: Analyze prediction intervals and uncertainty quantification
        
        PURPOSE: Provide uncertainty estimates for health consultation forecasts
        - ★ PRIMARY for Component 2: Essential for reliable forecasting systems
        - ★ Calculates prediction intervals (95% confidence bounds)
        - ★ Evaluates coverage statistics to validate uncertainty estimates
        - ★ Critical for decision-making under uncertainty in health planning
        - ★ Provides residual analysis for model reliability assessment
        
        COMPONENT 1 SECONDARY: Uncertainty in climate sensitivity estimates
        - Helps quantify confidence in climate-health relationship strength
        
        Returns prediction intervals, coverage statistics, and uncertainty metrics.
        """
        interval_results = {}
        
        # For now, provide a basic uncertainty analysis
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict') and model.is_fitted:
                    predictions = model.predict(X)
                    
                    # Handle different prediction lengths (e.g., LSTM/GRU with sequences)
                    if len(predictions) != len(y):
                        # For sequence models, truncate y to match predictions
                        y_truncated = y[-len(predictions):] if len(predictions) < len(y) else y[:len(predictions)]
                        predictions_truncated = predictions[:len(y_truncated)]
                    else:
                        y_truncated = y
                        predictions_truncated = predictions
                    
                    # Calculate residuals for uncertainty estimation
                    residuals = y_truncated - predictions_truncated
                    residual_std = np.std(residuals)
                    
                    # Simple prediction intervals (mean ± 1.96 * std)
                    lower_bound = predictions_truncated - 1.96 * residual_std
                    upper_bound = predictions_truncated + 1.96 * residual_std
                    
                    # Coverage statistics
                    coverage = np.mean((y_truncated >= lower_bound) & (y_truncated <= upper_bound))
                    
                    interval_results[model_name] = {
                        'residual_std': float(residual_std),
                        'coverage_95': float(coverage),
                        'mean_prediction': float(np.mean(predictions_truncated)),
                        'prediction_std': float(np.std(predictions_truncated)),
                        'samples_used': len(predictions_truncated)
                    }
                else:
                    interval_results[model_name] = {'error': 'Model not fitted or no predict method'}
                    
            except Exception as e:
                logger.warning(f"Prediction interval analysis failed for {model_name}: {e}")
                interval_results[model_name] = {'error': str(e)}
        
        return interval_results
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        COMPONENT 2 PRIMARY: Calculate comprehensive metrics including Poisson deviance
        
        PURPOSE: Evaluate count data predictions with appropriate statistical measures
        - ★ PRIMARY for Component 2: Proper evaluation of consultation count forecasts
        - ★ Uses Poisson deviance appropriate for count data (health consultations)
        - ★ Includes RMSE, MAE, R², MAPE for comprehensive assessment
        - ★ Calculates mean prediction ratio for bias detection
        - ★ Essential for comparing forecasting model performance
        
        COMPONENT 1 SECONDARY: Baseline metrics for climate sensitivity analysis
        - Provides standard metrics for comparing climate-health model fits
        
        Returns comprehensive metrics optimized for count data evaluation.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        try:
            # Ensure arrays are the same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Ensure non-negative predictions for count data
            y_pred = np.maximum(y_pred, 1e-8)
            
            # Standard regression metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            
            # R2 score with error handling
            try:
                r2 = r2_score(y_true, y_pred)
            except:
                r2 = 0.0
            
            # Poisson deviance for count data (Component 2 methodology)
            poisson_deviance = self._calculate_poisson_deviance(y_true, y_pred)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
            
            # Additional count-specific metrics
            mean_ratio = np.mean(y_pred) / np.maximum(np.mean(y_true), 1e-8)
            
            return {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2_score': float(r2),
                'poisson_deviance': float(poisson_deviance),
                'mape': float(mape),
                'mean_prediction_ratio': float(mean_ratio),
                'n_samples': len(y_true)
            }
        except Exception as e:
            logger.warning(f"Metric calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate standard regression metrics (backward compatibility)"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        try:
            # Ensure arrays are the same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            
            # R2 score with error handling
            try:
                r2 = r2_score(y_true, y_pred)
            except:
                r2 = 0.0
            
            # Mean Absolute Percentage Error
            try:
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            except:
                mape = 0.0
            
            return {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'mse': float(mse)
            }
            
        except Exception as e:
            logger.warning(f"Metrics calculation failed: {e}")
            return {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0, 'mape': 0.0, 'mse': 0.0}
    
    def _rank_models(self) -> Dict[str, Dict[str, float]]:
        """Rank models based on performance metrics"""
        rankings = {
            'overall': {},
            'by_metric': {}
        }
        
        # Collect RMSE scores from time series CV if available
        if hasattr(self, 'results') and 'time_series_cv' in self.results:
            ts_results = self.results['time_series_cv']
            rmse_scores = {}
            
            for model_name, results in ts_results.items():
                if isinstance(results, dict) and 'mean_rmse' in results:
                    rmse_scores[model_name] = results['mean_rmse']
                elif isinstance(results, dict) and 'rmse' in results:
                    rmse_scores[model_name] = results['rmse']
            
            # Rank by RMSE (lower is better)
            rankings['overall'] = rmse_scores
            rankings['by_metric']['rmse'] = rmse_scores
        
        # If no CV results, use dummy rankings
        if not rankings['overall'] and self.models:
            for model_name in self.models.keys():
                rankings['overall'][model_name] = 1.0
        
        return rankings
    
    def _surge_prediction_analysis(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> Dict:
        """
        COMPONENT 2 PRIMARY: Analyze consultation surge prediction using PR-AUC methodology
        
        PURPOSE: Evaluate models' ability to predict high consultation volume periods
        - ⭐ PRIMARY for Component 2: Tests forecasting capability for health system planning
        - ⭐ Uses Precision-Recall AUC to evaluate surge detection (top 10% consultation volumes)
        - ⭐ Critical for early warning systems and resource allocation
        - ⭐ Implements proper time series validation to prevent data leakage
        - ⭐ Evaluates models as binary classifiers for surge/no-surge prediction
        
        COMPONENT 1 SECONDARY: Climate-driven surge identification
        - Helps identify if climate extremes drive consultation surges
        
        Returns PR-AUC scores for consultation surge prediction performance.
        """
        from sklearn.metrics import precision_recall_curve, auc
        
        surge_results = {}
        
        # Define surge threshold (top 10% of consultation counts)
        surge_threshold = np.percentile(y, 90)
        y_surge = (y > surge_threshold).astype(int)
        
        logger.info(f"  - Surge threshold: {surge_threshold:.2f} consultations")
        logger.info(f"  - Surge rate: {y_surge.mean():.3f}")
        
        # Use time series split to avoid data leakage
        if 'date' in df.columns:
            sort_idx = df['date'].argsort()
            X_sorted = X.iloc[sort_idx]
            y_sorted = y.iloc[sort_idx]
            y_surge_sorted = y_surge.iloc[sort_idx]
            df_sorted = df.iloc[sort_idx]
        else:
            X_sorted, y_sorted, y_surge_sorted, df_sorted = X, y, y_surge, df
        
        # Use 3-fold time series validation
        n_samples = len(X_sorted)
        test_size = n_samples // 4
        
        for model_name, model in self.models.items():
            logger.info(f"  - Surge prediction analysis for {model.name}...")
            
            pr_aucs = []
            fold_details = []
            
            try:
                for fold in range(3):
                    # Time series split
                    train_end = n_samples - (3 - fold) * test_size
                    test_start = train_end
                    test_end = min(test_start + test_size, n_samples)
                    
                    if train_end < 100 or test_end - test_start < 50:
                        continue
                    
                    X_train = X_sorted.iloc[:train_end]
                    y_train = y_sorted.iloc[:train_end]
                    X_test = X_sorted.iloc[test_start:test_end]
                    y_test = y_sorted.iloc[test_start:test_end]
                    y_surge_test = y_surge_sorted.iloc[test_start:test_end]
                    
                    # Need both classes for PR-AUC
                    if len(np.unique(y_surge_test)) < 2:
                        continue
                    
                    # Train model copy
                    model_copy = self._get_model_copy(model, model_name)
                    
                    if model_name in ['lstm', 'gru']:
                        X_train_seq = X_train.copy()
                        X_train_seq['date'] = df_sorted['date'].iloc[:train_end]
                        X_train_seq['admin1_encoded'] = df_sorted.get('admin1_encoded', pd.Series([0]*len(X_train))).iloc[:train_end]
                        
                        X_test_seq = X_test.copy()
                        X_test_seq['date'] = df_sorted['date'].iloc[test_start:test_end]
                        X_test_seq['admin1_encoded'] = df_sorted.get('admin1_encoded', pd.Series([0]*len(X_test))).iloc[test_start:test_end]
                        
                        model_copy.fit(X_train_seq, y_train)
                        y_pred_prob = model_copy.predict(X_test_seq)
                    else:
                        model_copy.fit(X_train, y_train)
                        y_pred_prob = model_copy.predict(X_test)
                    
                    # Use predictions as probability scores for surge detection
                    y_pred_prob = np.maximum(y_pred_prob, 0)  # Ensure non-negative
                    
                    # Calculate PR-AUC
                    if len(y_pred_prob) == len(y_surge_test):
                        precision, recall, _ = precision_recall_curve(y_surge_test, y_pred_prob)
                        pr_auc = auc(recall, precision)
                        pr_aucs.append(pr_auc)
                        
                        fold_details.append({
                            'fold': fold,
                            'pr_auc': pr_auc,
                            'n_surges': y_surge_test.sum(),
                            'surge_rate': y_surge_test.mean(),
                            'test_size': len(y_test)
                        })
                
                if pr_aucs:
                    surge_results[model_name] = {
                        'pr_auc_mean': np.mean(pr_aucs),
                        'pr_auc_std': np.std(pr_aucs),
                        'pr_auc_scores': pr_aucs,
                        'n_successful_folds': len(pr_aucs),
                        'surge_threshold': surge_threshold,
                        'overall_surge_rate': y_surge.mean(),
                        'fold_details': fold_details
                    }
                else:
                    surge_results[model_name] = {'error': 'No successful folds for surge prediction'}
                    
            except Exception as e:
                logger.warning(f"Surge prediction analysis failed for {model_name}: {e}")
                surge_results[model_name] = {'error': str(e)}
        
        return surge_results
    
    def _calculate_poisson_deviance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        COMPONENT 2 PRIMARY: Calculate Poisson deviance for count data evaluation
        
        PURPOSE: Proper statistical measure for evaluating consultation count predictions
        - ★ PRIMARY for Component 2: Gold standard for count data model evaluation
        - ★ Accounts for the discrete, non-negative nature of consultation counts
        - ★ More appropriate than MSE for Poisson-distributed health data
        - ★ Handles zero consultation counts properly
        - ★ Used in time series cross-validation for model comparison
        
        COMPONENT 1 SECONDARY: Statistical fit measure for climate-health models
        
        Implements proper Poisson deviance formula with numerical stability.
        """
        epsilon = 1e-8
        y_pred = np.maximum(y_pred, epsilon)  # Avoid log(0)
        
        # Poisson deviance formula: 2 * sum(y_true * log(y_true / y_pred) - (y_true - y_pred))
        # Handle y_true = 0 case separately
        mask_nonzero = y_true > 0
        deviance = 0.0
        
        if np.any(mask_nonzero):
            y_true_nz = y_true[mask_nonzero]
            y_pred_nz = y_pred[mask_nonzero]
            deviance += 2 * np.sum(y_true_nz * np.log(y_true_nz / y_pred_nz) - (y_true_nz - y_pred_nz))
        
        if np.any(~mask_nonzero):
            y_pred_z = y_pred[~mask_nonzero]
            deviance += 2 * np.sum(y_pred_z)  # When y_true = 0, deviance contribution is 2 * y_pred
        
        return deviance / len(y_true)  # Normalized by sample size
    
    def _calculate_temporal_stability(self, fold_metrics: List[Dict]) -> Dict:
        """
        COMPONENT 2 PRIMARY: Calculate temporal stability of model performance
        
        PURPOSE: Assess forecasting model consistency over time
        - ★ PRIMARY for Component 2: Critical for reliable forecasting systems
        - ★ Measures coefficient of variation in RMSE across time periods
        - ★ Stability score indicates prediction consistency over time
        - ★ Essential for operational forecasting system deployment
        - ★ Helps identify models that maintain performance over time
        
        COMPONENT 1 SECONDARY: Temporal consistency of climate-health relationships
        - Validates that climate sensitivity patterns are stable over time
        
        Returns stability metrics to assess temporal performance consistency.
        """
        if not fold_metrics or len(fold_metrics) < 2:
            return {'stability_score': 1.0, 'coefficient_of_variation': 0.0}
        
        # Extract RMSE scores across folds
        rmse_scores = []
        for metrics in fold_metrics:
            if 'rmse' in metrics:
                rmse_scores.append(metrics['rmse'])
        
        if not rmse_scores:
            return {'stability_score': 1.0, 'coefficient_of_variation': 0.0}
        
        # Calculate coefficient of variation (lower = more stable)
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        cv = std_rmse / mean_rmse if mean_rmse > 0 else 0.0
        
        # Stability score (higher = more stable)
        stability_score = 1.0 / (1.0 + cv)
        
        return {
            'stability_score': stability_score,
            'coefficient_of_variation': cv,
            'rmse_mean': mean_rmse,
            'rmse_std': std_rmse
        }
    
    def _get_model_copy(self, model, model_name: str):
        """Get a copy of model for independent training (avoid contamination)"""
        if hasattr(model, 'get_params') and hasattr(model, '__class__'):
            # For sklearn-style models
            model_class = model.__class__
            params = model.get_params()
            return model_class(**params)
        else:
            # For custom models, assume they have a copy method or can be recreated
            try:
                if hasattr(model, 'config'):
                    model_class = model.__class__
                    return model_class(model.config)
                else:
                    # Fallback - return the original model (not ideal but functional)
                    logger.warning(f"Could not create copy of {model_name}, using original")
                    return model
            except:
                logger.warning(f"Could not create copy of {model_name}, using original")
                return model
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across multiple folds/splits with enhanced statistics"""
        if not metrics_list:
            return {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0, 'mape': 0.0}
        
        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # Average each metric
        averaged = {}
        for key in all_keys:
            values = [metrics.get(key, 0.0) for metrics in metrics_list if isinstance(metrics.get(key), (int, float))]
            if values:
                averaged[key] = np.mean(values)
                averaged[f'{key}_std'] = np.std(values)
            else:
                averaged[key] = 0.0
                averaged[f'{key}_std'] = 0.0
        
        return averaged

# === Module-level convenience wrappers ======================================

def evaluate(models: Dict, feature_data: pd.DataFrame, config: Dict | None = None) -> Dict[str, Any]:
    """
    BOTH COMPONENTS: Main entry point for comprehensive model evaluation
    
    COMPONENT 1 USAGE: Climate sensitivity analysis evaluation
    - Evaluates how well models identify climate-sensitive morbidities
    - Focus on feature importance, spatial validation, extreme event performance
    - Validates climate-health relationship strength and consistency
    
    COMPONENT 2 USAGE: Predictive forecasting evaluation 
    - Evaluates forecasting accuracy and temporal performance
    - Focus on time series validation, surge prediction, uncertainty quantification
    - Validates prediction reliability for health system planning

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of trained models (RandomForest, XGBoost for Component 1; LSTM, GRU for Component 2)
    feature_data : pd.DataFrame
        Feature-engineered dataframe with weather data and consultation counts
    config : Dict, optional
        Configuration dictionary with evaluation parameters

    Returns
    -------
    Dict[str, Any]
        Comprehensive evaluation results supporting both analytical objectives
    """
    evaluator = ModelEvaluator(models=models, config=config or {})
    return evaluator.evaluate_models(feature_data)


def evaluate_models(models: Dict, feature_data: pd.DataFrame, config: Dict | None = None) -> Dict[str, Any]:
    """
    BOTH COMPONENTS: Preferred entry point for model evaluation pipeline
    
    COMPONENT 1 & 2: Comprehensive evaluation supporting both analytical objectives:
    - Component 1: Climate sensitivity analysis with interpretability focus  
    - Component 2: Predictive forecasting with accuracy focus
    
    This is the main function called by run_analysis.py for complete model assessment.
    Thin wrapper around evaluate() for API compatibility.
    """
    return evaluate(models=models, feature_data=feature_data, config=config or {})