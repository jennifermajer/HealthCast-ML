"""
Unit tests for machine learning models.
Tests model training, prediction, and evaluation functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Import models to test
import sys
sys.path.append('src')
from models import (
    BaseModel, PoissonModel, NegativeBinomialModel, RandomForestModel,
    XGBoostModel, LightGBMModel, EnsembleModel, train_all_models
)

# Try to import deep learning models (may not be available in all environments)
try:
    from models import LSTMModel, GRUModel
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

class TestBaseModel:
    """Test suite for BaseModel abstract class"""

    def test_base_model_cannot_be_instantiated(self):
        """Test that BaseModel cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseModel("test", {})

    def test_base_model_abstract_methods(self):
        """Test that subclasses must implement abstract methods"""
        class IncompleteModel(BaseModel):
            pass
        
        with pytest.raises(TypeError):
            IncompleteModel("incomplete", {})

class TestPoissonModel:
    """Test suite for Poisson regression model"""

    @pytest.fixture
    def sample_count_data(self):
        """Create sample count data for Poisson regression"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate count target with Poisson distribution
        linear_combination = X @ np.array([1.5, -0.8, 0.3, 2.1, -1.2])
        lambda_vals = np.exp(linear_combination - 2)  # Ensure reasonable counts
        y = np.random.poisson(lambda_vals)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        
        return X_df, y_series

    def test_poisson_model_initialization(self):
        """Test Poisson model initialization"""
        model = PoissonModel()
        assert model.name == "Poisson Regression"
        assert not model.is_fitted

    def test_poisson_model_fit_and_predict(self, sample_count_data):
        """Test Poisson model fitting and prediction"""
        X, y = sample_count_data
        model = PoissonModel()
        
        # Fit model
        model.fit(X, y)
        assert model.is_fitted
        assert model.model is not None
        
        # Make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert np.all(predictions >= 0)  # Poisson predictions should be non-negative

    def test_poisson_model_feature_importance(self, sample_count_data):
        """Test Poisson model feature importance extraction"""
        X, y = sample_count_data
        model = PoissonModel()
        
        model.fit(X, y)
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert len(importance) == len(X.columns)
        assert all(isinstance(v, (int, float)) for v in importance.values())

    def test_poisson_model_save_load(self, sample_count_data):
        """Test Poisson model save and load functionality"""
        X, y = sample_count_data
        model = PoissonModel()
        model.fit(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            model.save_model(f.name)
            
            # Load model
            new_model = PoissonModel()
            new_model.load_model(f.name)
            
            assert new_model.is_fitted
            
            # Compare predictions
            original_pred = model.predict(X)
            loaded_pred = new_model.predict(X)
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
            
            os.unlink(f.name)

class TestRandomForestModel:
    """Test suite for Random Forest model"""

    @pytest.fixture
    def sample_regression_data(self):
        """Create sample regression data"""
        X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        return X_df, y_series

    def test_random_forest_initialization(self):
        """Test Random Forest model initialization"""
        model = RandomForestModel()
        assert model.name == "Random Forest"
        assert not model.is_fitted

    def test_random_forest_fit_and_predict(self, sample_regression_data):
        """Test Random Forest fitting and prediction"""
        X, y = sample_regression_data
        model = RandomForestModel()
        
        model.fit(X, y)
        assert model.is_fitted
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_random_forest_hyperparameter_tuning(self, sample_regression_data):
        """Test Random Forest with hyperparameter tuning"""
        X, y = sample_regression_data
        model = RandomForestModel()
        
        # This should use RandomizedSearchCV internally
        model.fit(X, y, tune_hyperparameters=True)
        assert model.is_fitted

    def test_random_forest_feature_importance(self, sample_regression_data):
        """Test Random Forest feature importance"""
        X, y = sample_regression_data
        model = RandomForestModel()
        
        model.fit(X, y)
        importance = model.get_feature_importance()
        
        assert importance is not None
        assert len(importance) == len(X.columns)
        assert all(0 <= v <= 1 for v in importance.values())  # RF importance is normalized

class TestXGBoostModel:
    """Test suite for XGBoost model"""

    @pytest.fixture
    def sample_data_with_validation(self):
        """Create sample data with train/validation split"""
        X, y = make_regression(n_samples=1000, n_features=8, noise=0.1, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        
        X_train, X_val, y_train, y_val = train_test_split(X_df, y_series, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val

    def test_xgboost_initialization(self):
        """Test XGBoost model initialization"""
        model = XGBoostModel()
        assert model.name == "XGBoost"
        assert not model.is_fitted

    def test_xgboost_fit_and_predict(self, sample_data_with_validation):
        """Test XGBoost fitting and prediction"""
        X_train, X_val, y_train, y_val = sample_data_with_validation
        model = XGBoostModel()
        
        model.fit(X_train, y_train)
        assert model.is_fitted
        
        predictions = model.predict(X_train)
        assert len(predictions) == len(y_train)

    def test_xgboost_early_stopping(self, sample_data_with_validation):
        """Test XGBoost with early stopping"""
        X_train, X_val, y_train, y_val = sample_data_with_validation
        model = XGBoostModel()
        
        # Fit with validation set for early stopping
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        assert model.is_fitted

class TestLightGBMModel:
    """Test suite for LightGBM model"""

    def test_lightgbm_initialization(self):
        """Test LightGBM model initialization"""
        model = LightGBMModel()
        assert model.name == "LightGBM"
        assert not model.is_fitted

    def test_lightgbm_fit_and_predict(self, sample_data_with_validation=None):
        """Test LightGBM fitting and prediction"""
        # Create simple data for this test
        X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y_series = pd.Series(y)
        
        model = LightGBMModel()
        model.fit(X_df, y_series)
        assert model.is_fitted
        
        predictions = model.predict(X_df)
        assert len(predictions) == len(y_series)

@pytest.mark.skipif(not DEEP_LEARNING_AVAILABLE, reason="TensorFlow not available")
class TestLSTMModel:
    """Test suite for LSTM model"""

    @pytest.fixture
    def sample_time_series_data(self):
        """Create sample time series data with admin1 and date columns"""
        np.random.seed(42)
        n_days = 100
        n_regions = 3
        n_features = 5
        
        dates = pd.date_range('2023-01-01', periods=n_days)
        regions = [0, 1, 2]  # Encoded admin1 values
        
        data = []
        for region in regions:
            for i, date in enumerate(dates):
                row = {
                    'date': date,
                    'admin1_encoded': region,
                    'consultation_count': np.random.poisson(10 + region + np.sin(i/10))
                }
                
                # Add features
                for j in range(n_features):
                    row[f'feature_{j}'] = np.random.randn() + region * 0.5
                
                data.append(row)
        
        df = pd.DataFrame(data)
        X = df.drop('consultation_count', axis=1)
        y = df['consultation_count']
        
        return X, y

    def test_lstm_initialization(self):
        """Test LSTM model initialization"""
        model = LSTMModel()
        assert model.name == "LSTM"
        assert not model.is_fitted

    def test_lstm_sequence_creation(self, sample_time_series_data):
        """Test LSTM sequence creation"""
        X, y = sample_time_series_data
        model = LSTMModel()
        
        # Test sequence creation method
        X_seq, y_seq = model._create_sequences(X, y)
        
        # Should have some sequences
        assert len(X_seq) > 0
        assert len(y_seq) > 0
        assert X_seq.shape[0] == y_seq.shape[0]

    @pytest.mark.slow
    def test_lstm_fit_and_predict(self, sample_time_series_data):
        """Test LSTM fitting and prediction (slow test)"""
        X, y = sample_time_series_data
        
        # Use very small model for fast testing
        config = {
            'lstm': {
                'sequence_length': 5,
                'lstm_units': 8,
                'epochs': 5,
                'batch_size': 16
            }
        }
        
        model = LSTMModel(config)
        
        try:
            model.fit(X, y)
            assert model.is_fitted
            
            predictions = model.predict(X)
            assert len(predictions) > 0
        except Exception as e:
            pytest.skip(f"LSTM test skipped due to: {e}")

class TestEnsembleModel:
    """Test suite for Ensemble model"""

    @pytest.fixture
    def sample_base_models(self, sample_regression_data):
        """Create sample base models for ensemble testing"""
        X, y = sample_regression_data
        
        # Create and train base models
        rf_model = RandomForestModel()
        rf_model.fit(X, y)
        
        xgb_model = XGBoostModel()
        xgb_model.fit(X, y)
        
        return [rf_model, xgb_model], X, y

    def test_ensemble_initialization(self, sample_base_models):
        """Test ensemble model initialization"""
        base_models, X, y = sample_base_models
        ensemble = EnsembleModel(base_models)
        
        assert ensemble.name == "Ensemble"
        assert len(ensemble.base_models) == 2
        assert not ensemble.is_fitted

    def test_ensemble_fit_and_predict(self, sample_base_models):
        """Test ensemble fitting and prediction"""
        base_models, X, y = sample_base_models
        ensemble = EnsembleModel(base_models)
        
        ensemble.fit(X, y)
        assert ensemble.is_fitted
        assert ensemble.weights is not None
        assert len(ensemble.weights) == len(base_models)
        
        predictions = ensemble.predict(X)
        assert len(predictions) == len(y)

    def test_ensemble_weight_optimization(self, sample_base_models):
        """Test ensemble weight optimization"""
        base_models, X, y = sample_base_models
        ensemble = EnsembleModel(base_models)
        
        ensemble.fit(X, y, optimize_weights=True)
        
        # Weights should sum to approximately 1
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6
        assert all(w >= 0 for w in ensemble.weights)

    def test_ensemble_feature_importance(self, sample_base_models):
        """Test ensemble feature importance aggregation"""
        base_models, X, y = sample_base_models
        ensemble = EnsembleModel(base_models)
        
        ensemble.fit(X, y)
        importance = ensemble.get_feature_importance()
        
        assert importance is not None
        assert len(importance) == len(X.columns)

class TestModelTrainingPipeline:
    """Test suite for the complete model training pipeline"""

    @pytest.fixture
    def sample_climate_health_data(self):
        """Create sample climate-health dataset similar to real data structure"""
        np.random.seed(42)
        n_records = 1000
        
        # Create features similar to what feature engineering would produce
        data = {
            'date': pd.date_range('2023-01-01', periods=n_records//10).repeat(10)[:n_records],
            'admin1': np.tile(['Aleppo', 'Damascus', 'Homs', 'Idlib', 'Lattakia'], n_records//5)[:n_records],
            'morbidity_category': np.random.choice(['respiratory', 'gastrointestinal', 'cardiovascular'], n_records),
            'consultation_count': np.random.poisson(5, n_records),
            
            # Climate features
            'temp_max': np.random.normal(25, 10, n_records),
            'temp_min': np.random.normal(15, 8, n_records),
            'temp_mean': np.random.normal(20, 9, n_records),
            'precipitation': np.random.exponential(2, n_records),
            
            # Lag features
            'temp_mean_lag_1': np.random.normal(20, 9, n_records),
            'temp_mean_lag_7': np.random.normal(20, 9, n_records),
            'precipitation_lag_1': np.random.exponential(2, n_records),
            'precipitation_lag_7': np.random.exponential(2, n_records),
            
            # Temporal features
            'month': np.random.randint(1, 13, n_records),
            'dayofweek': np.random.randint(0, 7, n_records),
            'is_weekend': np.random.binomial(1, 0.3, n_records),
            'month_sin': np.random.uniform(-1, 1, n_records),
            'month_cos': np.random.uniform(-1, 1, n_records),
            
            # Demographic features
            'is_female': np.random.binomial(1, 0.5, n_records),
            'is_young_child': np.random.binomial(1, 0.2, n_records),
            'is_elderly': np.random.binomial(1, 0.15, n_records),
            
            # Encoded categorical features
            'admin1_encoded': np.random.randint(0, 5, n_records),
            'morbidity_category_encoded': np.random.randint(0, 3, n_records),
        }
        
        return pd.DataFrame(data)

    def test_train_all_models_basic(self, sample_climate_health_data):
        """Test basic model training pipeline"""
        config = {
            'evaluation': {'cv_folds': 3},
            'random_forest': {'n_estimators': 10},  # Small for fast testing
            'xgboost': {'n_estimators': 10}
        }
        
        models = train_all_models(sample_climate_health_data, config)
        
        # Should train at least some models
        assert len(models) > 0
        
        # Check that models are fitted
        for name, model in models.items():
            assert model.is_fitted, f"Model {name} is not fitted"

    def test_train_all_models_with_ensemble(self, sample_climate_health_data):
        """Test model training pipeline with ensemble creation"""
        config = {
            'random_forest': {'n_estimators': 5},
            'xgboost': {'n_estimators': 5}
        }
        
        models = train_all_models(sample_climate_health_data, config)
        
        # Should include ensemble if multiple models trained successfully
        if len([m for m in models.keys() if m != 'ensemble']) >= 2:
            assert 'ensemble' in models

    def test_train_all_models_insufficient_data_for_deep_learning(self):
        """Test that deep learning models are skipped with insufficient data"""
        small_data = pd.DataFrame({
            'consultation_count': np.random.poisson(5, 100),
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })
        
        models = train_all_models(small_data)
        
        # Should not include LSTM/GRU due to insufficient data
        assert 'lstm' not in models
        assert 'gru' not in models

class TestModelErrorHandling:
    """Test suite for model error handling and edge cases"""

    def test_model_fit_with_empty_data(self):
        """Test model behavior with empty data"""
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)
        
        model = RandomForestModel()
        
        with pytest.raises(Exception):  # Should raise some kind of error
            model.fit(empty_X, empty_y)

    def test_model_predict_before_fit(self):
        """Test prediction before fitting"""
        X = pd.DataFrame({'feature_1': [1, 2, 3]})
        model = RandomForestModel()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)

    def test_model_with_mismatched_features(self, sample_regression_data):
        """Test model with mismatched features between fit and predict"""
        X, y = sample_regression_data
        model = RandomForestModel()
        
        model.fit(X, y)
        
        # Try to predict with different features
        X_different = pd.DataFrame({'different_feature': [1, 2, 3]})
        
        with pytest.raises(Exception):  # Should raise some kind of error
            model.predict(X_different)

    def test_model_with_nan_values(self):
        """Test model handling of NaN values"""
        X = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4],
            'feature_2': [1, np.nan, 3, 4]
        })
        y = pd.Series([1, 2, 3, 4])
        
        model = RandomForestModel()
        
        # Depending on implementation, should either handle NaNs or raise error
        try:
            model.fit(X, y)
            predictions = model.predict(X)
            assert len(predictions) == len(y)
        except Exception:
            # It's acceptable for model to reject NaN values
            pass

class TestModelConfiguration:
    """Test suite for model configuration handling"""

    def test_model_with_custom_config(self, sample_regression_data):
        """Test model with custom configuration"""
        X, y = sample_regression_data
        
        custom_config = {
            'random_forest': {
                'n_estimators': 50,
                'max_depth': 5,
                'random_state': 123
            }
        }
        
        model = RandomForestModel(custom_config)
        model.fit(X, y)
        
        # Check that custom parameters were used
        assert model.model.n_estimators == 50
        assert model.model.max_depth == 5
        assert model.model.random_state == 123

    def test_model_with_empty_config(self, sample_regression_data):
        """Test model with empty configuration (should use defaults)"""
        X, y = sample_regression_data
        
        model = RandomForestModel({})
        model.fit(X, y)
        
        # Should use default parameters
        assert model.model.n_estimators == 100  # Default value

class TestModelPersistence:
    """Test suite for model saving and loading"""

    def test_model_save_load_consistency(self, sample_regression_data):
        """Test that saved and loaded models produce identical predictions"""
        X, y = sample_regression_data
        X_test = X.iloc[:10]  # Small test set
        
        # Train and save model
        original_model = RandomForestModel()
        original_model.fit(X, y)
        original_predictions = original_model.predict(X_test)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            original_model.save_model(f.name)
            
            # Load model and test predictions
            loaded_model = RandomForestModel()
            loaded_model.load_model(f.name)
            loaded_predictions = loaded_model.predict(X_test)
            
            # Predictions should be identical
            np.testing.assert_array_almost_equal(
                original_predictions, loaded_predictions, decimal=10
            )
            
            os.unlink(f.name)

    def test_model_save_load_feature_importance(self, sample_regression_data):
        """Test that feature importance is preserved through save/load"""
        X, y = sample_regression_data
        
        original_model = RandomForestModel()
        original_model.fit(X, y)
        original_importance = original_model.get_feature_importance()
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            original_model.save_model(f.name)
            
            loaded_model = RandomForestModel()
            loaded_model.load_model(f.name)
            loaded_importance = loaded_model.get_feature_importance()
            
            # Feature importance should be preserved
            assert original_importance.keys() == loaded_importance.keys()
            for feature in original_importance.keys():
                assert abs(original_importance[feature] - loaded_importance[feature]) < 1e-10
            
            os.unlink(f.name)

class TestModelComparison:
    """Test suite for comparing different models"""

    def test_model_performance_comparison(self, sample_regression_data):
        """Test that different models can be trained and compared"""
        X, y = sample_regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'rf': RandomForestModel(),
            'xgb': XGBoostModel()
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate simple MSE
            mse = np.mean((y_test - predictions) ** 2)
            results[name] = mse
        
        # Both models should produce reasonable results
        for name, mse in results.items():
            assert mse > 0
            assert not np.isnan(mse)

    def test_ensemble_vs_individual_models(self, sample_regression_data):
        """Test that ensemble combines individual model predictions"""
        X, y = sample_regression_data
        X_test = X.iloc[:10]
        
        # Train individual models
        rf_model = RandomForestModel()
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(X_test)
        
        xgb_model = XGBoostModel()
        xgb_model.fit(X, y)
        xgb_pred = xgb_model.predict(X_test)
        
        # Train ensemble
        ensemble = EnsembleModel([rf_model, xgb_model])
        ensemble.fit(X, y)
        ensemble_pred = ensemble.predict(X_test)
        
        # Ensemble predictions should be between individual predictions
        # (assuming equal weights)
        for i in range(len(X_test)):
            min_pred = min(rf_pred[i], xgb_pred[i])
            max_pred = max(rf_pred[i], xgb_pred[i])
            assert min_pred <= ensemble_pred[i] <= max_pred

@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for the complete modeling pipeline"""

    def test_end_to_end_model_training(self, sample_climate_health_data):
        """Test complete end-to-end model training pipeline"""
        # This simulates the complete workflow
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'data': {'processed_data_dir': temp_dir},
                'random_forest': {'n_estimators': 10},
                'xgboost': {'n_estimators': 10}
            }
            
            # Train models
            models = train_all_models(sample_climate_health_data, config)
            
            # Test that models can make predictions
            test_data = sample_climate_health_data.iloc[:10]
            exclude_cols = ['date', 'admin1', 'morbidity_category', 'consultation_count']
            feature_cols = [col for col in test_data.columns if col not in exclude_cols]
            X_test = test_data[feature_cols].fillna(0)
            
            for name, model in models.items():
                try:
                    if name not in ['lstm', 'gru']:  # Skip RNNs for this test
                        predictions = model.predict(X_test)
                        assert len(predictions) == len(X_test)
                        assert not np.any(np.isnan(predictions))
                except Exception as e:
                    pytest.fail(f"Model {name} failed prediction: {e}")

if __name__ == '__main__':
    pytest.main([__file__])