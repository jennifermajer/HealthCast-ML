"""
Climate-Health Machine Learning Models

This module implements models for two main project components:

COMPONENT 1: CLIMATE SENSITIVITY ANALYSIS
- Objective: Identify climate-sensitive morbidities by linking health consultations 
  to historical weather data (temperature, precipitation) at national and governorate levels
- Primary Models: RandomForest, XGBoost (for interpretability and feature importance)
- Focus: Understanding which diseases are most affected by climate variables

COMPONENT 2: PREDICTIVE MODELING & FORECASTING  
- Objective: Develop predictive models to quantify and forecast the impact of temperature 
  and precipitation changes on consultations for climate-sensitive morbidities
- Primary Models: All models including deep learning (LSTM, GRU) for time-series forecasting
- Focus: Accurate prediction and forecasting of health consultation volumes

All models inherit from BaseModel and support both analytical objectives.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from abc import ABC, abstractmethod
import joblib
from pathlib import Path

# Statistical models
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, NegativeBinomial

# Tree-based models
import xgboost as xgb
import lightgbm as lgb

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Utilities
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all climate-health models
    
    Supports both project components:
    - Component 1: Climate sensitivity analysis (feature importance, interpretability)
    - Component 2: Predictive modeling (forecasting, accuracy)
    
    All models must implement fit() and predict() methods.
    Feature importance is used primarily for Component 1 climate sensitivity analysis.
    """
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_importance_ = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores"""
        return self.feature_importance_
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_importance': self.feature_importance_
        }, filepath)
        logger.info(f"✓ {self.name} model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data.get('scaler')
        self.config = data.get('config', {})
        self.feature_importance_ = data.get('feature_importance')
        self.is_fitted = True
        logger.info(f"✓ {self.name} model loaded from {filepath}")


# =============================================================================
# COMPONENT 1 & 2: BASELINE STATISTICAL MODELS
# =============================================================================
# These models serve both components:
# - Component 1: Baseline for climate sensitivity analysis
# - Component 2: Fast, interpretable forecasting models


class PoissonModel(BaseModel):
    """
    Poisson regression for count data (health consultations)
    
    COMPONENT 1 USE: Baseline climate sensitivity analysis
    - Good for identifying basic climate-health relationships
    - Handles count data appropriately (non-negative integers)
    
    COMPONENT 2 USE: Fast forecasting baseline  
    - Quick training for rapid prototyping
    - Interpretable coefficients for policy communication
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("Poisson Regression", config)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'PoissonModel':
        """Fit Poisson regression model"""
        logger.info(f"Training {self.name}...")
        
        # Add constant term for statsmodels
        X_with_const = sm.add_constant(X)
        
        # Fit Poisson GLM
        self.model = sm.GLM(y, X_with_const, family=Poisson()).fit()
        
        # Extract feature importance (coefficient magnitudes)
        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': self.model.params[1:],  # Skip intercept
            'p_value': self.model.pvalues[1:],
            'importance': np.abs(self.model.params[1:])
        })
        
        self.feature_importance_ = dict(zip(coef_df['feature'], coef_df['importance']))
        self.is_fitted = True
        
        logger.info(f"✓ {self.name} training completed")
        logger.info(f"✓ AIC: {self.model.aic:.2f}, Deviance: {self.model.deviance:.2f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Poisson model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_with_const = sm.add_constant(X)
        return self.model.predict(X_with_const)

class NegativeBinomialModel(BaseModel):
    """
    Negative Binomial regression for overdispersed count data
    
    COMPONENT 1 USE: Better baseline for climate sensitivity when data shows overdispersion
    - Handles variance > mean common in health consultation data
    - More robust climate-health relationship estimation than Poisson
    
    COMPONENT 2 USE: Improved forecasting baseline
    - Better handling of consultation count variability
    - More accurate predictions when health data is overdispersed
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("Negative Binomial Regression", config)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'NegativeBinomialModel':
        """Fit Negative Binomial regression model"""
        logger.info(f"Training {self.name}...")
        
        X_with_const = sm.add_constant(X)
        
        # Fit Negative Binomial GLM
        self.model = sm.GLM(y, X_with_const, family=NegativeBinomial()).fit()
        
        # Extract feature importance
        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': self.model.params[1:],
            'p_value': self.model.pvalues[1:],
            'importance': np.abs(self.model.params[1:])
        })
        
        self.feature_importance_ = dict(zip(coef_df['feature'], coef_df['importance']))
        self.is_fitted = True
        
        logger.info(f"✓ {self.name} training completed")
        logger.info(f"✓ AIC: {self.model.aic:.2f}, Deviance: {self.model.deviance:.2f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Negative Binomial model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_with_const = sm.add_constant(X)
        return self.model.predict(X_with_const)


# =============================================================================
# COMPONENT 1 PRIMARY: TREE-BASED MODELS FOR CLIMATE SENSITIVITY ANALYSIS
# =============================================================================
# These are the PRIMARY models for Component 1 climate sensitivity analysis:
# - Excellent feature importance extraction
# - Handle non-linear climate-health relationships
# - Interpretable for identifying climate-sensitive morbidities


class RandomForestModel(BaseModel):
    """
    Random Forest - PRIMARY MODEL for Component 1 climate sensitivity analysis
    
    COMPONENT 1 PRIMARY USE: Climate sensitivity analysis
    - ★ BEST for feature importance: identifies which climate variables matter most
    - ★ Handles non-linear climate-health relationships (e.g., temperature thresholds)
    - ★ Robust to outliers in weather data
    - ★ Provides variable importance rankings for morbidity-climate linkages
    
    COMPONENT 2 SECONDARY USE: Reliable forecasting  
    - Good prediction accuracy for health consultation forecasting
    - Less prone to overfitting than individual decision trees
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("Random Forest", config)
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'RandomForestModel':
        """Fit Random Forest model with hyperparameter tuning"""
        logger.info(f"Training {self.name}...")
        
        # Get model parameters
        model_params = {**self.default_params, **self.config.get('random_forest', {})}
        
        # Hyperparameter tuning if requested
        if kwargs.get('tune_hyperparameters', False):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            search = RandomizedSearchCV(
                rf, param_grid, n_iter=20, 
                cv=tscv, scoring='neg_mean_squared_error',
                random_state=42, n_jobs=-1
            )
            
            search.fit(X, y)
            self.model = search.best_estimator_
            logger.info(f"✓ Best parameters: {search.best_params_}")
            
        else:
            self.model = RandomForestRegressor(**model_params)
            self.model.fit(X, y)
        
        # Extract feature importance
        self.feature_importance_ = dict(zip(X.columns, self.model.feature_importances_))
        self.is_fitted = True
        
        logger.info(f"✓ {self.name} training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Random Forest"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)

class XGBoostModel(BaseModel):
    """
    XGBoost - CO-PRIMARY MODEL for Component 1 climate sensitivity analysis
    
    COMPONENT 1 PRIMARY USE: Advanced climate sensitivity analysis
    - ★ EXCELLENT feature importance with SHAP support
    - ★ Captures complex climate-health interactions (e.g., temperature × precipitation)
    - ★ Handles missing climate data gracefully
    - ★ Provides gain/cover/frequency importance metrics
    
    COMPONENT 2 PRIMARY USE: High-accuracy forecasting
    - ★ Often best performing model for health consultation prediction
    - ★ Good for both short-term and medium-term forecasting
    - Regularization prevents overfitting in time series
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("XGBoost", config)
        self.default_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'XGBoostModel':
        """Fit XGBoost model"""
        logger.info(f"Training {self.name}...")
        
        # Get model parameters
        model_params = {**self.default_params, **self.config.get('xgboost', {})}
        
        self.model = xgb.XGBRegressor(**model_params)
        
        # Add early stopping if validation set provided
        eval_set = kwargs.get('eval_set')
        if eval_set is not None:
            X_val, y_val = eval_set
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X, y)
        
        # Extract feature importance
        self.feature_importance_ = dict(zip(X.columns, self.model.feature_importances_))
        self.is_fitted = True
        
        logger.info(f"✓ {self.name} training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)

class LightGBMModel(BaseModel):
    """
    LightGBM - FAST MODEL for Component 1 & 2
    
    COMPONENT 1 USE: Quick climate sensitivity analysis
    - ★ FASTEST tree-based model for large climate datasets  
    - ★ Good feature importance for rapid climate variable screening
    - Handles categorical variables (governorates, seasons) well
    
    COMPONENT 2 USE: Fast forecasting
    - ★ Quick training for real-time prediction systems
    - ★ Memory efficient for large consultation datasets
    - Good accuracy with faster training than XGBoost
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("LightGBM", config)
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': 6,
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LightGBMModel':
        """Fit LightGBM model"""
        logger.info(f"Training {self.name}...")
        
        model_params = {**self.default_params, **self.config.get('lightgbm', {})}
        
        self.model = lgb.LGBMRegressor(**model_params)
        
        # Add early stopping if validation set provided
        eval_set = kwargs.get('eval_set')
        if eval_set is not None:
            X_val, y_val = eval_set
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(10)]
            )
        else:
            self.model.fit(X, y)
        
        # Extract feature importance
        self.feature_importance_ = dict(zip(X.columns, self.model.feature_importances_))
        self.is_fitted = True
        
        logger.info(f"✓ {self.name} training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)


# =============================================================================
# COMPONENT 2 PRIMARY: DEEP LEARNING MODELS FOR TIME-SERIES FORECASTING  
# =============================================================================
# These models are PRIMARY for Component 2 predictive modeling:
# - Advanced time-series forecasting capabilities
# - Capture temporal dependencies in consultation patterns
# - Best for medium to long-term forecasting


class LSTMModel(BaseModel):
    """
    LSTM - PRIMARY MODEL for Component 2 time-series forecasting
    
    COMPONENT 2 PRIMARY USE: Advanced time-series forecasting
    - ★ BEST for capturing temporal patterns in health consultations
    - ★ Excels at medium to long-term forecasting (weeks to months)
    - ★ Learns complex seasonal patterns and climate-health lag effects
    - ★ Handles irregular time-series patterns well
    
    COMPONENT 1 LIMITED USE: Temporal climate sensitivity (advanced)
    - Can identify temporal climate-health relationships
    - Less interpretable than tree-based models for variable importance
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("LSTM", config)
        self.default_params = {
            'sequence_length': 14,  # Look back 14 days
            'lstm_units': 50,
            'dropout_rate': 0.2,
            'dense_units': 32,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10
        }
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for LSTM input"""
        sequence_length = self.config.get('lstm', {}).get('sequence_length', self.default_params['sequence_length'])
        
        # Sort by admin1 and date to ensure proper sequencing
        if 'date' in X.columns and 'admin1_encoded' in X.columns:
            sort_cols = ['admin1_encoded', 'date']
            X_sorted = X.sort_values(sort_cols).drop(columns=['date'])
        else:
            X_sorted = X
        
        X_sequences = []
        y_sequences = []
        
        # Create sequences for each admin1 region
        if 'admin1_encoded' in X_sorted.columns:
            for admin1 in X_sorted['admin1_encoded'].unique():
                admin_data = X_sorted[X_sorted['admin1_encoded'] == admin1].drop(columns=['admin1_encoded'])
                admin_y = y[X_sorted['admin1_encoded'] == admin1] if y is not None else None
                
                for i in range(sequence_length, len(admin_data)):
                    X_sequences.append(admin_data.iloc[i-sequence_length:i].values)
                    if admin_y is not None:
                        y_sequences.append(admin_y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences) if y_sequences else None
        
        return X_sequences, y_sequences
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LSTMModel':
        """Fit LSTM model"""
        logger.info(f"Training {self.name}...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X.select_dtypes(include=[np.number])),
            columns=X.select_dtypes(include=[np.number]).columns,
            index=X.index
        )
        
        # Add back non-numeric columns
        for col in X.columns:
            if col not in X_scaled.columns:
                X_scaled[col] = X[col]
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        if len(X_seq) == 0:
            raise ValueError("No sequences created. Check data structure.")
        
        logger.info(f"Created {len(X_seq)} sequences of length {X_seq.shape[1]} with {X_seq.shape[2]} features")
        
        # Get model parameters
        params = {**self.default_params, **self.config.get('lstm', {})}
        
        # Build LSTM model
        model = Sequential([
            LSTM(params['lstm_units'], return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
            Dropout(params['dropout_rate']),
            LSTM(params['lstm_units']//2),
            Dropout(params['dropout_rate']),
            Dense(params['dense_units'], activation='relu'),
            BatchNormalization(),
            Dropout(params['dropout_rate']),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=params['patience'], restore_best_weights=True),
            ReduceLROnPlateau(patience=params['patience']//2, factor=0.5)
        ]
        
        # Split for validation
        val_split = 0.2
        split_idx = int(len(X_seq) * (1 - val_split))
        
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        self.model = model
        self.history = history
        self.is_fitted = True
        
        logger.info(f"✓ {self.name} training completed")
        logger.info(f"✓ Final validation loss: {min(history.history['val_loss']):.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X.select_dtypes(include=[np.number])),
            columns=X.select_dtypes(include=[np.number]).columns,
            index=X.index
        )
        
        # Add back non-numeric columns
        for col in X.columns:
            if col not in X_scaled.columns:
                X_scaled[col] = X[col]
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            return np.array([])
        
        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()

class GRUModel(BaseModel):
    """
    GRU - FASTER ALTERNATIVE for Component 2 time-series forecasting
    
    COMPONENT 2 PRIMARY USE: Fast time-series forecasting
    - ★ FASTER training than LSTM with similar performance
    - ★ Good for short to medium-term forecasting (days to weeks)
    - ★ Better for real-time prediction systems due to speed
    - ★ Simpler architecture, less prone to overfitting
    
    COMPONENT 1 LIMITED USE: Quick temporal climate analysis
    - Faster than LSTM for temporal pattern discovery
    - Less interpretable for climate variable importance
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("GRU", config)
        self.default_params = {
            'sequence_length': 14,
            'gru_units': 50,
            'dropout_rate': 0.2,
            'dense_units': 32,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10
        }
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for GRU input (same as LSTM)"""
        sequence_length = self.config.get('gru', {}).get('sequence_length', self.default_params['sequence_length'])
        
        if 'date' in X.columns and 'admin1_encoded' in X.columns:
            sort_cols = ['admin1_encoded', 'date']
            X_sorted = X.sort_values(sort_cols).drop(columns=['date'])
        else:
            X_sorted = X
        
        X_sequences = []
        y_sequences = []
        
        if 'admin1_encoded' in X_sorted.columns:
            for admin1 in X_sorted['admin1_encoded'].unique():
                admin_data = X_sorted[X_sorted['admin1_encoded'] == admin1].drop(columns=['admin1_encoded'])
                admin_y = y[X_sorted['admin1_encoded'] == admin1] if y is not None else None
                
                for i in range(sequence_length, len(admin_data)):
                    X_sequences.append(admin_data.iloc[i-sequence_length:i].values)
                    if admin_y is not None:
                        y_sequences.append(admin_y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences) if y_sequences else None
        
        return X_sequences, y_sequences
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'GRUModel':
        """Fit GRU model"""
        logger.info(f"Training {self.name}...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X.select_dtypes(include=[np.number])),
            columns=X.select_dtypes(include=[np.number]).columns,
            index=X.index
        )
        
        for col in X.columns:
            if col not in X_scaled.columns:
                X_scaled[col] = X[col]
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        if len(X_seq) == 0:
            raise ValueError("No sequences created. Check data structure.")
        
        logger.info(f"Created {len(X_seq)} sequences of length {X_seq.shape[1]} with {X_seq.shape[2]} features")
        
        # Get model parameters
        params = {**self.default_params, **self.config.get('gru', {})}
        
        # Build GRU model
        model = Sequential([
            GRU(params['gru_units'], return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
            Dropout(params['dropout_rate']),
            GRU(params['gru_units']//2),
            Dropout(params['dropout_rate']),
            Dense(params['dense_units'], activation='relu'),
            BatchNormalization(),
            Dropout(params['dropout_rate']),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=params['patience'], restore_best_weights=True),
            ReduceLROnPlateau(patience=params['patience']//2, factor=0.5)
        ]
        
        # Split for validation
        val_split = 0.2
        split_idx = int(len(X_seq) * (1 - val_split))
        
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        self.model = model
        self.history = history
        self.is_fitted = True
        
        logger.info(f"✓ {self.name} training completed")
        logger.info(f"✓ Final validation loss: {min(history.history['val_loss']):.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with GRU"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X.select_dtypes(include=[np.number])),
            columns=X.select_dtypes(include=[np.number]).columns,
            index=X.index
        )
        
        for col in X.columns:
            if col not in X_scaled.columns:
                X_scaled[col] = X[col]
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            return np.array([])
        
        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models"""
    
    def __init__(self, base_models: List[BaseModel], config: Dict = None):
        super().__init__("Ensemble", config)
        self.base_models = base_models
        self.weights = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'EnsembleModel':
        """Fit all base models and learn optimal weights"""
        logger.info(f"Training {self.name} with {len(self.base_models)} base models...")
        
        # Train all base models
        predictions = []
        
        for model in self.base_models:
            logger.info(f"Training base model: {model.name}")
            model.fit(X, y, **kwargs)
            
            # Get predictions for weight learning
            pred = model.predict(X)
            predictions.append(pred)
        
        # Learn optimal weights using simple least squares
        predictions_matrix = np.column_stack(predictions)
        
        # Simple equal weighting (could be enhanced with more sophisticated methods)
        self.weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        # Alternative: optimize weights based on individual model performance
        if kwargs.get('optimize_weights', False):
            from scipy.optimize import minimize
            
            def objective(weights):
                ensemble_pred = np.dot(predictions_matrix, weights)
                return mean_squared_error(y, ensemble_pred)
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(len(self.base_models))]
            
            result = minimize(objective, self.weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                self.weights = result.x
                logger.info(f"✓ Optimized ensemble weights: {dict(zip([m.name for m in self.base_models], self.weights))}")
        
        # Calculate combined feature importance
        self._compute_ensemble_feature_importance(X)
        
        self.is_fitted = True
        logger.info(f"✓ {self.name} training completed")
        
        return self
    
    def _compute_ensemble_feature_importance(self, X: pd.DataFrame):
        """Compute weighted average of feature importances"""
        importance_dict = {}
        
        for model, weight in zip(self.base_models, self.weights):
            model_importance = model.get_feature_importance()
            if model_importance:
                for feature, importance in model_importance.items():
                    if feature in importance_dict:
                        importance_dict[feature] += weight * importance
                    else:
                        importance_dict[feature] = weight * importance
        
        self.feature_importance_ = importance_dict
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions with enhanced methodology"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from all base models
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Enhanced ensemble combination
        if len(predictions) == 0:
            return np.zeros(len(X))
        
        predictions_matrix = np.column_stack(predictions)
        
        # Apply weights and combine predictions
        if len(self.weights) == len(predictions):
            ensemble_pred = np.dot(predictions_matrix, self.weights)
        else:
            # Fallback to simple average if weights are mismatched
            ensemble_pred = np.mean(predictions_matrix, axis=1)
        
        # Ensure non-negative predictions for count data
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        return ensemble_pred

def train_all_models(df: pd.DataFrame, config: Dict = None) -> Dict[str, BaseModel]:
    """
    Train all models on the feature-engineered dataset
    
    Args:
        df: Feature-engineered dataframe
        config: Configuration dictionary
        
    Returns:
        Dictionary of trained models
    """
    logger.info("🤖 Starting model training pipeline...")
    
    if config is None:
        # Load default config
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    
    # Prepare data for modeling
    # Separate features from target and identifiers
    exclude_cols = ['date', 'admin1', 'category_canonical_disease_imc', 'consultation_count']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['consultation_count'].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    logger.info(f"📊 Training data shape: X={X.shape}, y={y.shape}")
    logger.info(f"📊 Features: {len(feature_cols)}")
    logger.info(f"📊 Target range: {y.min()} to {y.max()}")
    
    # Initialize models
    # Define all available models
    all_models = {
        'poisson': PoissonModel(config),
        'negative_binomial': NegativeBinomialModel(config),
        'random_forest': RandomForestModel(config),
        'xgboost': XGBoostModel(config),
        'lightgbm': LightGBMModel(config),
    }
    
    # Add deep learning models if TensorFlow is available and data is sufficient
    if len(df) > 1000:  # Only use deep learning for larger datasets
        try:
            all_models['lstm'] = LSTMModel(config)
            all_models['gru'] = GRUModel(config)
        except Exception as e:
            logger.warning(f"Skipping deep learning models: {e}")
    
    # Apply model filtering based on configuration
    models_to_skip = config.get('skip_models', [])
    models_to_train = config.get('models_to_train', None)
    
    if models_to_train is not None:
        # Only train specified models
        models = {name: model for name, model in all_models.items() 
                 if name in models_to_train and name not in models_to_skip}
        logger.info(f"🎯 Training only specified models: {list(models.keys())}")
    else:
        # Train all models except those in skip list
        models = {name: model for name, model in all_models.items() 
                 if name not in models_to_skip}
        if models_to_skip:
            logger.info(f"⏭️ Skipping models: {models_to_skip}")
    
    logger.info(f"🔄 Models to train: {list(models.keys())}")
    
    # Train individual models
    trained_models = {}
    
    for name, model in models.items():
        try:
            logger.info(f"🔄 Training {model.name}...")
            
            # Special handling for deep learning models that need different data format
            if name in ['lstm', 'gru']:
                # Deep learning models need date and admin1 for sequencing
                X_with_ids = X.copy()
                X_with_ids['date'] = df['date']
                # Find the admin1 encoded column (could be admin1_encoded)
                admin1_col = None
                for col in df.columns:
                    if 'admin1' in col and 'encoded' in col:
                        admin1_col = col
                        break
                X_with_ids['admin1_encoded'] = df.get(admin1_col, 0) if admin1_col else 0
                model.fit(X_with_ids, y)
            else:
                model.fit(X, y)
            
            trained_models[name] = model
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model.name}: {e}")
            continue
    
    # Create ensemble model if we have multiple successful models and it's not skipped
    models_to_skip = config.get('skip_models', [])
    if len(trained_models) >= 2 and 'ensemble' not in models_to_skip:
        try:
            logger.info("🔄 Training ensemble model...")
            ensemble_base_models = [model for model in trained_models.values() 
                                  if model.name not in ['LSTM', 'GRU']]  # Exclude RNNs from ensemble
            
            if len(ensemble_base_models) >= 2:
                ensemble = EnsembleModel(ensemble_base_models, config)
                ensemble.fit(X, y, optimize_weights=True)
                trained_models['ensemble'] = ensemble
                
        except Exception as e:
            logger.error(f"❌ Failed to train ensemble model: {e}")
    
    logger.info(f"✅ Model training completed: {len(trained_models)} models trained successfully")
    
    # Save models
    results_dir = Path('results/models')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in trained_models.items():
        try:
            model_path = results_dir / f'{name}_model.joblib'
            model.save_model(str(model_path))
        except Exception as e:
            logger.warning(f"Failed to save {name} model: {e}")
    
    return trained_models

def load_trained_models(model_dir: str = 'results/models') -> Dict[str, BaseModel]:
    """
    Load previously trained models
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        Dictionary of loaded models
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    models = {}
    model_classes = {
        'poisson': PoissonModel,
        'negative_binomial': NegativeBinomialModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'lstm': LSTMModel,
        'gru': GRUModel,
        'ensemble': EnsembleModel
    }
    
    for model_file in model_dir.glob('*_model.joblib'):
        model_name = model_file.stem.replace('_model', '')
        
        if model_name in model_classes:
            try:
                # Special handling for ensemble model which requires base_models
                if model_name == 'ensemble':
                    # Create empty ensemble model for loading
                    model = EnsembleModel([], {})  # Empty base_models list, will be loaded from file
                else:
                    model = model_classes[model_name]()
                
                model.load_model(str(model_file))
                models[model_name] = model
                logger.info(f"✓ Loaded {model.name} model")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name} model: {e}")
                # For ensemble, provide helpful error message
                if model_name == 'ensemble':
                    logger.warning("   Ensemble model requires base models to be loaded first. Skipping...")
    
    logger.info(f"✅ Loaded {len(models)} models")
    return models


def train_hierarchical_models(df: pd.DataFrame, config: Dict = None) -> Dict[str, Any]:
    """
    Train hierarchical models at both morbidity-group and specific levels
    Implements Component 2 methodology for partial pooling
    
    Args:
        df: Feature-engineered dataframe
        config: Configuration dictionary
        
    Returns:
        Dictionary containing hierarchical model results
    """
    logger.info("🌳 Starting hierarchical model training...")
    
    if config is None:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    
    # Define morbidity groupings for hierarchical structure
    morbidity_groups = _create_morbidity_groups(df)
    
    # Prepare data
    exclude_cols = ['date', 'admin1', 'category_canonical_disease_imc', 'consultation_count']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['consultation_count']
    morbidity_categories = df['category_canonical_disease_imc']
    
    hierarchical_results = {
        'group_models': {},
        'specific_models': {},
        'hierarchical_structure': morbidity_groups,
        'partial_pooling_weights': {}
    }
    
    # Train group-level models (Component 2 methodology)
    logger.info("🏗️ Training group-level models...")
    for group_name, group_morbidities in morbidity_groups.items():
        if not group_morbidities:
            continue
            
        # Filter data for this group
        group_mask = morbidity_categories.isin(group_morbidities)
        X_group = X[group_mask]
        y_group = y[group_mask]
        
        if len(X_group) < 50:  # Skip if insufficient data
            continue
        
        # Train interpretable baseline models for this group
        group_models = {}
        
        # Poisson regression for group
        try:
            poisson_model = PoissonModel(config)
            poisson_model.fit(X_group, y_group)
            group_models['poisson'] = poisson_model
        except Exception as e:
            logger.warning(f"Group Poisson model failed for {group_name}: {e}")
        
        # XGBoost for capturing nonlinear relationships
        try:
            xgb_model = XGBoostModel(config)
            xgb_model.fit(X_group, y_group)
            group_models['xgboost'] = xgb_model
        except Exception as e:
            logger.warning(f"Group XGBoost model failed for {group_name}: {e}")
        
        hierarchical_results['group_models'][group_name] = {
            'models': group_models,
            'n_morbidities': len(group_morbidities),
            'n_samples': len(X_group),
            'morbidities': group_morbidities
        }
        
        logger.info(f"✓ Trained group models for {group_name}: {len(group_models)} models, {len(X_group)} samples")
    
    # Train morbidity-specific models with partial pooling
    logger.info("🎯 Training morbidity-specific models with partial pooling...")
    
    for morbidity in df['category_canonical_disease_imc'].unique():
        morbidity_mask = morbidity_categories == morbidity
        X_specific = X[morbidity_mask]
        y_specific = y[morbidity_mask]
        
        if len(X_specific) < 20:  # Skip very small categories
            continue
        
        # Find which group this morbidity belongs to
        parent_group = None
        for group_name, group_morbidities in morbidity_groups.items():
            if morbidity in group_morbidities:
                parent_group = group_name
                break
        
        specific_models = {}
        pooling_weights = {}
        
        # Train specific models
        model_types = ['poisson', 'xgboost'] if len(X_specific) >= 100 else ['poisson']
        
        for model_type in model_types:
            try:
                # Train morbidity-specific model
                if model_type == 'poisson':
                    specific_model = PoissonModel(config)
                elif model_type == 'xgboost':
                    specific_model = XGBoostModel(config)
                
                specific_model.fit(X_specific, y_specific)
                
                # Calculate partial pooling weight based on sample size
                # More data = more weight on specific model, less on group model
                alpha = min(len(X_specific) / 200.0, 0.8)  # Cap at 80% specific weight
                
                pooling_weights[model_type] = {
                    'specific_weight': alpha,
                    'group_weight': 1 - alpha,
                    'sample_size': len(X_specific)
                }
                
                specific_models[model_type] = specific_model
                
            except Exception as e:
                logger.warning(f"Specific {model_type} model failed for {morbidity}: {e}")
                continue
        
        if specific_models:
            hierarchical_results['specific_models'][morbidity] = {
                'models': specific_models,
                'parent_group': parent_group,
                'n_samples': len(X_specific),
                'pooling_weights': pooling_weights
            }
            
            hierarchical_results['partial_pooling_weights'][morbidity] = pooling_weights
    
    logger.info(f"✅ Hierarchical training completed:")
    logger.info(f"   • Group models: {len(hierarchical_results['group_models'])}")
    logger.info(f"   • Specific models: {len(hierarchical_results['specific_models'])}")
    
    return hierarchical_results


def create_enhanced_ensemble(base_models: Dict[str, BaseModel], 
                           hierarchical_results: Dict = None,
                           config: Dict = None) -> EnsembleModel:
    """
    Create enhanced ensemble combining interpretable and flexible models
    Implements Component 2 ensemble methodology
    
    Args:
        base_models: Dictionary of trained base models
        hierarchical_results: Results from hierarchical modeling
        config: Configuration dictionary
        
    Returns:
        Enhanced ensemble model
    """
    logger.info("🤝 Creating enhanced ensemble model...")
    
    if not base_models:
        raise ValueError("No base models provided for ensemble")
    
    # Separate models by type for balanced ensemble
    interpretable_models = []
    flexible_models = []
    
    interpretable_types = ['poisson', 'negative_binomial']
    flexible_types = ['random_forest', 'xgboost', 'lightgbm']
    
    for name, model in base_models.items():
        if name in interpretable_types:
            interpretable_models.append(model)
        elif name in flexible_types:
            flexible_models.append(model)
    
    # Create balanced ensemble (Component 2 methodology)
    selected_models = []
    model_weights = []
    
    # Add interpretable models with equal weight among them
    if interpretable_models:
        interpretable_weight = 0.3 / len(interpretable_models)  # 30% total for interpretable
        for model in interpretable_models:
            selected_models.append(model)
            model_weights.append(interpretable_weight)
    
    # Add flexible models with equal weight among them
    if flexible_models:
        flexible_weight = 0.7 / len(flexible_models)  # 70% total for flexible
        for model in flexible_models:
            selected_models.append(model)
            model_weights.append(flexible_weight)
    
    # If only one type available, distribute equally
    if not interpretable_models or not flexible_models:
        equal_weight = 1.0 / len(selected_models)
        model_weights = [equal_weight] * len(selected_models)
    
    # Create ensemble
    ensemble = EnsembleModel(config or {})
    ensemble.fit_ensemble(selected_models, np.array(model_weights))
    
    logger.info(f"✅ Enhanced ensemble created with {len(selected_models)} models")
    logger.info(f"   • Model weights: {[f'{w:.3f}' for w in model_weights]}")
    
    return ensemble


def _create_morbidity_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Create morbidity groupings for hierarchical modeling
    Maps actual morbidities in data to disease groups
    """
    # Get actual morbidities from data
    actual_morbidities = df['category_canonical_disease_imc'].unique()
    
    # Define grouping patterns
    group_patterns = {
        'respiratory': ['respiratory', 'pneumonia', 'asthma', 'cough', 'breathing'],
        'gastrointestinal': ['diarrhea', 'gastro', 'stomach', 'abdominal', 'vomiting'],
        'vector_borne': ['malaria', 'dengue', 'fever', 'mosquito'],
        'skin_conditions': ['skin', 'dermatitis', 'rash', 'eczema'],
        'infectious': ['infection', 'sepsis', 'bacterial', 'viral'],
        'other': []  # Will contain unmapped conditions
    }
    
    # Map actual morbidities to groups
    morbidity_groups = {group: [] for group in group_patterns.keys()}
    mapped_morbidities = set()
    
    for morbidity in actual_morbidities:
        morbidity_lower = morbidity.lower()
        assigned = False
        
        for group_name, patterns in group_patterns.items():
            if group_name == 'other':
                continue
                
            # Check if any pattern matches the morbidity
            if any(pattern in morbidity_lower for pattern in patterns):
                morbidity_groups[group_name].append(morbidity)
                mapped_morbidities.add(morbidity)
                assigned = True
                break
        
        # If not assigned to any specific group, add to 'other'
        if not assigned:
            morbidity_groups['other'].append(morbidity)
    
    # Remove empty groups
    morbidity_groups = {k: v for k, v in morbidity_groups.items() if v}
    
    logger.info(f"🏗️ Created morbidity groups: {list(morbidity_groups.keys())}")
    for group, morbidities in morbidity_groups.items():
        logger.info(f"   • {group}: {len(morbidities)} conditions")
    
    return morbidity_groups