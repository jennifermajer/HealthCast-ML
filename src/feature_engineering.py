"""
Feature engineering for climate-health analysis.
Creates temporal, climate, and demographic features for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from date column
    
    Args:
        df: Input dataframe with 'date' column
        
    Returns:
        Dataframe with additional temporal features
    """
    logger.info("Creating temporal features...")
    
    df = df.copy()
    
    # Basic date components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['dayofyear'] = df['date'].dt.dayofyear
    df['week'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Seasonal features
    # Syria has distinct seasons - create seasonal indicators
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring' 
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    df['season'] = df['month'].apply(get_season)
    
    # Cyclical encoding for periodic features
    # Month (important for seasonal disease patterns)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Day of year (for longer seasonal patterns)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    # Day of week (for weekly patterns in healthcare utilization)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Time-based indicators
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)  # Saturday, Sunday
    df['is_month_start'] = (df['day'] <= 7).astype(int)
    df['is_month_end'] = (df['day'] >= 24).astype(int)
    
    # Ramadan indicator (approximate - varies by year)
    # This is a simplified approach - in practice, you'd want exact Ramadan dates
    df['is_ramadan_approx'] = 0  # Placeholder for more sophisticated Ramadan dating
    
    logger.info(f"✓ Created {len([col for col in df.columns if col.endswith(('_sin', '_cos', 'season', 'is_'))])} temporal features")
    
    return df

def create_climate_lag_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Create lagged climate features to capture delayed health effects
    
    Args:
        df: Input dataframe with climate variables
        config: Configuration dict with lag parameters
        
    Returns:
        Dataframe with lagged climate features
    """
    logger.info("Creating climate lag features...")
    
    df = df.copy()
    df = df.sort_values(['admin1', 'date']).reset_index(drop=True)
    
    # Get lag configurations
    temp_lags = config.get('features', {}).get('temperature_lags', [1, 3, 7, 14])
    precip_lags = config.get('features', {}).get('precipitation_lags', [1, 3, 7])
    
    climate_vars = ['temp_max', 'temp_min', 'temp_mean', 'temp_range', 'precipitation']
    
    # Create lagged features for each admin1 region
    for var in climate_vars:
        if var in df.columns:
            # Determine which lags to use
            if 'temp' in var:
                lags = temp_lags
            else:
                lags = precip_lags
                
            for lag in lags:
                lag_col = f'{var}_lag_{lag}'
                df[lag_col] = df.groupby('admin1')[var].shift(lag)
                
                # Rolling averages for longer patterns
                if lag >= 7:
                    roll_col = f'{var}_roll_{lag}'
                    df[roll_col] = df.groupby('admin1')[var].rolling(
                        window=lag, min_periods=1
                    ).mean().reset_index(0, drop=True)
    
    # Climate anomalies (deviation from seasonal normal)
    for var in ['temp_mean', 'precipitation']:
        if var in df.columns:
            # Calculate seasonal averages
            seasonal_mean = df.groupby(['admin1', 'month'])[var].transform('mean')
            df[f'{var}_anomaly'] = df[var] - seasonal_mean
            
            # Standardized anomalies (z-scores)
            seasonal_std = df.groupby(['admin1', 'month'])[var].transform('std')
            df[f'{var}_zscore'] = (df[var] - seasonal_mean) / (seasonal_std + 1e-8)
    
    # Climate extremes indicators
    if 'temp_max' in df.columns:
        # Heat wave indicator (temperature > 95th percentile for 3+ consecutive days)
        temp_95 = df.groupby(['admin1', 'month'])['temp_max'].transform(lambda x: x.quantile(0.95))
        df['is_hot_day'] = (df['temp_max'] > temp_95).astype(int)
        
        # Rolling sum for consecutive hot days
        df['hot_days_3day'] = df.groupby('admin1')['is_hot_day'].rolling(
            window=3, min_periods=1
        ).sum().reset_index(0, drop=True)
        
        df['is_heatwave'] = (df['hot_days_3day'] >= 3).astype(int)
    
    if 'precipitation' in df.columns:
        # Heavy rain indicator (precipitation > 90th percentile)
        precip_90 = df.groupby(['admin1', 'month'])['precipitation'].transform(lambda x: x.quantile(0.90))
        df['is_heavy_rain'] = (df['precipitation'] > precip_90).astype(int)
        
        # Drought indicator (no precipitation for 7+ days)
        df['dry_spell'] = (df['precipitation'] == 0).astype(int)
        df['dry_spell_length'] = df.groupby('admin1')['dry_spell'].rolling(
            window=7, min_periods=1
        ).sum().reset_index(0, drop=True)
        
        df['is_drought_period'] = (df['dry_spell_length'] >= 7).astype(int)
    
    # Log number of features created
    lag_features = [col for col in df.columns if any(pattern in col for pattern in ['_lag_', '_roll_', '_anomaly', '_zscore', 'is_hot', 'is_heavy', 'is_drought'])]
    logger.info(f"✓ Created {len(lag_features)} climate lag and extreme weather features")
    
    return df

def create_morbidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from morbidity classifications
    
    Args:
        df: Input dataframe with morbidity columns
        
    Returns:
        Dataframe with morbidity-based features
    """
    logger.info("Creating morbidity classification features...")
    
    df = df.copy()
    
    # Create morbidity category indicators
    # These will be key target variables for climate sensitivity analysis
    
    # Use canonical_disease_imc as primary classification
    if 'canonical_disease_imc' in df.columns:
        df['category_canonical_disease_imc'] = df['canonical_disease_imc'].fillna('Unknown')
        
        # Create binary indicators for major disease categories
        # These categories are commonly climate-sensitive
        climate_sensitive_categories = [
            'respiratory', 'diarrheal', 'vector-borne', 'heat-related',
            'cardiovascular', 'renal', 'dermatological'
        ]
        
        for category in climate_sensitive_categories:
            # Create indicator if category appears in the morbidity text (case-insensitive)
            df[f'is_{category}'] = df['category_canonical_disease_imc'].str.contains(
                category, case=False, na=False
            ).astype(int)
    
    # ICD-11 based features
    if 'icd11_title' in df.columns:
        df['has_icd11'] = df['icd11_title'].notna().astype(int)
        
        # Extract ICD-11 chapter/category if available
        # ICD-11 codes typically start with chapter indicators
        df['icd11_chapter'] = df['icd11_title'].str[:2].fillna('Unknown')
    
    # Age-morbidity interactions
    if 'age_group' in df.columns and 'standard_disease_imc' in df.columns:
        # Create age-specific morbidity categories
        df['age_morbidity'] = df['age_group'].astype(str) + '_' + df['category_canonical_disease_imc'].astype(str)
    
    # Facility type - morbidity patterns
    if 'facility_type' in df.columns:
        df['facility_morbidity'] = df['facility_type'].astype(str) + '_' + df['category_canonical_disease_imc'].astype(str)
    
    logger.info(f"✓ Created morbidity classification features")
    logger.info(f"✓ Primary morbidity categories: {df['category_canonical_disease_imc'].nunique()}")
    
    return df

def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from demographic variables
    
    Args:
        df: Input dataframe with demographic columns
        
    Returns:
        Dataframe with demographic features
    """
    logger.info("Creating demographic features...")
    
    df = df.copy()
    
    # Age group processing
    if 'age_group' in df.columns:
        df['age_group_clean'] = df['age_group'].fillna('Unknown')
        
        # Create age category indicators
        # Identify vulnerable age groups (very young and elderly)
        young_indicators = ['infant', '0-1', '0-5', '<1', '<5', 'newborn']
        elderly_indicators = ['60+', '65+', '70+', 'elderly', 'senior']
        
        df['is_young_child'] = df['age_group_clean'].str.contains(
            '|'.join(young_indicators), case=False, na=False
        ).astype(int)
        
        df['is_elderly'] = df['age_group_clean'].str.contains(
            '|'.join(elderly_indicators), case=False, na=False
        ).astype(int)
        
        df['is_vulnerable_age'] = ((df['is_young_child'] == 1) | (df['is_elderly'] == 1)).astype(int)
    
    # Additional age group features if available
    age_cols = [col for col in df.columns if col.startswith('age_group') and col != 'age_group']
    for col in age_cols:
        if col in df.columns:
            df[f'{col}_clean'] = df[col].fillna('Unknown')
    
    # Sex-based features
    if 'sex' in df.columns:
        df['sex_clean'] = df['sex'].fillna('Unknown')
        # Sex is already binary (M/F), so just ensure clean encoding
        df['is_female'] = (df['sex_clean'] == 'F').astype(int)
        df['is_male'] = (df['sex_clean'] == 'M').astype(int)
    
    # Geographic features at different administrative levels
    admin_levels = ['admin0', 'admin1', 'admin2', 'admin3']
    for level in admin_levels:
        if level in df.columns:
            df[f'{level}_clean'] = df[level].fillna('Unknown')
    
    # Health facility features
    if 'facility_type' in df.columns:
        df['facility_type_clean'] = df['facility_type'].fillna('Unknown')
        
        # Create facility category indicators
        primary_care = ['clinic', 'health center', 'primary', 'dispensary']
        secondary_care = ['hospital', 'secondary', 'district']
        emergency_care = ['emergency', 'trauma', 'urgent']
        
        df['is_primary_care'] = df['facility_type_clean'].str.contains(
            '|'.join(primary_care), case=False, na=False
        ).astype(int)
        
        df['is_secondary_care'] = df['facility_type_clean'].str.contains(
            '|'.join(secondary_care), case=False, na=False
        ).astype(int)
        
        df['is_emergency_care'] = df['facility_type_clean'].str.contains(
            '|'.join(emergency_care), case=False, na=False
        ).astype(int)
    
    if 'orgunit' in df.columns:
        df['orgunit_clean'] = df['orgunit'].fillna('Unknown')
        # Could create facility-specific indicators if needed for analysis
    
    logger.info(f"✓ Created demographic and facility features")
    
    return df

def create_consultation_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated consultation counts and rates
    
    Args:
        df: Input dataframe with all features
        
    Returns:
        Dataframe with aggregated features for modeling
    """
    logger.info("Creating consultation aggregation features...")
    
    # Create daily aggregates by admin1 and morbidity category
    grouping_cols = ['date', 'admin1', 'category_canonical_disease_imc']
    
    # Basic consultation counts
    daily_agg = df.groupby(grouping_cols).agg({
        # Count consultations
        'morbidity': 'count',
        # Climate variables (take first value since they're the same for each date-admin1)
        'temp_max': 'first',
        'temp_min': 'first', 
        'temp_mean': 'first',
        'precipitation': 'first',
        # Demographics - get distributions
        'is_female': 'mean',  # Proportion female
        'is_young_child': 'mean',  # Proportion young children
        'is_elderly': 'mean',  # Proportion elderly
        'is_vulnerable_age': 'mean'  # Proportion vulnerable ages
    }).reset_index()
    
    # Rename consultation count column
    daily_agg = daily_agg.rename(columns={'morbidity': 'consultation_count'})
    
    # Add climate lag features to aggregated data
    climate_lag_cols = [col for col in df.columns if any(pattern in col for pattern in ['_lag_', '_roll_', '_anomaly', '_zscore', 'is_hot', 'is_heavy', 'is_drought'])]
    
    if climate_lag_cols:
        climate_features = df.groupby(grouping_cols)[climate_lag_cols].first().reset_index()
        daily_agg = daily_agg.merge(climate_features, on=grouping_cols, how='left')
    
    # Add temporal features
    temporal_cols = [col for col in df.columns if any(pattern in col for pattern in ['year', 'month', 'day', 'season', '_sin', '_cos', 'is_weekend', 'quarter'])]
    
    if temporal_cols:
        temporal_features = df.groupby(grouping_cols)[temporal_cols].first().reset_index()
        daily_agg = daily_agg.merge(temporal_features, on=grouping_cols, how='left')
    
    # Sort by date and admin1 for time series features
    daily_agg = daily_agg.sort_values(['admin1', 'category_canonical_disease_imc', 'date']).reset_index(drop=True)
    
    # Create rolling consultation features (7-day and 14-day windows)
    for window in [7, 14]:
        rolling_mean = daily_agg.groupby(['admin1', 'category_canonical_disease_imc'])['consultation_count'].rolling(
            window=window, min_periods=1
        ).mean().reset_index(level=[0, 1], drop=True)
        daily_agg[f'consultation_roll_{window}'] = rolling_mean.values
        
        # Rolling standard deviation
        rolling_std = daily_agg.groupby(['admin1', 'category_canonical_disease_imc'])['consultation_count'].rolling(
            window=window, min_periods=1
        ).std().reset_index(level=[0, 1], drop=True)
        daily_agg[f'consultation_std_{window}'] = rolling_std.values
    
    # Create lag features for consultation counts (to capture autoregressive patterns)
    for lag in [1, 7, 14]:
        daily_agg[f'consultation_lag_{lag}'] = daily_agg.groupby(['admin1', 'category_canonical_disease_imc'])['consultation_count'].shift(lag)
    
    # Create anomaly indicators for consultation counts
    # Consultation rate compared to seasonal average
    seasonal_mean = daily_agg.groupby(['admin1', 'category_canonical_disease_imc', 'month'])['consultation_count'].transform('mean')
    seasonal_std = daily_agg.groupby(['admin1', 'category_canonical_disease_imc', 'month'])['consultation_count'].transform('std')
    
    daily_agg['consultation_seasonal_anomaly'] = daily_agg['consultation_count'] - seasonal_mean
    daily_agg['consultation_zscore'] = (daily_agg['consultation_count'] - seasonal_mean) / (seasonal_std + 1e-8)
    
    # High consultation day indicator (above 90th percentile)
    consultation_90 = daily_agg.groupby(['admin1', 'category_canonical_disease_imc'])['consultation_count'].transform(lambda x: x.quantile(0.90))
    daily_agg['is_high_consultation_day'] = (daily_agg['consultation_count'] > consultation_90).astype(int)
    
    logger.info(f"✓ Created aggregated dataset: {len(daily_agg):,} daily records")
    logger.info(f"✓ Morbidity categories: {daily_agg['category_canonical_disease_imc'].nunique()}")
    logger.info(f"✓ Admin1 regions: {daily_agg['admin1'].nunique()}")
    
    return daily_agg

def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical features for machine learning
    
    Args:
        df: Input dataframe with categorical features
        
    Returns:
        Tuple of (encoded dataframe, encoders dictionary)
    """
    logger.info("Encoding categorical features...")
    
    df = df.copy()
    encoders = {}
    
    # Identify categorical columns to encode
    categorical_cols = [
        'admin1', 'category_canonical_disease_imc', 'season',
        'age_group_clean', 'sex_clean', 'facility_type_clean'
    ]
    
    # Preserve these identifier columns for evaluation - don't drop them
    preserve_cols = ['admin1', 'category_canonical_disease_imc']
    
    # Filter to columns that exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    for col in categorical_cols:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Use label encoding for tree-based models
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
            # Only drop the original categorical column if it's not needed for evaluation
            if col not in preserve_cols:
                df = df.drop(columns=[col])
            
            logger.info(f"✓ Encoded {col}: {len(le.classes_)} unique values")
    
    logger.info(f"✓ Encoded {len(categorical_cols)} categorical features")
    
    return df, encoders

def create_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """
    Main feature engineering pipeline
    
    Args:
        df: Input merged health-climate dataframe
        config: Configuration dictionary with feature parameters
        
    Returns:
        Dataframe with all engineered features ready for modeling
    """
    logger.info("🔧 Starting comprehensive feature engineering...")
    
    if config is None:
        # Load default config if not provided
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from data_processing import load_config
        config = load_config()
    
    # Step 1: Create temporal features
    df = create_temporal_features(df)
    
    # Step 2: Create climate lag and extreme weather features
    df = create_climate_lag_features(df, config)
    
    # Step 3: Create morbidity classification features
    df = create_morbidity_features(df)
    
    # Step 4: Create demographic features
    df = create_demographic_features(df)
    
    # Step 5: Create consultation aggregates (this creates the final modeling dataset)
    df_aggregated = create_consultation_aggregates(df)
    
    # Step 6: Encode categorical features
    df_final, encoders = encode_categorical_features(df_aggregated)
    
    # Remove rows with insufficient data (e.g., missing lag features at the beginning)
    initial_len = len(df_final)
    # Keep rows with sufficient lag features available - use max lag from config
    max_temp_lag = max(config.get('features', {}).get('temperature_lags', [7]))
    max_precip_lag = max(config.get('features', {}).get('precipitation_lags', [7]))
    max_lag = max(max_temp_lag, max_precip_lag, 7)  # Ensure at least 7 days
    
    # Dynamic column selection based on available lags
    required_cols = []
    if f'consultation_lag_{max_lag}' in df_final.columns:
        required_cols.append(f'consultation_lag_{max_lag}')
    if f'temp_mean_lag_{max_lag}' in df_final.columns:
        required_cols.append(f'temp_mean_lag_{max_lag}')
    
    # If specific max lag columns don't exist, use any available lag columns
    if not required_cols:
        consultation_lags = [col for col in df_final.columns if col.startswith('consultation_lag_')]
        temp_lags = [col for col in df_final.columns if col.startswith('temp_mean_lag_')]
        if consultation_lags:
            required_cols.append(consultation_lags[-1])  # Use highest lag available
        if temp_lags:
            required_cols.append(temp_lags[-1])  # Use highest lag available
    
    if required_cols:
        df_final = df_final.dropna(subset=required_cols)
    
    final_len = len(df_final)
    
    logger.info(f"✓ Removed {initial_len - final_len:,} rows with insufficient lag data")
    
    # Store encoders for later use
    df_final.attrs['encoders'] = encoders
    
    # Save feature pipeline metadata for results viewer
    import joblib
    import os
    from pathlib import Path
    
    # Create cache directory
    cache_dir = Path('data/processed')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature metadata
    feature_metadata = {
        'feature_columns': list(df_final.columns),
        'feature_dtypes': df_final.dtypes.to_dict(),
        'encoders': encoders,
        'target_column': 'consultation_count',
        'identifier_columns': ['date', 'admin1', 'category_canonical_disease_imc'],
        'feature_groups': get_feature_groups(),
        'dataset_shape': df_final.shape
    }
    
    joblib.dump(feature_metadata, cache_dir / 'feature_pipeline_metadata.joblib')
    logger.info(f"✓ Feature pipeline metadata saved to {cache_dir / 'feature_pipeline_metadata.joblib'}")
    
    # Log final feature summary
    feature_cols = [col for col in df_final.columns if col not in ['date', 'admin1', 'category_canonical_disease_imc']]
    numeric_features = len([col for col in feature_cols if df_final[col].dtype in ['int64', 'float64']])
    categorical_features = len([col for col in feature_cols if col.endswith('_encoded')])
    
    logger.info("✅ Feature engineering completed!")
    logger.info(f"✅ Final dataset shape: {df_final.shape}")
    logger.info(f"✅ Total features: {len(feature_cols)} ({numeric_features} numeric, {categorical_features} categorical)")
    logger.info(f"✅ Date range: {df_final['date'].min()} to {df_final['date'].max()}")
    logger.info(f"✅ Target variable (consultation_count) range: {df_final['consultation_count'].min()} to {df_final['consultation_count'].max()}")
    
    return df_final

def get_feature_groups() -> Dict[str, List[str]]:
    """
    Define feature groups for model interpretation and feature selection
    
    Returns:
        Dictionary mapping feature group names to column patterns
    """
    return {
        'climate_current': ['temp_max', 'temp_min', 'temp_mean', 'temp_range', 'precipitation'],
        'climate_lags': ['_lag_', '_roll_'],
        'climate_anomalies': ['_anomaly', '_zscore'],
        'climate_extremes': ['is_hot', 'is_heavy', 'is_drought', 'is_heatwave'],
        'temporal_linear': ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter'],
        'temporal_cyclical': ['_sin', '_cos'],
        'temporal_indicators': ['is_weekend', 'is_month_', 'season'],
        'demographics': ['is_female', 'is_male', 'is_young_child', 'is_elderly', 'is_vulnerable_age'],
        'morbidity': ['category_canonical_disease_imc', 'is_respiratory', 'is_diarrheal', 'has_icd11'],
        'facility': ['facility_type', 'is_primary_care', 'is_secondary_care', 'is_emergency_care'],
        'consultation_history': ['consultation_lag_', 'consultation_roll_', 'consultation_std_'],
        'consultation_anomalies': ['consultation_seasonal_anomaly', 'consultation_zscore', 'is_high_consultation_day'],
        'geographic': ['admin1_encoded', 'admin0_encoded']
    }

def select_features_by_group(df: pd.DataFrame, feature_groups: List[str]) -> List[str]:
    """
    Select features based on feature group names
    
    Args:
        df: Dataframe with features
        feature_groups: List of feature group names to include
        
    Returns:
        List of column names matching the selected feature groups
    """
    group_patterns = get_feature_groups()
    selected_features = []
    
    for group in feature_groups:
        if group in group_patterns:
            patterns = group_patterns[group]
            for pattern in patterns:
                matching_cols = [col for col in df.columns if pattern in col]
                selected_features.extend(matching_cols)
    
    # Remove duplicates and ensure columns exist
    selected_features = list(set(selected_features))
    selected_features = [col for col in selected_features if col in df.columns]
    
    return selected_features