"""
Unit tests for data processing functions.
Tests data loading, validation, merging, and quality checks.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
import yaml
from unittest.mock import patch, MagicMock

# Import functions to test
import sys
sys.path.append('src')
from data_processing import (
    load_config, get_data_paths, validate_health_data, validate_climate_data,
    standardize_geographic_names, merge_health_climate, load_and_merge_data,
    get_data_summary
)

class TestDataProcessing:
    """Test suite for data processing functions"""

    @pytest.fixture
    def sample_health_data(self):
        """Create sample health consultation data for testing"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n_records = len(dates) * 10  # ~10 records per day
        
        # Create realistic health data
        data = {
            'date': np.random.choice(dates, n_records),
            'admin1': np.random.choice(['Aleppo', 'Damascus', 'Homs', 'Idlib', 'Lattakia'], n_records),
            'morbidity': np.random.choice([
                'Upper Respiratory Infection', 'Diarrhea', 'Hypertension', 
                'Diabetes', 'Pneumonia', 'Skin Infection'
            ], n_records),
            'standard_disease_imc': np.random.choice([
                'respiratory_infection', 'gastrointestinal_disease', 
                'cardiovascular_disease', 'endocrine_disorder',
                'respiratory_disease', 'dermatological_condition'
            ], n_records),
            'icd11_title': np.random.choice([
                'Acute upper respiratory infections', 'Diarrhoea or gastroenteritis',
                'Essential hypertension', 'Type 2 diabetes mellitus',
                'Pneumonia', 'Cellulitis'
            ], n_records),
            'age_group': np.random.choice(['0-5', '6-17', '18-59', '60+'], n_records),
            'sex': np.random.choice(['M', 'F'], n_records),
            'admin0': 'Syria',
            'admin2': np.random.choice(['District1', 'District2', 'District3'], n_records),
            'orgunit': np.random.choice(['Hospital A', 'Clinic B', 'Health Center C'], n_records),
            'facility_type': np.random.choice(['Hospital', 'Clinic', 'Health Center'], n_records)
        }
        
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_climate_data(self):
        """Create sample climate data for testing"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        regions = ['Aleppo', 'Damascus', 'Homs', 'Idlib', 'Lattakia']
        
        data = []
        for date in dates:
            for region in regions:
                # Create realistic Syrian climate patterns
                month = date.month
                base_temp = 15 + 10 * np.sin(2 * np.pi * (month - 1) / 12)  # Seasonal variation
                
                data.append({
                    'date': date,
                    'admin1': region,
                    'temp_max': base_temp + np.random.normal(10, 3),
                    'temp_min': base_temp + np.random.normal(-5, 2),
                    'precipitation': np.random.exponential(2) if np.random.random() < 0.3 else 0
                })
        
        df = pd.DataFrame(data)
        df['temp_mean'] = (df['temp_max'] + df['temp_min']) / 2
        df['temp_range'] = df['temp_max'] - df['temp_min']
        
        return df

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return {
            'data': {
                'use_synthetic': True,
                'public': {
                    'health_data_path': 'test_health.csv',
                    'climate_data_path': 'test_climate.csv'
                },
                'private': {
                    'health_data_path': 'private_health.csv',
                    'climate_data_path': 'private_climate.csv'
                },
                'processed_data_dir': 'test_processed'
            },
            'features': {
                'temperature_lags': [1, 3, 7],
                'precipitation_lags': [1, 3, 7]
            }
        }

    @pytest.fixture
    def temp_config_file(self, sample_config):
        """Create temporary config file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            return f.name

    def test_validate_health_data_valid(self, sample_health_data):
        """Test health data validation with valid data"""
        validated_df = validate_health_data(sample_health_data)
        
        # Check that validation passes
        assert len(validated_df) > 0
        assert 'date' in validated_df.columns
        assert validated_df['date'].dtype == 'datetime64[ns]'
        assert 'admin1' in validated_df.columns
        assert 'sex' in validated_df.columns
        assert validated_df['sex'].isin(['M', 'F']).all()

    def test_validate_health_data_missing_columns(self):
        """Test health data validation with missing required columns"""
        invalid_data = pd.DataFrame({
            'date': ['2023-01-01'],
            'admin1': ['Aleppo']
            # Missing required columns: morbidity, standard_disease_imc, etc.
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_health_data(invalid_data)

    def test_validate_health_data_invalid_dates(self, sample_health_data):
        """Test health data validation with invalid dates"""
        # Introduce some invalid dates
        sample_health_data.loc[0:5, 'date'] = 'invalid_date'
        
        validated_df = validate_health_data(sample_health_data)
        
        # Should remove invalid dates
        assert len(validated_df) == len(sample_health_data) - 6

    def test_validate_climate_data_valid(self, sample_climate_data):
        """Test climate data validation with valid data"""
        validated_df = validate_climate_data(sample_climate_data)
        
        assert len(validated_df) > 0
        assert 'temp_mean' in validated_df.columns
        assert 'temp_range' in validated_df.columns
        assert (validated_df['temp_max'] >= validated_df['temp_min']).all()
        assert (validated_df['precipitation'] >= 0).all()

    def test_validate_climate_data_extreme_values(self, sample_climate_data):
        """Test climate data validation with extreme values"""
        # Introduce extreme values
        sample_climate_data.loc[0, 'temp_max'] = 100  # Too hot
        sample_climate_data.loc[1, 'temp_min'] = -50  # Too cold
        sample_climate_data.loc[2, 'precipitation'] = -5  # Negative precipitation
        
        validated_df = validate_climate_data(sample_climate_data)
        
        # Should remove rows with extreme values
        assert len(validated_df) < len(sample_climate_data)

    def test_standardize_geographic_names(self, sample_health_data, sample_climate_data):
        """Test geographic name standardization"""
        # Introduce some inconsistent naming
        sample_health_data.loc[sample_health_data['admin1'] == 'Aleppo', 'admin1'] = 'aleppo'
        sample_climate_data.loc[sample_climate_data['admin1'] == 'Damascus', 'admin1'] = 'DAMASCUS'
        
        health_std, climate_std = standardize_geographic_names(sample_health_data, sample_climate_data)
        
        # Check that standardization worked
        common_regions = set(health_std['admin1'].unique()) & set(climate_std['admin1'].unique())
        assert len(common_regions) > 0
        
        # Should have title case
        assert 'Aleppo' in health_std['admin1'].unique()
        assert 'Damascus' in climate_std['admin1'].unique()

    def test_merge_health_climate(self, sample_health_data, sample_climate_data):
        """Test merging health and climate data"""
        merged_df = merge_health_climate(sample_health_data, sample_climate_data)
        
        # Check merge results
        assert len(merged_df) > 0
        assert 'date' in merged_df.columns
        assert 'admin1' in merged_df.columns
        assert 'morbidity' in merged_df.columns
        assert 'temp_max' in merged_df.columns
        assert 'precipitation' in merged_df.columns

    def test_get_data_paths_synthetic(self, sample_config):
        """Test getting data paths for synthetic data"""
        sample_config['data']['use_synthetic'] = True
        paths = get_data_paths(sample_config)
        
        assert paths['health'] == 'test_health.csv'
        assert paths['climate'] == 'test_climate.csv'

    def test_get_data_paths_private(self, sample_config):
        """Test getting data paths for private data"""
        sample_config['data']['use_synthetic'] = False
        paths = get_data_paths(sample_config)
        
        assert paths['health'] == 'private_health.csv'
        assert paths['climate'] == 'private_climate.csv'

    @patch.dict(os.environ, {'USE_SYNTHETIC': 'true'})
    def test_load_config_with_environment(self, temp_config_file):
        """Test loading config with environment variable override"""
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = """
data:
  use_synthetic: false
  public:
    health_data_path: test_health.csv
"""
            with patch('yaml.safe_load') as mock_yaml:
                mock_yaml.return_value = {
                    'data': {
                        'use_synthetic': False,
                        'public': {'health_data_path': 'test_health.csv'}
                    }
                }
                
                config = load_config()
                assert config['data']['use_synthetic'] == True  # Should be overridden by env var

    def test_get_data_summary(self, sample_health_data, sample_climate_data):
        """Test data summary generation"""
        merged_df = merge_health_climate(sample_health_data, sample_climate_data)
        summary = get_data_summary(merged_df)
        
        # Check summary structure
        assert 'total_records' in summary
        assert 'date_range' in summary
        assert 'geographic_coverage' in summary
        assert 'health_data' in summary
        assert 'climate_data' in summary
        
        # Check specific values
        assert summary['total_records'] > 0
        assert 'admin1_regions' in summary['geographic_coverage']
        assert summary['geographic_coverage']['admin1_regions'] > 0

    @pytest.mark.integration
    def test_load_and_merge_data_integration(self, sample_health_data, sample_climate_data, sample_config):
        """Integration test for complete data loading pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary data files
            health_path = os.path.join(temp_dir, 'test_health.csv')
            climate_path = os.path.join(temp_dir, 'test_climate.csv')
            config_path = os.path.join(temp_dir, 'config.yaml')
            
            # Save test data
            sample_health_data.to_csv(health_path, index=False)
            sample_climate_data.to_csv(climate_path, index=False)
            
            # Update config paths
            sample_config['data']['public']['health_data_path'] = health_path
            sample_config['data']['public']['climate_data_path'] = climate_path
            sample_config['data']['processed_data_dir'] = temp_dir
            
            # Save config
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Mock the config loading to use our test config
            with patch('src.data_processing.load_config', return_value=sample_config):
                result_df = load_and_merge_data(config_path)
            
            # Verify results
            assert len(result_df) > 0
            assert 'consultation_count' in result_df.columns or 'morbidity' in result_df.columns
            assert 'temp_max' in result_df.columns
            assert 'precipitation' in result_df.columns

class TestDataQuality:
    """Test suite for data quality checks"""

    def test_health_data_completeness(self, sample_health_data):
        """Test completeness of health data"""
        # Check required columns are present
        required_cols = ['date', 'admin1', 'morbidity', 'standard_disease_imc', 'age_group', 'sex']
        missing_cols = [col for col in required_cols if col not in sample_health_data.columns]
        assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"

    def test_climate_data_ranges(self, sample_climate_data):
        """Test that climate data is within reasonable ranges"""
        # Temperature checks for Syria
        assert sample_climate_data['temp_max'].min() >= -20, "Temperature too low"
        assert sample_climate_data['temp_max'].max() <= 60, "Temperature too high"
        assert sample_climate_data['precipitation'].min() >= 0, "Negative precipitation"

    def test_date_consistency(self, sample_health_data, sample_climate_data):
        """Test date consistency between datasets"""
        health_dates = set(sample_health_data['date'].dt.date)
        climate_dates = set(sample_climate_data['date'].dt.date)
        
        # Should have some overlap
        overlap = health_dates.intersection(climate_dates)
        assert len(overlap) > 0, "No date overlap between datasets"

class TestErrorHandling:
    """Test suite for error handling and edge cases"""

    def test_empty_dataframe_validation(self):
        """Test validation with empty dataframes"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            validate_health_data(empty_df)

    def test_all_missing_values_column(self, sample_health_data):
        """Test handling of columns with all missing values"""
        sample_health_data['all_missing'] = np.nan
        
        # Should not raise error, just log warning
        result = validate_health_data(sample_health_data)
        assert 'all_missing' in result.columns

    def test_invalid_sex_values(self, sample_health_data):
        """Test handling of invalid sex values"""
        sample_health_data.loc[0:5, 'sex'] = 'X'  # Invalid sex
        
        result = validate_health_data(sample_health_data)
        # Should remove rows with invalid sex
        assert len(result) == len(sample_health_data) - 6

    def test_duplicate_date_admin1_combinations(self, sample_climate_data):
        """Test handling of duplicate date-admin1 combinations"""
        # Create duplicates
        duplicate_row = sample_climate_data.iloc[0:1].copy()
        climate_with_dupes = pd.concat([sample_climate_data, duplicate_row])
        
        # Should handle duplicates gracefully
        result = validate_climate_data(climate_with_dupes)
        assert len(result) > 0

class TestConfigurationHandling:
    """Test suite for configuration handling"""

    def test_missing_config_file(self):
        """Test handling of missing configuration file"""
        with pytest.raises(FileNotFoundError):
            with patch('builtins.open', side_effect=FileNotFoundError):
                load_config()

    def test_invalid_yaml_config(self):
        """Test handling of invalid YAML configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            
            with pytest.raises(Exception):  # Could be various YAML errors
                with patch('builtins.open', return_value=f):
                    load_config()

    def test_missing_config_keys(self):
        """Test handling of missing configuration keys"""
        incomplete_config = {'data': {'use_synthetic': True}}
        
        # Should not raise error, but return incomplete config
        paths = get_data_paths(incomplete_config)
        # Should handle missing keys gracefully or use defaults

if __name__ == '__main__':
    pytest.main([__file__])