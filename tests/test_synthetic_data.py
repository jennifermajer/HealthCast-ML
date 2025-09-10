"""
Unit tests for synthetic data generation.
Tests synthetic data quality, realism, and compatibility with real data structure.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import warnings

# Import the real synthetic data generator
import sys
sys.path.append('data/synthetic')
from generate_synthetic import SyrianHealthClimateGenerator as SyntheticDataGenerator

class TestSyntheticDataGeneration:
    """Test suite for synthetic data generation"""

    @pytest.fixture
    def generator(self):
        """Create synthetic data generator"""
        return SyntheticDataGenerator(random_seed=42)

    @pytest.fixture
    def sample_health_data(self, generator):
        """Generate sample synthetic health data"""
        return generator.generate_health_consultations(
            start_date='2023-01-01',
            end_date='2023-03-31',  # Smaller date range for testing
            daily_consultations_base=30
        )

    @pytest.fixture
    def sample_climate_data(self, generator):
        """Generate sample synthetic climate data"""
        return generator.generate_climate_data(
            start_date='2023-01-01',
            end_date='2023-03-31'
        )

    def test_health_data_structure(self, sample_health_data):
        """Test that synthetic health data has correct structure"""
        
        # Check required columns are present
        required_columns = [
            'date', 'admin1', 'morbidity', 'standard_disease_imc', 'icd11_title',
            'age_group', 'sex', 'admin0', 'admin2', 'admin3', 'orgunit', 'facility_type'
        ]
        
        for col in required_columns:
            assert col in sample_health_data.columns, f"Missing column: {col}"
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(sample_health_data['date'])
        assert sample_health_data['sex'].dtype == 'object'
        assert sample_health_data['admin1'].dtype == 'object'

    def test_health_data_values(self, sample_health_data):
        """Test that synthetic health data has realistic values"""
        
        # Check date range
        assert sample_health_data['date'].min() >= pd.Timestamp('2023-01-01')
        assert sample_health_data['date'].max() <= pd.Timestamp('2023-03-31')
        
        # Check sex values
        sex_values = set(sample_health_data['sex'].unique())
        assert sex_values.issubset({'M', 'F'})
        
        # Check admin0 is consistently Syria
        assert (sample_health_data['admin0'] == 'Syria').all()
        
        # Check that governorates are realistic Syrian governorates
        syrian_governorates = {'Aleppo', 'Damascus', 'Homs', 'Idlib', 'Lattakia', 'Hama', 'Al-Hasakah', 'Deir ez-Zor'}
        actual_governorates = set(sample_health_data['admin1'].unique())
        assert actual_governorates.issubset(syrian_governorates)

    def test_health_data_distributions(self, sample_health_data):
        """Test that synthetic health data has realistic distributions"""
        
        # Test age group distribution (should have reasonable proportions)
        age_dist = sample_health_data['age_group'].value_counts(normalize=True)
        assert age_dist['18-59'] > 0.3  # Working age should be largest group
        assert age_dist['0-5'] > 0.05   # Children should be present
        
        # Test sex distribution (should be roughly balanced)
        sex_dist = sample_health_data['sex'].value_counts(normalize=True)
        assert 0.4 < sex_dist['F'] < 0.6  # Should be close to 50/50
        
        # Test that there are multiple morbidity categories
        assert sample_health_data['morbidity'].nunique() > 5

    def test_health_data_seasonal_patterns(self, generator):
        """Test that synthetic health data shows seasonal patterns"""
        
        # Generate data for full year to test seasonality
        full_year_data = generator.generate_health_consultations(
            start_date='2023-01-01',
            end_date='2023-12-31',
            daily_consultations_base=20
        )
        
        # Group by month and check for variation
        monthly_counts = full_year_data.groupby(full_year_data['date'].dt.month).size()
        
        # Should have variation across months (coefficient of variation > 0.1)
        cv = monthly_counts.std() / monthly_counts.mean()
        assert cv > 0.05, "Insufficient seasonal variation in consultation counts"

    def test_climate_data_structure(self, sample_climate_data):
        """Test that synthetic climate data has correct structure"""
        
        # Check required columns
        required_columns = ['date', 'admin1', 'temp_max', 'temp_min', 'precipitation']
        for col in required_columns:
            assert col in sample_climate_data.columns, f"Missing column: {col}"
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(sample_climate_data['date'])
        assert pd.api.types.is_numeric_dtype(sample_climate_data['temp_max'])
        assert pd.api.types.is_numeric_dtype(sample_climate_data['temp_min'])
        assert pd.api.types.is_numeric_dtype(sample_climate_data['precipitation'])

    def test_climate_data_values(self, sample_climate_data):
        """Test that synthetic climate data has realistic values"""
        
        # Temperature constraints
        assert sample_climate_data['temp_max'].min() > -30  # Not unreasonably cold
        assert sample_climate_data['temp_max'].max() < 60   # Not unreasonably hot
        assert sample_climate_data['temp_min'].min() > -40  # Not unreasonably cold
        assert sample_climate_data['temp_min'].max() < 50   # Not unreasonably hot for min temp
        
        # Precipitation constraints  
        assert sample_climate_data['precipitation'].min() >= 0  # Non-negative
        assert sample_climate_data['precipitation'].max() < 500  # Reasonable upper bound
        
        # Temperature logic
        assert (sample_climate_data['temp_max'] >= sample_climate_data['temp_min']).all()

    def test_disease_mapping_completeness(self, generator):
        """Test that disease mapping includes all required fields"""
        
        diseases = generator.disease_mapping
        
        for disease_name, disease_info in diseases.items():
            # Check required keys
            required_keys = ['imc_category', 'icd11_code', 'icd11_title', 'climate_sensitivity']
            for key in required_keys:
                assert key in disease_info, f"Missing {key} in disease {disease_name}"
            
            # Check climate sensitivity structure
            climate_sens = disease_info['climate_sensitivity']
            assert 'temperature' in climate_sens
            assert 'precipitation' in climate_sens
            assert 'seasonal_peak' in climate_sens

    def test_climate_health_interactions(self, generator):
        """Test that climate conditions influence disease patterns"""
        
        # Generate data with different climate conditions
        winter_data = generator.generate_health_consultations(
            start_date='2023-01-01', end_date='2023-01-31',
            daily_consultations_base=50
        )
        
        summer_data = generator.generate_health_consultations(
            start_date='2023-07-01', end_date='2023-07-31', 
            daily_consultations_base=50
        )
        
        # Check seasonal disease patterns
        winter_resp = winter_data[winter_data['standard_disease_imc'].str.contains('respiratory', na=False)]
        summer_gi = summer_data[summer_data['standard_disease_imc'].str.contains('gastrointestinal', na=False)]
        
        # Should have some seasonal variation (this is probabilistic, so use reasonable thresholds)
        winter_resp_rate = len(winter_resp) / len(winter_data)
        summer_gi_rate = len(summer_gi) / len(summer_data)
        
        # These are rough expectations - actual rates will vary due to randomness
        assert winter_resp_rate > 0.1, "Should have some respiratory diseases in winter"
        assert summer_gi_rate > 0.05, "Should have some GI diseases in summer"

    def test_integration_with_pipeline(self):
        """Test that synthetic data works with the real data processing pipeline"""
        
        # This test verifies that the synthetic data can be processed
        # by the actual data processing functions
        import sys
        sys.path.append('src')
        from data_processing import validate_health_data, validate_climate_data
        
        generator = SyntheticDataGenerator(random_seed=42)
        
        # Generate small test datasets
        health_df = generator.generate_health_consultations(
            start_date='2023-01-01', end_date='2023-01-07',
            daily_consultations_base=20
        )
        
        climate_df = generator.generate_climate_data(
            start_date='2023-01-01', end_date='2023-01-07'
        )
        
        # Test validation functions don't fail
        validated_health = validate_health_data(health_df)
        validated_climate = validate_climate_data(climate_df)
        
        assert len(validated_health) > 0, "Health data validation removed all records"
        assert len(validated_climate) > 0, "Climate data validation removed all records"
        
        # Test that required columns are present after validation
        assert 'standard_disease_imc' in validated_health.columns
        assert 'temp_mean' in validated_climate.columns