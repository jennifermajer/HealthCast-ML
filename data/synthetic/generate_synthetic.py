#!/usr/bin/env python3
"""
Generate synthetic health consultation and climate data for Climate-Health ML project.

This script creates realistic synthetic datasets that preserve the statistical
properties and patterns needed for climate-health analysis while protecting
sensitive patient information.

Key Features:
- Dual disease mapping: Local → IMC canonical → ICD-11
- Climate-health interactions with lagged effects
- Generic geographic and demographic patterns
- Seasonal patterns and extreme weather events
"""

import pandas as pd
import numpy as np
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthClimateGenerator:
    """Generate realistic synthetic health and climate data"""
    
    def __init__(self, random_seed: int = 42):
        """Initialize generator with random seed for reproducibility"""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Generic regions with realistic characteristics
        self.regions = {
            'Central Region': {
                'population_weight': 0.25, 'climate_zone': 'continental',
                'base_temp': 18, 'temp_range': 20, 'precip_rate': 0.30,
                'urban': 0.8, 'health_burden': 0.7
            },
            'Capital Region': {
                'population_weight': 0.20, 'climate_zone': 'mountain',
                'base_temp': 17, 'temp_range': 18, 'precip_rate': 0.25,
                'urban': 0.9, 'health_burden': 0.3
            },
            'Western Region': {
                'population_weight': 0.15, 'climate_zone': 'continental',
                'base_temp': 16, 'temp_range': 19, 'precip_rate': 0.35,
                'urban': 0.6, 'health_burden': 0.6
            },
            'Northern Region': {
                'population_weight': 0.12, 'climate_zone': 'mountain',
                'base_temp': 17, 'temp_range': 18, 'precip_rate': 0.40,
                'urban': 0.4, 'health_burden': 0.9
            },
            'Coastal Region': {
                'population_weight': 0.08, 'climate_zone': 'coastal',
                'base_temp': 20, 'temp_range': 12, 'precip_rate': 0.45,
                'urban': 0.7, 'health_burden': 0.2
            },
            'Southern Region': {
                'population_weight': 0.08, 'climate_zone': 'continental',
                'base_temp': 18, 'temp_range': 19, 'precip_rate': 0.30,
                'urban': 0.5, 'health_burden': 0.5
            },
            'Eastern Region': {
                'population_weight': 0.06, 'climate_zone': 'continental',
                'base_temp': 19, 'temp_range': 22, 'precip_rate': 0.25,
                'urban': 0.3, 'health_burden': 0.4
            },
            'Desert Region': {
                'population_weight': 0.06, 'climate_zone': 'desert',
                'base_temp': 21, 'temp_range': 20, 'precip_rate': 0.20,
                'urban': 0.4, 'health_burden': 0.8
            }
        }
        
        # Disease mapping: Country-specific → IMC canonical → ICD-11
        self.disease_mapping = self._create_disease_mapping()
        
        # Demographic patterns
        self.age_groups = {
            '0-5': {'weight': 0.15, 'vulnerability': 1.3, 'seasonal_variation': 1.4},
            '6-17': {'weight': 0.20, 'vulnerability': 0.9, 'seasonal_variation': 1.2},
            '18-59': {'weight': 0.50, 'vulnerability': 0.8, 'seasonal_variation': 0.9},
            '60+': {'weight': 0.15, 'vulnerability': 1.4, 'seasonal_variation': 1.3}
        }
        
        # Health facility types
        self.facility_types = {
            'Hospital': {'weight': 0.30, 'severe_cases': 1.5},
            'Primary Health Center': {'weight': 0.35, 'severe_cases': 0.8},
            'Clinic': {'weight': 0.25, 'severe_cases': 0.7},
            'Mobile Clinic': {'weight': 0.10, 'severe_cases': 0.9}
        }
    
    def _create_disease_mapping(self) -> Dict:
        """Create comprehensive disease mapping system"""
        
        # Climate-sensitive diseases with their characteristics
        diseases = {
            # Respiratory diseases (climate-sensitive)
            'Upper Respiratory Infection': {
                'imc_category': 'respiratory_infection',
                'icd11_code': 'CA40.Z',
                'icd11_title': 'Acute upper respiratory infections, unspecified',
                'climate_sensitivity': {
                    'temperature': 'cold_increases', 'precipitation': 'slight_increase',
                    'seasonal_peak': 'winter', 'lag_days': 3
                },
                'age_preference': {'0-5': 1.8, '6-17': 1.4, '18-59': 0.9, '60+': 1.2},
                'baseline_rate': 0.18  # 18% of consultations
            },
            'Pneumonia': {
                'imc_category': 'respiratory_disease',
                'icd11_code': 'CA40.0',
                'icd11_title': 'Pneumonia, organism unspecified',
                'climate_sensitivity': {
                    'temperature': 'cold_increases', 'precipitation': 'increases',
                    'seasonal_peak': 'winter', 'lag_days': 7
                },
                'age_preference': {'0-5': 2.0, '6-17': 0.8, '18-59': 0.7, '60+': 2.2},
                'baseline_rate': 0.08
            },
            'Asthma': {
                'imc_category': 'respiratory_disease',
                'icd11_code': 'CA23',
                'icd11_title': 'Asthma',
                'climate_sensitivity': {
                    'temperature': 'extremes_increase', 'precipitation': 'dust_storms',
                    'seasonal_peak': 'spring_summer', 'lag_days': 1
                },
                'age_preference': {'0-5': 1.5, '6-17': 1.3, '18-59': 0.9, '60+': 1.1},
                'baseline_rate': 0.06
            },
            'Chronic Obstructive Pulmonary Disease': {
                'imc_category': 'respiratory_disease',
                'icd11_code': 'CA22',
                'icd11_title': 'Chronic obstructive pulmonary disease',
                'climate_sensitivity': {
                    'temperature': 'heat_increases', 'precipitation': 'dust_increases',
                    'seasonal_peak': 'summer', 'lag_days': 1
                },
                'age_preference': {'0-5': 0.1, '6-17': 0.2, '18-59': 1.0, '60+': 3.5},
                'baseline_rate': 0.05
            },
            
            # Gastrointestinal diseases (climate-sensitive)
            'Acute Diarrhea': {
                'imc_category': 'gastrointestinal_disease',
                'icd11_code': 'ME05.0',
                'icd11_title': 'Acute diarrhoea',
                'climate_sensitivity': {
                    'temperature': 'heat_increases', 'precipitation': 'heavy_rain_increases',
                    'seasonal_peak': 'summer', 'lag_days': 2
                },
                'age_preference': {'0-5': 2.5, '6-17': 1.2, '18-59': 0.8, '60+': 1.3},
                'baseline_rate': 0.12
            },
            'Gastroenteritis': {
                'imc_category': 'gastrointestinal_disease',
                'icd11_code': 'ME05',
                'icd11_title': 'Gastroenteritis or colitis of infectious origin',
                'climate_sensitivity': {
                    'temperature': 'heat_increases', 'precipitation': 'flooding_increases',
                    'seasonal_peak': 'summer', 'lag_days': 3
                },
                'age_preference': {'0-5': 1.8, '6-17': 1.1, '18-59': 0.9, '60+': 1.2},
                'baseline_rate': 0.08
            },
            
            # Cardiovascular diseases (heat-sensitive)
            'Hypertension': {
                'imc_category': 'cardiovascular_disease',
                'icd11_code': 'BA00',
                'icd11_title': 'Essential hypertension',
                'climate_sensitivity': {
                    'temperature': 'heat_increases', 'precipitation': 'neutral',
                    'seasonal_peak': 'summer', 'lag_days': 1
                },
                'age_preference': {'0-5': 0.1, '6-17': 0.3, '18-59': 1.2, '60+': 2.8},
                'baseline_rate': 0.10
            },
            
            # Mental health (climate affects stress)
            'Anxiety Disorder': {
                'imc_category': 'mental_health_disorder',
                'icd11_code': '6B00',
                'icd11_title': 'Anxiety disorders',
                'climate_sensitivity': {
                    'temperature': 'extremes_increase', 'precipitation': 'neutral',
                    'seasonal_peak': 'all_seasons', 'lag_days': 5
                },
                'age_preference': {'0-5': 0.3, '6-17': 0.8, '18-59': 1.3, '60+': 1.0},
                'baseline_rate': 0.07
            },
            'Depression': {
                'imc_category': 'mental_health_disorder',
                'icd11_code': '6A70',
                'icd11_title': 'Single episode depressive disorder',
                'climate_sensitivity': {
                    'temperature': 'cold_increases', 'precipitation': 'neutral',
                    'seasonal_peak': 'winter', 'lag_days': 7
                },
                'age_preference': {'0-5': 0.2, '6-17': 0.6, '18-59': 1.2, '60+': 1.1},
                'baseline_rate': 0.05
            },
            
            # Other conditions
            'Type 2 Diabetes': {
                'imc_category': 'endocrine_disorder',
                'icd11_code': '5A11',
                'icd11_title': 'Type 2 diabetes mellitus',
                'climate_sensitivity': {
                    'temperature': 'neutral', 'precipitation': 'neutral',
                    'seasonal_peak': 'none', 'lag_days': 0
                },
                'age_preference': {'0-5': 0.1, '6-17': 0.4, '18-59': 1.0, '60+': 3.0},
                'baseline_rate': 0.08
            },
            'Skin Infection': {
                'imc_category': 'dermatological_condition',
                'icd11_code': '1F03',
                'icd11_title': 'Cellulitis',
                'climate_sensitivity': {
                    'temperature': 'heat_increases', 'precipitation': 'humidity_increases',
                    'seasonal_peak': 'summer', 'lag_days': 2
                },
                'age_preference': {'0-5': 1.4, '6-17': 1.2, '18-59': 0.9, '60+': 1.0},
                'baseline_rate': 0.06
            },
            'Urinary Tract Infection': {
                'imc_category': 'genitourinary_disease',
                'icd11_code': 'GC08',
                'icd11_title': 'Urinary tract infection, site not specified',
                'climate_sensitivity': {
                    'temperature': 'heat_increases', 'precipitation': 'neutral',
                    'seasonal_peak': 'summer', 'lag_days': 1
                },
                'age_preference': {'0-5': 1.2, '6-17': 0.7, '18-59': 1.1, '60+': 1.5},
                'baseline_rate': 0.05
            }
        }
        
        return diseases
    
    def generate_climate_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic climate data for geographic regions"""
        logger.info(f"Generating climate data from {start_date} to {end_date}")
        
        dates = pd.date_range(start_date, end_date, freq='D')
        data = []
        
        for date in dates:
            day_of_year = date.dayofyear
            
            for region_name, region_params in self.regions.items():
                # Seasonal temperature variation
                seasonal_temp = region_params['base_temp'] + \
                    region_params['temp_range'] * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                
                # Daily temperature variation with realistic noise
                temp_max = seasonal_temp + np.random.normal(8, 3)
                temp_min = seasonal_temp + np.random.normal(-8, 2.5)
                
                # Ensure temp_max > temp_min
                if temp_max <= temp_min:
                    temp_max = temp_min + np.random.uniform(2, 8)
                
                # Climate zone adjustments
                if region_params['climate_zone'] == 'desert':
                    temp_max += np.random.normal(3, 2)  # Hotter days
                    temp_min -= np.random.normal(2, 1)  # Cooler nights
                elif region_params['climate_zone'] == 'coastal':
                    temp_range = temp_max - temp_min
                    temp_max = temp_min + temp_range * 0.7  # More moderate
                
                # Precipitation patterns
                # Winter peak (December-February)
                winter_factor = 1 + 0.6 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
                precip_prob = region_params['precip_rate'] * winter_factor
                
                # Add random weather events
                if np.random.random() < 0.02:  # 2% chance of weather system
                    precip_prob *= 2
                
                if np.random.random() < precip_prob:
                    # Exponential distribution for realistic rain amounts
                    precipitation = np.random.exponential(5)
                    # Occasional heavy rain events
                    if np.random.random() < 0.1:
                        precipitation *= np.random.uniform(2, 5)
                else:
                    precipitation = 0
                
                data.append({
                    'date': date,
                    'admin1': region_name,
                    'temp_max': round(temp_max, 1),
                    'temp_min': round(temp_min, 1),
                    'precipitation': round(precipitation, 1)
                })
        
        climate_df = pd.DataFrame(data)
        logger.info(f"Generated {len(climate_df)} climate observations")
        return climate_df
    
    def generate_health_consultations(self, start_date: str, end_date: str, 
                                    daily_consultations_base: int = 60,
                                    climate_df: pd.DataFrame = None) -> pd.DataFrame:
        """Generate realistic health consultation data with climate interactions"""
        logger.info(f"Generating health consultations from {start_date} to {end_date}")
        
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Create climate lookup if provided
        climate_lookup = {}
        if climate_df is not None:
            for _, row in climate_df.iterrows():
                climate_lookup[(row['date'], row['admin1'])] = {
                    'temp_max': row['temp_max'],
                    'temp_min': row['temp_min'],
                    'precipitation': row['precipitation']
                }
        
        data = []
        
        for date in dates:
            # Seasonal variation in overall consultation numbers
            seasonal_factor = 1 + 0.25 * np.sin(2 * np.pi * (date.dayofyear - 90) / 365)
            
            # Weekend effect
            if date.weekday() >= 5:  # Weekend
                weekend_factor = 0.7
            else:
                weekend_factor = 1.0
            
            # Conflict effect (higher consultations in conflict areas)
            # Ramadan effect (reduced consultations during fasting hours)
            ramadan_factor = 1.0  # Simplified - could add Islamic calendar
            
            for region_name, region_params in self.regions.items():
                # Calculate daily consultations for this region
                base_consultations = int(
                    daily_consultations_base * 
                    region_params['population_weight'] * 
                    seasonal_factor * 
                    weekend_factor * 
                    ramadan_factor *
                    (1 + region_params['health_burden'] * 0.3)  # Higher health burden increases needs
                )
                
                # Get climate conditions for this date/location
                climate_conditions = climate_lookup.get((date, region_name), {
                    'temp_max': 20, 'temp_min': 10, 'precipitation': 0
                })
                
                # Generate individual consultations
                for _ in range(base_consultations):
                    # Select disease based on climate conditions and seasonality
                    disease_name = self._select_disease_with_climate_influence(
                        date, climate_conditions, region_params
                    )
                    disease_info = self.disease_mapping[disease_name]
                    
                    # Select age group based on disease preferences
                    age_group = self._select_age_group(disease_info)
                    
                    # Select facility type
                    facility_type = np.random.choice(
                        list(self.facility_types.keys()),
                        p=[f['weight'] for f in self.facility_types.values()]
                    )
                    
                    # Generate generic facility name
                    facility_id = ord(region_name[0]) - ord('A') + 1  # Use first letter for consistency
                    facility_number = np.random.randint(1, 10)
                    facility_name = f"Health Facility {chr(64 + facility_id)}{facility_number}"
                    
                    # Create age group classifications to match events.csv structure
                    age_group1 = self._map_age_group1(age_group)
                    age_group2 = self._map_age_group2(age_group)
                    age_group3 = self._map_age_group3(age_group)
                    age_group_original = self._map_age_group_original(age_group)
                    
                    # Additional fields to match events.csv
                    admission_date = date if np.random.random() < 0.1 else ""  # 10% have admission dates
                    disability = np.random.choice([0, 1], p=[0.95, 0.05])  # 5% with disability
                    case_type = np.random.choice(['Non-trauma', 'Trauma'], p=[0.8, 0.2])
                    visit_number = np.random.choice(['1st Visit', '2nd Visit', '3rd Visit'], p=[0.8, 0.15, 0.05])
                    visit_type = np.random.choice(['New', 'Follow-up'], p=[0.7, 0.3])
                    
                    data.append({
                        'Organisation unit name': facility_name,
                        'Date of visit': date.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        'Admission Date': admission_date.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if admission_date else "",
                        'Age Group (0-59, 5-17, 18-49, 50 and Above) - SY': age_group_original,
                        'Age group 1 (0-5,6-17,18-59,60+) - SY': age_group1, 
                        'Age group 2 (0-11m, 1-4, 5-14,15-49,50-60,60+) - SY': age_group2,
                        'Age group 3 (< 5 , 5-14,15-18,19-49,50+) - SY': age_group3,
                        'Gender': 'Male' if np.random.choice(['M', 'F'], p=[0.48, 0.52]) == 'M' else 'Female',
                        'Morbidity Classification - SY': disease_name,
                        'Patient Presents With Disability (Y/N)': disability,
                        'Type of Case (Trauma/Non-trauma)': case_type,
                        'Visit Number - SYR': visit_number,
                        'Visit Type': visit_type,
                        # Add admin levels as requested
                        'admin0': 'Generic Country',
                        'admin1': region_name,
                        'admin2': f"{region_name.replace(' Region', '')} District {np.random.randint(1, 4)}",
                        'admin3': f"Subdistrict {chr(65 + np.random.randint(0, 5))}",
                        # Keep additional fields for compatibility
                        'canonical_disease_imc': disease_info['imc_category'],
                        'icd11_title': disease_info['icd11_title'],
                        'facility_type': facility_type
                    })
        
        health_df = pd.DataFrame(data)
        logger.info(f"Generated {len(health_df)} health consultations")
        return health_df
    
    def _select_disease_with_climate_influence(self, date: datetime, 
                                             climate_conditions: Dict,
                                             region_params: Dict) -> str:
        """Select disease based on climate conditions and seasonal patterns"""
        
        month = date.month
        temp_max = climate_conditions.get('temp_max', 20)
        precipitation = climate_conditions.get('precipitation', 0)
        
        # Calculate weights for each disease based on climate
        disease_weights = []
        disease_names = list(self.disease_mapping.keys())
        
        for disease_name in disease_names:
            disease_info = self.disease_mapping[disease_name]
            base_rate = disease_info['baseline_rate']
            climate_sens = disease_info['climate_sensitivity']
            
            weight = base_rate
            
            # Temperature effects
            if climate_sens['temperature'] == 'cold_increases':
                if temp_max < 10:  # Cold day
                    weight *= 1.5
                elif temp_max < 15:
                    weight *= 1.2
            elif climate_sens['temperature'] == 'heat_increases':
                if temp_max > 35:  # Very hot day
                    weight *= 1.6
                elif temp_max > 30:
                    weight *= 1.3
            elif climate_sens['temperature'] == 'extremes_increase':
                if temp_max > 35 or temp_max < 5:
                    weight *= 1.4
            
            # Precipitation effects
            if climate_sens['precipitation'] == 'heavy_rain_increases':
                if precipitation > 15:  # Heavy rain
                    weight *= 1.4
                elif precipitation > 5:
                    weight *= 1.2
            elif climate_sens['precipitation'] == 'increases':
                if precipitation > 0:
                    weight *= 1.2
            
            # Seasonal effects
            seasonal_peak = climate_sens['seasonal_peak']
            if seasonal_peak == 'winter' and month in [12, 1, 2]:
                weight *= 1.4
            elif seasonal_peak == 'summer' and month in [6, 7, 8]:
                weight *= 1.4
            elif seasonal_peak == 'spring_summer' and month in [4, 5, 6, 7]:
                weight *= 1.3
            
            # Urban/rural differences (some diseases more common in crowded areas)
            if region_params['urban'] > 0.7:
                if 'respiratory' in disease_info['imc_category']:
                    weight *= 1.1
            
            disease_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(disease_weights)
        disease_probs = [w/total_weight for w in disease_weights]
        
        return np.random.choice(disease_names, p=disease_probs)
    
    def _select_age_group(self, disease_info: Dict) -> str:
        """Select age group based on disease age preferences"""
        age_groups = list(self.age_groups.keys())
        base_weights = [self.age_groups[ag]['weight'] for ag in age_groups]
        
        # Apply disease-specific age preferences
        age_prefs = disease_info.get('age_preference', {})
        adjusted_weights = []
        for i, age_group in enumerate(age_groups):
            weight = base_weights[i] * age_prefs.get(age_group, 1.0)
            adjusted_weights.append(weight)
        
        # Normalize
        total_weight = sum(adjusted_weights)
        age_probs = [w/total_weight for w in adjusted_weights]
        
        return np.random.choice(age_groups, p=age_probs)
    
    def _map_age_group_original(self, age_group: str) -> str:
        """Map to original age group format from events.csv"""
        mapping = {
            '0-5': '0-5 y',
            '6-17': '6-17 y',
            '18-59': '18-59 y',
            '60+': '≥60 y'
        }
        return mapping.get(age_group, '')
    
    def _map_age_group1(self, age_group: str) -> str:
        """Map to age group 1 format from events.csv"""
        mapping = {
            '0-5': '0-5 y',
            '6-17': '6-17 y',
            '18-59': '18-59 y',
            '60+': '≥60 y'
        }
        return mapping.get(age_group, '')
    
    def _map_age_group2(self, age_group: str) -> str:
        """Map to age group 2 format from events.csv"""
        mapping = {
            '0-5': np.random.choice(['0-11 m', '1-4 y'], p=[0.2, 0.8]),  # Realistic split
            '6-17': '5-14 y',
            '18-59': np.random.choice(['15-49 y', '50-60 y'], p=[0.8, 0.2]),
            '60+': '≥60 y'
        }
        return mapping.get(age_group, '')
    
    def _map_age_group3(self, age_group: str) -> str:
        """Map to age group 3 format from events.csv"""
        mapping = {
            '0-5': '< 5 y',
            '6-17': np.random.choice(['5-14 y', '15-18 y'], p=[0.7, 0.3]),
            '18-59': np.random.choice(['19-49 y', '50+ y'], p=[0.8, 0.2]),
            '60+': '50+ y'
        }
        return mapping.get(age_group, '')

def create_external_reference_data():
    """Create external reference data files"""
    logger.info("Creating external reference data files...")
    
    external_dir = Path('data/external')
    external_dir.mkdir(exist_ok=True)
    
    # 1. Regional mapping
    region_mapping = {
        "regions": {
            "Central Region": {"code": "R01", "population": 4868000, "area_km2": 18500},
            "Capital Region": {"code": "R02", "population": 1711000, "area_km2": 8500},
            "Western Region": {"code": "R03", "population": 1763000, "area_km2": 12200},
            "Northern Region": {"code": "R04", "population": 1501000, "area_km2": 9100},
            "Coastal Region": {"code": "R05", "population": 1008000, "area_km2": 5300},
            "Southern Region": {"code": "R06", "population": 1593000, "area_km2": 11900},
            "Eastern Region": {"code": "R07", "population": 1512000, "area_km2": 15000},
            "Desert Region": {"code": "R08", "population": 1239000, "area_km2": 22000}
        },
        "metadata": {
            "source": "Synthetic Regional Statistics",
            "note": "Population and area figures are synthetic for research purposes",
            "coordinate_system": "WGS84"
        }
    }
    
    with open(external_dir / 'region_mapping.json', 'w') as f:
        json.dump(region_mapping, f, indent=2)
    
    # 2. Sample climate data (small subset for quick testing)
    generator = HealthClimateGenerator(random_seed=42)
    sample_climate = generator.generate_climate_data('2023-01-01', '2023-01-31')
    sample_climate.to_csv(external_dir / 'sample_noaa_data.csv', index=False)
    
    logger.info("✓ Created external reference data files")

def main():
    """Main function to generate all synthetic datasets"""
    parser = argparse.ArgumentParser(description='Generate synthetic climate-health data')
    parser.add_argument('--start-date', default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--daily-consultations', type=int, default=60, 
                       help='Base number of daily consultations')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', default='data/synthetic', help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("🏥 Starting Climate-Health Synthetic Data Generation")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Random seed: {args.random_seed}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = HealthClimateGenerator(random_seed=args.random_seed)
    
    # Generate climate data
    logger.info("🌡️ Generating climate data...")
    climate_df = generator.generate_climate_data(args.start_date, args.end_date)
    
    # Generate health consultations with climate interactions
    logger.info("🏥 Generating health consultation data...")
    health_df = generator.generate_health_consultations(
        args.start_date, args.end_date, 
        daily_consultations_base=args.daily_consultations,
        climate_df=climate_df
    )
    
    # Save datasets
    health_output = output_dir / 'synthetic_consultations.csv'
    climate_output = output_dir / 'synthetic_climate.csv'  # Additional climate file
    
    health_df.to_csv(health_output, index=False)
    climate_df.to_csv(climate_output, index=False)
    
    logger.info(f"✅ Health data saved: {health_output} ({len(health_df):,} records)")
    logger.info(f"✅ Climate data saved: {climate_output} ({len(climate_df):,} records)")
    
    # Generate external reference data
    create_external_reference_data()
    
    # Summary statistics
    logger.info("📊 Dataset Summary:")
    logger.info(f"  Health consultations: {len(health_df):,}")
    logger.info(f"  Climate observations: {len(climate_df):,}")
    logger.info(f"  Date range: {health_df['Date of visit'].min()} to {health_df['Date of visit'].max()}")
    logger.info(f"  Regions: {health_df['admin1'].nunique()}")
    logger.info(f"  Disease categories: {health_df['canonical_disease_imc'].nunique()}")
    logger.info(f"  Unique morbidities: {health_df['Morbidity Classification - SY'].nunique()}")
    
    logger.info("🎉 Synthetic data generation completed successfully!")

if __name__ == "__main__":
    main()