#!/usr/bin/env python3
"""
Open-Meteo Weather Data Processor

Processes Open-Meteo weather data files and converts them to the expected format
for the climate-health analysis pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def process_open_meteo_data(file_path: str, admin1_region: str = "Raqqa") -> pd.DataFrame:
    """
    Process Open-Meteo weather data file and convert to expected format.
    
    Args:
        file_path: Path to Open-Meteo CSV file
        admin1_region: Admin1 region name to assign to this weather data
        
    Returns:
        Processed DataFrame with standardized columns
    """
    logger.info(f"📊 Processing Open-Meteo data from {file_path}")
    
    # Read the Open-Meteo file
    df = pd.read_csv(file_path, skiprows=2)  # Skip lat/long header rows
    
    logger.info(f"📈 Loaded {len(df)} weather records")
    
    # Column mapping from Open-Meteo to expected format
    column_mapping = {
        'time': 'date',
        'temperature_2m_max (°C)': 'temp_max',
        'temperature_2m_min (°C)': 'temp_min', 
        'precipitation_sum (mm)': 'precipitation'
    }
    
    # Rename columns
    processed_df = df.rename(columns=column_mapping)
    
    # Add admin1 region (since Open-Meteo data is for single location)
    processed_df['admin1'] = admin1_region
    
    # Convert date to datetime
    processed_df['date'] = pd.to_datetime(processed_df['date'])
    
    # Select only required columns
    required_cols = ['date', 'admin1', 'temp_max', 'temp_min', 'precipitation']
    processed_df = processed_df[required_cols]
    
    # Data validation
    processed_df = validate_weather_data(processed_df)
    
    logger.info(f"✅ Processed weather data: {len(processed_df)} records for {admin1_region}")
    logger.info(f"📅 Date range: {processed_df['date'].min()} to {processed_df['date'].max()}")
    
    return processed_df

def validate_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean weather data.
    
    Args:
        df: Weather DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("🔍 Validating weather data...")
    
    initial_count = len(df)
    
    # Remove rows with missing critical values
    df = df.dropna(subset=['temp_max', 'temp_min', 'precipitation'])
    
    # Temperature validation (reasonable for Syria: -15 to 50°C)
    df = df[(df['temp_max'] >= -15) & (df['temp_max'] <= 50)]
    df = df[(df['temp_min'] >= -20) & (df['temp_min'] <= 45)]
    df = df[df['temp_max'] >= df['temp_min']]  # Max should be >= Min
    
    # Precipitation validation (0 to 200mm/day reasonable max)
    df = df[(df['precipitation'] >= 0) & (df['precipitation'] <= 200)]
    
    # Calculate derived variables
    df['temp_mean'] = (df['temp_max'] + df['temp_min']) / 2
    df['temp_range'] = df['temp_max'] - df['temp_min']
    
    removed_count = initial_count - len(df)
    if removed_count > 0:
        logger.warning(f"⚠️ Removed {removed_count} invalid weather records")
    
    logger.info(f"✅ Weather data validation complete: {len(df)} records")
    logger.info(f"🌡️ Temperature range: {df['temp_min'].min():.1f}°C to {df['temp_max'].max():.1f}°C")
    logger.info(f"🌧️ Precipitation range: {df['precipitation'].min():.1f}mm to {df['precipitation'].max():.1f}mm")
    
    return df

def map_facilities_to_regions() -> Dict[str, str]:
    """
    Map facility names to admin1 regions based on Syrian geography.
    
    Returns:
        Dictionary mapping facility patterns to admin1 regions
    """
    return {
        # Raqqa Governorate
        'Tabqa': 'Raqqa',
        'Raqqa': 'Raqqa', 
        'RAQA': 'Raqqa',
        'Qobeih': 'Raqqa',
        'Qahtaniya': 'Raqqa',
        'Akeirishi': 'Raqqa',
        'Harmoshiya': 'Raqqa',
        
        # Aleppo Governorate  
        'Afrin': 'Aleppo',
        'Kobani': 'Aleppo',
        'Kobane': 'Aleppo',
        'Ain Arab': 'Aleppo',
        
        # Al-Hasakah Governorate
        'Hasakah': 'Al-Hasakah',
        'Qamishli': 'Al-Hasakah',
        'Malki': 'Al-Hasakah',
        'Derbesiye': 'Al-Hasakah',
        
        # Deir ez-Zor Governorate
        'Deir': 'Deir ez-Zor',
        'Mayadin': 'Deir ez-Zor',
        
        # Generic patterns
        'Hospital': 'Raqqa',  # Default for hospitals without clear geographic indicator
        'PHC': 'Raqqa',      # Default for PHCs without clear geographic indicator
        'MCH': 'Raqqa',      # Default for MCH without clear geographic indicator
    }

def assign_weather_to_facilities(health_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign weather data to health facilities based on geographic proximity.
    
    For now, assigns the single Open-Meteo location data to facilities based on
    likely regional mapping.
    
    Args:
        health_df: Health consultation DataFrame
        weather_df: Weather DataFrame
        
    Returns:
        Health DataFrame with admin1 regions assigned
    """
    logger.info("🗺️ Assigning weather regions to health facilities...")
    
    # Get facility-to-region mapping
    region_mapping = map_facilities_to_regions()
    
    # Function to determine region from facility name
    def get_region_from_facility(facility_name):
        if pd.isna(facility_name):
            return 'Unknown'
            
        facility_upper = str(facility_name).upper()
        
        # Check each pattern
        for pattern, region in region_mapping.items():
            if pattern.upper() in facility_upper:
                return region
                
        return 'Unknown'
    
    # Apply mapping
    health_df = health_df.copy()
    if 'admin1' not in health_df.columns:
        health_df['admin1'] = health_df['facility_name'].apply(get_region_from_facility)
    
    # Log mapping results
    region_counts = health_df['admin1'].value_counts()
    logger.info(f"📊 Facility-to-region mapping results:")
    for region, count in region_counts.head(10).items():
        logger.info(f"   • {region}: {count:,} consultations")
    
    unknown_count = region_counts.get('Unknown', 0)
    if unknown_count > 0:
        logger.warning(f"⚠️ {unknown_count:,} consultations could not be mapped to regions")
        
        # Show some examples of unmapped facilities
        unknown_facilities = health_df[health_df['admin1'] == 'Unknown']['facility_name'].value_counts().head(5)
        logger.warning("📋 Examples of unmapped facilities:")
        for facility, count in unknown_facilities.items():
            logger.warning(f"   • {facility}: {count} consultations")
    
    return health_df

def expand_weather_data_to_regions(weather_df: pd.DataFrame, target_regions: list) -> pd.DataFrame:
    """
    Expand single-location weather data to multiple regions.
    
    For now, duplicates the weather data for each region. In a full implementation,
    you would have separate weather data for each region.
    
    Args:
        weather_df: Weather DataFrame for single location
        target_regions: List of admin1 regions to expand to
        
    Returns:
        Expanded weather DataFrame covering all regions
    """
    logger.info(f"🌍 Expanding weather data to {len(target_regions)} regions...")
    
    expanded_dfs = []
    
    for region in target_regions:
        if region != 'Unknown':
            region_df = weather_df.copy()
            region_df['admin1'] = region
            expanded_dfs.append(region_df)
    
    if expanded_dfs:
        expanded_weather = pd.concat(expanded_dfs, ignore_index=True)
        logger.info(f"✅ Weather data expanded: {len(expanded_weather)} records across {len(target_regions)} regions")
        return expanded_weather
    else:
        logger.warning("⚠️ No valid regions to expand weather data to")
        return weather_df

def load_and_process_internal_weather_data(weather_file_path: str, health_df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete pipeline to load and process internal weather data for merging.
    
    Args:
        weather_file_path: Path to Open-Meteo weather file
        health_df: Health DataFrame to determine required regions
        
    Returns:
        Processed weather DataFrame ready for merging
    """
    logger.info("🌡️ Loading and processing internal weather data...")
    
    # Process the Open-Meteo file
    weather_df = process_open_meteo_data(weather_file_path, admin1_region="Raqqa")
    
    # Get unique regions from health data
    if 'admin1' in health_df.columns:
        target_regions = health_df['admin1'].unique().tolist()
    else:
        # Assign regions to facilities first
        health_df_with_regions = assign_weather_to_facilities(health_df, weather_df)
        target_regions = health_df_with_regions['admin1'].unique().tolist()
    
    # Remove 'Unknown' from target regions
    target_regions = [r for r in target_regions if r != 'Unknown']
    
    # Expand weather data to all regions
    if len(target_regions) > 1:
        expanded_weather = expand_weather_data_to_regions(weather_df, target_regions)
    else:
        expanded_weather = weather_df
    
    logger.info(f"✅ Internal weather data processing complete: {len(expanded_weather)} records")
    
    return expanded_weather

if __name__ == "__main__":
    # Test the processor
    import logging
    logging.basicConfig(level=logging.INFO)
    
    weather_file = "data/internal/raw_dat/open-meteo-34.97N38.08E916m.csv"
    
    if Path(weather_file).exists():
        weather_df = process_open_meteo_data(weather_file)
        print(f"Processed weather data shape: {weather_df.shape}")
        print(weather_df.head())
    else:
        print(f"Weather file not found: {weather_file}")