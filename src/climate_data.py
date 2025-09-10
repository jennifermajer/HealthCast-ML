"""
Climate Data Import Module

This module provides comprehensive climate data fetching capabilities for both project components:

COMPONENT 1: CLIMATE SENSITIVITY ANALYSIS
- Fetches historical climate data for identifying climate-sensitive morbidities
- Supports multiple geographic levels (country, governorate, district)
- Provides climate extreme detection and categorization

COMPONENT 2: PREDICTIVE MODELING & FORECASTING  
- Supplies climate features for forecasting models
- Ensures consistent temporal resolution for time series analysis
- Provides real-time and historical data for model training/validation

Supports two primary data sources:
1. Open-Meteo API: Country-level historical weather data (single location)
2. NASA POWER API: Multi-location data for governorate/district level analysis
"""

import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from pathlib import Path
import time
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Location:
    """Location data structure for climate data fetching"""
    name: str
    lat: float
    lon: float
    admin_level: str = "unknown"  # country, governorate, district
    admin1: Optional[str] = None  # parent administrative unit
    country_code: Optional[str] = None

class ClimateDataError(Exception):
    """Custom exception for climate data operations"""
    pass

class ClimateDataFetcher:
    """
    BOTH COMPONENTS: Main class for fetching climate data from multiple sources
    
    Supports both analytical objectives:
    - Component 1: Historical climate data for sensitivity analysis
    - Component 2: Consistent climate features for forecasting models
    """
    
    def __init__(self, default_source: str = "open_meteo", cache_dir: str = "data/cache/climate"):
        self.default_source = default_source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints
        self.open_meteo_url = "https://archive-api.open-meteo.com/v1/archive"
        self.nasa_power_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests
        
    def fetch_climate_data(self, 
                          locations: Union[Location, List[Location]], 
                          date_range: Tuple[str, str],
                          source: str = None,
                          variables: List[str] = None) -> pd.DataFrame:
        """
        BOTH COMPONENTS: Main entry point for fetching climate data
        
        PURPOSE: Fetch climate data supporting both analytical objectives
        - Component 1: Historical climate data for climate-health relationship analysis
        - Component 2: Consistent climate features for predictive model training
        
        Args:
            locations: Single location or list of locations to fetch data for
            date_range: Tuple of (start_date, end_date) as strings 'YYYY-MM-DD'
            source: Data source ('open_meteo', 'nasa_power', or 'demo')
            variables: List of climate variables to fetch
            
        Returns:
            DataFrame with standardized climate data
        """
        source = source or self.default_source
        variables = variables or self._get_default_variables()
        
        # Ensure locations is a list
        if isinstance(locations, Location):
            locations = [locations]
            
        logger.info(f"🌡️ Fetching climate data from {source} for {len(locations)} location(s)")
        logger.info(f"📅 Date range: {date_range[0]} to {date_range[1]}")
        
        # Validate date range
        start_date, end_date = self._validate_date_range(date_range, source)
        
        all_climate_data = []
        
        for location in locations:
            logger.info(f"📍 Fetching data for: {location.name} (lat: {location.lat}, lon: {location.lon})")
            
            try:
                # Check cache first
                cache_key = self._get_cache_key(location, (start_date, end_date), source)
                cached_data = self._load_from_cache(cache_key)
                
                if cached_data is not None:
                    logger.info(f"📁 Using cached data for {location.name}")
                    location_data = cached_data
                else:
                    # Fetch fresh data
                    if source == "open_meteo":
                        location_data = self._fetch_open_meteo(location, (start_date, end_date), variables)
                    elif source == "nasa_power":
                        location_data = self._fetch_nasa_power(location, (start_date, end_date), variables)
                    elif source == "demo":
                        location_data = self._generate_demo_data(location, (start_date, end_date))
                    else:
                        raise ClimateDataError(f"Unknown data source: {source}")
                    
                    # Cache the results
                    self._save_to_cache(cache_key, location_data)
                
                # Add location metadata
                if location_data is not None and len(location_data) > 0:
                    location_data['location_name'] = location.name
                    location_data['admin_level'] = location.admin_level
                    location_data['admin1'] = location.admin1 or location.name
                    location_data['latitude'] = location.lat
                    location_data['longitude'] = location.lon
                    location_data['climate_source'] = source
                    
                    all_climate_data.append(location_data)
                    
            except Exception as e:
                logger.error(f"❌ Failed to fetch climate data for {location.name}: {e}")
                # Generate demo data as fallback
                try:
                    demo_data = self._generate_demo_data(location, (start_date, end_date))
                    demo_data['location_name'] = location.name
                    demo_data['admin_level'] = location.admin_level
                    demo_data['admin1'] = location.admin1 or location.name
                    demo_data['latitude'] = location.lat
                    demo_data['longitude'] = location.lon
                    demo_data['climate_source'] = 'demo_fallback'
                    all_climate_data.append(demo_data)
                    logger.warning(f"⚠️ Using demo data as fallback for {location.name}")
                except Exception as fallback_error:
                    logger.error(f"❌ Even demo data generation failed for {location.name}: {fallback_error}")
                    continue
        
        # Combine all location data
        if not all_climate_data:
            raise ClimateDataError("Failed to fetch climate data for any location")
            
        combined_data = pd.concat(all_climate_data, ignore_index=True)
        
        # Standardize and enhance the data
        standardized_data = self._standardize_climate_data(combined_data)
        
        logger.info(f"✅ Successfully fetched climate data: {len(standardized_data)} rows")
        return standardized_data

    def _fetch_open_meteo(self, location: Location, date_range: Tuple[str, str], variables: List[str]) -> pd.DataFrame:
        """
        COMPONENT 1 & 2: Fetch data from Open-Meteo Historical Weather API
        
        Open-Meteo provides free historical weather data with good coverage.
        Best suited for country-level analysis due to single-location limitation.
        """
        self._rate_limit()
        
        # Open-Meteo parameters mapping
        param_mapping = {
            'temperature_mean': 'temperature_2m_mean',
            'temperature_min': 'temperature_2m_min', 
            'temperature_max': 'temperature_2m_max',
            'humidity': 'relativehumidity_2m_mean',
            'precipitation': 'precipitation_sum',
            'wind_speed': 'windspeed_10m_mean',
            'pressure': 'surface_pressure_mean'
        }
        
        # Build parameter list
        params = []
        for var in variables:
            if var in param_mapping:
                params.append(param_mapping[var])
        
        # API request parameters
        request_params = {
            'latitude': location.lat,
            'longitude': location.lon,
            'start_date': date_range[0],
            'end_date': date_range[1],
            'daily': ','.join(params),
            'timezone': 'UTC'
        }
        
        try:
            response = requests.get(self.open_meteo_url, params=request_params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if 'daily' not in data:
                raise ClimateDataError("Invalid response from Open-Meteo API")
            
            # Convert to DataFrame
            daily_data = data['daily']
            df = pd.DataFrame()
            
            df['date'] = pd.to_datetime(daily_data['time'])
            
            # Map variables back to standard names
            reverse_mapping = {v: k for k, v in param_mapping.items()}
            
            for api_param, values in daily_data.items():
                if api_param != 'time' and api_param in reverse_mapping:
                    df[reverse_mapping[api_param]] = values
            
            # Fill missing columns with NaN
            for var in variables:
                if var not in df.columns:
                    df[var] = np.nan
                    
            return df
            
        except requests.exceptions.RequestException as e:
            raise ClimateDataError(f"Open-Meteo API request failed: {e}")
        except (KeyError, ValueError) as e:
            raise ClimateDataError(f"Failed to parse Open-Meteo response: {e}")

    def _fetch_nasa_power(self, location: Location, date_range: Tuple[str, str], variables: List[str]) -> pd.DataFrame:
        """
        COMPONENT 1 & 2: Fetch data from NASA POWER API
        
        NASA POWER provides high-quality satellite-derived meteorological data.
        Supports multiple locations for governorate/district-level analysis.
        """
        self._rate_limit()
        
        # NASA POWER parameter mapping
        param_mapping = {
            'temperature_mean': 'T2M',
            'temperature_min': 'T2M_MIN',
            'temperature_max': 'T2M_MAX', 
            'humidity': 'RH2M',
            'precipitation': 'PRECTOTCORR',
            'wind_speed': 'WS2M',
            'pressure': 'PS'  # Surface pressure
        }
        
        # Build parameter list for NASA POWER
        nasa_params = []
        for var in variables:
            if var in param_mapping:
                nasa_params.append(param_mapping[var])
        
        # Validate date range for NASA POWER (1981-01-01 onwards)
        start_date = max(pd.to_datetime(date_range[0]), pd.to_datetime('1981-01-01'))
        end_date = min(pd.to_datetime(date_range[1]), pd.to_datetime('now') - pd.Timedelta(days=7))
        
        if start_date > end_date:
            raise ClimateDataError("Invalid date range for NASA POWER API")
        
        # Format dates for NASA API
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # Build API URL
        url = (f"{self.nasa_power_url}?"
               f"parameters={','.join(nasa_params)}&"
               f"community=ag&"
               f"longitude={location.lon}&"
               f"latitude={location.lat}&"
               f"start={start_str}&"
               f"end={end_str}&"
               f"format=json")
        
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            
            if 'properties' not in data or 'parameter' not in data['properties']:
                raise ClimateDataError("Invalid response from NASA POWER API")
            
            # Extract parameter data
            param_data = data['properties']['parameter']
            
            # Create DataFrame
            dates = []
            values_dict = {var: [] for var in variables}
            
            # Get date range from first parameter
            first_param = list(param_data.keys())[0]
            date_keys = sorted(param_data[first_param].keys())
            
            reverse_mapping = {v: k for k, v in param_mapping.items()}
            
            for date_key in date_keys:
                dates.append(pd.to_datetime(date_key, format='%Y%m%d'))
                
                for nasa_param, daily_value in param_data.items():
                    if nasa_param in reverse_mapping:
                        var_name = reverse_mapping[nasa_param]
                        values_dict[var_name].append(daily_value.get(date_key, np.nan))
            
            # Create DataFrame
            df = pd.DataFrame({'date': dates})
            for var in variables:
                if var in values_dict:
                    df[var] = values_dict[var]
                else:
                    df[var] = np.nan
                    
            return df
            
        except requests.exceptions.RequestException as e:
            raise ClimateDataError(f"NASA POWER API request failed: {e}")
        except (KeyError, ValueError) as e:
            raise ClimateDataError(f"Failed to parse NASA POWER response: {e}")

    def _generate_demo_data(self, location: Location, date_range: Tuple[str, str]) -> pd.DataFrame:
        """
        BOTH COMPONENTS: Generate realistic demo climate data for testing/fallback
        
        Creates climatologically realistic synthetic data with seasonal patterns
        suitable for both climate sensitivity analysis and forecasting model development.
        """
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Seasonal patterns based on day of year
        day_of_year = dates.dayofyear
        
        # Base climate patterns (adjust based on location if needed)
        # These patterns are designed for Middle Eastern climate (like Syria)
        
        # Temperature with seasonal variation (hot summers, mild winters)
        temp_base = 20 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temperature_mean = temp_base + np.random.normal(0, 3, n)
        
        # Temperature range varies by season
        temp_range = 8 + 4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temperature_min = temperature_mean - temp_range/2 + np.random.normal(0, 2, n)
        temperature_max = temperature_mean + temp_range/2 + np.random.normal(0, 2, n)
        
        # Humidity with inverse seasonal pattern (higher in winter)
        humidity_base = 65 - 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        humidity = np.clip(humidity_base + np.random.normal(0, 10, n), 20, 90)
        
        # Precipitation with winter peaks and occasional summer storms
        precip_seasonal = 2 + 3 * np.sin(2 * np.pi * (day_of_year - 300) / 365)
        precipitation = np.maximum(0, precip_seasonal + np.random.exponential(1, n))
        
        # Add occasional heavy precipitation events
        heavy_rain_prob = 0.05  # 5% chance of heavy rain
        heavy_rain_mask = np.random.random(n) < heavy_rain_prob
        precipitation[heavy_rain_mask] += np.random.exponential(10, sum(heavy_rain_mask))
        
        # Wind speed (log-normal distribution)
        wind_speed = np.maximum(0, np.random.lognormal(np.log(8), 0.4, n))
        
        # Pressure (normal around sea level pressure)
        pressure = np.random.normal(1013, 8, n)
        
        df = pd.DataFrame({
            'date': dates,
            'temperature_mean': temperature_mean,
            'temperature_min': temperature_min, 
            'temperature_max': temperature_max,
            'humidity': humidity,
            'precipitation': precipitation,
            'wind_speed': wind_speed,
            'pressure': pressure
        })
        
        return df

    def _standardize_climate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BOTH COMPONENTS: Standardize climate data format and add derived variables
        
        PURPOSE: Create consistent climate features for both analytical objectives
        - Component 1: Standardized variables for climate-health correlation analysis
        - Component 2: Consistent feature engineering for predictive models
        """
        df = df.copy()
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Add time-based features (useful for both components)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring', 
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Temperature categories (Component 1: climate sensitivity analysis)
        df['temp_category'] = pd.cut(df['temperature_mean'], 
                                   bins=[-np.inf, 5, 15, 25, 30, 35, np.inf],
                                   labels=['Cold', 'Cool', 'Moderate', 'Warm', 'Hot', 'Extreme'])
        
        # Precipitation categories
        df['precip_category'] = pd.cut(df['precipitation'],
                                     bins=[-0.1, 0, 5, 25, 50, np.inf],
                                     labels=['No Rain', 'Light', 'Moderate', 'Heavy', 'Extreme'])
        
        # Humidity categories
        df['humidity_category'] = pd.cut(df['humidity'],
                                       bins=[0, 40, 60, 80, 100],
                                       labels=['Dry', 'Moderate', 'Humid', 'Very Humid'])
        
        # Climate extremes indicators (Component 1: extreme event analysis)
        df['is_heatwave'] = (df['temperature_max'] > 35).astype(int)
        df['is_heavy_rain'] = (df['precipitation'] > 25).astype(int)
        df['is_drought_day'] = ((df['precipitation'] == 0) & 
                               (df['humidity'] < 30)).astype(int)
        
        # Heat index calculation (simplified)
        df['heat_index'] = df['temperature_mean'] + (df['humidity'] / 10)
        
        # Rolling averages for trend analysis (Component 2: forecasting features)
        if len(df) >= 7:
            df['temp_mean_7d'] = df.groupby(['location_name'])['temperature_mean'].rolling(7, min_periods=1).mean().values
            df['precip_sum_7d'] = df.groupby(['location_name'])['precipitation'].rolling(7, min_periods=1).sum().values
            df['humidity_mean_7d'] = df.groupby(['location_name'])['humidity'].rolling(7, min_periods=1).mean().values
        
        if len(df) >= 30:
            df['temp_mean_30d'] = df.groupby(['location_name'])['temperature_mean'].rolling(30, min_periods=1).mean().values
            df['precip_sum_30d'] = df.groupby(['location_name'])['precipitation'].rolling(30, min_periods=1).sum().values
        
        # Sort by location and date
        df = df.sort_values(['location_name', 'date']).reset_index(drop=True)
        
        return df

    def _validate_date_range(self, date_range: Tuple[str, str], source: str) -> Tuple[str, str]:
        """Validate and adjust date range based on data source limitations"""
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        # Source-specific validations
        if source == "nasa_power":
            earliest_date = pd.to_datetime('1981-01-01')
            latest_date = pd.to_datetime('now') - pd.Timedelta(days=7)
            
            if start_date < earliest_date:
                logger.warning(f"NASA POWER data starts from 1981-01-01. Adjusting start date from {start_date.date()} to {earliest_date.date()}")
                start_date = earliest_date
                
            if end_date > latest_date:
                logger.warning(f"NASA POWER data is ~7 days behind. Adjusting end date from {end_date.date()} to {latest_date.date()}")
                end_date = latest_date
                
        elif source == "open_meteo":
            # Open-Meteo historical data goes back to 1940 and is current up to yesterday
            earliest_date = pd.to_datetime('1940-01-01')
            latest_date = pd.to_datetime('now') - pd.Timedelta(days=1)
            
            if start_date < earliest_date:
                logger.warning(f"Open-Meteo data starts from 1940-01-01. Adjusting start date from {start_date.date()} to {earliest_date.date()}")
                start_date = earliest_date
                
            if end_date > latest_date:
                logger.warning(f"Open-Meteo data is ~1 day behind. Adjusting end date from {end_date.date()} to {latest_date.date()}")
                end_date = latest_date
        
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

    def _get_default_variables(self) -> List[str]:
        """Get default climate variables for fetching"""
        return [
            'temperature_mean', 'temperature_min', 'temperature_max',
            'humidity', 'precipitation', 'wind_speed', 'pressure'
        ]
    
    def _rate_limit(self):
        """Simple rate limiting for API requests"""
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _get_cache_key(self, location: Location, date_range: Tuple[str, str], source: str) -> str:
        """Generate cache key for location and date range"""
        return f"{source}_{location.name}_{location.lat}_{location.lon}_{date_range[0]}_{date_range[1]}.csv"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and recent"""
        cache_file = self.cache_dir / cache_key
        
        if cache_file.exists():
            try:
                # Check if cache is recent (within 7 days)
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age < 7 * 24 * 3600:  # 7 days in seconds
                    return pd.read_csv(cache_file, parse_dates=['date'])
                else:
                    logger.info(f"Cache file {cache_key} is older than 7 days, fetching fresh data")
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache"""
        try:
            cache_file = self.cache_dir / cache_key
            data.to_csv(cache_file, index=False)
            logger.debug(f"Saved data to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save to cache {cache_key}: {e}")

# ============================================================================
# LOCATION MANAGEMENT SYSTEM
# ============================================================================

class LocationManager:
    """
    BOTH COMPONENTS: Manage geographic locations for climate data fetching
    
    Supports multiple administrative levels:
    - Country level: For national analysis and Open-Meteo integration
    - Governorate level: For regional climate sensitivity analysis  
    - District level: For detailed spatial forecasting models
    """
    
    def __init__(self):
        self.locations = {}
        self._load_default_locations()
    
    def _load_default_locations(self):
        """Load default locations for common countries/regions"""
        
        # Syria governorates (example for Middle East region)
        syria_governorates = {
            'Damascus': Location('Damascus', 33.5138, 36.2765, 'governorate', 'Damascus', 'SY'),
            'Aleppo': Location('Aleppo', 36.2021, 37.1343, 'governorate', 'Aleppo', 'SY'),
            'Homs': Location('Homs', 34.7394, 36.7163, 'governorate', 'Homs', 'SY'),
            'Hama': Location('Hama', 35.1520, 36.7490, 'governorate', 'Hama', 'SY'),
            'Latakia': Location('Latakia', 35.5138, 35.7719, 'governorate', 'Latakia', 'SY'),
            'Idlib': Location('Idlib', 35.9333, 36.6333, 'governorate', 'Idlib', 'SY'),
            'Daraa': Location('Daraa', 32.6189, 36.1021, 'governorate', 'Daraa', 'SY'),
            'Deir_ez_Zor': Location('Deir ez-Zor', 35.3394, 40.1467, 'governorate', 'Deir ez-Zor', 'SY'),
            'Al_Hasakah': Location('Al-Hasakah', 36.5000, 40.7500, 'governorate', 'Al-Hasakah', 'SY'),
            'Ar_Raqqah': Location('Ar-Raqqah', 35.9500, 39.0167, 'governorate', 'Ar-Raqqah', 'SY'),
            'As_Suwayda': Location('As-Suwayda', 32.7094, 36.5694, 'governorate', 'As-Suwayda', 'SY'),
            'Quneitra': Location('Quneitra', 33.1267, 35.8242, 'governorate', 'Quneitra', 'SY'),
            'Rif_Dimashq': Location('Rif Dimashq', 33.4500, 36.5000, 'governorate', 'Rif Dimashq', 'SY'),
            'Tartus': Location('Tartus', 34.8833, 35.8833, 'governorate', 'Tartus', 'SY')
        }
        
        # Country-level locations
        country_capitals = {
            'Syria': Location('Syria', 33.5138, 36.2765, 'country', None, 'SY'),
            'Jordan': Location('Jordan', 31.9539, 35.9106, 'country', None, 'JO'),
            'Lebanon': Location('Lebanon', 33.8547, 35.8623, 'country', None, 'LB'),
            'Iraq': Location('Iraq', 33.3128, 44.3615, 'country', None, 'IQ'),
            'Turkey': Location('Turkey', 39.9334, 32.8597, 'country', None, 'TR')
        }
        
        self.locations.update(syria_governorates)
        self.locations.update(country_capitals)
    
    def add_location(self, location: Location):
        """Add a custom location"""
        self.locations[location.name] = location
    
    def get_location(self, name: str) -> Optional[Location]:
        """Get location by name"""
        return self.locations.get(name)
    
    def get_locations_by_level(self, admin_level: str) -> List[Location]:
        """Get all locations at a specific administrative level"""
        return [loc for loc in self.locations.values() if loc.admin_level == admin_level]
    
    def get_country_locations(self, country_code: str) -> List[Location]:
        """Get all locations within a country"""
        return [loc for loc in self.locations.values() if loc.country_code == country_code]
    
    def list_available_locations(self) -> pd.DataFrame:
        """Return DataFrame of all available locations"""
        data = []
        for loc in self.locations.values():
            data.append({
                'name': loc.name,
                'latitude': loc.lat,
                'longitude': loc.lon,
                'admin_level': loc.admin_level,
                'admin1': loc.admin1,
                'country_code': loc.country_code
            })
        return pd.DataFrame(data)

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def fetch_multi_location_climate(locations: List[Location], 
                                date_range: Tuple[str, str],
                                source: str = "open_meteo",
                                variables: List[str] = None) -> pd.DataFrame:
    """
    BOTH COMPONENTS: Convenience function to fetch climate data for multiple locations
    
    This is the main function that should be used by data_processing.py for integration.
    
    Args:
        locations: List of Location objects
        date_range: Tuple of (start_date, end_date) strings
        source: Data source ('open_meteo', 'nasa_power', 'demo')
        variables: Climate variables to fetch
        
    Returns:
        Combined DataFrame with climate data for all locations
    """
    fetcher = ClimateDataFetcher(default_source=source)
    return fetcher.fetch_climate_data(locations, date_range, source, variables)

def get_syria_climate_data(date_range: Tuple[str, str], 
                          level: str = "country",
                          source: str = "open_meteo") -> pd.DataFrame:
    """
    BOTH COMPONENTS: Convenience function to fetch Syria climate data
    
    Args:
        date_range: Tuple of (start_date, end_date) strings
        level: Geographic level ('country', 'governorate') 
        source: Data source ('open_meteo', 'nasa_power', 'demo')
        
    Returns:
        Climate data for Syria at specified geographic level
    """
    location_manager = LocationManager()
    
    if level == "country":
        locations = [location_manager.get_location('Syria')]
    elif level == "governorate":
        locations = location_manager.get_country_locations('SY')
        locations = [loc for loc in locations if loc.admin_level == 'governorate']
    else:
        raise ValueError(f"Invalid level: {level}. Use 'country' or 'governorate'")
    
    # Filter out None locations
    locations = [loc for loc in locations if loc is not None]
    
    if not locations:
        raise ValueError(f"No locations found for level: {level}")
    
    return fetch_multi_location_climate(locations, date_range, source)

def create_custom_location(name: str, lat: float, lon: float, 
                          admin_level: str = "custom",
                          admin1: str = None,
                          country_code: str = None) -> Location:
    """
    BOTH COMPONENTS: Create a custom location for climate data fetching
    
    Allows users to define custom locations for specialized analysis.
    """
    return Location(name, lat, lon, admin_level, admin1, country_code)

# ============================================================================
# INTEGRATION WITH EXISTING DATA_PROCESSING MODULE
# ============================================================================

def save_climate_data_for_processing(climate_data: pd.DataFrame, 
                                   output_path: str = "data/processed/climate_data.csv"):
    """
    BOTH COMPONENTS: Save climate data in format compatible with data_processing.py
    
    PURPOSE: Bridge between climate data fetching and existing data processing pipeline
    - Saves standardized climate data for merge with health consultation data
    - Ensures consistent format for both Component 1 and Component 2 analysis
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with proper formatting
    climate_data.to_csv(output_path, index=False)
    logger.info(f"✅ Climate data saved to {output_path}")
    logger.info(f"📊 Data shape: {climate_data.shape}")
    logger.info(f"📅 Date range: {climate_data['date'].min()} to {climate_data['date'].max()}")
    
    return str(output_path)

def validate_climate_data_for_merge(climate_data: pd.DataFrame, 
                                  health_data: pd.DataFrame = None) -> Dict[str, Any]:
    """
    BOTH COMPONENTS: Validate climate data before merging with health data
    
    PURPOSE: Ensure data quality for both analytical objectives
    - Component 1: Validates climate data completeness for correlation analysis
    - Component 2: Ensures temporal alignment for forecasting models
    """
    validation_report = {
        'data_shape': climate_data.shape,
        'date_range': (climate_data['date'].min(), climate_data['date'].max()),
        'locations': climate_data['location_name'].nunique() if 'location_name' in climate_data.columns else 0,
        'missing_values': climate_data.isnull().sum().to_dict(),
        'issues': [],
        'recommendations': []
    }
    
    # Check for missing data
    missing_pct = (climate_data.isnull().sum() / len(climate_data)) * 100
    high_missing = missing_pct[missing_pct > 20]
    
    if len(high_missing) > 0:
        validation_report['issues'].append(f"High missing data in: {high_missing.to_dict()}")
        validation_report['recommendations'].append("Consider using multiple data sources or imputation")
    
    # Check temporal coverage
    expected_days = (climate_data['date'].max() - climate_data['date'].min()).days + 1
    actual_days = climate_data['date'].nunique()
    
    if actual_days < expected_days * 0.95:  # Less than 95% coverage
        validation_report['issues'].append(f"Temporal gaps detected: {actual_days}/{expected_days} days")
        validation_report['recommendations'].append("Fill temporal gaps or adjust analysis period")
    
    # Check for extreme values
    numeric_cols = climate_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['temperature_mean', 'temperature_min', 'temperature_max']:
            extreme_temp = climate_data[(climate_data[col] < -50) | (climate_data[col] > 60)]
            if len(extreme_temp) > 0:
                validation_report['issues'].append(f"Extreme temperature values in {col}")
        
        if col == 'precipitation':
            extreme_precip = climate_data[climate_data[col] > 500]  # > 500mm/day is very rare
            if len(extreme_precip) > 0:
                validation_report['issues'].append(f"Extreme precipitation values detected")
    
    # If health data provided, check alignment
    if health_data is not None:
        if 'date' in health_data.columns:
            health_dates = set(pd.to_datetime(health_data['date']).dt.date)
            climate_dates = set(climate_data['date'].dt.date)
            
            overlap = len(health_dates.intersection(climate_dates))
            overlap_pct = (overlap / len(health_dates)) * 100
            
            validation_report['health_climate_overlap'] = {
                'overlap_days': overlap,
                'overlap_percentage': overlap_pct
            }
            
            if overlap_pct < 90:
                validation_report['issues'].append(f"Poor temporal overlap with health data: {overlap_pct:.1f}%")
                validation_report['recommendations'].append("Extend climate data date range to match health data")
    
    return validation_report

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize location manager
    location_manager = LocationManager()
    
    # Example 1: Country-level data for Syria (Open-Meteo)
    print("📍 Fetching country-level climate data for Syria...")
    syria_country = get_syria_climate_data(
        date_range=("2023-01-01", "2023-12-31"),
        level="country",
        source="open_meteo"
    )
    print(f"✅ Retrieved {len(syria_country)} days of country-level data")
    
    # Example 2: Multi-location governorate data (NASA POWER)
    print("\n📍 Fetching governorate-level climate data...")
    key_governorates = [
        location_manager.get_location('Damascus'),
        location_manager.get_location('Aleppo'),
        location_manager.get_location('Homs')
    ]
    
    governorate_data = fetch_multi_location_climate(
        locations=key_governorates,
        date_range=("2023-06-01", "2023-08-31"),
        source="nasa_power"
    )
    print(f"✅ Retrieved {len(governorate_data)} rows of governorate-level data")
    
    # Example 3: Save for integration with data processing
    print("\n💾 Saving climate data for processing...")
    save_climate_data_for_processing(syria_country, "data/processed/syria_climate_2023.csv")
    
    # Example 4: Data validation
    print("\n🔍 Validating climate data...")
    validation = validate_climate_data_for_merge(syria_country)
    print(f"Data validation: {len(validation['issues'])} issues found")
    
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    print("\n🎉 Climate data module test completed!")