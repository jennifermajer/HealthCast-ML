"""
Data loading, validation, and merging functions for climate-health analysis.
Handles both private IMC data and public synthetic data, including Syria internal data.
"""

import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
from dotenv import load_dotenv
from datetime import datetime
import warnings

# Import taxonomy processor for disease classification
try:
    from taxonomy_processor import TaxonomyProcessor
except ImportError:
    TaxonomyProcessor = None
    logger.warning("⚠️ TaxonomyProcessor not available - using fallback taxonomy")

# Import taxonomy cache for performance optimization
try:
    from taxonomy_cache import TaxonomyCache, apply_cached_taxonomy_mapping
except ImportError:
    TaxonomyCache = None
    apply_cached_taxonomy_mapping = None
    logger.warning("⚠️ TaxonomyCache not available")

# Import climate data module for comprehensive weather data fetching
try:
    from climate_data import (
        ClimateDataFetcher, LocationManager, Location,
        fetch_multi_location_climate, get_syria_climate_data,
        save_climate_data_for_processing, validate_climate_data_for_merge
    )
    CLIMATE_MODULE_AVAILABLE = True
except ImportError:
    ClimateDataFetcher = None
    LocationManager = None
    CLIMATE_MODULE_AVAILABLE = False
    logger.warning("⚠️ Climate data module not available - using legacy climate data loading")

logger = logging.getLogger(__name__)

def load_config() -> Dict:
    """Load configuration with environment variable substitution"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment variables
    load_dotenv()
    
    # Override with environment variables
    use_synthetic = os.getenv('USE_SYNTHETIC', 'true').lower() == 'true'
    config['data']['use_synthetic'] = use_synthetic
    
    return config

def get_data_paths(config: Dict) -> Dict[str, str]:
    """Get appropriate data paths based on configuration"""
    if config['data']['use_synthetic']:
        return {
            'health': config['data']['public']['health_data_path'],
            'climate': config['data']['public']['climate_data_path']
        }
    else:
        return {
            'health': config['data']['private']['health_data_path'],
            'climate': config['data']['private']['climate_data_path']
        }

def validate_health_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate health consultation data format and content
    
    Args:
        df: Raw health consultation dataframe
        
    Returns:
        Validated and cleaned dataframe
    """
    logger.info("Validating health consultation data...")
    
    # Required columns for analysis
    required_cols = [
        'date', 'admin1',  # Core location-time
        'morbidity', 'canonical_disease_imc', 'icd11_title',  # Disease classification
        'age_group', 'sex'  # Demographics
    ]
    
    # Optional columns that enhance analysis if present
    optional_cols = [
        'age_group_new', 'age_group2', 'age_group3', 'age_group4',  # Age variants
        'admin0', 'admin2', 'admin3',  # Additional geographic levels
        'orgunit', 'facility_type'  # Health facility info
    ]
    
    # Check for required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Log available optional columns
    available_optional = [col for col in optional_cols if col in df.columns]
    logger.info(f"Available optional columns: {available_optional}")
    
    # Data type validation and cleaning
    original_len = len(df)
    
    # 1. Date processing
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    logger.info(f"Date validation: {original_len - len(df)} invalid dates removed")
    
    # 2. Geographic validation
    # admin1 should be Syrian governorate
    df = df.dropna(subset=['admin1'])
    df['admin1'] = df['admin1'].str.strip().str.title()
    logger.info(f"Geographic data: {df['admin1'].nunique()} unique admin1 values")
    
    # 3. Disease classification validation
    # Remove rows where all disease classifications are missing
    disease_cols = ['morbidity', 'canonical_disease_imc', 'icd11_title']
    df = df.dropna(subset=disease_cols, how='all')
    
    # Clean disease classification strings
    for col in disease_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'NaN', ''], np.nan)
    
    logger.info(f"Disease classification: {df['canonical_disease_imc'].nunique()} unique IMC categories")
    logger.info(f"ICD-11 mapping: {df['icd11_title'].notna().sum()} rows with ICD-11 codes")
    
    # 4. Demographic validation
    # Sex standardization
    if 'sex' in df.columns:
        df['sex'] = df['sex'].str.upper().str.strip()
        valid_sex = df['sex'].isin(['M', 'F', 'MALE', 'FEMALE'])
        df = df[valid_sex]
        # Standardize to M/F
        df['sex'] = df['sex'].map({
            'M': 'M', 'MALE': 'M',
            'F': 'F', 'FEMALE': 'F'
        })
    
    # Age group validation
    if 'age_group' in df.columns:
        df = df.dropna(subset=['age_group'])
        logger.info(f"Age groups: {sorted(df['age_group'].unique())}")
    
    # 5. Facility information (if available)
    if 'facility_type' in df.columns:
        df['facility_type'] = df['facility_type'].str.strip().str.title()
        logger.info(f"Facility types: {df['facility_type'].value_counts().to_dict()}")
    
    # Final validation summary
    final_len = len(df)
    logger.info(f"✓ Health data validated: {final_len:,} consultations ({original_len - final_len:,} removed)")
    logger.info(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"✓ Geographic coverage: {df['admin1'].nunique()} admin1 regions")
    logger.info(f"✓ Sex distribution: {df['sex'].value_counts().to_dict()}")
    
    return df

def validate_climate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate climate data format and content
    
    Args:
        df: Raw climate dataframe
        
    Returns:
        Validated and cleaned dataframe
    """
    logger.info("Validating climate data...")
    
    required_cols = ['date', 'admin1', 'temp_max', 'temp_min', 'precipitation']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required climate columns: {missing_cols}")
    
    original_len = len(df)
    
    # Date processing
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Geographic matching
    df['admin1'] = df['admin1'].str.strip().str.title()
    
    # Climate variable validation
    # Temperature should be reasonable for Syria (-10 to 50°C)
    df = df[(df['temp_max'] >= -10) & (df['temp_max'] <= 50)]
    df = df[(df['temp_min'] >= -15) & (df['temp_min'] <= 45)]
    df = df[df['temp_max'] >= df['temp_min']]  # Max should be >= Min
    
    # Precipitation should be non-negative and reasonable (<500mm/day)
    df = df[(df['precipitation'] >= 0) & (df['precipitation'] <= 500)]
    
    # Create additional climate variables
    df['temp_mean'] = (df['temp_max'] + df['temp_min']) / 2
    df['temp_range'] = df['temp_max'] - df['temp_min']
    
    final_len = len(df)
    logger.info(f"✓ Climate data validated: {final_len:,} observations ({original_len - final_len:,} removed)")
    logger.info(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"✓ Geographic coverage: {df['admin1'].nunique()} admin1 regions")
    logger.info(f"✓ Temperature range: {df['temp_mean'].min():.1f}°C to {df['temp_mean'].max():.1f}°C")
    logger.info(f"✓ Precipitation range: {df['precipitation'].min():.1f}mm to {df['precipitation'].max():.1f}mm")
    
    return df

# ============================================================================
# NEW CLIMATE DATA INTEGRATION FUNCTIONS
# ============================================================================

def fetch_climate_data_api(date_range: Tuple[str, str],
                          locations: Optional[List[str]] = None,
                          admin_level: str = "country",
                          source: str = "open_meteo",
                          cache_dir: str = "data/cache/climate") -> pd.DataFrame:
    """
    BOTH COMPONENTS: Fetch climate data using the comprehensive climate data module
    
    PURPOSE: Primary function for fetching fresh climate data from APIs
    - Component 1: Historical climate data for climate-health relationship analysis
    - Component 2: Consistent climate features for predictive model training
    
    Args:
        date_range: Tuple of (start_date, end_date) as strings 'YYYY-MM-DD'
        locations: List of location names (None for default Syria locations)
        admin_level: Geographic level ('country', 'governorate', 'district')
        source: Data source ('open_meteo', 'nasa_power', 'demo')
        cache_dir: Directory for caching API responses
        
    Returns:
        Standardized climate DataFrame ready for merging with health data
    """
    if not CLIMATE_MODULE_AVAILABLE:
        logger.error("❌ Climate data module not available. Install required dependencies or use legacy climate data loading.")
        raise ImportError("Climate data module not available")
    
    logger.info(f"🌡️ Fetching climate data from {source} API...")
    logger.info(f"📅 Date range: {date_range[0]} to {date_range[1]}")
    logger.info(f"📍 Admin level: {admin_level}")
    
    try:
        # Initialize location manager
        location_manager = LocationManager()
        
        # Determine locations to fetch
        if locations is None:
            if admin_level == "country":
                # Default to Syria country-level
                location_objects = [location_manager.get_location('Syria')]
            elif admin_level == "governorate":
                # Default to Syria governorates
                location_objects = location_manager.get_country_locations('SY')
                location_objects = [loc for loc in location_objects if loc and loc.admin_level == 'governorate']
            else:
                raise ValueError(f"No default locations for admin level: {admin_level}")
        else:
            # Convert location names to location objects
            location_objects = []
            for loc_name in locations:
                loc_obj = location_manager.get_location(loc_name)
                if loc_obj:
                    location_objects.append(loc_obj)
                else:
                    logger.warning(f"⚠️ Location '{loc_name}' not found in location database")
        
        # Filter out None locations
        location_objects = [loc for loc in location_objects if loc is not None]
        
        if not location_objects:
            raise ValueError("No valid locations found for climate data fetching")
        
        logger.info(f"📍 Fetching data for {len(location_objects)} location(s): {[loc.name for loc in location_objects]}")
        
        # Fetch climate data
        climate_data = fetch_multi_location_climate(
            locations=location_objects,
            date_range=date_range,
            source=source
        )
        
        # Convert to format compatible with existing merge functions
        climate_data_formatted = _format_climate_for_merge(climate_data, admin_level)
        
        # Validate the data
        climate_data_validated = validate_climate_data(climate_data_formatted)
        
        logger.info(f"✅ Successfully fetched {len(climate_data_validated)} rows of climate data")
        
        # Optionally save for future use
        cache_path = save_climate_data_for_processing(
            climate_data_validated, 
            f"{cache_dir}/fetched_climate_{date_range[0]}_{date_range[1]}_{admin_level}_{source}.csv"
        )
        logger.info(f"💾 Climate data cached to: {cache_path}")
        
        return climate_data_validated
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch climate data from API: {e}")
        logger.info("🔄 Falling back to demo data generation...")
        
        # Fallback to demo data
        return _generate_fallback_climate_data(date_range, locations or ['Syria'], admin_level)

def load_climate_data_enhanced(file_path: Optional[str] = None,
                             date_range: Optional[Tuple[str, str]] = None,
                             admin_level: str = "country",
                             source: str = "open_meteo",
                             use_api: bool = False) -> pd.DataFrame:
    """
    BOTH COMPONENTS: Enhanced climate data loading with API integration option
    
    PURPOSE: Flexible climate data loading supporting both file-based and API sources
    - Component 1: Load historical climate data for sensitivity analysis
    - Component 2: Load consistent climate features for forecasting models
    
    Args:
        file_path: Path to existing climate data file (None to use API)
        date_range: Date range for API fetching (required if use_api=True)
        admin_level: Geographic level for API fetching
        source: Data source for API fetching
        use_api: Whether to fetch from API instead of loading from file
        
    Returns:
        Validated climate DataFrame ready for analysis
    """
    logger.info("📊 Loading climate data with enhanced integration...")
    
    if use_api or file_path is None:
        # Use API fetching
        if date_range is None:
            raise ValueError("date_range is required when using API or when file_path is None")
        
        logger.info("🌐 Using API-based climate data fetching...")
        return fetch_climate_data_api(
            date_range=date_range,
            admin_level=admin_level,
            source=source
        )
    
    else:
        # Load from file (legacy mode)
        logger.info(f"📁 Loading climate data from file: {file_path}")
        
        if not Path(file_path).exists():
            logger.warning(f"⚠️ Climate data file not found: {file_path}")
            
            if date_range:
                logger.info("🔄 Falling back to API fetching...")
                return fetch_climate_data_api(
                    date_range=date_range,
                    admin_level=admin_level,
                    source=source
                )
            else:
                raise FileNotFoundError(f"Climate data file not found and no date range provided for API fallback: {file_path}")
        
        # Load and validate file data
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            validated_df = validate_climate_data(df)
            
            logger.info(f"✅ Successfully loaded climate data from file: {len(validated_df)} rows")
            return validated_df
            
        except Exception as e:
            logger.error(f"❌ Failed to load climate data from file: {e}")
            
            if date_range:
                logger.info("🔄 Falling back to API fetching...")
                return fetch_climate_data_api(
                    date_range=date_range,
                    admin_level=admin_level,
                    source=source
                )
            else:
                raise

def validate_climate_health_compatibility(health_df: pd.DataFrame, 
                                        climate_df: pd.DataFrame) -> Dict[str, any]:
    """
    BOTH COMPONENTS: Validate compatibility between health and climate data before merging
    
    PURPOSE: Ensure successful integration for both analytical objectives
    - Component 1: Validate data alignment for climate sensitivity analysis
    - Component 2: Ensure temporal consistency for forecasting models
    
    Returns comprehensive validation report with recommendations.
    """
    if not CLIMATE_MODULE_AVAILABLE:
        logger.warning("⚠️ Climate validation module not available, using basic validation")
        return {"status": "basic_validation", "issues": [], "recommendations": []}
    
    logger.info("🔍 Validating climate-health data compatibility...")
    
    try:
        # Use the comprehensive validation from climate_data module
        validation_report = validate_climate_data_for_merge(climate_df, health_df)
        
        # Add additional checks specific to this project
        if len(validation_report.get('issues', [])) == 0:
            logger.info("✅ Climate-health data compatibility validation passed")
        else:
            logger.warning(f"⚠️ Found {len(validation_report['issues'])} compatibility issues")
            for issue in validation_report['issues']:
                logger.warning(f"  - {issue}")
        
        if validation_report.get('recommendations'):
            logger.info("💡 Recommendations:")
            for rec in validation_report['recommendations']:
                logger.info(f"  - {rec}")
        
        return validation_report
        
    except Exception as e:
        logger.error(f"❌ Climate-health compatibility validation failed: {e}")
        return {"status": "validation_failed", "error": str(e), "issues": ["Validation failed"], "recommendations": []}

def _format_climate_for_merge(climate_data: pd.DataFrame, admin_level: str) -> pd.DataFrame:
    """
    BOTH COMPONENTS: Format climate data from API for compatibility with existing merge functions
    
    PURPOSE: Bridge between new climate API and existing data processing pipeline
    """
    logger.info("🔄 Formatting climate data for merge compatibility...")
    
    # Create a copy to avoid modifying original data
    formatted_data = climate_data.copy()
    
    # Map location fields to admin1 field expected by existing merge function
    if admin_level == "country":
        # For country-level data, use location_name as admin1
        formatted_data['admin1'] = formatted_data['location_name']
    elif admin_level in ["governorate", "district"]:
        # For subnational data, use admin1 field or location_name
        if 'admin1' not in formatted_data.columns:
            formatted_data['admin1'] = formatted_data['location_name']
    
    # Ensure required columns exist with proper names
    column_mapping = {
        'temperature_mean': 'temp_mean',
        'temperature_min': 'temp_min', 
        'temperature_max': 'temp_max'
        # Add more mappings as needed
    }
    
    for new_col, old_col in column_mapping.items():
        if new_col in formatted_data.columns and old_col not in formatted_data.columns:
            formatted_data[old_col] = formatted_data[new_col]
    
    # Ensure we have the minimum required columns for existing validation
    required_cols = ['date', 'admin1', 'temp_max', 'temp_min', 'precipitation']
    
    for col in required_cols:
        if col not in formatted_data.columns:
            if col.startswith('temp_'):
                formatted_data[col] = np.nan
                logger.warning(f"⚠️ Missing temperature column {col}, filled with NaN")
            elif col == 'precipitation':
                formatted_data[col] = 0.0
                logger.warning(f"⚠️ Missing precipitation column, filled with 0.0")
    
    logger.info(f"✅ Climate data formatted: {len(formatted_data)} rows, {len(formatted_data.columns)} columns")
    
    return formatted_data

def _generate_fallback_climate_data(date_range: Tuple[str, str], 
                                  locations: List[str], 
                                  admin_level: str) -> pd.DataFrame:
    """
    BOTH COMPONENTS: Generate fallback demo climate data when API fails
    
    PURPOSE: Ensure analysis can continue even if external APIs are unavailable
    """
    logger.info("🎲 Generating fallback demo climate data...")
    
    if not CLIMATE_MODULE_AVAILABLE:
        # Very basic fallback if climate module not available
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        demo_data = []
        for location in locations:
            for date in dates:
                demo_data.append({
                    'date': date,
                    'admin1': location,
                    'temp_max': 25 + np.random.normal(0, 5),
                    'temp_min': 15 + np.random.normal(0, 3),
                    'precipitation': max(0, np.random.exponential(2))
                })
        
        df = pd.DataFrame(demo_data)
        df['temp_mean'] = (df['temp_max'] + df['temp_min']) / 2
        
        return df
    
    else:
        # Use the sophisticated demo data generation from climate module
        try:
            location_manager = LocationManager()
            location_objects = []
            
            for loc_name in locations:
                # Try to get existing location, or create a simple one
                loc_obj = location_manager.get_location(loc_name)
                if not loc_obj:
                    # Create a simple location (default to Syria coordinates)
                    loc_obj = Location(loc_name, 33.5138, 36.2765, admin_level)
                location_objects.append(loc_obj)
            
            # Generate demo data using climate module
            climate_data = fetch_multi_location_climate(
                locations=location_objects,
                date_range=date_range,
                source="demo"
            )
            
            return _format_climate_for_merge(climate_data, admin_level)
            
        except Exception as e:
            logger.error(f"❌ Even fallback demo data generation failed: {e}")
            raise

def standardize_geographic_names(health_df: pd.DataFrame, climate_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize geographic names between health and climate datasets
    
    Args:
        health_df: Health consultation dataframe
        climate_df: Climate dataframe
        
    Returns:
        Tuple of standardized dataframes
    """
    logger.info("Standardizing geographic names...")
    
    # Get unique admin1 values from both datasets
    health_admin1 = set(health_df['admin1'].unique())
    climate_admin1 = set(climate_df['admin1'].unique())
    
    logger.info(f"Health data admin1 regions: {len(health_admin1)}")
    logger.info(f"Climate data admin1 regions: {len(climate_admin1)}")
    
    # Find common regions
    common_admin1 = health_admin1.intersection(climate_admin1)
    logger.info(f"Common admin1 regions: {len(common_admin1)}")
    
    # Regions only in health data
    health_only = health_admin1 - climate_admin1
    if health_only:
        logger.warning(f"Admin1 regions only in health data: {health_only}")
    
    # Regions only in climate data
    climate_only = climate_admin1 - health_admin1
    if climate_only:
        logger.warning(f"Admin1 regions only in climate data: {climate_only}")
    
    # Filter to common regions for analysis
    health_df_filtered = health_df[health_df['admin1'].isin(common_admin1)].copy()
    climate_df_filtered = climate_df[climate_df['admin1'].isin(common_admin1)].copy()
    
    logger.info(f"✓ Geographic standardization complete")
    logger.info(f"✓ Health consultations in common regions: {len(health_df_filtered):,}")
    logger.info(f"✓ Climate observations in common regions: {len(climate_df_filtered):,}")
    
    return health_df_filtered, climate_df_filtered

def merge_health_climate(health_df: pd.DataFrame, climate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge health consultation and climate data by date and admin1
    
    Args:
        health_df: Validated health consultation dataframe
        climate_df: Validated climate dataframe
        
    Returns:
        Merged dataframe with health and climate variables
    """
    logger.info("Merging health and climate data...")
    
    # Ensure common geographic names
    health_df, climate_df = standardize_geographic_names(health_df, climate_df)
    
    # Ensure date columns have compatible types (keep as datetime for .dt accessor compatibility)
    health_df['date'] = pd.to_datetime(health_df['date'])
    climate_df['date'] = pd.to_datetime(climate_df['date'])
    
    # Perform the merge
    merged_df = health_df.merge(
        climate_df,
        on=['date', 'admin1'],
        how='inner'
    )
    
    # Calculate merge statistics
    health_records = len(health_df)
    climate_records = len(climate_df)
    merged_records = len(merged_df)
    
    logger.info(f"✓ Merge completed:")
    logger.info(f"  - Health consultations: {health_records:,}")
    logger.info(f"  - Climate observations: {climate_records:,}")
    logger.info(f"  - Merged records: {merged_records:,}")
    logger.info(f"  - Match rate: {merged_records/health_records*100:.1f}% of health consultations")
    
    # Date range of merged data
    logger.info(f"✓ Merged date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    logger.info(f"✓ Admin1 regions in merged data: {merged_df['admin1'].nunique()}")
    
    return merged_df

def load_and_merge_data(config_path: str = 'config.yaml', 
                        data_mode: Optional[str] = None,
                        sample_size: Optional[int] = None,
                        climate_api_source: Optional[str] = None,
                        climate_date_range: Optional[Tuple[str, str]] = None,
                        use_climate_api: bool = False) -> pd.DataFrame:
    """
    Main data loading and merging function with enhanced climate data integration.
    
    Args:
        config_path: Path to configuration file
        data_mode: Force specific data mode ('syria', 'synthetic', 'private'). 
                  If None, auto-detects based on available data.
        sample_size: Optional sample size for Syria data (for testing)
        climate_api_source: Climate data source ('open_meteo', 'nasa_power', 'demo')
        climate_date_range: Date range for API fetching (start_date, end_date)
        use_climate_api: Whether to fetch climate data from API instead of files
        
    Returns:
        Complete merged dataset ready for feature engineering with enhanced climate data
    """
    logger.info("🔄 Starting data loading and merging process...")
    
    # Load configuration
    config = load_config()
    
    # Auto-detect data mode if not specified, but respect USE_SYNTHETIC setting
    if data_mode is None:
        # Check if synthetic mode is explicitly requested
        use_synthetic = config.get('data', {}).get('use_synthetic', True)
        if use_synthetic:
            data_mode = 'synthetic'
            logger.info(f"🔍 Using synthetic data mode (USE_SYNTHETIC=true)")
        else:
            data_mode = detect_data_mode()
            logger.info(f"🔍 Auto-detected data mode: {data_mode}")
    else:
        logger.info(f"📊 Using specified data mode: {data_mode}")
    
    # Route to appropriate data loader
    if data_mode == 'internal':
        logger.info("📊 Using internal data mode")
        return load_and_merge_internal_data(config, sample_size)
    
    elif data_mode == 'synthetic':
        logger.info("📊 Using synthetic data mode") 
        config['data']['use_synthetic'] = True
        
    else:  # private/legacy mode
        logger.info("📊 Using private/legacy data mode")
        config['data']['use_synthetic'] = False
    
    # Legacy synthetic/private data loading path
    paths = get_data_paths(config)
    data_type = "synthetic" if config['data']['use_synthetic'] else "private"
    
    # Check file existence
    for data_source, path in paths.items():
        if not Path(path).exists():
            if config['data']['use_synthetic']:
                raise FileNotFoundError(
                    f"Synthetic {data_source} data file not found: {path}\n"
                    f"Run data/synthetic/generate_synthetic.py first"
                )
            else:
                raise FileNotFoundError(
                    f"Private {data_source} data file not found: {path}\n"
                    f"Please place your data files in data/raw/"
                )
    
    # Load health data
    logger.info(f"📈 Loading health data from {paths['health']}")
    health_df = pd.read_csv(paths['health'])
    health_df = validate_health_data(health_df)
    
    # Load climate data  
    logger.info(f"🌡️ Loading climate data from {paths['climate']}")
    climate_df = pd.read_csv(paths['climate'])
    climate_df = validate_climate_data(climate_df)
    
    # Merge datasets
    logger.info("🔗 Merging health and climate datasets...")
    merged_df = merge_health_climate(health_df, climate_df)
    
    # Save processed data
    processed_dir = Path(config['data']['processed_data_dir'])
    processed_dir.mkdir(exist_ok=True)
    
    output_path = processed_dir / 'merged_dataset.csv'
    merged_df.to_csv(output_path, index=False)
    logger.info(f"💾 Merged data saved to {output_path}")
    
    logger.info("✅ Data loading and merging completed successfully!")
    
    return merged_df

def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive summary statistics of the merged dataset
    
    Args:
        df: Merged health-climate dataframe
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df['date'].min(),
            'end': df['date'].max(),
            'days': (df['date'].max() - df['date'].min()).days
        },
        'geographic_coverage': {
            'admin1_regions': df['admin1'].nunique(),
            'regions_list': sorted(df['admin1'].unique().tolist())
        },
        'health_data': {
            'unique_morbidities': df['morbidity'].nunique() if 'morbidity' in df.columns else None,
            'unique_imc_categories': df['canonical_disease_imc'].nunique() if 'canonical_disease_imc' in df.columns else None,
            'icd11_coverage': df['icd11_title'].notna().sum() if 'icd11_title' in df.columns else None,
            'sex_distribution': df['sex'].value_counts().to_dict() if 'sex' in df.columns else None,
            'age_groups': sorted(df['age_group'].unique().tolist()) if 'age_group' in df.columns else None
        },
        'climate_data': {
            'temp_stats': {
                'min': df['temp_min'].min(),
                'max': df['temp_max'].max(), 
                'mean': df['temp_mean'].mean()
            },
            'precipitation_stats': {
                'min': df['precipitation'].min(),
                'max': df['precipitation'].max(),
                'mean': df['precipitation'].mean()
            }
        }
    }
    
    return summary


# ================================================================================
# INTERNAL DATA PROCESSING FUNCTIONS
# ================================================================================

def load_syria_data(config: Dict, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load and process Syria internal health data (events.csv).
    
    Args:
        config: Configuration dictionary
        sample_size: Optional limit on number of records to load for testing
        
    Returns:
        Processed DataFrame with standardized columns
    """
    logger.info("📥 Loading Syria internal health data...")
    
    # Define file path from environment variable or default
    events_path = Path(os.getenv('HEALTH_DATA_PATH', 'data/internal/raw_dat/events.csv'))
    
    if not events_path.exists():
        raise FileNotFoundError(f"Syria events file not found at {events_path}")
    
    logger.info(f"📂 Loading data from {events_path}")
    
    try:
        # Load data with explicit encoding handling
        df = pd.read_csv(events_path, encoding='utf-8')
        
        # Sample data if requested
        if sample_size and len(df) > sample_size:
            logger.info(f"📊 Sampling {sample_size} records from {len(df)} total records")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
        logger.info(f"📊 Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Process and standardize the data
        df = process_syria_data(df)
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading Syria data: {e}")
        raise


def process_syria_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and standardize Syria health data.
    
    Args:
        df: Raw Syria events DataFrame
        
    Returns:
        Processed DataFrame with standardized columns
    """
    logger.info("🔄 Processing Syria health data...")
    
    processed_df = df.copy()
    
    # ---- Column Standardization ----
    # Map Syria column names to standard names
    column_mapping = {
        'Organisation unit name': 'facility_name',
        'Date of visit': 'date',
        'Admission Date': 'admission_date',
        'Age Group (0-59, 5-17, 18-49, 50 and Above) - SY': 'age_group_original',
        'Age group 1 (0-5,6-17,18-59,60+) - SY': 'age_group_1',
        'Age group 2 (0-11m, 1-4, 5-14,15-49,50-60,60+) - SY': 'age_group_2', 
        'Age group 3 (< 5 , 5-14,15-18,19-49,50+) - SY': 'age_group_3',
        'Gender': 'sex',
        'Morbidity Classification - SY': 'morbidity',
        'Patient Presents With Disability (Y/N)': 'has_disability',
        'Type of Case (Trauma/Non-trauma)': 'case_type',
        'Visit Number - SYR': 'visit_number',
        'Visit Type': 'visit_type'
    }
    
    # Rename columns
    for old_name, new_name in column_mapping.items():
        if old_name in processed_df.columns:
            processed_df = processed_df.rename(columns={old_name: new_name})
    
    # ---- Date Processing ----
    processed_df = process_syria_dates(processed_df)
    
    # ---- Age Group Harmonization ----
    processed_df = harmonize_age_groups(processed_df)
    
    # ---- Geographic Processing ----
    processed_df = process_geography(processed_df)
    
    # ---- Disease Classification ----
    # Use cached taxonomy mapping to avoid expensive re-computation
    try:
        from taxonomy_cache import apply_cached_taxonomy_mapping
        processed_df = apply_cached_taxonomy_mapping(processed_df, country='base')
    except ImportError:
        logger.warning("⚠️ Taxonomy cache not available, using direct mapping")
        processed_df = classify_diseases_dual_taxonomy(processed_df, country='base')
    
    # ---- Demographic Standardization ----
    processed_df = standardize_demographics(processed_df)
    
    # ---- Create Consultation Count ----
    processed_df['consultation_count'] = 1  # Each row is one consultation
    
    logger.info(f"✅ Syria data processing completed: {len(processed_df)} records processed")
    
    return processed_df


def process_syria_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Process and standardize date columns."""
    logger.info("📅 Processing date columns...")
    
    processed_df = df.copy()
    
    # Process main date column
    if 'date' in processed_df.columns:
        processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
        
        # Use admission date as fallback if visit date is missing
        if 'admission_date' in processed_df.columns:
            admission_dates = pd.to_datetime(processed_df['admission_date'], errors='coerce')
            processed_df['date'] = processed_df['date'].fillna(admission_dates)
    
    # Remove records with invalid dates
    initial_count = len(processed_df)
    processed_df = processed_df.dropna(subset=['date'])
    removed_count = initial_count - len(processed_df)
    
    if removed_count > 0:
        logger.warning(f"⚠️ Removed {removed_count} records with invalid dates")
    
    # Extract date components
    processed_df['year'] = processed_df['date'].dt.year
    processed_df['month'] = processed_df['date'].dt.month
    processed_df['day'] = processed_df['date'].dt.day
    processed_df['dayofweek'] = processed_df['date'].dt.dayofweek
    processed_df['dayofyear'] = processed_df['date'].dt.dayofyear
    
    logger.info(f"✅ Date processing completed. Date range: {processed_df['date'].min()} to {processed_df['date'].max()}")
    
    return processed_df


def harmonize_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize multiple age group systems into standardized categories.
    
    Syria data has 4 different age grouping systems - we'll standardize to the
    IMC standard: 0-5, 6-17, 18-59, 60+
    """
    logger.info("👥 Harmonizing age groups...")
    
    processed_df = df.copy()
    
    # Use age_group_1 as primary (0-5,6-17,18-59,60+)
    if 'age_group_1' in processed_df.columns:
        processed_df['age_group'] = processed_df['age_group_1'].str.strip()
        
        # Standardize age group names
        age_group_mapping = {
            '0-5 y': '0-5',
            '6-17 y': '6-17', 
            '18-59 y': '18-59',
            '≥60 y': '60+',
            '60+ y': '60+',
            # Handle variations
            '0-5': '0-5',
            '6-17': '6-17',
            '18-59': '18-59',
            '60+': '60+'
        }
        
        processed_df['age_group'] = processed_df['age_group'].map(age_group_mapping)
        
        # Fill missing values using other age group columns as fallbacks
        mask_missing = processed_df['age_group'].isna()
        
        if 'age_group_2' in processed_df.columns and mask_missing.sum() > 0:
            # Map age_group_2 to standard categories
            age2_mapping = {
                '0-11m': '0-5',
                '1-4': '0-5', 
                '5-14': '6-17',
                '15-49': '18-59',
                '50-60': '18-59',  # Could be 60+ depending on exact age
                '60+': '60+'
            }
            
            fallback_ages = processed_df.loc[mask_missing, 'age_group_2'].map(age2_mapping)
            processed_df.loc[mask_missing, 'age_group'] = fallback_ages
    
    # Create binary age indicators for ML features
    processed_df['is_young_child'] = (processed_df['age_group'] == '0-5').astype(int)
    processed_df['is_child'] = (processed_df['age_group'] == '6-17').astype(int) 
    processed_df['is_adult'] = (processed_df['age_group'] == '18-59').astype(int)
    processed_df['is_elderly'] = (processed_df['age_group'] == '60+').astype(int)
    
    # Vulnerable age groups (standard epidemiological definition)
    processed_df['is_vulnerable_age'] = (
        (processed_df['age_group'].isin(['0-5', '60+'])).astype(int)
    )
    
    age_dist = processed_df['age_group'].value_counts()
    logger.info(f"✅ Age group harmonization completed. Distribution: {age_dist.to_dict()}")
    
    return processed_df


def process_geography(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic geographic processing and standardization.
    Maps facilities to administrative levels using configurable mappings.
    """
    logger.info("🗺️ Processing geographic information...")
    
    processed_df = df.copy()
    
    # Initialize admin columns if not present
    admin_cols = ['admin0', 'admin1', 'admin2', 'admin3']
    for col in admin_cols:
        if col not in processed_df.columns:
            processed_df[col] = 'Unknown'
    
    # Auto-detect country from existing admin0 column or set default
    if 'admin0' in processed_df.columns and not processed_df['admin0'].isna().all():
        detected_country = processed_df['admin0'].mode().iloc[0] if len(processed_df['admin0'].mode()) > 0 else 'Unknown'
    else:
        # Try to infer from facility names or admin1 patterns
        detected_country = infer_country_from_data(processed_df)
        processed_df['admin0'] = detected_country
    
    logger.info(f"🔍 Processing for country/region: {detected_country}")
    
    # Apply generic geographic standardization
    processed_df = standardize_geographic_names_single(processed_df)
    
    # Apply facility-to-admin mapping if facility data exists
    if 'facility_name' in processed_df.columns:
        processed_df = map_facilities_to_admin(processed_df)
    
    # Clean and standardize facility names
    processed_df = clean_facility_names(processed_df)
    
    # Log geographic distribution
    log_geographic_summary(processed_df)
    
    return processed_df

def infer_country_from_data(df: pd.DataFrame) -> str:
    """
    Infer country/region from available data patterns (generic approach).
    """
    # Check admin1 patterns first
    if 'admin1' in df.columns:
        unique_admin1 = df['admin1'].dropna().unique()
        if len(unique_admin1) > 0:
            # Return the most common pattern or 'Multiple Regions' if diverse
            return 'Region' if len(unique_admin1) > 1 else str(unique_admin1[0])
    
    # Check facility name patterns
    if 'facility_name' in df.columns:
        facilities = df['facility_name'].dropna().unique()
        if len(facilities) > 0:
            return 'Health System'  # Generic identifier
    
    return 'Unknown'

def standardize_geographic_names_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic geographic name standardization for single dataframe.
    """
    logger.info("🗺️ Standardizing geographic names...")
    
    # Basic cleaning for all admin levels
    admin_cols = ['admin1', 'admin2', 'admin3']
    for col in admin_cols:
        if col in df.columns:
            # Clean and standardize formatting
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
                .str.replace(r'^(state|province|governorate|district)\s+', '', case=False, regex=True)  # Remove prefixes
                .str.replace(r'\s+(state|province|governorate|district)$', '', case=False, regex=True)  # Remove suffixes
                .str.title()  # Title case
            )
    
    return df

def map_facilities_to_admin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map facilities to administrative levels using pattern matching.
    """
    logger.info("🏥 Mapping facilities to administrative levels...")
    
    # Try to extract admin info from facility names using generic patterns
    mask_unknown = df['admin1'] == 'Unknown'
    
    if mask_unknown.sum() > 0:
        logger.info(f"🔍 Applying pattern matching for {mask_unknown.sum()} unmapped facilities...")
        
        # Extract potential admin names from facility names
        # Look for common patterns like "City Hospital", "Region Health Center", etc.
        facility_names = df.loc[mask_unknown, 'facility_name'].dropna()
        
        for idx, facility_name in facility_names.items():
            if pd.isna(facility_name):
                continue
                
            facility_str = str(facility_name).strip()
            
            # Try to extract location from facility name patterns
            # Pattern 1: "Location Hospital/Clinic/Center"
            location_patterns = [
                r'^([A-Za-z\s-]+?)\s+(Hospital|Clinic|Center|Health|Medical)',
                r'([A-Za-z\s-]+?)\s+(General|National|Regional|Central)\s+Hospital',
                r'([A-Za-z\s-]+?)\s+(PHC|HC)$'
            ]
            
            for pattern in location_patterns:
                import re
                match = re.search(pattern, facility_str, re.IGNORECASE)
                if match:
                    potential_location = match.group(1).strip().title()
                    if len(potential_location) > 2:  # Avoid very short matches
                        df.loc[idx, 'admin1'] = potential_location
                        break
    
    return df

def clean_facility_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize facility names.
    """
    if 'facility_name' not in df.columns:
        return df
    
    logger.info("🏥 Cleaning facility names...")
    
    # Create cleaned facility name column
    df['facility_name_clean'] = (
        df['facility_name']
        .astype(str)
        .str.strip()
        .str.upper()
        # Remove common prefixes/suffixes
        .str.replace(r'^(PHC|HEALTH CENTER|HC|CLINIC)\s*', '', regex=True)
        .str.replace(r'\s*(PHC|HEALTH CENTER|HC|CLINIC)$', '', regex=True)
        # Normalize whitespace
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    
    return df

def log_geographic_summary(df: pd.DataFrame):
    """
    Log summary of geographic processing results.
    """
    geo_vars = ['admin0', 'admin1', 'admin2', 'admin3']
    available_geo_vars = [var for var in geo_vars if var in df.columns]
    
    if available_geo_vars:
        logger.info(f"🗺️ Geographic variables processed: {', '.join(available_geo_vars)}")
        
        for var in available_geo_vars:
            unique_count = df[var].nunique()
            non_unknown = (df[var] != 'Unknown').sum()
            coverage = (non_unknown / len(df)) * 100 if len(df) > 0 else 0
            logger.info(f"   • {var}: {unique_count} unique values, {coverage:.1f}% coverage")
        
        # Show top admin1 regions if available
        if 'admin1' in df.columns and (df['admin1'] != 'Unknown').any():
            admin1_dist = df[df['admin1'] != 'Unknown']['admin1'].value_counts().head(5)
            logger.info(f"   • Top admin1 regions: {dict(admin1_dist)}")
    
    logger.info("✅ Geographic processing completed")


def classify_diseases_dual_taxonomy(df: pd.DataFrame, country: str = 'base') -> pd.DataFrame:
    """
    Apply dual taxonomy mapping: IMC custom taxonomy (base.yaml + country.yaml) + ICD-11 API taxonomy.
    Maps from morbidity -> canonical_disease_imc (IMC) + icd11_* variables (ICD-11).
    """
    logger.info(f"🏷️ Applying dual taxonomy classification for {country}...")
    logger.info("📋 Taxonomy 1: IMC custom taxonomy (base.yaml + country-specific)")
    logger.info("📋 Taxonomy 2: ICD-11 API taxonomy")
    
    processed_df = df.copy()
    
    if 'morbidity' not in processed_df.columns:
        logger.warning("⚠️ No morbidity column found - skipping disease classification")
        return processed_df
    
    try:
        # TAXONOMY 1: IMC Custom Taxonomy (base.yaml + country.yaml)
        logger.info("🎯 Step 1: Applying IMC custom taxonomy mapping...")
        if TaxonomyProcessor:
            # Use advanced taxonomy processor if available
            taxonomy_processor = TaxonomyProcessor()
            processed_df = taxonomy_processor.classify_diseases_dataframe(
                processed_df, 
                disease_col='morbidity',
                which=country  # Use country-specific + base taxonomy
            )
        else:
            # Use fallback IMC taxonomy mapping
            processed_df = apply_imc_taxonomy_fallback(processed_df, country)
        
        # TAXONOMY 2: ICD-11 API Taxonomy
        logger.info("🌐 Step 2: Applying ICD-11 API taxonomy mapping...")
        processed_df = apply_icd11_taxonomy_mapping(processed_df)
        
        # Add epidemiological feature flags based on both taxonomies
        logger.info("🏥 Step 3: Adding epidemiological feature flags...")
        processed_df = add_epidemiological_features_dual(processed_df)
        
        # Log classification results for both taxonomies
        log_dual_taxonomy_results(processed_df)
        
    except Exception as e:
        logger.error(f"❌ Error in dual taxonomy classification: {e}")
        logger.info("🔄 Applying minimal classification fallback...")
        processed_df = apply_minimal_dual_taxonomy(processed_df)
    
    return processed_df

def apply_imc_taxonomy_fallback(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Fallback IMC taxonomy mapping when TaxonomyProcessor is not available.
    Loads base.yaml + country.yaml mappings.
    """
    logger.info(f"🔧 Using fallback IMC taxonomy for {country}...")
    
    # Load base taxonomy mappings
    base_mappings = load_base_taxonomy_mappings()
    country_mappings = load_country_taxonomy_mappings(country)
    
    # Merge base + country mappings (country overrides base)
    combined_synonyms = {**base_mappings.get('synonyms', {}), **country_mappings.get('synonyms', {})}
    combined_categories = {**base_mappings.get('categories', {}), **country_mappings.get('categories', {})}
    
    # Apply morbidity -> canonical_disease_imc mapping
    df['canonical_disease_imc'] = df['morbidity'].map(combined_synonyms).fillna(df['morbidity'])
    
    # Apply canonical_disease_imc -> category mapping
    df['category_canonical_disease_imc'] = df['canonical_disease_imc'].map(combined_categories).fillna('Uncategorized')
    
    # Add backward compatibility
    df['canonical_disease'] = df['canonical_disease_imc']
    
    mapped_count = (df['canonical_disease_imc'] != df['morbidity']).sum()
    logger.info(f"✅ IMC taxonomy: {mapped_count}/{len(df)} diseases mapped to canonical forms")
    
    return df

def load_base_taxonomy_mappings() -> dict:
    """
    Load base taxonomy mappings from base.yaml or fallback.
    """
    try:
        # Try to load base.yaml
        import yaml
        base_path = Path('taxonomy/base.yaml')
        if base_path.exists():
            with open(base_path, 'r') as f:
                base_taxonomy = yaml.safe_load(f)
                logger.info(f"📂 Loaded base taxonomy from {base_path}")
                return base_taxonomy
    except Exception as e:
        logger.warning(f"⚠️ Could not load base.yaml: {e}")
    
    # Fallback base mappings
    logger.info("🔧 Using fallback base taxonomy mappings")
    return {
        'synonyms': {
            'ARI': 'Acute respiratory infection',
            'Diarrhoea': 'Diarrhea', 
            'Diarrhea': 'Diarrhea',
            'AWD': 'Acute Watery Diarrhea',
            'Acute watery diarrhea': 'Acute Watery Diarrhea',
            'High blood pressure': 'Hypertension',
            'Diabetes mellitus': 'Diabetes',
            'SAM': 'Severe Acute Malnutrition',
            'Severe acute malnutrition': 'Severe Acute Malnutrition',
            'Malnutrition': 'Malnutrition',
            'Trauma': 'Injury',
            'TB': 'Tuberculosis',
            'UTI': 'Urinary Tract Infection'
        },
        'categories': {
            'Acute respiratory infection': 'Respiratory',
            'Diarrhea': 'Gastrointestinal', 
            'Acute Watery Diarrhea': 'Gastrointestinal',
            'Cholera': 'Gastrointestinal',
            'Fever': 'General',
            'Malaria': 'Vector-borne',
            'Dengue': 'Vector-borne',
            'Hypertension': 'Cardiovascular',
            'Diabetes': 'Metabolic',
            'Malnutrition': 'Nutrition',
            'Severe Acute Malnutrition': 'Nutrition',
            'Injury': 'Trauma',
            'Tuberculosis': 'Respiratory',
            'Measles': 'Vaccine-preventable',
            'Polio': 'Vaccine-preventable',
            'Meningitis': 'Neurological',
            'Pneumonia': 'Respiratory'
        }
    }

def load_country_taxonomy_mappings(country: str) -> dict:
    """
    Load country-specific taxonomy mappings from country.yaml files.
    """
    try:
        import yaml
        country_path = Path(f'taxonomy/{country}.yaml')
        if country_path.exists():
            with open(country_path, 'r') as f:
                country_taxonomy = yaml.safe_load(f)
                logger.info(f"📂 Loaded {country} taxonomy from {country_path}")
                return country_taxonomy
    except Exception as e:
        logger.warning(f"⚠️ Could not load {country}.yaml: {e}")
    
    # Return empty mappings if country file not found
    return {'synonyms': {}, 'categories': {}}

def apply_icd11_taxonomy_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ICD-11 taxonomy mapping from API data.
    Maps from original morbidity -> icd11_code, icd11_title, icd11_category.
    """
    logger.info("🌐 Applying ICD-11 API taxonomy mapping...")
    
    # Initialize ICD-11 columns
    df['icd11_code'] = 'Unclassified'
    df['icd11_title'] = 'Unclassified' 
    df['icd11_category'] = 'Unclassified'
    df['confidence'] = 'none'
    
    try:
        # Try to load ICD-11 mappings from API data file
        icd11_mappings = load_icd11_api_mappings()
        
        if icd11_mappings:
            # Apply ICD-11 mappings to original morbidity values
            mapped_count = 0
            for i, morbidity in enumerate(df['morbidity']):
                if pd.notna(morbidity) and str(morbidity) in icd11_mappings:
                    mapping = icd11_mappings[str(morbidity)]
                    df.loc[i, 'icd11_code'] = mapping.get('icd11_code', 'Unclassified')
                    df.loc[i, 'icd11_title'] = mapping.get('icd11_title', 'Unclassified')
                    df.loc[i, 'icd11_category'] = mapping.get('icd11_category', 'Unclassified')
                    df.loc[i, 'confidence'] = mapping.get('confidence', 'medium')
                    mapped_count += 1
            
            logger.info(f"✅ ICD-11 API taxonomy: {mapped_count}/{len(df)} diseases mapped")
        else:
            logger.warning("⚠️ No ICD-11 API mappings available, using basic mappings")
            df = apply_basic_icd11_mappings(df)
            
    except Exception as e:
        logger.error(f"❌ Error in ICD-11 mapping: {e}")
        df = apply_basic_icd11_mappings(df)
    
    return df

def load_icd11_api_mappings() -> dict:
    """
    Load ICD-11 API mappings from disease_mappings.yml or API cache.
    """
    try:
        import yaml
        icd11_path = Path('taxonomy/icd11/disease_mappings.yml')
        if icd11_path.exists():
            with open(icd11_path, 'r') as f:
                icd11_data = yaml.safe_load(f)
                logger.info(f"📂 Loaded ICD-11 mappings from {icd11_path}")
                
                # Extract mappings from all confidence levels
                all_mappings = {}
                for confidence_level in ['high_confidence', 'medium_confidence', 'needs_review']:
                    if 'mappings' in icd11_data and confidence_level in icd11_data['mappings']:
                        level_mappings = icd11_data['mappings'][confidence_level]
                        for disease, mapping in level_mappings.items():
                            all_mappings[disease] = {**mapping, 'confidence': confidence_level.replace('_', ' ')}
                
                logger.info(f"📊 Loaded {len(all_mappings)} ICD-11 API mappings")
                return all_mappings
    except Exception as e:
        logger.warning(f"⚠️ Could not load ICD-11 API mappings: {e}")
    
    return {}

def apply_basic_icd11_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic ICD-11 mappings as fallback.
    """
    logger.info("🔧 Using basic ICD-11 mappings as fallback")
    
    basic_mappings = {
        'Acute respiratory infection': {'code': 'CA07', 'title': 'Upper respiratory tract infection', 'category': 'Respiratory diseases'},
        'Pneumonia': {'code': 'CA40', 'title': 'Pneumonia', 'category': 'Respiratory diseases'},
        'Diarrhea': {'code': '1A40', 'title': 'Acute diarrhoea', 'category': 'Infectious diseases'},
        'Acute Watery Diarrhea': {'code': '1A40', 'title': 'Acute diarrhoea', 'category': 'Infectious diseases'},
        'Cholera': {'code': '1A00', 'title': 'Cholera', 'category': 'Infectious diseases'},
        'Malaria': {'code': '1F40', 'title': 'Malaria', 'category': 'Infectious diseases'},
        'Dengue': {'code': '1D26', 'title': 'Dengue', 'category': 'Infectious diseases'},
        'Measles': {'code': '1F03', 'title': 'Measles', 'category': 'Infectious diseases'},
        'Tuberculosis': {'code': '1B10', 'title': 'Tuberculosis', 'category': 'Infectious diseases'},
        'Hypertension': {'code': 'BA00', 'title': 'Hypertension', 'category': 'Cardiovascular diseases'},
        'Diabetes': {'code': '5A10', 'title': 'Diabetes mellitus', 'category': 'Metabolic diseases'},
        'Malnutrition': {'code': '5B50', 'title': 'Malnutrition', 'category': 'Nutritional diseases'},
        'Severe Acute Malnutrition': {'code': '5B51', 'title': 'Severe malnutrition', 'category': 'Nutritional diseases'},
        'Injury': {'code': 'QA00', 'title': 'Injury', 'category': 'Injuries'}
    }
    
    # Apply basic mappings to canonical diseases
    mapped_count = 0
    for disease, mapping in basic_mappings.items():
        mask = (df['canonical_disease_imc'] == disease) | (df['morbidity'].str.contains(disease, case=False, na=False))
        if mask.any():
            df.loc[mask, 'icd11_code'] = mapping['code']
            df.loc[mask, 'icd11_title'] = mapping['title']
            df.loc[mask, 'icd11_category'] = mapping['category']
            df.loc[mask, 'confidence'] = 'medium'
            mapped_count += mask.sum()
    
    logger.info(f"✅ Basic ICD-11 mapping: {mapped_count} records mapped")
    return df

def add_epidemiological_features_dual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add epidemiological feature flags based on both IMC taxonomy and ICD-11 data.
    """
    logger.info("🏥 Adding epidemiological features from dual taxonomy...")
    
    # Initialize all feature flags
    feature_flags = {
        'vaccine_preventable': False,
        'climate_sensitive': False,
        'outbreak_prone': False,
        'trauma_related': False,
        'amr_relevant': False,
        'epidemic_prone': False,  # Alias for outbreak_prone
        'pediatric_case': False,
        'elderly_care_case': False,
        'pregnancy_related': False,
        'maternal_health_case': False,
        'malnutrition_complication': False,
        'chronic_condition': False,
        'ncd_related': False,
        'emergency_syndrome_case': False
    }
    
    for flag in feature_flags:
        df[flag] = feature_flags[flag]
    
    # Define feature mappings based on both canonical diseases and ICD-11 categories
    feature_mappings = {
        'vaccine_preventable': {
            'diseases': ['Measles', 'Polio', 'Diphtheria', 'Pertussis', 'Mumps', 'Rubella', 'Hepatitis A', 'Hepatitis B', 'Yellow Fever'],
            'icd11_codes': ['1F03', '1C80', '1C15', '1C16', '1F04', '1F05', '1E50.0', '1E51.0']
        },
        'climate_sensitive': {
            'diseases': ['Malaria', 'Dengue', 'Chikungunya', 'Zika', 'Cholera', 'Diarrhea', 'Acute Watery Diarrhea'],
            'icd11_codes': ['1F40', '1D26', '1D27', '1D28', '1A00', '1A40']
        },
        'outbreak_prone': {
            'diseases': ['Cholera', 'Measles', 'Meningitis', 'Yellow Fever', 'Viral Hemorrhagic Fever', 'Ebola', 'Plague', 'Anthrax'],
            'icd11_codes': ['1A00', '1F03', '8B00', '1D42', '1D44', '1C12', '1C11']
        },
        'trauma_related': {
            'diseases': ['Injury', 'Trauma', 'Fracture', 'Burn', 'Wound'],
            'icd11_codes': ['QA00', 'QB00', 'QC00', 'QD00']
        },
        'amr_relevant': {
            'diseases': ['Tuberculosis', 'Pneumonia', 'Typhoid', 'Sepsis', 'Bacterial infection'],
            'icd11_codes': ['1B10', 'CA40', '1A02', '1G40']
        }
    }
    
    # Apply feature flags based on canonical diseases and ICD-11 codes
    for feature, mappings in feature_mappings.items():
        # Check canonical diseases
        for disease in mappings['diseases']:
            mask = df['canonical_disease_imc'].str.contains(disease, case=False, na=False)
            df.loc[mask, feature] = True
        
        # Check ICD-11 codes
        for code in mappings['icd11_codes']:
            mask = df['icd11_code'] == code
            df.loc[mask, feature] = True
    
    # Set epidemic_prone as alias for outbreak_prone
    df['epidemic_prone'] = df['outbreak_prone']
    
    # Add demographic risk flags based on age groups if available
    if 'age_group' in df.columns:
        df['pediatric_case'] = df['age_group'].isin(['0-5', '6-17'])
        df['elderly_care_case'] = df['age_group'].isin(['60+'])
    
    # Disease-based risk classifications
    malnutrition_patterns = ['malnutrition', 'underweight', 'stunting', 'wasting', 'marasmus', 'kwashiorkor']
    chronic_patterns = ['diabetes', 'hypertension', 'cardiovascular', 'asthma', 'copd', 'chronic', 'cancer', 'epilepsy']
    emergency_patterns = ['severe', 'critical', 'emergency', 'shock', 'sepsis', 'coma', 'respiratory failure']
    
    for pattern in malnutrition_patterns:
        mask = df['canonical_disease_imc'].str.contains(pattern, case=False, na=False)
        df.loc[mask, 'malnutrition_complication'] = True
    
    for pattern in chronic_patterns:
        mask = df['canonical_disease_imc'].str.contains(pattern, case=False, na=False)
        df.loc[mask, 'chronic_condition'] = True
        df.loc[mask, 'ncd_related'] = True
    
    for pattern in emergency_patterns:
        mask = df['canonical_disease_imc'].str.contains(pattern, case=False, na=False)
        df.loc[mask, 'emergency_syndrome_case'] = True
    
    # Log feature flag results
    flag_counts = {flag: df[flag].sum() for flag in feature_flags.keys()}
    logger.info("🏷️ Epidemiological feature flags applied:")
    for flag, count in flag_counts.items():
        if count > 0:
            logger.info(f"   • {flag}: {count} cases")
    
    return df

def log_dual_taxonomy_results(df: pd.DataFrame):
    """
    Log results from dual taxonomy classification.
    """
    # IMC Taxonomy results
    unique_canonicals = df['canonical_disease_imc'].nunique()
    unique_categories = df['category_canonical_disease_imc'].nunique()
    imc_mapped = (df['canonical_disease_imc'] != df['morbidity']).sum()
    
    # ICD-11 Taxonomy results  
    icd11_mapped = (df['icd11_code'] != 'Unclassified').sum()
    high_confidence = (df['confidence'] == 'high confidence').sum()
    medium_confidence = (df['confidence'] == 'medium confidence').sum()
    
    logger.info("✅ Dual taxonomy classification completed:")
    logger.info(f"📋 IMC Taxonomy Results:")
    logger.info(f"   • {unique_canonicals} unique canonical diseases")
    logger.info(f"   • {unique_categories} disease categories") 
    logger.info(f"   • {imc_mapped}/{len(df)} diseases mapped from morbidity")
    
    logger.info(f"🌐 ICD-11 Taxonomy Results:")
    logger.info(f"   • {icd11_mapped}/{len(df)} diseases mapped to ICD-11")
    logger.info(f"   • {high_confidence} high confidence mappings")
    logger.info(f"   • {medium_confidence} medium confidence mappings")
    
    # Show top categories from both systems
    if unique_categories > 0:
        top_imc_categories = df['category_canonical_disease_imc'].value_counts().head(3)
        logger.info(f"   • Top IMC categories: {dict(top_imc_categories)}")
    
    if icd11_mapped > 0:
        top_icd11_categories = df['icd11_category'].value_counts().head(3)
        logger.info(f"   • Top ICD-11 categories: {dict(top_icd11_categories)}")

def apply_minimal_dual_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal dual taxonomy classification for error recovery.
    """
    logger.info("🔧 Applying minimal dual taxonomy classification...")
    
    # IMC taxonomy fallback
    df['canonical_disease_imc'] = df['morbidity']
    df['category_canonical_disease_imc'] = 'Uncategorized'
    df['canonical_disease'] = df['morbidity']
    
    # ICD-11 taxonomy fallback
    df['icd11_code'] = 'Unclassified'
    df['icd11_title'] = 'Unclassified'
    df['icd11_category'] = 'Unclassified'
    df['confidence'] = 'none'
    
    # Minimal feature flags
    feature_flags = ['vaccine_preventable', 'climate_sensitive', 'outbreak_prone', 'trauma_related', 'epidemic_prone']
    for flag in feature_flags:
        df[flag] = False
    
    return df


def standardize_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize demographic variables for ML pipeline compatibility."""
    logger.info("👥 Standardizing demographic variables...")
    
    processed_df = df.copy()
    
    # Standardize sex/gender
    if 'sex' in processed_df.columns:
        processed_df['sex'] = processed_df['sex'].str.upper().str.strip()
        
        # Create binary indicators
        processed_df['is_female'] = (processed_df['sex'] == 'FEMALE').astype(int)
        processed_df['is_male'] = (processed_df['sex'] == 'MALE').astype(int)
    
    # Standardize case type (trauma vs non-trauma)
    if 'case_type' in processed_df.columns:
        processed_df['case_type'] = processed_df['case_type'].str.lower().str.strip()
        processed_df['is_trauma'] = (processed_df['case_type'] == 'trauma').astype(int)
        processed_df['is_non_trauma'] = (processed_df['case_type'] == 'non-trauma').astype(int)
    
    # Standardize visit type
    if 'visit_type' in processed_df.columns:
        processed_df['visit_type'] = processed_df['visit_type'].str.lower().str.strip()
        processed_df['is_new_visit'] = (processed_df['visit_type'] == 'new').astype(int)
        processed_df['is_followup_visit'] = (processed_df['visit_type'] == 'follow-up').astype(int)
    
    # Disability indicator
    if 'has_disability' in processed_df.columns:
        processed_df['has_disability'] = processed_df['has_disability'].astype(str).str.upper()
        processed_df['has_disability_binary'] = (processed_df['has_disability'] == 'Y').astype(int)
    
    logger.info("✅ Demographic standardization completed")
    
    return processed_df


def load_and_merge_internal_data(config: Dict, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Complete pipeline for loading and processing Syria data for ML.
    
    Args:
        config: Configuration dictionary
        sample_size: Optional sample size for testing
        
    Returns:
        Processed DataFrame ready for feature engineering
    """
    logger.info("🚀 Starting internal data loading and processing pipeline...")
    
    # Load raw internal data
    internal_df = load_syria_data(config, sample_size)
    
    # Get climate data paths from environment variable or config
    climate_data_path = os.getenv('CLIMATE_DATA_PATH') or config.get('data', {}).get('private', {}).get('climate_data_path')
    
    # Check for Open-Meteo data file
    open_meteo_path = Path("data/internal/raw_dat/open-meteo-34.97N38.08E916m.csv")
    
    if open_meteo_path.exists():
        logger.info("🌡️ Loading Open-Meteo climate data...")
        from open_meteo_processor import load_and_process_internal_weather_data
        climate_df = load_and_process_internal_weather_data(str(open_meteo_path), internal_df)
        
        # Assign regions to health facilities
        from open_meteo_processor import assign_weather_to_facilities
        internal_df = assign_weather_to_facilities(internal_df, climate_df)
        
        # Merge with climate data
        merged_df = merge_health_climate(internal_df, climate_df)
    elif climate_data_path and Path(climate_data_path).exists():
        logger.info("🌡️ Loading climate data...")
        climate_df = pd.read_csv(climate_data_path)
        
        # Merge with climate data
        merged_df = merge_health_climate(internal_df, climate_df)
    else:
        logger.warning("⚠️ Climate data not available - using health data only")
        merged_df = internal_df
    
    # Save processed data
    processed_dir = Path('data/processed')
    processed_dir.mkdir(exist_ok=True)
    
    output_path = processed_dir / 'internal_processed_dataset.csv'
    merged_df.to_csv(output_path, index=False)
    logger.info(f"💾 Processed internal data saved to {output_path}")
    
    logger.info(f"✅ Internal data processing completed: {len(merged_df)} records ready for ML")
    
    return merged_df


def detect_data_mode() -> str:
    """
    Auto-detect whether to use synthetic or internal data.
    
    Returns:
        'internal' if internal data is available, 'synthetic' otherwise
    """
    internal_events_path = Path(os.getenv('HEALTH_DATA_PATH', 'data/internal/raw_dat/events.csv'))
    
    if internal_events_path.exists():
        file_size = internal_events_path.stat().st_size / (1024 * 1024)  # MB
        if file_size > 1:  # At least 1MB suggests real data
            return 'internal'
    
    return 'synthetic'