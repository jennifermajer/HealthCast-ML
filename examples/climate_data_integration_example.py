"""
Climate Data Integration Example

This script demonstrates how to use the new comprehensive climate data module
to fetch weather data for climate-health analysis supporting both project components.

COMPONENT 1: CLIMATE SENSITIVITY ANALYSIS
- Fetch multi-location climate data for identifying climate-sensitive morbidities
- Compare different data sources and geographic levels

COMPONENT 2: PREDICTIVE MODELING & FORECASTING  
- Fetch consistent climate features for forecasting model training
- Demonstrate temporal alignment with health consultation data
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from climate_data import (
    ClimateDataFetcher, LocationManager, Location,
    fetch_multi_location_climate, get_syria_climate_data,
    save_climate_data_for_processing, validate_climate_data_for_merge
)

from data_processing import (
    fetch_climate_data_api, load_climate_data_enhanced,
    validate_climate_health_compatibility
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_1_country_level_open_meteo():
    """
    COMPONENT 1 & 2: Example 1 - Country-level climate data using Open-Meteo
    
    Perfect for national-level analysis or when Open-Meteo is preferred.
    Open-Meteo provides free, reliable historical weather data.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Country-level Syria data using Open-Meteo API")
    print("="*60)
    
    try:
        # Fetch country-level climate data for Syria
        climate_data = get_syria_climate_data(
            date_range=("2023-01-01", "2023-12-31"),
            level="country",
            source="open_meteo"
        )
        
        print(f"✅ Successfully fetched {len(climate_data)} rows of climate data")
        print(f"📊 Columns: {list(climate_data.columns)}")
        print(f"📅 Date range: {climate_data['date'].min()} to {climate_data['date'].max()}")
        print(f"🌡️ Temperature range: {climate_data['temperature_mean'].min():.1f}°C to {climate_data['temperature_mean'].max():.1f}°C")
        print(f"🌧️ Precipitation range: {climate_data['precipitation'].min():.1f}mm to {climate_data['precipitation'].max():.1f}mm")
        
        # Save for later use
        save_path = save_climate_data_for_processing(
            climate_data, 
            "data/processed/syria_country_climate_2023_openmeteo.csv"
        )
        print(f"💾 Data saved to: {save_path}")
        
        return climate_data
        
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")
        return None

def example_2_governorate_level_nasa():
    """
    COMPONENT 1 PRIMARY: Example 2 - Governorate-level climate data using NASA POWER
    
    Ideal for Component 1 spatial climate sensitivity analysis.
    NASA POWER supports multiple locations for detailed geographic analysis.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Governorate-level Syria data using NASA POWER")
    print("="*60)
    
    try:
        # Fetch governorate-level climate data for key Syrian governorates
        climate_data = get_syria_climate_data(
            date_range=("2023-06-01", "2023-08-31"),  # Summer period
            level="governorate", 
            source="nasa_power"
        )
        
        print(f"✅ Successfully fetched {len(climate_data)} rows of climate data")
        print(f"📍 Locations: {sorted(climate_data['location_name'].unique())}")
        print(f"🗺️ Geographic coverage: {climate_data['location_name'].nunique()} governorates")
        print(f"📊 Data shape: {climate_data.shape}")
        
        # Show temperature variation across governorates
        temp_by_location = climate_data.groupby('location_name')['temperature_mean'].agg(['mean', 'std']).round(2)
        print(f"\n🌡️ Temperature variation by governorate:")
        print(temp_by_location)
        
        return climate_data
        
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")
        return None

def example_3_custom_locations():
    """
    COMPONENT 1 & 2: Example 3 - Custom locations for specialized analysis
    
    Demonstrates how to fetch climate data for custom geographic points.
    Useful for detailed analysis of specific health facilities or regions.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom locations with multiple data sources")
    print("="*60)
    
    try:
        # Initialize location manager
        location_manager = LocationManager()
        
        # Create custom locations (e.g., major cities in the region)
        custom_locations = [
            Location("Beirut", 33.8938, 35.5018, "city", "Beirut", "LB"),
            Location("Amman", 31.9539, 35.9106, "city", "Amman", "JO"),
            Location("Baghdad", 33.3152, 44.3661, "city", "Baghdad", "IQ")
        ]
        
        # Add to location manager
        for loc in custom_locations:
            location_manager.add_location(loc)
        
        # Fetch climate data for custom locations
        climate_data = fetch_multi_location_climate(
            locations=custom_locations,
            date_range=("2023-07-01", "2023-07-31"),  # July 2023
            source="nasa_power"
        )
        
        print(f"✅ Successfully fetched climate data for custom locations")
        print(f"📍 Locations: {sorted(climate_data['location_name'].unique())}")
        print(f"🌡️ Average temperatures by location:")
        
        avg_temps = climate_data.groupby('location_name')['temperature_mean'].mean().round(1)
        for location, temp in avg_temps.items():
            print(f"  {location}: {temp}°C")
        
        return climate_data
        
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")
        return None

def example_4_integration_with_data_processing():
    """
    COMPONENT 1 & 2: Example 4 - Integration with existing data processing pipeline
    
    Demonstrates how the new climate module integrates with the existing
    data_processing.py functions for seamless climate-health analysis.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Integration with data processing pipeline")
    print("="*60)
    
    try:
        # Method 1: Use the enhanced data processing functions directly
        climate_data = fetch_climate_data_api(
            date_range=("2023-01-01", "2023-03-31"),
            admin_level="country",
            source="open_meteo"
        )
        
        print(f"✅ Fetched via data_processing API: {len(climate_data)} rows")
        
        # Method 2: Enhanced loading with fallback
        climate_data_enhanced = load_climate_data_enhanced(
            file_path=None,  # No file, will use API
            date_range=("2023-01-01", "2023-02-28"),
            admin_level="country", 
            source="demo",  # Use demo for reliable testing
            use_api=True
        )
        
        print(f"✅ Enhanced loading: {len(climate_data_enhanced)} rows")
        
        # Simulate health data for compatibility validation
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range("2023-01-01", "2023-02-28", freq='D')
        mock_health_data = pd.DataFrame({
            'date': dates,
            'admin1': ['Syria'] * len(dates),
            'consultation_count': np.random.poisson(100, len(dates))
        })
        
        # Validate compatibility
        validation_report = validate_climate_health_compatibility(
            mock_health_data, 
            climate_data_enhanced
        )
        
        print(f"🔍 Validation report:")
        print(f"  Issues found: {len(validation_report.get('issues', []))}")
        print(f"  Overlap: {validation_report.get('health_climate_overlap', {}).get('overlap_percentage', 'N/A')}%")
        
        if validation_report.get('recommendations'):
            print(f"  Recommendations: {validation_report['recommendations'][:2]}")  # Show first 2
        
        return climate_data_enhanced
        
    except Exception as e:
        logger.error(f"Example 4 failed: {e}")
        return None

def example_5_comparison_data_sources():
    """
    COMPONENT 1: Example 5 - Compare different climate data sources
    
    Demonstrates differences between Open-Meteo and NASA POWER for the same location.
    Useful for Component 1 sensitivity analysis to understand data source impact.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Compare Open-Meteo vs NASA POWER data sources")
    print("="*60)
    
    date_range = ("2023-06-01", "2023-06-07")  # Short range for comparison
    
    try:
        # Fetch same data from both sources
        openmeteo_data = get_syria_climate_data(
            date_range=date_range,
            level="country",
            source="open_meteo"
        )
        
        nasa_data = get_syria_climate_data(
            date_range=date_range,
            level="country", 
            source="nasa_power"
        )
        
        # Compare key metrics
        import pandas as pd
        
        comparison = pd.DataFrame({
            'Open-Meteo Temperature': openmeteo_data['temperature_mean'].round(2),
            'NASA Temperature': nasa_data['temperature_mean'].round(2),
            'Open-Meteo Precipitation': openmeteo_data['precipitation'].round(2),
            'NASA Precipitation': nasa_data['precipitation'].round(2),
            'Date': openmeteo_data['date'].dt.strftime('%Y-%m-%d')
        })
        
        print("📊 Data source comparison:")
        print(comparison.to_string(index=False))
        
        # Calculate correlation between sources
        temp_corr = openmeteo_data['temperature_mean'].corr(nasa_data['temperature_mean'])
        precip_corr = openmeteo_data['precipitation'].corr(nasa_data['precipitation'])
        
        print(f"\n🔗 Source correlations:")
        print(f"  Temperature correlation: {temp_corr:.3f}")
        print(f"  Precipitation correlation: {precip_corr:.3f}")
        
        return comparison
        
    except Exception as e:
        logger.error(f"Example 5 failed: {e}")
        return None

def main():
    """Run all climate data integration examples"""
    print("🌡️ Climate Data Integration Examples")
    print("Supporting both Component 1 (Climate Sensitivity) and Component 2 (Forecasting)")
    
    # Ensure output directory exists
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Run examples
    examples = [
        ("Country-level Open-Meteo", example_1_country_level_open_meteo),
        ("Governorate-level NASA POWER", example_2_governorate_level_nasa),
        ("Custom locations", example_3_custom_locations),
        ("Data processing integration", example_4_integration_with_data_processing),
        ("Source comparison", example_5_comparison_data_sources)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n🚀 Running: {name}")
            result = example_func()
            results[name] = result
            if result is not None:
                print(f"✅ {name}: SUCCESS")
            else:
                print(f"❌ {name}: FAILED")
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
            results[name] = None
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)
    
    print(f"✅ Successful examples: {successful}/{total}")
    
    if successful > 0:
        print("\n🎉 Climate data integration is working!")
        print("🔗 You can now use these functions in your climate-health analysis:")
        print("   • fetch_climate_data_api() - API-based climate data fetching")
        print("   • get_syria_climate_data() - Convenient Syria-specific data")
        print("   • load_climate_data_enhanced() - Enhanced file + API loading")
        print("   • validate_climate_health_compatibility() - Data validation")
    else:
        print("⚠️  No examples were successful. Check your environment and API connectivity.")
    
    print(f"\n📚 For more details, see:")
    print(f"   • src/climate_data.py - Comprehensive climate data module")
    print(f"   • src/data_processing.py - Integration with existing pipeline")

if __name__ == "__main__":
    main()