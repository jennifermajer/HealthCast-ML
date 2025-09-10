# Data Directory Guide

## Directory Structure

### `raw/` - Private Data (Not in Repository)
**This directory is gitignored.** Place your confidential datasets here:

#### Required Files for IMC Internal Use:
- **`imc_consultations.csv`** - Line-listed health consultation data
  - Columns: `date`, `governorate`, `age`, `sex`, `morbidity`
  - Date format: YYYY-MM-DD
  - Expected size: ~500,000 consultations
  - Coverage: 2018-2024

- **`noaa_climate_full.csv`** - Complete NOAA climate data  
  - Columns: `date`, `governorate`, `temp_max`, `temp_min`, `precipitation`
  - Date format: YYYY-MM-DD
  - Coverage: Daily data for all Syrian governorates

#### Data Privacy Note:
- Never commit files from `raw/` to version control
- Use `.env` file to configure data paths
- Real data stays on your local machine only

### `processed/` - Generated Datasets (Not in Repository)
Auto-generated intermediate files (also gitignored):
- `merged_dataset.csv` - Health + climate data combined
- `feature_matrix.csv` - ML-ready dataset with engineered features
- Other intermediate processing outputs

### `synthetic/` - Public Synthetic Data (In Repository)
**Safe for public sharing:**
- `synthetic_consultations.csv` - Synthetic health data mimicking real structure
- `generate_synthetic.py` - Script to create synthetic data
- Used by default for public replication

### `external/` - Public Reference Data (In Repository)
**Non-sensitive reference data:**
- `sample_noaa_data.csv` - Small climate data sample for testing
- `governorate_mapping.json` - Governorate names and codes
- Other public datasets

## Usage Instructions

### For IMC Internal Use (Real Data):
1. Place real data files in `data/raw/`
2. Copy `.env.example` to `.env`
3. Set `USE_SYNTHETIC=false` in `.env`
4. Run analysis: `python run_analysis.py`

### For External Researchers (Synthetic Data):
1. Clone repository (synthetic data included)
2. Run analysis immediately: `python run_analysis.py`
3. Default configuration uses synthetic data automatically

### Data Validation
The pipeline automatically validates:
- Required columns are present
- Date formats are correct
- Data ranges are reasonable
- No personal information is logged

## 🌡️ Climate Data Integration

The project includes a comprehensive climate data module (`src/climate_data.py`) that automatically fetches weather data from multiple APIs, eliminating the need for manual climate data preparation.

### Available Data Sources

#### Open-Meteo API (Recommended for single locations)
- **Coverage**: Global historical weather data from 1940 to present
- **Cost**: Free with no API key required
- **Best for**: Country-level analysis, single location studies
- **Variables**: Temperature (min/max/mean), precipitation, humidity, wind speed, pressure

#### NASA POWER API (Recommended for multi-location analysis)
- **Coverage**: Global satellite-derived data from 1981 to present  
- **Cost**: Free with no API key required
- **Best for**: Multi-location analysis, governorate/district level studies
- **Variables**: Temperature, precipitation, humidity, wind speed, solar radiation, pressure

#### Synthetic Data Generation
- **Purpose**: Testing, development, and fallback when APIs are unavailable
- **Quality**: Climatologically realistic with seasonal patterns and extreme events

### Usage Examples

#### Basic Climate Data Fetching
```python
from src.climate_data import get_syria_climate_data

# Simple country-level analysis
climate_data = get_syria_climate_data(
    date_range=("2020-01-01", "2023-12-31"),
    level="country",
    source="open_meteo"
)

# Multi-location governorate analysis  
climate_data = get_syria_climate_data(
    date_range=("2020-01-01", "2023-12-31"),
    level="governorate",
    source="nasa_power"
)
```

#### Custom Locations for Other Regions
```python
from src.climate_data import LocationManager, Location, fetch_multi_location_climate

# Create custom locations for your study area
custom_locations = [
    Location("MyCity", latitude=40.7128, longitude=-74.0060),
    Location("MyRegion", latitude=41.2033, longitude=-77.1945)
]

# Fetch climate data using either Open-Meteo or NASA POWER
climate_data = fetch_multi_location_climate(
    locations=custom_locations,
    date_range=("2020-01-01", "2023-12-31"),
    source="open_meteo"  # or "nasa_power"
)
```

### Automated Features
- **Climate Extremes Detection**: Automatic identification of heatwaves, heavy precipitation, drought conditions
- **Temporal Feature Engineering**: Seasonal patterns, rolling averages, climate anomaly detection
- **Data Quality & Validation**: Automatic data quality checks and missing data handling

### Integration with Analysis Pipeline
```bash
# Climate data is automatically processed and saved in the correct format
python run_analysis.py --cache --synthetic  # Uses synthetic climate data
python run_analysis.py --cache             # Auto-fetches real climate data
```

The climate module automatically handles:
- ✅ Global coordinate support for any latitude/longitude
- ✅ Automatic data source selection based on coverage and availability  
- ✅ Standardized output format regardless of input source
- ✅ Built-in caching and error handling with fallback to synthetic data