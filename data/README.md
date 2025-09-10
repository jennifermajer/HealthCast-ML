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