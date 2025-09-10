"""
Taxonomy-aware disease classification system for health data processing.

Python implementation of the R taxonomy processing system, providing:
- Disease name normalization and canonical mapping
- Category classification based on YAML taxonomies
- Epidemic grouping and surveillance classifications
- Climate sensitivity analysis
- Age risk stratification
- Seasonal pattern identification

Based on IMC Global Base Canonicals aligned with ICD-11 and WHO surveillance standards.
"""

import yaml
import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from functools import lru_cache
import warnings

logger = logging.getLogger(__name__)

class TaxonomyProcessor:
    """
    Comprehensive disease taxonomy processing system.
    
    Handles loading of YAML-based taxonomy files and provides methods for:
    - Disease name normalization and canonical mapping
    - Category classification
    - Epidemic and syndromic grouping
    - Climate sensitivity analysis
    """
    
    def __init__(self, taxonomy_dir: str = "taxonomy"):
        """
        Initialize taxonomy processor.
        
        Args:
            taxonomy_dir: Path to directory containing taxonomy YAML files
        """
        self.taxonomy_dir = Path(taxonomy_dir)
        self._taxonomies_cache = {}
        self._synonym_indices_cache = {}
        self._category_lookups_cache = {}
        
        logger.info(f"🔧 Initializing TaxonomyProcessor with directory: {self.taxonomy_dir}")
        
    def _normalize_disease_name(self, name: str) -> str:
        """
        Normalize disease name for matching.
        
        Replicates the R .norm() function behavior:
        - Convert to lowercase
        - Remove extra whitespace
        - Remove special characters
        - Standardize common abbreviations
        """
        if pd.isna(name) or not isinstance(name, str):
            return ""
            
        # Convert to lowercase and strip
        normalized = str(name).lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove special characters but keep letters, numbers, spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Handle common abbreviations and variations
        replacements = {
            'diarrhoea': 'diarrhea',
            'uri': 'upper respiratory infection',
            'lrti': 'lower respiratory tract infection',
            'urti': 'upper respiratory tract infection',
            'ari': 'acute respiratory infection',
            'copd': 'chronic obstructive pulmonary disease',
            'ncd': 'non communicable disease'
        }
        
        for abbrev, full_form in replacements.items():
            normalized = normalized.replace(abbrev, full_form)
            
        return normalized.strip()
    
    def _find_taxonomy_file(self, filename: str) -> Optional[Path]:
        """Find taxonomy file, trying both .yml and .yaml extensions."""
        base_name = Path(filename).stem
        
        # Try both extensions
        for ext in ['.yml', '.yaml']:
            for candidate in [
                self.taxonomy_dir / f"{base_name}{ext}",
                Path(f"{base_name}{ext}"),
                Path("config") / f"{base_name}{ext}",
                Path("data") / "taxonomy" / f"{base_name}{ext}"
            ]:
                if candidate.exists():
                    return candidate
                    
        return None
    
    @lru_cache(maxsize=32)
    def load_taxonomy(self, which: str = "base") -> Dict:
        """
        Load taxonomy from YAML file with caching.
        
        Args:
            which: Name of taxonomy file (without extension)
            
        Returns:
            Dictionary containing taxonomy structure
        """
        cache_key = f"tax_{which}"
        if cache_key in self._taxonomies_cache:
            return self._taxonomies_cache[cache_key]
            
        # Find and load taxonomy file
        tax_file = self._find_taxonomy_file(which)
        
        if tax_file is None:
            logger.warning(f"⚠️ Taxonomy file {which} not found, creating minimal taxonomy")
            return self._create_minimal_taxonomy()
            
        try:
            with open(tax_file, 'r', encoding='utf-8') as f:
                taxonomy = yaml.safe_load(f)
                
            logger.info(f"✅ Loaded taxonomy from {tax_file}")
            
            # Handle inheritance
            if 'meta' in taxonomy and 'inherits' in taxonomy['meta']:
                parent = taxonomy['meta']['inherits']
                if parent:
                    parent_tax = self.load_taxonomy(parent)
                    # Merge parent taxonomy (parent first, then child overrides)
                    merged = self._merge_taxonomies(parent_tax, taxonomy)
                    taxonomy = merged
                    
            self._taxonomies_cache[cache_key] = taxonomy
            return taxonomy
            
        except Exception as e:
            logger.error(f"❌ Error loading taxonomy {tax_file}: {e}")
            return self._create_minimal_taxonomy()
    
    def _create_minimal_taxonomy(self) -> Dict:
        """Create minimal taxonomy structure as fallback."""
        return {
            'canonicals': {
                "Acute Respiratory Infection": {
                    "category": "Respiratory Infectious Diseases",
                    "severity": "Medium",
                    "attributes": ["Outbreak-prone", "Climate-sensitive"]
                },
                "Diarrhea": {
                    "category": "Enteric & Food/Water-borne Infections", 
                    "severity": "Medium",
                    "attributes": ["Water-borne", "Climate-sensitive"]
                },
                "Fever": {
                    "category": "General Symptoms",
                    "severity": "Low"
                },
                "Hypertension": {
                    "category": "Non-Communicable Diseases",
                    "severity": "Medium"
                }
            },
            'synonyms': {
                "ARI": "Acute Respiratory Infection",
                "Diarrhoea": "Diarrhea",
                "High Blood Pressure": "Hypertension"
            },
            'categories': {
                "Respiratory Infectious Diseases": ["Acute Respiratory Infection"],
                "Enteric & Food/Water-borne Infections": ["Diarrhea"],
                "General Symptoms": ["Fever"],
                "Non-Communicable Diseases": ["Hypertension"]
            }
        }
    
    def _merge_taxonomies(self, parent: Dict, child: Dict) -> Dict:
        """Merge child taxonomy into parent taxonomy."""
        merged = parent.copy()
        
        for key, value in child.items():
            if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key].update(value)
            else:
                merged[key] = value
                
        return merged
    
    def build_synonym_index(self, which: str = "base") -> Dict[str, str]:
        """
        Build normalized synonym index for disease name mapping.
        
        Args:
            which: Taxonomy name to use
            
        Returns:
            Dictionary mapping normalized names to canonical names
        """
        cache_key = f"syn_idx_{which}"
        if cache_key in self._synonym_indices_cache:
            return self._synonym_indices_cache[cache_key]
            
        taxonomy = self.load_taxonomy(which)
        synonym_index = {}
        
        # Add direct synonyms
        synonyms = taxonomy.get('synonyms', {})
        for synonym, canonical in synonyms.items():
            normalized_syn = self._normalize_disease_name(synonym)
            if normalized_syn:
                synonym_index[normalized_syn] = canonical
                
        # Add canonicals mapping to themselves
        canonicals = taxonomy.get('canonicals', {})
        for canonical in canonicals.keys():
            normalized_canonical = self._normalize_disease_name(canonical)
            if normalized_canonical:
                synonym_index[normalized_canonical] = canonical
                
        logger.info(f"✅ Built synonym index with {len(synonym_index)} mappings for '{which}' taxonomy")
        
        self._synonym_indices_cache[cache_key] = synonym_index
        return synonym_index
    
    def canonicalize_disease(self, disease_name: str, which: str = "base") -> str:
        """
        Map disease name to canonical form.
        
        Args:
            disease_name: Input disease name
            which: Taxonomy to use
            
        Returns:
            Canonical disease name or original if no mapping found
        """
        if pd.isna(disease_name) or not isinstance(disease_name, str):
            return str(disease_name) if not pd.isna(disease_name) else ""
            
        synonym_index = self.build_synonym_index(which)
        normalized = self._normalize_disease_name(disease_name)
        
        return synonym_index.get(normalized, str(disease_name))
    
    def get_disease_category(self, canonical_name: str, which: str = "base") -> str:
        """
        Get category for canonical disease name.
        
        Args:
            canonical_name: Canonical disease name
            which: Taxonomy to use
            
        Returns:
            Disease category or "Uncategorized"
        """
        cache_key = f"cat_lookup_{which}"
        if cache_key not in self._category_lookups_cache:
            taxonomy = self.load_taxonomy(which)
            canonicals = taxonomy.get('canonicals', {})
            
            category_lookup = {}
            for disease, info in canonicals.items():
                if isinstance(info, dict) and 'category' in info:
                    category_lookup[disease] = info['category']
                    
            self._category_lookups_cache[cache_key] = category_lookup
            
        category_lookup = self._category_lookups_cache[cache_key]
        return category_lookup.get(str(canonical_name), "Uncategorized")
    
    def get_disease_attributes(self, canonical_name: str, which: str = "base") -> Dict:
        """
        Get all attributes for a canonical disease.
        
        Args:
            canonical_name: Canonical disease name
            which: Taxonomy to use
            
        Returns:
            Dictionary of disease attributes
        """
        taxonomy = self.load_taxonomy(which)
        canonicals = taxonomy.get('canonicals', {})
        
        disease_info = canonicals.get(str(canonical_name), {})
        if not isinstance(disease_info, dict):
            return {}
            
        return {
            'category': disease_info.get('category', 'Uncategorized'),
            'severity': disease_info.get('severity', 'Unknown'),
            'attributes': disease_info.get('attributes', []),
            'epidemic_groups': disease_info.get('epidemic_groups', []),
            'seasonal': disease_info.get('seasonal', []),
            'age_risks': disease_info.get('age_risks', []),
            'climate_groups': disease_info.get('climate_groups', []),
            'icd11_codes': disease_info.get('icd11_codes', [])
        }
    
    def classify_diseases_dataframe(self, df: pd.DataFrame, 
                                  disease_col: str,
                                  which: str = "base") -> pd.DataFrame:
        """
        Apply comprehensive disease classification to DataFrame.
        
        Args:
            df: Input DataFrame
            disease_col: Name of column containing disease names
            which: Taxonomy to use
            
        Returns:
            DataFrame with additional classification columns
        """
        if disease_col not in df.columns:
            logger.error(f"❌ Disease column '{disease_col}' not found in DataFrame")
            return df
            
        logger.info(f"🏷️ Applying comprehensive disease classification to {len(df)} records...")
        
        result_df = df.copy()
        
        # 1. Canonicalize disease names
        result_df['canonical_disease_imc'] = df[disease_col].apply(
            lambda x: self.canonicalize_disease(x, which)
        )
        
        # 2. Get categories
        result_df['category_canonical_disease_imc'] = result_df['canonical_disease_imc'].apply(
            lambda x: self.get_disease_category(x, which)
        )
        
        # 3. Add comprehensive attributes
        taxonomy = self.load_taxonomy(which)
        
        # Initialize boolean columns
        result_df['is_epidemic_prone'] = False
        result_df['is_vaccine_preventable'] = False
        result_df['is_climate_sensitive'] = False
        result_df['is_outbreak_prone'] = False
        result_df['is_water_borne'] = False
        result_df['is_vector_borne'] = False
        result_df['is_respiratory'] = False
        result_df['is_enteric'] = False
        result_df['severity_level'] = 'Unknown'
        
        # Apply attributes for each disease
        for idx, canonical in enumerate(result_df['canonical_disease_imc']):
            attrs = self.get_disease_attributes(canonical, which)
            
            # Boolean attributes
            attributes = attrs.get('attributes', [])
            result_df.loc[idx, 'is_epidemic_prone'] = 'Epidemic-prone' in attributes or 'Outbreak-prone' in attributes
            result_df.loc[idx, 'is_vaccine_preventable'] = 'VPD' in attributes
            result_df.loc[idx, 'is_climate_sensitive'] = 'Climate-sensitive' in attributes
            result_df.loc[idx, 'is_outbreak_prone'] = 'Outbreak-prone' in attributes
            result_df.loc[idx, 'is_water_borne'] = 'Water-borne' in attributes
            result_df.loc[idx, 'is_vector_borne'] = 'Vector-borne' in attributes
            
            # Category-based flags
            category = attrs.get('category', '')
            result_df.loc[idx, 'is_respiratory'] = 'Respiratory' in category
            result_df.loc[idx, 'is_enteric'] = 'Enteric' in category or 'Food/Water-borne' in category
            
            # Severity
            result_df.loc[idx, 'severity_level'] = attrs.get('severity', 'Unknown')
        
        # Add climate sensitivity groups
        self._add_climate_groups(result_df, which)
        
        # Add epidemic groups
        self._add_epidemic_groups(result_df, which)
        
        # Add seasonal patterns
        self._add_seasonal_patterns(result_df, which)
        
        # Add age risk groups
        self._add_age_risk_groups(result_df, which)
        
        # Add standard compatibility columns
        result_df['canonical_disease'] = result_df['canonical_disease_imc']
        result_df['register'] = 'register'
        
        logger.info(f"✅ Disease classification completed for {len(result_df)} records")
        logger.info(f"📊 Categories found: {result_df['category_canonical_disease_imc'].nunique()} unique")
        logger.info(f"📊 Canonical diseases: {result_df['canonical_disease_imc'].nunique()} unique")
        
        return result_df
    
    def _add_climate_groups(self, df: pd.DataFrame, which: str = "base"):
        """Add climate sensitivity group classifications."""
        taxonomy = self.load_taxonomy(which)
        
        # Initialize climate group columns
        df['climate_water_food_safety'] = False
        df['climate_vector_borne'] = False
        df['climate_heat_air_quality'] = False
        df['climate_nutrition_food_security'] = False
        
        canonicals = taxonomy.get('canonicals', {})
        
        for idx, canonical in enumerate(df['canonical_disease_imc']):
            if canonical in canonicals:
                climate_groups = canonicals[canonical].get('climate_groups', [])
                
                df.loc[idx, 'climate_water_food_safety'] = any(
                    'Water' in group or 'Food' in group for group in climate_groups
                )
                df.loc[idx, 'climate_vector_borne'] = any(
                    'Vector' in group for group in climate_groups
                )
                df.loc[idx, 'climate_heat_air_quality'] = any(
                    'Heat' in group or 'Air' in group for group in climate_groups
                )
                df.loc[idx, 'climate_nutrition_food_security'] = any(
                    'Nutrition' in group or 'Food_Security' in group for group in climate_groups
                )
    
    def _add_epidemic_groups(self, df: pd.DataFrame, which: str = "base"):
        """Add epidemic group classifications."""
        taxonomy = self.load_taxonomy(which)
        
        # Initialize epidemic group columns
        df['epidemic_vaccine_preventable'] = False
        df['epidemic_water_food_borne'] = False
        df['epidemic_respiratory'] = False
        df['epidemic_vector_borne'] = False
        
        canonicals = taxonomy.get('canonicals', {})
        
        for idx, canonical in enumerate(df['canonical_disease_imc']):
            if canonical in canonicals:
                epidemic_groups = canonicals[canonical].get('epidemic_groups', [])
                
                df.loc[idx, 'epidemic_vaccine_preventable'] = 'Vaccine_Preventable' in epidemic_groups
                df.loc[idx, 'epidemic_water_food_borne'] = 'Water_Food_Borne' in epidemic_groups
                df.loc[idx, 'epidemic_respiratory'] = 'Respiratory_Epidemic' in epidemic_groups
                df.loc[idx, 'epidemic_vector_borne'] = 'Vector_Borne' in epidemic_groups
    
    def _add_seasonal_patterns(self, df: pd.DataFrame, which: str = "base"):
        """Add seasonal pattern classifications."""
        taxonomy = self.load_taxonomy(which)
        
        # Initialize seasonal columns
        df['seasonal_winter'] = False
        df['seasonal_spring'] = False  
        df['seasonal_summer'] = False
        df['seasonal_autumn'] = False
        
        canonicals = taxonomy.get('canonicals', {})
        
        for idx, canonical in enumerate(df['canonical_disease_imc']):
            if canonical in canonicals:
                seasonal = canonicals[canonical].get('seasonal', [])
                
                df.loc[idx, 'seasonal_winter'] = 'Winter' in seasonal
                df.loc[idx, 'seasonal_spring'] = 'Spring' in seasonal
                df.loc[idx, 'seasonal_summer'] = 'Summer' in seasonal
                df.loc[idx, 'seasonal_autumn'] = 'Autumn' in seasonal
    
    def _add_age_risk_groups(self, df: pd.DataFrame, which: str = "base"):
        """Add age risk group classifications."""
        taxonomy = self.load_taxonomy(which)
        
        # Initialize age risk columns
        df['age_risk_under_5'] = False
        df['age_risk_elderly'] = False
        df['age_risk_reproductive_age'] = False
        
        canonicals = taxonomy.get('canonicals', {})
        
        for idx, canonical in enumerate(df['canonical_disease_imc']):
            if canonical in canonicals:
                age_risks = canonicals[canonical].get('age_risks', [])
                
                df.loc[idx, 'age_risk_under_5'] = 'Under_5' in age_risks
                df.loc[idx, 'age_risk_elderly'] = 'Elderly' in age_risks
                df.loc[idx, 'age_risk_reproductive_age'] = 'Reproductive_Age' in age_risks

    def get_category_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics by disease category.
        
        Args:
            df: DataFrame with classified diseases
            
        Returns:
            Summary DataFrame with category statistics
        """
        if 'category_canonical_disease_imc' not in df.columns:
            logger.error("❌ No category column found. Run classify_diseases_dataframe first.")
            return pd.DataFrame()
            
        summary = df.groupby('category_canonical_disease_imc').agg({
            'canonical_disease_imc': 'count',
            'is_climate_sensitive': 'sum',
            'is_epidemic_prone': 'sum',
            'is_vaccine_preventable': 'sum'
        }).rename(columns={
            'canonical_disease_imc': 'total_cases',
            'is_climate_sensitive': 'climate_sensitive_cases',
            'is_epidemic_prone': 'epidemic_prone_cases', 
            'is_vaccine_preventable': 'vaccine_preventable_cases'
        })
        
        # Calculate percentages
        summary['climate_sensitive_pct'] = (summary['climate_sensitive_cases'] / summary['total_cases'] * 100).round(1)
        summary['epidemic_prone_pct'] = (summary['epidemic_prone_cases'] / summary['total_cases'] * 100).round(1)
        summary['vaccine_preventable_pct'] = (summary['vaccine_preventable_cases'] / summary['total_cases'] * 100).round(1)
        
        return summary.sort_values('total_cases', ascending=False)


# Convenience functions for backward compatibility
def canonicalize_diseases(diseases: Union[str, List[str], pd.Series], 
                         taxonomy: str = "base") -> Union[str, List[str], pd.Series]:
    """
    Convenience function to canonicalize disease names.
    
    Args:
        diseases: Disease name(s) to canonicalize
        taxonomy: Taxonomy name to use
        
    Returns:
        Canonicalized disease name(s)
    """
    processor = TaxonomyProcessor()
    
    if isinstance(diseases, str):
        return processor.canonicalize_disease(diseases, taxonomy)
    elif isinstance(diseases, (list, pd.Series)):
        return [processor.canonicalize_disease(disease, taxonomy) for disease in diseases]
    else:
        raise ValueError("Input must be string, list, or pandas Series")


def classify_dataframe(df: pd.DataFrame, 
                      disease_col: str,
                      taxonomy: str = "base") -> pd.DataFrame:
    """
    Convenience function to classify diseases in a DataFrame.
    
    Args:
        df: Input DataFrame
        disease_col: Name of disease column
        taxonomy: Taxonomy name to use
        
    Returns:
        DataFrame with classification columns added
    """
    processor = TaxonomyProcessor()
    return processor.classify_diseases_dataframe(df, disease_col, taxonomy)


if __name__ == "__main__":
    # Example usage and testing
    processor = TaxonomyProcessor()
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'disease': ['URTI', 'Acute diarrhea', 'Fever', 'Hypertension', 'Unknown disease'],
        'patient_id': [1, 2, 3, 4, 5]
    })
    
    print("🧪 Testing taxonomy processor...")
    classified = processor.classify_diseases_dataframe(sample_data, 'disease')
    print("\n📊 Classification results:")
    print(classified[['disease', 'canonical_disease_imc', 'category_canonical_disease_imc']].to_string())
    
    print("\n📈 Category summary:")
    summary = processor.get_category_summary(classified)
    print(summary.to_string())