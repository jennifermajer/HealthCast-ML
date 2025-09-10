"""
Taxonomy Mapping Cache System

Provides caching functionality for expensive taxonomy mapping operations to avoid
recalculating morbidity -> canonical disease mappings and ICD-11 classifications
on every run.
"""

import os
import json
import pandas as pd
import numpy as np
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TaxonomyCache:
    """
    Manages caching of taxonomy mappings to avoid expensive re-computation.
    """
    
    def __init__(self, cache_dir: str = "data/cache", enabled: bool = True):
        """
        Initialize taxonomy cache.
        
        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.cache_file = self.cache_dir / "taxonomy_mappings.json"
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"🗄️ Taxonomy cache enabled: {self.cache_dir}")
        else:
            logger.info("🚫 Taxonomy cache disabled")
    
    def _generate_cache_key(self, morbidities: pd.Series, taxonomy_config: Dict) -> str:
        """
        Generate a cache key based on unique morbidities and taxonomy configuration.
        
        Args:
            morbidities: Series of morbidity values
            taxonomy_config: Taxonomy configuration parameters
            
        Returns:
            Cache key string
        """
        # Get unique morbidities and sort for consistent hashing
        unique_morbidities = sorted(morbidities.dropna().unique().tolist())
        
        # Create hash from morbidities and config
        cache_data = {
            'morbidities': unique_morbidities,
            'config': taxonomy_config
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _load_cache(self) -> Dict:
        """Load cache data from disk."""
        if not self.enabled or not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            logger.info(f"📥 Loaded taxonomy cache from {self.cache_file}")
            return cache_data
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cache: {e}")
            return {}
    
    def _save_cache(self, cache_data: Dict):
        """Save cache data to disk."""
        if not self.enabled:
            return
        
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved taxonomy cache to {self.cache_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")
    
    def get_cached_mappings(self, morbidities: pd.Series, taxonomy_config: Dict) -> Optional[Dict]:
        """
        Get cached taxonomy mappings for the given morbidities.
        
        Args:
            morbidities: Series of morbidity values
            taxonomy_config: Taxonomy configuration
            
        Returns:
            Cached mappings dictionary or None if not found
        """
        if not self.enabled:
            return None
        
        cache_key = self._generate_cache_key(morbidities, taxonomy_config)
        cache_data = self._load_cache()
        
        if cache_key in cache_data:
            cached_entry = cache_data[cache_key]
            
            # Check if cache is still valid (not expired)
            cache_timestamp = datetime.fromisoformat(cached_entry['timestamp'])
            if datetime.now() - cache_timestamp < timedelta(days=30):  # 30-day cache expiry
                logger.info(f"✅ Found valid cached taxonomy mappings (key: {cache_key[:8]}...)")
                return cached_entry['mappings']
            else:
                logger.info(f"⏰ Cache expired for key {cache_key[:8]}...")
        
        return None
    
    def save_mappings(self, morbidities: pd.Series, taxonomy_config: Dict, mappings: Dict):
        """
        Save taxonomy mappings to cache.
        
        Args:
            morbidities: Series of morbidity values
            taxonomy_config: Taxonomy configuration
            mappings: Mapping results to cache
        """
        if not self.enabled:
            return
        
        cache_key = self._generate_cache_key(morbidities, taxonomy_config)
        cache_data = self._load_cache()
        
        # Store the mappings with timestamp
        cache_data[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'mappings': mappings,
            'config': taxonomy_config,
            'unique_morbidities_count': len(morbidities.dropna().unique())
        }
        
        self._save_cache(cache_data)
        logger.info(f"💾 Cached taxonomy mappings (key: {cache_key[:8]}...)")
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.enabled and self.cache_file.exists():
            self.cache_file.unlink()
            logger.info("🗑️ Taxonomy cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get information about the current cache."""
        if not self.enabled:
            return {"enabled": False}
        
        cache_data = self._load_cache()
        
        info = {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "cache_file": str(self.cache_file),
            "cache_exists": self.cache_file.exists(),
            "num_cached_keys": len(cache_data),
        }
        
        if cache_data:
            timestamps = [entry['timestamp'] for entry in cache_data.values()]
            info["oldest_entry"] = min(timestamps)
            info["newest_entry"] = max(timestamps)
            info["total_unique_morbidities"] = sum(
                entry.get('unique_morbidities_count', 0) for entry in cache_data.values()
            )
        
        return info


def apply_cached_taxonomy_mapping(df: pd.DataFrame, country: str = 'base', 
                                config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Apply taxonomy mapping with caching support.
    
    This function wraps the original classify_diseases_dual_taxonomy function
    with caching to avoid re-computing expensive mappings.
    
    Args:
        df: DataFrame with morbidity column
        country: Country taxonomy to use
        config: Configuration dictionary
        
    Returns:
        DataFrame with taxonomy mappings applied
    """
    if config is None:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    
    # Check if taxonomy application is enabled
    taxonomy_config = config.get('taxonomy', {})
    if not taxonomy_config.get('apply_taxonomy', True):
        logger.info("🚫 Taxonomy mapping disabled by configuration")
        return df
    
    # Check if caching is enabled
    use_cache = taxonomy_config.get('use_cache', True)
    cache_dir = taxonomy_config.get('cache_dir', 'data/cache')
    
    cache = TaxonomyCache(cache_dir=cache_dir, enabled=use_cache)
    
    if 'morbidity' not in df.columns:
        logger.warning("⚠️ No morbidity column found - skipping taxonomy mapping")
        return df
    
    # Create configuration for cache key
    cache_config = {
        'country': country,
        'apply_taxonomy': True,
    }
    
    # Check cache first
    cached_mappings = cache.get_cached_mappings(df['morbidity'], cache_config)
    
    if cached_mappings is not None:
        logger.info("🎯 Using cached taxonomy mappings")
        return _apply_cached_mappings_to_dataframe(df, cached_mappings)
    
    # No cache found, need to compute mappings
    logger.info("🔄 Computing taxonomy mappings (no cache found)")
    
    # Import the original function
    from data_processing import classify_diseases_dual_taxonomy
    
    # Apply original taxonomy mapping
    mapped_df = classify_diseases_dual_taxonomy(df, country=country)
    
    # Extract mappings for caching
    mappings = _extract_mappings_from_dataframe(df, mapped_df)
    
    # Save to cache
    cache.save_mappings(df['morbidity'], cache_config, mappings)
    
    return mapped_df


def _apply_cached_mappings_to_dataframe(df: pd.DataFrame, mappings: Dict) -> pd.DataFrame:
    """Apply cached mappings to dataframe."""
    result_df = df.copy()
    
    # Initialize all the taxonomy columns with default values
    taxonomy_columns = [
        'canonical_disease_imc', 'category_canonical_disease_imc', 'canonical_disease',
        'icd11_code', 'icd11_title', 'icd11_category', 'confidence'
    ]
    
    for col in taxonomy_columns:
        if col not in result_df.columns:
            result_df[col] = 'Uncategorized' if 'category' in col else 'Unclassified'
    
    # Apply mappings row by row
    for idx, morbidity in enumerate(result_df['morbidity']):
        if pd.notna(morbidity) and str(morbidity) in mappings:
            mapping = mappings[str(morbidity)]
            for col, value in mapping.items():
                if col in taxonomy_columns:
                    result_df.loc[idx, col] = value
    
    # Add epidemiological feature flags (simplified version)
    _add_epidemiological_flags_from_cache(result_df)
    
    return result_df


def _extract_mappings_from_dataframe(original_df: pd.DataFrame, mapped_df: pd.DataFrame) -> Dict:
    """Extract mappings from mapped dataframe for caching."""
    mappings = {}
    
    # Columns to cache
    cache_columns = [
        'canonical_disease_imc', 'category_canonical_disease_imc', 'canonical_disease',
        'icd11_code', 'icd11_title', 'icd11_category', 'confidence'
    ]
    
    for idx, morbidity in enumerate(original_df['morbidity']):
        if pd.notna(morbidity):
            morbidity_str = str(morbidity)
            if morbidity_str not in mappings:
                mapping = {}
                for col in cache_columns:
                    if col in mapped_df.columns:
                        mapping[col] = mapped_df.iloc[idx][col]
                mappings[morbidity_str] = mapping
    
    return mappings


def _add_epidemiological_flags_from_cache(df: pd.DataFrame):
    """Add basic epidemiological flags based on cached canonical diseases."""
    # Initialize flags
    flags = [
        'vaccine_preventable', 'climate_sensitive', 'outbreak_prone', 
        'trauma_related', 'epidemic_prone'
    ]
    
    for flag in flags:
        if flag not in df.columns:
            df[flag] = False
    
    # Define simple mappings based on canonical diseases
    flag_mappings = {
        'vaccine_preventable': ['Measles', 'Polio', 'Diphtheria', 'Pertussis', 'Mumps', 'Rubella'],
        'climate_sensitive': ['Malaria', 'Dengue', 'Cholera', 'Diarrhea', 'Acute Watery Diarrhea'],
        'outbreak_prone': ['Cholera', 'Measles', 'Meningitis', 'Yellow Fever'],
        'trauma_related': ['Injury', 'Trauma', 'Fracture', 'Burn', 'Wound']
    }
    
    for flag, diseases in flag_mappings.items():
        for disease in diseases:
            mask = df['canonical_disease_imc'].str.contains(disease, case=False, na=False)
            df.loc[mask, flag] = True
    
    # Set epidemic_prone as alias for outbreak_prone
    df['epidemic_prone'] = df['outbreak_prone']


if __name__ == "__main__":
    # Test the cache system
    cache = TaxonomyCache()
    info = cache.get_cache_info()
    print("Cache info:", info)