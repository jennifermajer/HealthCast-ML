#!/usr/bin/env python3
"""
Analyze Syria facility names to help create geographic mappings.
This script examines facility names in the events.csv to identify patterns
and suggest admin1/admin2 mappings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from collections import Counter
import re

def analyze_facility_names(sample_size: int = 10000):
    """Analyze facility names to identify geographic patterns."""
    
    print("🔍 Analyzing Syria facility names for geographic mapping...")
    
    # Load sample of events data
    events_path = Path("data/internal/raw_dat/events.csv")
    if not events_path.exists():
        print(f"❌ Events file not found: {events_path}")
        return
        
    print(f"📂 Loading sample from {events_path}")
    df = pd.read_csv(events_path, encoding='utf-8', nrows=sample_size)
    
    facility_col = 'Organisation unit name'
    if facility_col not in df.columns:
        print(f"❌ Column '{facility_col}' not found")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Analyze facility names
    facilities = df[facility_col].dropna().unique()
    print(f"📊 Found {len(facilities)} unique facilities in sample")
    
    # Count occurrences
    facility_counts = df[facility_col].value_counts()
    print(f"📊 Total consultations across {len(facility_counts)} facilities")
    
    # Analyze name patterns
    print("\n🏥 TOP 20 FACILITIES BY CONSULTATION COUNT:")
    print("=" * 60)
    for facility, count in facility_counts.head(20).items():
        print(f"{count:6,} | {facility}")
    
    # Extract potential admin regions from facility names
    print("\n🗺️ GEOGRAPHIC PATTERN ANALYSIS:")
    print("=" * 60)
    
    # Common Syrian governorates/cities
    syrian_regions = {
        'Aleppo': ['Aleppo', 'Halab'],
        'Damascus': ['Damascus', 'Dimashq'],
        'Homs': ['Homs', 'Hims'], 
        'Hama': ['Hama', 'Hamah'],
        'Lattakia': ['Lattakia', 'Latakia'],
        'Deir ez-Zor': ['Deir', 'Deir ez-Zor', 'Der'],
        'Ar-Raqqa': ['Raqqa', 'Rakka', 'Tabqa'],
        'Idlib': ['Idlib', 'Edlib'],
        'Daraa': ['Daraa', 'Deraa'],
        'As-Suwayda': ['Suwayda', 'Sweida'],
        'Quneitra': ['Quneitra', 'Qunaytirah'],
        'Tartous': ['Tartous', 'Tartus'],
        'Al-Hasakah': ['Hasakah', 'Hasaka']
    }
    
    # Create mapping suggestions
    region_matches = {}
    
    for admin1, patterns in syrian_regions.items():
        matches = []
        for facility in facilities:
            facility_clean = str(facility).lower()
            for pattern in patterns:
                if pattern.lower() in facility_clean:
                    matches.append(facility)
                    break
        
        if matches:
            region_matches[admin1] = {
                'facilities': matches,
                'count': df[df[facility_col].isin(matches)][facility_col].count()
            }
    
    print("SUGGESTED ADMIN1 MAPPINGS:")
    print("-" * 40)
    total_mapped = 0
    
    for admin1, info in sorted(region_matches.items(), 
                              key=lambda x: x[1]['count'], reverse=True):
        print(f"\n{admin1}: {info['count']:,} consultations")
        for facility in sorted(info['facilities'])[:10]:  # Show top 10
            count = facility_counts.get(facility, 0)
            print(f"  • {facility} ({count:,})")
        if len(info['facilities']) > 10:
            print(f"  ... and {len(info['facilities'])-10} more facilities")
        total_mapped += info['count']
    
    unmapped_count = len(df) - total_mapped
    print(f"\n📊 MAPPING COVERAGE:")
    print(f"✅ Mapped: {total_mapped:,} consultations ({total_mapped/len(df)*100:.1f}%)")
    print(f"⚠️ Unmapped: {unmapped_count:,} consultations ({unmapped_count/len(df)*100:.1f}%)")
    
    # Show unmapped facilities
    mapped_facilities = set()
    for info in region_matches.values():
        mapped_facilities.update(info['facilities'])
    
    unmapped_facilities = set(facilities) - mapped_facilities
    if unmapped_facilities:
        print(f"\n⚠️ TOP 10 UNMAPPED FACILITIES:")
        print("-" * 30)
        unmapped_counts = facility_counts[facility_counts.index.isin(unmapped_facilities)]
        for facility, count in unmapped_counts.head(10).items():
            print(f"  {count:4,} | {facility}")
    
    # Generate Python mapping code
    print(f"\n💻 SUGGESTED PYTHON MAPPING CODE:")
    print("=" * 50)
    print("facility_admin_mapping = {")
    
    for admin1, info in sorted(region_matches.items()):
        for facility in sorted(info['facilities'])[:5]:  # Top 5 per region
            print(f"    '{facility}': {{")
            print(f"        'admin1': '{admin1}',")
            print(f"        'admin2': 'Unknown',  # TODO: Add admin2 mapping")
            print(f"        'admin3': 'Unknown'   # TODO: Add admin3 mapping")
            print(f"    }},")
    print("}")
    
    print(f"\n✅ Analysis complete!")
    

if __name__ == "__main__":
    analyze_facility_names()