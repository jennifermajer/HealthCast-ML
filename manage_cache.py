#!/usr/bin/env python3
"""
Taxonomy Cache Management Utility

Provides command-line interface for managing the taxonomy mapping cache.
"""

import argparse
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from taxonomy_cache import TaxonomyCache
    import yaml
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you are running from the project root directory")
    sys.exit(1)

def load_config():
    """Load configuration file."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️ Could not load config.yaml: {e}")
        return {}

def cmd_info(args):
    """Show cache information."""
    config = load_config()
    taxonomy_config = config.get('taxonomy', {})
    
    cache_dir = taxonomy_config.get('cache_dir', 'data/cache')
    use_cache = taxonomy_config.get('use_cache', True)
    
    cache = TaxonomyCache(cache_dir=cache_dir, enabled=use_cache)
    info = cache.get_cache_info()
    
    print("📊 Taxonomy Cache Information")
    print("=" * 40)
    print(f"Status:        {'Enabled' if info['enabled'] else 'Disabled'}")
    
    if info['enabled']:
        print(f"Cache dir:     {info['cache_dir']}")
        print(f"Cache file:    {info['cache_file']}")
        print(f"Cache exists:  {'Yes' if info['cache_exists'] else 'No'}")
        print(f"Cached keys:   {info['num_cached_keys']}")
        
        if info['num_cached_keys'] > 0:
            print(f"Oldest entry:  {info.get('oldest_entry', 'Unknown')}")
            print(f"Newest entry:  {info.get('newest_entry', 'Unknown')}")
            print(f"Morbidities:   {info.get('total_unique_morbidities', 'Unknown')}")
    
    return True

def cmd_clear(args):
    """Clear the cache."""
    config = load_config()
    taxonomy_config = config.get('taxonomy', {})
    
    cache_dir = taxonomy_config.get('cache_dir', 'data/cache')
    use_cache = taxonomy_config.get('use_cache', True)
    
    if not use_cache:
        print("⚠️ Cache is disabled in configuration")
        return False
    
    cache = TaxonomyCache(cache_dir=cache_dir, enabled=True)
    
    if args.force or input("🗑️ Are you sure you want to clear the taxonomy cache? [y/N]: ").lower() == 'y':
        cache.clear_cache()
        print("✅ Taxonomy cache cleared")
        return True
    else:
        print("❌ Cache clear cancelled")
        return False

def cmd_enable(args):
    """Enable cache in configuration."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        if 'taxonomy' not in config:
            config['taxonomy'] = {}
        
        config['taxonomy']['use_cache'] = True
        
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Cache enabled in configuration")
        return True
    except Exception as e:
        print(f"❌ Failed to enable cache: {e}")
        return False

def cmd_disable(args):
    """Disable cache in configuration."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        if 'taxonomy' not in config:
            config['taxonomy'] = {}
        
        config['taxonomy']['use_cache'] = False
        
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Cache disabled in configuration")
        return True
    except Exception as e:
        print(f"❌ Failed to disable cache: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Taxonomy Cache Management Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_cache.py info       # Show cache information
  python manage_cache.py clear      # Clear cache (with confirmation)
  python manage_cache.py clear -f   # Force clear without confirmation
  python manage_cache.py enable     # Enable cache in config
  python manage_cache.py disable    # Disable cache in config
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show cache information')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the cache')
    clear_parser.add_argument('-f', '--force', action='store_true', 
                             help='Force clear without confirmation')
    
    # Enable command
    enable_parser = subparsers.add_parser('enable', help='Enable cache in configuration')
    
    # Disable command
    disable_parser = subparsers.add_parser('disable', help='Disable cache in configuration')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    commands = {
        'info': cmd_info,
        'clear': cmd_clear,
        'enable': cmd_enable,
        'disable': cmd_disable,
    }
    
    try:
        success = commands[args.command](args)
        return 0 if success else 1
    except KeyError:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        return 1
    except Exception as e:
        print(f"❌ Error executing command: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())