#!/usr/bin/env python3
"""
System Stability Monitor and Optimization

Monitors system resources and provides recommendations to prevent crashes
during large machine learning operations.
"""

import psutil
import os
import time
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class SystemStabilityMonitor:
    """Monitor and optimize system resources for stability"""
    
    def __init__(self):
        self.memory_threshold = 85  # Percentage
        self.cpu_threshold = 90     # Percentage
        self.disk_threshold = 95    # Percentage
        self.monitoring = False
        
    def check_system_health(self) -> Dict[str, any]:
        """Check overall system health"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'memory': self._check_memory(),
            'cpu': self._check_cpu(),
            'disk': self._check_disk(),
            'running_processes': self._check_running_processes(),
            'recommendations': []
        }
        
        # Generate recommendations
        health_report['recommendations'] = self._generate_recommendations(health_report)
        
        return health_report
    
    def _check_memory(self) -> Dict[str, any]:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'percent_used': memory.percent,
            'swap_used_gb': round(swap.used / (1024**3), 2),
            'swap_percent': swap.percent,
            'status': 'critical' if memory.percent > self.memory_threshold else 'normal'
        }
    
    def _check_cpu(self) -> Dict[str, any]:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        
        return {
            'usage_percent': cpu_percent,
            'core_count': cpu_count,
            'load_average_1m': load_avg[0],
            'load_average_5m': load_avg[1],
            'load_average_15m': load_avg[2],
            'status': 'critical' if cpu_percent > self.cpu_threshold else 'normal'
        }
    
    def _check_disk(self) -> Dict[str, any]:
        """Check disk usage"""
        disk = psutil.disk_usage('/')
        
        return {
            'total_gb': round(disk.total / (1024**3), 2),
            'used_gb': round(disk.used / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2),
            'percent_used': round((disk.used / disk.total) * 100, 2),
            'status': 'critical' if (disk.used / disk.total) * 100 > self.disk_threshold else 'normal'
        }
    
    def _check_running_processes(self) -> List[Dict[str, any]]:
        """Check for resource-intensive processes"""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
            try:
                pinfo = proc.info
                memory_pct = pinfo.get('memory_percent', 0) or 0
                cpu_pct = pinfo.get('cpu_percent', 0) or 0
                
                if memory_pct > 5 or cpu_pct > 10:  # High resource usage
                    processes.append({
                        'pid': pinfo['pid'],
                        'name': pinfo['name'],
                        'memory_percent': round(memory_pct, 2),
                        'cpu_percent': round(cpu_pct, 2)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by memory usage
        processes.sort(key=lambda x: x['memory_percent'], reverse=True)
        return processes[:10]  # Top 10
    
    def _generate_recommendations(self, health_report: Dict) -> List[str]:
        """Generate system optimization recommendations"""
        recommendations = []
        
        memory = health_report['memory']
        cpu = health_report['cpu']
        disk = health_report['disk']
        
        # Memory recommendations
        if memory['status'] == 'critical':
            recommendations.append(f"⚠️ CRITICAL: Memory usage at {memory['percent_used']:.1f}%. Consider:")
            recommendations.append("  • Close unnecessary applications")
            recommendations.append("  • Use --quick flag for faster analysis")
            recommendations.append("  • Process data in smaller chunks")
            
        # CPU recommendations
        if cpu['status'] == 'critical':
            recommendations.append(f"⚠️ CRITICAL: CPU usage at {cpu['usage_percent']:.1f}%. Consider:")
            recommendations.append("  • Reduce model complexity")
            recommendations.append("  • Use fewer CPU cores for training")
            recommendations.append("  • Run analysis during off-peak hours")
        
        # Disk recommendations  
        if disk['status'] == 'critical':
            recommendations.append(f"⚠️ CRITICAL: Disk usage at {disk['percent_used']:.1f}%. Consider:")
            recommendations.append("  • Clean up old log files and results")
            recommendations.append("  • Remove temporary files")
            recommendations.append("  • Move large files to external storage")
        
        # Process-specific recommendations
        high_memory_procs = [p for p in health_report['running_processes'] if p['memory_percent'] > 10]
        if high_memory_procs:
            recommendations.append("🔍 High memory processes detected:")
            for proc in high_memory_procs[:3]:  # Top 3
                recommendations.append(f"  • {proc['name']} (PID: {proc['pid']}): {proc['memory_percent']:.1f}% memory")
        
        return recommendations
    
    def optimize_for_ml_analysis(self) -> Dict[str, str]:
        """Optimize system settings for machine learning analysis"""
        optimizations = {}
        
        # Clear system caches (macOS)
        if os.uname().sysname == 'Darwin':
            try:
                subprocess.run(['sudo', 'purge'], capture_output=True)
                optimizations['cache_cleared'] = "✅ System caches cleared"
            except:
                optimizations['cache_cleared'] = "❌ Could not clear caches (requires sudo)"
        
        # Set environment variables for better memory management
        os.environ['OMP_NUM_THREADS'] = str(min(4, psutil.cpu_count()))
        os.environ['MKL_NUM_THREADS'] = str(min(4, psutil.cpu_count()))
        optimizations['thread_limiting'] = f"✅ Limited threads to {os.environ['OMP_NUM_THREADS']}"
        
        # Python garbage collection optimization
        import gc
        gc.collect()
        optimizations['garbage_collection'] = "✅ Forced garbage collection"
        
        return optimizations
    
    def monitor_process(self, pid: int, duration_minutes: int = 60):
        """Monitor a specific process for resource usage"""
        try:
            process = psutil.Process(pid)
            monitoring_data = []
            
            print(f"🔍 Monitoring process {process.name()} (PID: {pid}) for {duration_minutes} minutes...")
            
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            while time.time() < end_time:
                try:
                    data_point = {
                        'timestamp': datetime.now().isoformat(),
                        'memory_mb': round(process.memory_info().rss / (1024**2), 2),
                        'memory_percent': round(process.memory_percent(), 2),
                        'cpu_percent': round(process.cpu_percent(), 2)
                    }
                    monitoring_data.append(data_point)
                    
                    # Print periodic updates
                    if len(monitoring_data) % 60 == 0:  # Every minute
                        print(f"📊 Memory: {data_point['memory_mb']}MB ({data_point['memory_percent']:.1f}%), CPU: {data_point['cpu_percent']:.1f}%")
                    
                    time.sleep(1)
                    
                except psutil.NoSuchProcess:
                    print("❌ Process ended or not accessible")
                    break
                    
            return monitoring_data
            
        except psutil.NoSuchProcess:
            print(f"❌ Process with PID {pid} not found")
            return []
    
    def kill_hanging_processes(self, process_patterns: List[str]) -> Dict[str, List[int]]:
        """Kill processes matching certain patterns (use carefully!)"""
        killed_processes = {}
        
        for pattern in process_patterns:
            killed_pids = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    pinfo = proc.info
                    process_string = f"{pinfo['name']} {' '.join(pinfo['cmdline'] or [])}"
                    
                    if pattern.lower() in process_string.lower():
                        print(f"⚠️ Killing process: {pinfo['name']} (PID: {pinfo['pid']})")
                        proc.kill()
                        killed_pids.append(pinfo['pid'])
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            killed_processes[pattern] = killed_pids
        
        return killed_processes

def cleanup_temp_files():
    """Clean up temporary files to free disk space"""
    cleanup_paths = [
        Path.home() / '.cache',
        Path('/tmp'),
        Path.cwd() / '__pycache__',
        Path.cwd() / '.pytest_cache',
        Path.cwd() / 'logs' / '*.log',
        Path.cwd() / 'results' / 'temp'
    ]
    
    total_freed = 0
    
    for path in cleanup_paths:
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                path.unlink()
                total_freed += size
            elif path.is_dir() and 'cache' in path.name:
                # Clean cache directories
                for file in path.glob('**/*'):
                    if file.is_file():
                        size = file.stat().st_size
                        try:
                            file.unlink()
                            total_freed += size
                        except:
                            pass
    
    print(f"🧹 Freed {total_freed / (1024**2):.2f} MB of disk space")
    return total_freed

def main():
    """Main function for standalone usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='System Stability Monitor')
    parser.add_argument('--check', action='store_true', help='Check system health')
    parser.add_argument('--optimize', action='store_true', help='Optimize for ML analysis')
    parser.add_argument('--monitor', type=int, help='Monitor process by PID')
    parser.add_argument('--cleanup', action='store_true', help='Clean up temporary files')
    parser.add_argument('--kill-hanging', action='store_true', help='Kill hanging Python processes')
    
    args = parser.parse_args()
    
    monitor = SystemStabilityMonitor()
    
    if args.check:
        health = monitor.check_system_health()
        print("🏥 SYSTEM HEALTH REPORT")
        print("="*50)
        print(f"💾 Memory: {health['memory']['used_gb']:.1f}GB/{health['memory']['total_gb']:.1f}GB ({health['memory']['percent_used']:.1f}%)")
        print(f"⚡ CPU: {health['cpu']['usage_percent']:.1f}% ({health['cpu']['core_count']} cores)")
        print(f"💿 Disk: {health['disk']['used_gb']:.1f}GB/{health['disk']['total_gb']:.1f}GB ({health['disk']['percent_used']:.1f}%)")
        
        if health['recommendations']:
            print("\n📋 RECOMMENDATIONS:")
            for rec in health['recommendations']:
                print(rec)
    
    elif args.optimize:
        print("⚙️ Optimizing system for ML analysis...")
        optimizations = monitor.optimize_for_ml_analysis()
        for key, value in optimizations.items():
            print(f"{value}")
    
    elif args.monitor:
        monitor.monitor_process(args.monitor)
    
    elif args.cleanup:
        cleanup_temp_files()
    
    elif args.kill_hanging:
        print("⚠️ Killing hanging Python processes...")
        killed = monitor.kill_hanging_processes(['python', 'jupyter'])
        for pattern, pids in killed.items():
            if pids:
                print(f"Killed {len(pids)} processes matching '{pattern}': {pids}")
    
    else:
        # Interactive mode
        health = monitor.check_system_health()
        print("🖥️ SYSTEM STATUS")
        print("="*40)
        print(f"Memory: {health['memory']['percent_used']:.1f}% ({health['memory']['status']})")
        print(f"CPU: {health['cpu']['usage_percent']:.1f}% ({health['cpu']['status']})")
        print(f"Disk: {health['disk']['percent_used']:.1f}% ({health['disk']['status']})")

if __name__ == '__main__':
    main()