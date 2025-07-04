#!/usr/bin/env python
"""
Real-time GPU monitoring for H200 optimization
Shows utilization, memory, power, and temperature
"""

import subprocess
import time
import sys
from datetime import datetime

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            stats = []
            for line in lines:
                parts = line.split(', ')
                stats.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'gpu_util': float(parts[2]),
                    'mem_util': float(parts[3]),
                    'mem_used': float(parts[4]),
                    'mem_total': float(parts[5]),
                    'power': float(parts[6]),
                    'power_limit': float(parts[7]),
                    'temp': float(parts[8])
                })
            return stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None

def print_stats(stats):
    """Pretty print GPU statistics"""
    print("\033[H\033[J")  # Clear screen
    print(f"GPU Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    for gpu in stats:
        print(f"\nGPU {gpu['index']}: {gpu['name']}")
        print("-" * 40)
        
        # Utilization bars
        gpu_bar = "█" * int(gpu['gpu_util'] / 5) + "░" * (20 - int(gpu['gpu_util'] / 5))
        mem_bar = "█" * int(gpu['mem_util'] / 5) + "░" * (20 - int(gpu['mem_util'] / 5))
        power_bar = "█" * int((gpu['power'] / gpu['power_limit']) * 20) + "░" * (20 - int((gpu['power'] / gpu['power_limit']) * 20))
        
        print(f"GPU Util:    [{gpu_bar}] {gpu['gpu_util']:5.1f}%")
        print(f"Memory Util: [{mem_bar}] {gpu['mem_util']:5.1f}%")
        print(f"Memory:      {gpu['mem_used']/1024:6.1f} GB / {gpu['mem_total']/1024:6.1f} GB ({gpu['mem_used']/gpu['mem_total']*100:5.1f}%)")
        print(f"Power:       [{power_bar}] {gpu['power']:6.1f}W / {gpu['power_limit']:6.1f}W ({gpu['power']/gpu['power_limit']*100:5.1f}%)")
        print(f"Temperature: {gpu['temp']:5.1f}°C")
        
        # Warnings
        if gpu['gpu_util'] < 50:
            print("⚠️  Low GPU utilization - increase batch size or check data loading")
        if gpu['mem_used'] / gpu['mem_total'] < 0.1:
            print("⚠️  Low memory usage - scale up model or batch size")
        if gpu['power'] / gpu['power_limit'] < 0.5:
            print("⚠️  Low power usage - GPU is underutilized")
    
    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit")

def main():
    """Main monitoring loop"""
    print("Starting GPU monitoring...")
    print("Optimizing for H200 (140GB memory, 700W power)")
    
    try:
        while True:
            stats = get_gpu_stats()
            if stats:
                print_stats(stats)
            time.sleep(1)  # Update every second
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()