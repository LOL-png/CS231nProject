#!/usr/bin/env python3
"""
Create a zip archive of the 231nProjectV2 codebase, excluding:
- Model files (*.pth, *.pt, *.h5, *.ckpt)
- Images (*.png, *.jpg, *.jpeg, *.gif)
- Results and outputs
- Large data files
- Cache directories
- Wandb logs
"""

import os
import zipfile
from pathlib import Path
import argparse

# Patterns to exclude
EXCLUDE_EXTENSIONS = {
    # Model files
    '.pth', '.pt', '.h5', '.ckpt', '.pkl', '.pickle',
    '.safetensors', '.bin', '.model',
    
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
    '.svg', '.ico', '.webp',
    
    # Videos
    '.mp4', '.avi', '.mov', '.mkv', '.webm',
    
    # Data files
    '.npy', '.npz', '.hdf5', '.tfrecord', '.csv',
    '.parquet', '.feather', '.arrow',
    
    # Archives
    '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
    
    # Logs
    '.log',
    
    # Other large files
    '.obj', '.ply', '.stl', '.fbx', '.dae',
}

EXCLUDE_DIRS = {
    '__pycache__', '.ipynb_checkpoints', 'wandb', 'outputs',
    'checkpoints', 'cache', 'gpu_cache', 'gpu_cache_optimized',
    'gpu_cache_sweep', 'gpu_cache_advanced', 'results',
    'evaluation_results', '.git', 'data', 'dex-ycb',
    'visualization_results', 'test_debug',
}

EXCLUDE_PATTERNS = {
    'run-*', 'offline-run-*', 'sweep-*', '*_cache*',
    'train_gpu_cache_*', 'val_gpu_cache_*',
}

def should_exclude(file_path):
    """Check if a file should be excluded from the archive."""
    path = Path(file_path)
    
    # Check file extension
    if path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True
    
    # Check directory names
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
        # Check patterns
        for pattern in EXCLUDE_PATTERNS:
            if pattern.startswith('*') and pattern.endswith('*'):
                if pattern[1:-1] in part:
                    return True
            elif pattern.startswith('*'):
                if part.endswith(pattern[1:]):
                    return True
            elif pattern.endswith('*'):
                if part.startswith(pattern[:-1]):
                    return True
    
    # Exclude wandb files
    if 'wandb' in str(path) and path.suffix == '.wandb':
        return True
    
    # Exclude large data files in dex-ycb-toolkit
    if 'dex-ycb-toolkit/data' in str(path):
        return True
        
    return False

def create_archive(source_dir, output_file, verbose=False):
    """Create a zip archive of the source directory."""
    source_path = Path(source_dir)
    included_files = []
    excluded_files = []
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_path):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                file_path = os.path.join(root, file)
                if not should_exclude(file_path):
                    # Add file to archive with relative path
                    arcname = os.path.relpath(file_path, source_path.parent)
                    zipf.write(file_path, arcname)
                    included_files.append(arcname)
                    if verbose:
                        print(f"Added: {arcname}")
                else:
                    excluded_files.append(os.path.relpath(file_path, source_path.parent))
    
    return included_files, excluded_files

def main():
    parser = argparse.ArgumentParser(description='Create a code-only archive of 231nProjectV2')
    parser.add_argument('--output', '-o', default='231nProjectV2_code_only.zip',
                        help='Output zip file name')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print files being added')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be included without creating archive')
    
    args = parser.parse_args()
    
    # Get the project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir
    
    if args.dry_run:
        print("DRY RUN - Archive would include:\n")
        included = []
        excluded = []
        
        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_dir.parent)
                if not should_exclude(file_path):
                    included.append(rel_path)
                else:
                    excluded.append(rel_path)
        
        print(f"Would include {len(included)} files")
        print(f"Would exclude {len(excluded)} files")
        
        if args.verbose:
            print("\nIncluded files:")
            for f in sorted(included):
                print(f"  {f}")
    else:
        print(f"Creating archive: {args.output}")
        included, excluded = create_archive(project_dir, args.output, args.verbose)
        
        # Calculate sizes
        archive_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
        
        print(f"\nArchive created successfully!")
        print(f"Archive: {args.output} ({archive_size:.2f} MB)")
        print(f"Included: {len(included)} files")
        print(f"Excluded: {len(excluded)} files")
        
        # Show summary of excluded types
        ext_counts = {}
        for f in excluded:
            ext = Path(f).suffix.lower()
            if ext:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
        if ext_counts:
            print("\nExcluded file types:")
            for ext, count in sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {ext}: {count} files")

if __name__ == '__main__':
    main()