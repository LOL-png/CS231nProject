# Dataset Split Name Fix

## Problem
The DexYCB toolkit expects dataset names in the format `s0_train`, but our code was using various formats:
- `train` (without prefix)
- `s0-train` (with hyphen instead of underscore)
- `s0_train` (correct format)

## Available Dataset Names
From the factory.py, the valid dataset names are:
- `s0_train`, `s0_val`, `s0_test`
- `s1_train`, `s1_val`, `s1_test`
- `s2_train`, `s2_val`, `s2_test`
- `s3_train`, `s3_val`, `s3_test`

## Solution Applied
Updated `enhanced_dexycb.py` to:
1. Handle both formats: 'train' and 's0_train'
2. Convert hyphens to underscores
3. Default to 's0_' prefix if not provided

## How It Works
```python
# Handle both formats: 'train' and 's0_train'
if self.split.startswith('s'):
    # Already has the s0_ prefix
    dataset = get_dataset(self.split)
else:
    # Add s0_ prefix
    if self.split == 'train':
        dataset = get_dataset('s0_train')
    elif self.split == 'val':
        dataset = get_dataset('s0_val')
    else:
        dataset = get_dataset('s0_test')
```

## Usage Examples
All of these will now work:
- `split='train'` → loads `s0_train`
- `split='val'` → loads `s0_val`
- `split='s0_train'` → loads `s0_train`
- `split='s1_train'` → loads `s1_train`