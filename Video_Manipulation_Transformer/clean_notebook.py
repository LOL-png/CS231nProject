#!/usr/bin/env python
"""
Clean the notebook to only keep H200 optimized cells
"""

import json

# Read the notebook
with open('train_stage1_notebook.ipynb', 'r') as f:
    notebook = json.load(f)

# Keep only the first 13 cells (0-12)
notebook['cells'] = notebook['cells'][:13]

# Save the cleaned notebook
with open('train_stage1_notebook_clean.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook cleaned - saved as train_stage1_notebook_clean.ipynb")