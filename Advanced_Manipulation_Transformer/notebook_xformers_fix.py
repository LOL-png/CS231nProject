"""
Quick Fix for XFormers Error in Notebook
========================================

Add this to your notebook to fix the XFormers error.
"""

# ============================================
# ADD THIS TO YOUR NOTEBOOK TO FIX THE ERROR
# ============================================

from fix_model_xformers import fix_model_for_notebook

# Fix the model (removes XFormers and applies PyTorch optimizations)
print("Fixing model - removing XFormers and applying PyTorch native optimizations...")
model = fix_model_for_notebook(model)
print("✓ Model fixed! XFormers removed and native optimizations applied.")

# Now the debugger should work
debugger = ModelDebugger(model, save_dir=output_dir / "debug")
if config.debug.debug_initial_model:
    print("Analyzing initial model...")
    debugger.analyze_model(sample_batch)  # This should work now
    print("✓ Model analysis complete!")

# ============================================
# EXPLANATION OF THE FIX
# ============================================
"""
What happened:
1. Earlier in your notebook, MemoryOptimizer applied XFormers to the model
2. XFormers doesn't support cross-attention in TransformerEncoder
3. PyTorch native optimizations didn't remove the existing XFormers modules

What the fix does:
1. Finds all XFormersAttention modules in the model
2. Replaces them with standard PyTorch MultiheadAttention
3. Copies the weights to preserve training
4. Applies PyTorch native optimizations (SDPA, etc.)

Result:
- Same performance (SDPA = Flash Attention)
- Full compatibility with all PyTorch features
- No more NotImplementedError
"""

# ============================================
# ALTERNATIVE: PREVENT THE ISSUE
# ============================================
"""
To prevent this issue in future runs, replace your optimization section with:

# DON'T use this:
# memory_optimizer = MemoryOptimizer()
# model = memory_optimizer.optimize_model_for_h200(model, config)

# USE this instead:
from optimizations.pytorch_native_optimization import optimize_for_h200
model = optimize_for_h200(model)
"""

# ============================================
# DEBUGGING HELP
# ============================================

def check_model_status(model):
    """Check if model has XFormers or other issues"""
    has_xformers = False
    xformers_locations = []
    
    for name, module in model.named_modules():
        if type(module).__name__ == 'XFormersAttention':
            has_xformers = True
            xformers_locations.append(name)
    
    if has_xformers:
        print(f"⚠️  Model has {len(xformers_locations)} XFormersAttention modules:")
        for loc in xformers_locations[:5]:  # Show first 5
            print(f"    - {loc}")
        if len(xformers_locations) > 5:
            print(f"    ... and {len(xformers_locations) - 5} more")
        print("\nFix: Run fix_model_for_notebook(model)")
    else:
        print("✓ Model is clean - no XFormers modules found")
        
        # Check for standard attention
        multihead_count = sum(1 for _, m in model.named_modules() 
                            if isinstance(m, torch.nn.MultiheadAttention))
        print(f"✓ Has {multihead_count} standard MultiheadAttention modules")
        
        # Check if optimized
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("✓ PyTorch SDPA available - will use Flash Attention automatically")

# You can use this to check your model:
# check_model_status(model)