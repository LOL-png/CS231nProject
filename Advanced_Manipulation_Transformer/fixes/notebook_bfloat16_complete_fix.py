"""
Complete BFloat16 fixes for notebook cells
This file contains all the necessary fixes for BFloat16/Float32 compatibility issues
"""

# Cell fix for handling camera parameters in batch
CAMERA_PARAMS_FIX = """
def fix_batch_for_model(batch, model):
    '''Fix dtype issues in batch, especially for camera parameters'''
    # Get model dtype
    model_dtype = next(model.parameters()).dtype
    
    # Fix camera intrinsics if present
    if 'camera_intrinsics' in batch and isinstance(batch['camera_intrinsics'], torch.Tensor):
        batch['camera_intrinsics'] = batch['camera_intrinsics'].to(model_dtype)
    
    # Create camera_params dict if needed
    if 'camera_params' not in batch and 'camera_intrinsics' in batch:
        batch['camera_params'] = {'intrinsics': batch['camera_intrinsics']}
    
    # Fix camera_params dictionary
    if 'camera_params' in batch and isinstance(batch['camera_params'], dict):
        fixed_params = {}
        for key, value in batch['camera_params'].items():
            if isinstance(value, torch.Tensor):
                fixed_params[key] = value.to(model_dtype)
            else:
                fixed_params[key] = value
        batch['camera_params'] = fixed_params
    
    return batch
"""

# Updated cell 18 fix
CELL_18_FIX = """
# Debug initial model with memory-efficient approach
if config.debug.debug_initial_model:
    print("Analyzing initial model with memory-efficient debugger...")
    
    # Sample batch is already on GPU from GPU-cached dataset
    print("Using GPU-cached sample batch (no CPU-GPU transfer needed)")
    
    # IMPORTANT: Temporarily disable torch.compile for debugging
    # The debugger uses hooks that are incompatible with graph compilation
    was_compiled = hasattr(model, '_orig_mod')
    if was_compiled:
        print("Temporarily disabling torch.compile for debugging...")
        original_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        model = original_model
    
    # Convert sample batch to float32 for DINOv2 compatibility
    sample_batch_float32 = {}
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
            sample_batch_float32[k] = v.float()
        else:
            sample_batch_float32[k] = v
    
    # Fix camera parameters
    def fix_batch_for_model(batch, model):
        '''Fix dtype issues in batch, especially for camera parameters'''
        # Get model dtype
        model_dtype = next(model.parameters()).dtype
        
        # Fix camera intrinsics if present
        if 'camera_intrinsics' in batch and isinstance(batch['camera_intrinsics'], torch.Tensor):
            batch['camera_intrinsics'] = batch['camera_intrinsics'].to(model_dtype)
        
        # Create camera_params dict if needed
        if 'camera_params' not in batch and 'camera_intrinsics' in batch:
            batch['camera_params'] = {'intrinsics': batch['camera_intrinsics']}
        
        # Fix camera_params dictionary
        if 'camera_params' in batch and isinstance(batch['camera_params'], dict):
            fixed_params = {}
            for key, value in batch['camera_params'].items():
                if isinstance(value, torch.Tensor):
                    fixed_params[key] = value.to(model_dtype)
                else:
                    fixed_params[key] = value
            batch['camera_params'] = fixed_params
        
        return batch
    
    # Fix camera parameters
    sample_batch_float32 = fix_batch_for_model(sample_batch_float32, model)
    
    # Run analysis with memory cleanup
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Analyze model with float32 batch
    debugger.analyze_model(sample_batch_float32)
    
    # Check initial prediction diversity with limited batches
    print("\\nChecking initial prediction diversity (limited to 3 batches for memory)...")
    
    # Get 3 batches from GPU-cached dataloader
    simple_batches = []
    batch_iter = iter(val_loader)
    for i in range(3):
        try:
            batch = next(batch_iter)
            # Convert to float32
            batch_float32 = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    batch_float32[k] = v.float()
                else:
                    batch_float32[k] = v
            # Fix camera parameters
            batch_float32 = fix_batch_for_model(batch_float32, model)
            simple_batches.append(batch_float32)
        except StopIteration:
            break
    
    # Check diversity with GPU batches
    debugger.debug_prediction_diversity(simple_batches, num_batches=len(simple_batches))
    
    # Re-enable torch.compile if it was enabled
    if was_compiled:
        print("\\nRe-enabling torch.compile...")
        model = torch.compile(model, mode='default')
    
    # Clean up memory
    del simple_batches
    del sample_batch_float32
    gc.collect()
    torch.cuda.empty_cache()
    
    # Display debug visualizations if they exist
    from IPython.display import Image
    debug_files = ['gradient_flow.png', 'prediction_diversity.png']
    
    for file in debug_files:
        path = f"{config.output_dir}/debug/{file}"
        if os.path.exists(path):
            print(f"\\nDisplaying {file}:")
            display(Image(path))
    
    print(f"\\nInitial debugging complete")
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated")
    print(f"Peak Memory: {torch.cuda.max_memory_reserved() / 1e9:.1f} GB reserved")
"""

# Updated cell 19 fix
CELL_19_FIX = """
# Debug initial model
if config.debug.debug_initial_model:
    print("Analyzing initial model...")
    
    # Define helper function
    def fix_batch_for_model(batch, model):
        '''Fix dtype issues in batch, especially for camera parameters'''
        # Get model dtype
        model_dtype = next(model.parameters()).dtype
        
        # Fix camera intrinsics if present
        if 'camera_intrinsics' in batch and isinstance(batch['camera_intrinsics'], torch.Tensor):
            batch['camera_intrinsics'] = batch['camera_intrinsics'].to(model_dtype)
        
        # Create camera_params dict if needed
        if 'camera_params' not in batch and 'camera_intrinsics' in batch:
            batch['camera_params'] = {'intrinsics': batch['camera_intrinsics']}
        
        # Fix camera_params dictionary
        if 'camera_params' in batch and isinstance(batch['camera_params'], dict):
            fixed_params = {}
            for key, value in batch['camera_params'].items():
                if isinstance(value, torch.Tensor):
                    fixed_params[key] = value.to(model_dtype)
                else:
                    fixed_params[key] = value
            batch['camera_params'] = fixed_params
        
        return batch
    
    # CRITICAL FIX: Convert sample batch to float32 for DINOv2
    sample_batch_float32 = {}
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
            sample_batch_float32[k] = v.float()
        else:
            sample_batch_float32[k] = v
    
    # Fix camera parameters
    sample_batch_float32 = fix_batch_for_model(sample_batch_float32, model)
    
    debugger.analyze_model(sample_batch_float32)
    
    # Check initial prediction diversity
    print("\\nChecking initial prediction diversity...")
    
    # Create a wrapper for val_loader that converts BFloat16 to Float32
    class Float32BatchIterator:
        def __init__(self, loader, num_batches, model):
            self.loader = loader
            self.num_batches = num_batches
            self.model = model
            
        def __iter__(self):
            count = 0
            for batch in self.loader:
                if count >= self.num_batches:
                    break
                # Convert BFloat16 to Float32
                if 'image' in batch and batch['image'].dtype == torch.bfloat16:
                    batch['image'] = batch['image'].float()
                # Fix camera parameters
                batch = fix_batch_for_model(batch, self.model)
                yield batch
                count += 1
                
        def __len__(self):
            return self.num_batches
    
    val_loader_debug = Float32BatchIterator(val_loader, num_batches=5, model=model)
    debugger.debug_prediction_diversity(val_loader_debug, num_batches=5)
    
    # Display debug visualizations
    from IPython.display import Image
    debug_files = ['activation_distributions.png', 'gradient_flow.png', 
                   'parameter_distributions.png', 'prediction_diversity.png']
    
    for file in debug_files:
        path = f"{config.output_dir}/debug/{file}"
        if os.path.exists(path):
            display(Image(path))
"""