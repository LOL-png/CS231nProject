"""
GPU Prefetch DataLoader to eliminate CPU bottlenecks
Keeps next batch ready on GPU while current batch processes
"""

import torch
from torch.utils.data import DataLoader
import threading
import queue


class GPUPrefetchLoader:
    """
    DataLoader wrapper that prefetches batches to GPU
    Eliminates CPU-GPU transfer bottleneck
    """
    
    def __init__(self, dataloader, device='cuda', prefetch_batches=2):
        self.dataloader = dataloader
        self.device = device
        self.prefetch_batches = prefetch_batches
        
    def __iter__(self):
        # Create queue for prefetched batches
        batch_queue = queue.Queue(maxsize=self.prefetch_batches)
        
        # Function to load batches in background
        def prefetch_worker():
            for batch in self.dataloader:
                # Move batch to GPU in background thread
                gpu_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        # Non-blocking transfer to GPU
                        gpu_batch[key] = value.to(self.device, non_blocking=True)
                    else:
                        gpu_batch[key] = value
                
                # Put in queue (blocks if queue is full)
                batch_queue.put(gpu_batch)
            
            # Signal end of data
            batch_queue.put(None)
        
        # Start prefetch thread
        prefetch_thread = threading.Thread(target=prefetch_worker)
        prefetch_thread.start()
        
        # Yield batches from queue
        while True:
            batch = batch_queue.get()
            if batch is None:
                break
            
            # Ensure GPU transfer is complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            yield batch
        
        # Wait for thread to finish
        prefetch_thread.join()
    
    def __len__(self):
        return len(self.dataloader)


class CachedBatchLoader:
    """
    Caches entire dataset in GPU memory for ultimate performance
    Only suitable when dataset fits in GPU memory
    """
    
    def __init__(self, dataset, batch_size, device='cuda', shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        
        print("Caching entire dataset to GPU memory...")
        
        # Load all data to GPU
        self.cached_data = {}
        
        # Process in chunks to avoid CPU memory issues
        chunk_size = 1000
        for start_idx in range(0, len(dataset), chunk_size):
            end_idx = min(start_idx + chunk_size, len(dataset))
            
            # Load chunk
            chunk_data = [dataset[i] for i in range(start_idx, end_idx)]
            
            # Stack and move to GPU
            for key in chunk_data[0].keys():
                if key not in self.cached_data:
                    self.cached_data[key] = []
                
                if isinstance(chunk_data[0][key], torch.Tensor):
                    stacked = torch.stack([item[key] for item in chunk_data])
                    self.cached_data[key].append(stacked.to(device))
                else:
                    # Keep non-tensor data as is
                    self.cached_data[key].extend([item[key] for item in chunk_data])
            
            print(f"  Cached {end_idx}/{len(dataset)} samples to GPU")
        
        # Concatenate all chunks
        for key in list(self.cached_data.keys()):
            if isinstance(self.cached_data[key][0], torch.Tensor):
                self.cached_data[key] = torch.cat(self.cached_data[key], dim=0)
        
        self.num_samples = len(dataset)
        print(f"✓ Cached {self.num_samples} samples to GPU memory")
        
    def __iter__(self):
        # Generate indices
        if self.shuffle:
            indices = torch.randperm(self.num_samples, device=self.device)
        else:
            indices = torch.arange(self.num_samples, device=self.device)
        
        # Yield batches
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Create batch from cached data
            batch = {}
            for key, value in self.cached_data.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value[batch_indices]
                else:
                    # Handle non-tensor data
                    batch[key] = [value[idx] for idx in batch_indices.cpu().numpy()]
            
            yield batch
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size