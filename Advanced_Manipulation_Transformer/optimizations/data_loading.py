import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp
from typing import Optional
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class OptimizedDataLoader:
    """
    Optimized data loading for H200 with maximum throughput
    """
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        distributed: bool = False
    ) -> DataLoader:
        """
        Create optimized dataloader for H200
        """
        # Auto-detect optimal number of workers
        if num_workers is None:
            num_workers = min(mp.cpu_count() // 2, 16)
        
        # Create sampler
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # Sampler handles shuffling
        else:
            sampler = None
        
        # Create dataloader with optimizations
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else 2,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=True,  # For consistent batch sizes
            multiprocessing_context='spawn' if num_workers > 0 else None
        )
        
        # Wrap with prefetcher for GPU
        if torch.cuda.is_available():
            dataloader = DataPrefetcher(dataloader)
        
        return dataloader

class DataPrefetcher:
    """
    Prefetch data to GPU for zero CPU-GPU transfer overhead
    """
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.stream = torch.cuda.Stream()
    
    def __iter__(self):
        first = True
        for next_batch in self.loader:
            with torch.cuda.stream(self.stream):
                # Transfer to GPU asynchronously
                next_batch = self._transfer_to_gpu(next_batch)
            
            if not first:
                # Wait for previous transfer
                torch.cuda.current_stream().wait_stream(self.stream)
                batch = next_batch
            else:
                # First iteration
                batch = next_batch
                first = False
            
            yield batch
    
    def __len__(self):
        return len(self.loader)
    
    def _transfer_to_gpu(self, batch):
        """Transfer batch to GPU with non-blocking"""
        if isinstance(batch, torch.Tensor):
            return batch.cuda(non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._transfer_to_gpu(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._transfer_to_gpu(v) for v in batch]
        else:
            return batch

class CachedDataset(Dataset):
    """
    Dataset that caches data in memory for faster access
    """
    
    def __init__(self, base_dataset: Dataset, cache_size: Optional[int] = None):
        self.base_dataset = base_dataset
        self.cache = {}
        self.cache_size = cache_size or len(base_dataset)
        
        # Preload if small enough
        if len(base_dataset) <= 10000:
            logger.info("Preloading dataset into memory...")
            for i in tqdm(range(len(base_dataset))):
                self.cache[i] = base_dataset[i]
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        # Load and cache
        item = self.base_dataset[idx]
        
        # LRU cache management
        if len(self.cache) >= self.cache_size:
            # Remove oldest item (simple FIFO for now)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[idx] = item
        return item