import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DistributedTrainingSetup:
    """
    Distributed training setup for multi-GPU/node training
    """
    
    @staticmethod
    def init_distributed(backend: str = 'nccl') -> bool:
        """
        Initialize distributed training
        """
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            
            # Initialize process group
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            
            # Set device
            torch.cuda.set_device(rank)
            
            logger.info(f"Initialized distributed training: rank {rank}/{world_size}")
            return True
        
        return False
    
    @staticmethod
    def wrap_model_ddp(
        model: nn.Module,
        find_unused_parameters: bool = False,
        mixed_precision: bool = True
    ) -> DDP:
        """
        Wrap model with DistributedDataParallel
        """
        device_id = torch.cuda.current_device()
        
        model = DDP(
            model.cuda(device_id),
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=True
        )
        
        logger.info("Model wrapped with DistributedDataParallel")
        return model
    
    @staticmethod
    def wrap_model_fsdp(
        model: nn.Module,
        config: Dict,
        mixed_precision: bool = True
    ) -> FSDP:
        """
        Wrap model with FullyShardedDataParallel for memory efficiency
        """
        # Auto wrap policy for transformer layers
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={
                nn.TransformerEncoderLayer,
                nn.TransformerDecoderLayer
            },
            min_num_params=1e6  # Wrap layers with >1M params
        )
        
        # Mixed precision config
        mixed_precision_policy = None
        if mixed_precision:
            from torch.distributed.fsdp import MixedPrecision
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16
            )
        
        # Wrap model
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
            use_orig_params=True  # For optimizer compatibility
        )
        
        logger.info("Model wrapped with FullyShardedDataParallel")
        return model
    
    @staticmethod
    def all_reduce_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        All-reduce metrics across distributed processes
        """
        if not dist.is_initialized():
            return metrics
        
        reduced_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                # Move to current device if needed
                if not value.is_cuda:
                    value = value.cuda()
                
                # All-reduce
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value = value / dist.get_world_size()
                
                reduced_metrics[key] = value
            else:
                reduced_metrics[key] = value
        
        return reduced_metrics
    
    @staticmethod
    def save_checkpoint_distributed(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        checkpoint_path: str,
        **kwargs
    ):
        """
        Save checkpoint in distributed setting
        """
        # Only save on rank 0
        if dist.get_rank() != 0:
            return
        
        # Get state dict
        if isinstance(model, (DDP, FSDP)):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            **kwargs
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    @staticmethod
    def load_checkpoint_distributed(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str
    ) -> int:
        """
        Load checkpoint in distributed setting
        """
        # Load checkpoint
        map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        if isinstance(model, (DDP, FSDP)):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Loaded checkpoint from {checkpoint_path}, epoch {epoch}")
        
        return epoch
    
    @staticmethod
    def cleanup():
        """
        Cleanup distributed training
        """
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Cleaned up distributed training")