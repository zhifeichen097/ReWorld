# =============================================================================
# Sampler Classes
# =============================================================================

from typing import List, Optional

import torch
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler):
    """
    Distributed sampler with per-sample importance weighting.
    
    Each GPU rank gets a disjoint subset sampled according to weights.
    """
    
    def __init__(
        self, 
        weights: List[float], 
        num_samples_per_epoch: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True, 
        seed: int = 0,
    ):
        """
        Args:
            weights: Per-sample importance weights.
            num_samples_per_epoch: Number of samples per epoch.
            num_replicas: Number of distributed processes.
            rank: Current process rank.
            replacement: Whether to sample with replacement.
            seed: Random seed.
        """
        import torch.distributed as dist
        
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
            
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        total = num_samples_per_epoch if num_samples_per_epoch else len(self.weights)
        
        # Make divisible by num_replicas
        self.total_size = ((total + num_replicas - 1) // num_replicas) * num_replicas
        self.num_samples = self.total_size // num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = torch.multinomial(
            self.weights, 
            self.total_size, 
            replacement=self.replacement, 
            generator=g
        ).tolist()
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible sampling."""
        self.epoch = epoch
