# =============================================================================
# LMDB Dataset Classes
# =============================================================================

from typing import Dict, Any, Optional
import os

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb


class ODERegressionLMDBDataset(Dataset):
    """
    LMDB dataset for ODE regression training.
    
    Loads pre-computed latents and prompts from LMDB format.
    """
    
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        """
        Args:
            data_path: Path to LMDB directory.
            max_pair: Maximum number of samples to load.
        """
        self.env = lmdb.open(
            data_path, 
            readonly=True,
            lock=False, 
            readahead=False, 
            meminit=False
        )
        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self) -> int:
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - prompts: String prompt
                - ode_latent: Tensor of shape (num_denoising_steps, num_frames, 
                    num_channels, height, width), ordered from noise to clean.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    """
    Multi-shard LMDB dataset for distributed training.
    
    Splits data across multiple LMDB shards for parallel loading.
    """
    
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        """
        Args:
            data_path: Directory containing LMDB shard files.
            max_pair: Maximum number of samples to load.
        """
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width).
              Ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }
