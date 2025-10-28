"""
Expression embedding database for cosine similarity loss.
Keeps database in GPU VRAM for fast nearest-neighbor lookup.
"""

import torch
import torch.nn.functional as F
import h5py
import numpy as np
from pathlib import Path
from logger import logger


class ExpressionDatabase:
    """
    GPU-resident expression embedding database for fast cosine similarity lookup.

    Usage:
        db = ExpressionDatabase('expression_embeddings.h5')
        closest = db.get_closest(pred_embeddings)  # [B, T, 128]
    """

    def __init__(self, db_path: str, device: str = 'cuda'):
        """
        Load expression database into GPU memory.

        Args:
            db_path: Path to H5 database file
            device: Device to load database on (default: 'cuda')
        """
        self.db_path = Path(db_path)
        self.device = device

        logger.info(f"Loading expression database from {db_path}")

        with h5py.File(db_path, 'r') as f:
            embeddings = f['expression_embeddings'][:]  # [N, 128]
            self.num_embeddings = f.attrs['num_embeddings']
            self.embedding_dim = f.attrs['embedding_dim']

            logger.info(f"Database loaded: {self.num_embeddings} embeddings, dim={self.embedding_dim}")

        # Convert to torch tensor and move to GPU
        self.embeddings = torch.from_numpy(embeddings).float().to(device)

        # Normalize embeddings for cosine similarity (done once)
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)

        logger.info(f"Database on {device}: {self.embeddings.shape}, {self.embeddings.element_size() * self.embeddings.nelement() / 1024**2:.2f} MB")

    def get_closest(self, query_embeddings: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Find closest embeddings in database using cosine similarity.

        Args:
            query_embeddings: Query embeddings [B, T, 128] or [B, 128]
            k: Number of nearest neighbors (default: 1)

        Returns:
            closest_embeddings: [B, T, 128] or [B, 128] closest embeddings from database
        """
        original_shape = query_embeddings.shape
        is_3d = len(original_shape) == 3

        if is_3d:
            B, T, D = original_shape
            query_flat = query_embeddings.view(B * T, D)  # [B*T, 128]
        else:
            query_flat = query_embeddings  # [B, 128]

        # Normalize queries
        query_norm = F.normalize(query_flat, p=2, dim=1)  # [B*T, 128] or [B, 128]

        # Compute cosine similarity with all database embeddings
        # similarity = query_norm @ embeddings.T  # [B*T, N] or [B, N]
        similarity = torch.mm(query_norm, self.embeddings.T)  # [B*T, N] or [B, N]

        # Find top-k closest (highest cosine similarity)
        if k == 1:
            indices = similarity.argmax(dim=1)  # [B*T] or [B]
            closest = self.embeddings[indices]  # [B*T, 128] or [B, 128]
        else:
            _, indices = similarity.topk(k, dim=1)  # [B*T, k] or [B, k]
            closest = self.embeddings[indices]  # [B*T, k, 128] or [B, k, 128]
            # Average top-k embeddings
            closest = closest.mean(dim=1)  # [B*T, 128] or [B, 128]

        # Reshape back to original shape
        if is_3d:
            closest = closest.view(B, T, D)

        return closest

    def get_closest_batch(self, query_embeddings: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        """
        Find closest embeddings in batches to avoid OOM for large queries.

        Args:
            query_embeddings: Query embeddings [B, T, 128]
            batch_size: Batch size for processing (default: 1024)

        Returns:
            closest_embeddings: [B, T, 128] closest embeddings from database
        """
        B, T, D = query_embeddings.shape
        query_flat = query_embeddings.view(B * T, D)

        closest_list = []

        for i in range(0, len(query_flat), batch_size):
            batch = query_flat[i:i+batch_size]
            closest_batch = self.get_closest(batch, k=1)
            closest_list.append(closest_batch)

        closest = torch.cat(closest_list, dim=0)
        return closest.view(B, T, D)

    def append_embeddings(self, new_embeddings: np.ndarray):
        """
        Append new embeddings to the database (both in-memory and on disk).

        Args:
            new_embeddings: New embeddings to append [N, 128] numpy array
        """
        if new_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embeddings with dim {self.embedding_dim}, got {new_embeddings.shape[1]}")

        # Convert to torch and normalize
        new_embeddings_torch = torch.from_numpy(new_embeddings).float().to(self.device)
        new_embeddings_torch = F.normalize(new_embeddings_torch, p=2, dim=1)

        # Append to in-memory database
        self.embeddings = torch.cat([self.embeddings, new_embeddings_torch], dim=0)
        self.num_embeddings = len(self.embeddings)

        logger.info(f"Appended {len(new_embeddings)} embeddings to database (now {self.num_embeddings} total)")

        # Save updated database to disk
        self._save_to_disk()

    def _save_to_disk(self):
        """Save the current database to disk."""
        # Convert embeddings back to numpy (denormalize not needed, we keep normalized)
        embeddings_np = self.embeddings.cpu().numpy()

        with h5py.File(self.db_path, 'w') as f:
            f.create_dataset('expression_embeddings', data=embeddings_np, compression='gzip')
            f.attrs['num_embeddings'] = self.num_embeddings
            f.attrs['embedding_dim'] = self.embedding_dim

        logger.info(f"Saved database with {self.num_embeddings} embeddings to {self.db_path}")

    def __len__(self):
        return self.num_embeddings

    def __repr__(self):
        return f"ExpressionDatabase({self.num_embeddings} embeddings, {self.embedding_dim}D, device={self.device})"


def test_expression_db():
    """Test the expression database."""
    import time

    # Create test database
    logger.info("Creating test database...")
    num_db_embeddings = 10000
    db_embeddings = torch.randn(num_db_embeddings, 128)

    # Save to H5
    import h5py
    test_db_path = 'test_expr_db.h5'
    with h5py.File(test_db_path, 'w') as f:
        f.create_dataset('expression_embeddings', data=db_embeddings.numpy())
        f.attrs['num_embeddings'] = num_db_embeddings
        f.attrs['embedding_dim'] = 128

    # Load database
    db = ExpressionDatabase(test_db_path, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Test queries
    logger.info("Testing queries...")

    # 2D query
    query_2d = torch.randn(16, 128).to(db.device)
    start = time.time()
    closest_2d = db.get_closest(query_2d)
    elapsed_2d = time.time() - start
    logger.info(f"2D query: {query_2d.shape} -> {closest_2d.shape} in {elapsed_2d*1000:.2f}ms")

    # 3D query
    query_3d = torch.randn(4, 50, 128).to(db.device)
    start = time.time()
    closest_3d = db.get_closest(query_3d)
    elapsed_3d = time.time() - start
    logger.info(f"3D query: {query_3d.shape} -> {closest_3d.shape} in {elapsed_3d*1000:.2f}ms")

    # Test cosine similarity
    query_norm = F.normalize(query_3d.view(-1, 128), p=2, dim=1)
    closest_norm = F.normalize(closest_3d.view(-1, 128), p=2, dim=1)
    cos_sim = (query_norm * closest_norm).sum(dim=1).mean()
    logger.info(f"Average cosine similarity: {cos_sim.item():.4f}")

    # Clean up
    import os
    os.remove(test_db_path)
    logger.info("✅ Test complete")


if __name__ == '__main__':
    test_expression_db()
