import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import json
import torch

from ..utils.exceptions import (
    CheckpointNotFoundError,
    CheckpointLoadError,
    CheckpointSaveError
)

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints during training
    Supports versioning, best model tracking, and resume capability
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        save_best_only: bool = False
    ):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save checkpoints that improve metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = None
        self.best_checkpoint_path = None
        self.checkpoints = []

        logger.info(
            f"Checkpoint manager initialized: {checkpoint_dir} "
            f"(max={max_checkpoints}, best_only={save_best_only})"
        )

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        """
        Save a checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            **kwargs: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        try:
            checkpoint_name = f"checkpoint_epoch{epoch}_step{step}"
            checkpoint_path = self.checkpoint_dir / checkpoint_name

            # Check if should save based on metrics
            if self.save_best_only and metrics:
                metric_value = metrics.get("eval_loss") or metrics.get("loss")
                if metric_value is None:
                    logger.warning("No suitable metric found for best checkpoint tracking")
                elif self.best_metric is not None and metric_value >= self.best_metric:
                    logger.info(
                        f"Skipping checkpoint (metric {metric_value:.4f} >= "
                        f"best {self.best_metric:.4f})"
                    )
                    return None

            logger.info(f"Saving checkpoint to {checkpoint_path}")

            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Save model
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path / "model.pt")

            # Save optimizer
            torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")

            # Save metadata
            metadata = {
                "epoch": epoch,
                "step": step,
                "metrics": metrics or {},
                **kwargs
            }

            with open(checkpoint_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Update tracking
            self.checkpoints.append({
                "path": str(checkpoint_path),
                "epoch": epoch,
                "step": step,
                "metrics": metrics
            })

            # Update best checkpoint
            if metrics:
                metric_value = metrics.get("eval_loss") or metrics.get("loss")
                if metric_value and (self.best_metric is None or metric_value < self.best_metric):
                    self.best_metric = metric_value
                    self.best_checkpoint_path = str(checkpoint_path)
                    logger.info(f"New best checkpoint: {checkpoint_path} (metric={metric_value:.4f})")

            # Clean up old checkpoints
            self._cleanup_old_checkpoints()

            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise CheckpointSaveError(f"Checkpoint save failed: {e}")

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        model = None,
        optimizer = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Load a checkpoint

        Args:
            checkpoint_path: Path to checkpoint (uses latest if None)
            model: Model to load state into
            optimizer: Optimizer to load state into
            load_best: Load best checkpoint instead of latest

        Returns:
            Checkpoint metadata

        Raises:
            CheckpointNotFoundError: If checkpoint not found
            CheckpointLoadError: If loading fails
        """
        try:
            # Determine checkpoint to load
            if load_best and self.best_checkpoint_path:
                checkpoint_path = self.best_checkpoint_path
            elif checkpoint_path is None:
                checkpoint_path = self._get_latest_checkpoint()

            if checkpoint_path is None:
                raise CheckpointNotFoundError(
                    "No checkpoints found and no path provided"
                )

            checkpoint_path = Path(checkpoint_path)

            if not checkpoint_path.exists():
                raise CheckpointNotFoundError(str(checkpoint_path))

            logger.info(f"Loading checkpoint from {checkpoint_path}")

            # Load model
            if model is not None:
                if hasattr(model, "from_pretrained"):
                    # For HuggingFace models
                    model_state = model.__class__.from_pretrained(checkpoint_path)
                    model.load_state_dict(model_state.state_dict())
                else:
                    model_path = checkpoint_path / "model.pt"
                    model.load_state_dict(torch.load(model_path))

            # Load optimizer
            if optimizer is not None:
                optimizer_path = checkpoint_path / "optimizer.pt"
                if optimizer_path.exists():
                    optimizer.load_state_dict(torch.load(optimizer_path))

            # Load metadata
            metadata_path = checkpoint_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            logger.info(f"Checkpoint loaded: epoch={metadata.get('epoch')}, step={metadata.get('step')}")
            return metadata

        except CheckpointNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise CheckpointLoadError(f"Checkpoint load failed: {e}")

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        if not self.checkpoints:
            # Scan directory
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*"))
            if not checkpoints:
                return None

            # Sort by modification time
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            return str(latest)

        # Return most recent from tracking
        return self.checkpoints[-1]["path"]

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort by step (oldest first)
        self.checkpoints.sort(key=lambda x: x["step"])

        # Remove oldest checkpoints
        to_remove = self.checkpoints[: -self.max_checkpoints]

        for checkpoint in to_remove:
            path = Path(checkpoint["path"])

            # Don't remove best checkpoint
            if str(path) == self.best_checkpoint_path:
                continue

            try:
                if path.exists():
                    shutil.rmtree(path)
                    logger.info(f"Removed old checkpoint: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {path}: {e}")

        # Update tracking
        self.checkpoints = self.checkpoints[-self.max_checkpoints:]

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint"""
        return self.best_checkpoint_path

    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        return self.checkpoints.copy()

    def has_checkpoint(self) -> bool:
        """Check if any checkpoints exist"""
        return len(self.checkpoints) > 0 or len(list(self.checkpoint_dir.glob("checkpoint_*"))) > 0
