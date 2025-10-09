import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from .model import ModelWrapper, create_model
from .data_loader import DataLoaderFactory, create_data_loader
from .checkpoint_manager import CheckpointManager
from ..monitoring.metrics import get_metrics_collector
from ..monitoring.logger import get_job_logger
from ..utils.exceptions import TrainingFailedError

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer class for model fine-tuning
    Supports distributed training, checkpointing, and monitoring
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dir: str = "./output",
        num_gpus: int = 1,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trainer

        Args:
            model_name: Pretrained model name
            output_dir: Output directory for checkpoints and results
            num_gpus: Number of GPUs to use
            config: Training configuration
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.num_gpus = num_gpus
        self.config = config or {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model: Optional[ModelWrapper] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.metrics_collector = get_metrics_collector()

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        logger.info(
            f"Trainer initialized: model={model_name}, gpus={num_gpus}, "
            f"output={output_dir}"
        )

    def setup(
        self,
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500
    ):
        """
        Setup model, optimizer, and other components

        Args:
            num_labels: Number of output labels
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for scheduler
        """
        logger.info("Setting up trainer components...")

        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(
            model_name=self.model_name,
            num_labels=num_labels,
            device=device
        )

        # Setup multi-GPU if available
        if self.num_gpus > 1 and torch.cuda.device_count() > 1:
            device_ids = list(range(min(self.num_gpus, torch.cuda.device_count())))
            self.model.parallelize(device_ids)

        # Create optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate
        )

        # Create checkpoint manager
        checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            max_checkpoints=self.config.get("save_total_limit", 3)
        )

        logger.info("Trainer setup complete")

    def train(
        self,
        dataset_path: str,
        num_epochs: int = 3,
        max_steps: Optional[int] = None,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        save_checkpoints: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run training

        Args:
            dataset_path: Path to training data
            num_epochs: Number of training epochs
            max_steps: Maximum training steps (overrides epochs if set)
            eval_steps: Steps between evaluations
            save_steps: Steps between checkpoint saves
            logging_steps: Steps between logging
            save_checkpoints: Whether to save checkpoints
            **kwargs: Additional arguments

        Returns:
            Training results dictionary
        """
        try:
            logger.info("Starting training...")
            start_time = time.time()

            # Setup if not already done
            if self.model is None:
                self.setup()

            # Create data loader
            data_factory = DataLoaderFactory(
                tokenizer=self.model.tokenizer,
                max_length=self.config.get("max_seq_length", 128),
                batch_size=self.config.get("batch_size", 32)
            )

            train_loader, val_loader = data_factory.create_train_val_loaders(
                dataset_path,
                val_split=self.config.get("val_split", 0.1)
            )

            # Calculate training steps
            if max_steps:
                total_steps = max_steps
                num_epochs = (max_steps // len(train_loader)) + 1
            else:
                total_steps = len(train_loader) * num_epochs

            # Create scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.get("warmup_steps", 500),
                num_training_steps=total_steps
            )

            # Training loop
            logger.info(
                f"Training for {num_epochs} epochs, {total_steps} total steps"
            )

            for epoch in range(num_epochs):
                self.current_epoch = epoch
                epoch_metrics = self._train_epoch(
                    train_loader,
                    val_loader,
                    epoch,
                    max_steps,
                    eval_steps,
                    save_steps,
                    logging_steps,
                    save_checkpoints
                )

                if max_steps and self.global_step >= max_steps:
                    logger.info(f"Reached max_steps={max_steps}, stopping training")
                    break

            # Final evaluation
            logger.info("Running final evaluation...")
            final_metrics = self._evaluate(val_loader)

            # Save final model
            final_model_path = self.output_dir / "final_model"
            self.model.save_model(str(final_model_path))

            training_time = time.time() - start_time

            results = {
                "status": "completed",
                "epochs": self.current_epoch + 1,
                "steps": self.global_step,
                "training_time": training_time,
                "final_metrics": final_metrics,
                "model_path": str(final_model_path)
            }

            logger.info(
                f"Training completed in {training_time:.1f}s, "
                f"{self.global_step} steps"
            )

            return results

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise TrainingFailedError(str(e), self.current_epoch)

    def _train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epoch: int,
        max_steps: Optional[int],
        eval_steps: int,
        save_steps: int,
        logging_steps: int,
        save_checkpoints: bool
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.model.train()

        epoch_loss = 0.0
        steps_in_epoch = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model.model(**batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            steps_in_epoch += 1

            # Logging
            if self.global_step % logging_steps == 0:
                avg_loss = epoch_loss / steps_in_epoch
                lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}"
                })
                logger.debug(
                    f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}"
                )

            # Evaluation
            if self.global_step % eval_steps == 0:
                eval_metrics = self._evaluate(val_loader)
                logger.info(
                    f"Evaluation at step {self.global_step}: {eval_metrics}"
                )
                self.model.model.train()

            # Checkpointing
            if save_checkpoints and self.global_step % save_steps == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    step=self.global_step,
                    metrics={"loss": epoch_loss / steps_in_epoch}
                )

            # Check max steps
            if max_steps and self.global_step >= max_steps:
                break

        avg_epoch_loss = epoch_loss / steps_in_epoch
        return {"loss": avg_epoch_loss}

    def _evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run evaluation"""
        self.model.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                outputs = self.model.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                if "labels" in batch:
                    correct = (predictions == batch["labels"]).sum().item()
                    total_correct += correct
                    total_samples += batch["labels"].size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy
        }

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Resume training from checkpoint"""
        if self.checkpoint_manager is None:
            raise ValueError("Checkpoint manager not initialized")

        metadata = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model.model,
            optimizer=self.optimizer
        )

        self.global_step = metadata.get("step", 0)
        self.current_epoch = metadata.get("epoch", 0)

        logger.info(
            f"Resumed from checkpoint: epoch={self.current_epoch}, "
            f"step={self.global_step}"
        )
