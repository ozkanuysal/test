import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

from ..utils.exceptions import ModelLoadError, ModelSaveError

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Wrapper for transformer models
    Provides unified interface for model loading, saving, and inference
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        task_type: str = "classification",
        device: Optional[torch.device] = None,
        **model_kwargs
    ):
        """
        Initialize model wrapper

        Args:
            model_name: Pretrained model name or path
            num_labels: Number of output labels for classification
            task_type: Task type (classification, token_classification, qa)
            device: Device to load model on
            **model_kwargs: Additional model configuration arguments
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.task_type = task_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.config: Optional[AutoConfig] = None

        self.model_kwargs = model_kwargs

        logger.info(
            f"Initializing model wrapper for {model_name} "
            f"(task={task_type}, device={self.device})"
        )

    def load_model(self, model_path: Optional[str] = None):
        """
        Load model and tokenizer

        Args:
            model_path: Path to load model from (uses model_name if None)

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            model_source = model_path or self.model_name

            logger.info(f"Loading model from {model_source}")

            # Load configuration
            self.config = AutoConfig.from_pretrained(
                model_source,
                num_labels=self.num_labels,
                **self.model_kwargs
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)

            # Load model based on task type
            if self.task_type == "classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_source,
                    config=self.config
                )
            else:
                # Generic model loading
                self.model = AutoModel.from_pretrained(
                    model_source,
                    config=self.config
                )

            # Move to device
            self.model.to(self.device)

            logger.info(
                f"Model loaded successfully: {self.model.__class__.__name__} "
                f"on {self.device}"
            )

            # Log model info
            num_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            logger.info(
                f"Model parameters: {num_params:,} total, "
                f"{trainable_params:,} trainable"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Model loading failed: {e}")

    def save_model(self, save_path: str):
        """
        Save model and tokenizer

        Args:
            save_path: Directory to save model to

        Raises:
            ModelSaveError: If model saving fails
        """
        if self.model is None or self.tokenizer is None:
            raise ModelSaveError("Model or tokenizer not loaded")

        try:
            logger.info(f"Saving model to {save_path}")

            # Save model
            self.model.save_pretrained(save_path)

            # Save tokenizer
            self.tokenizer.save_pretrained(save_path)

            # Save config
            if self.config:
                self.config.save_pretrained(save_path)

            logger.info(f"Model saved successfully to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelSaveError(f"Model saving failed: {e}")

    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through model

        Args:
            **inputs: Model inputs (input_ids, attention_mask, etc.)

        Returns:
            Dictionary with model outputs
        """
        if self.model is None:
            raise ModelLoadError("Model not loaded")

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        return {
            "logits": outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            "attentions": outputs.attentions if hasattr(outputs, "attentions") else None,
        }

    def predict(self, texts: list) -> list:
        """
        Make predictions on input texts

        Args:
            texts: List of input texts

        Returns:
            List of predictions
        """
        if self.model is None or self.tokenizer is None:
            raise ModelLoadError("Model or tokenizer not loaded")

        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        outputs = self.forward(**inputs)
        logits = outputs["logits"]

        # Get predicted classes
        predictions = torch.argmax(logits, dim=-1)

        return predictions.cpu().tolist()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            return {"loaded": False}

        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "loaded": True,
            "model_name": self.model_name,
            "model_class": self.model.__class__.__name__,
            "task_type": self.task_type,
            "num_labels": self.num_labels,
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
        }

    def freeze_base_model(self):
        """Freeze base model parameters (for fine-tuning only classifier)"""
        if self.model is None:
            raise ModelLoadError("Model not loaded")

        # Freeze all parameters
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        logger.info("Frozen base model parameters")

        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def unfreeze_model(self):
        """Unfreeze all model parameters"""
        if self.model is None:
            raise ModelLoadError("Model not loaded")

        for param in self.model.parameters():
            param.requires_grad = True

        logger.info("Unfrozen all model parameters")

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        if self.model is None:
            raise ModelLoadError("Model not loaded")

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        else:
            logger.warning("Model does not support gradient checkpointing")

    def to_device(self, device: torch.device):
        """
        Move model to device

        Args:
            device: Target device
        """
        if self.model is None:
            raise ModelLoadError("Model not loaded")

        self.device = device
        self.model.to(device)
        logger.info(f"Moved model to {device}")

    def parallelize(self, device_ids: list):
        """
        Wrap model in DataParallel for multi-GPU training

        Args:
            device_ids: List of GPU device IDs
        """
        if self.model is None:
            raise ModelLoadError("Model not loaded")

        if len(device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            logger.info(f"Model parallelized across GPUs: {device_ids}")
        else:
            logger.warning("Only one GPU specified, skipping parallelization")


def create_model(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
    task_type: str = "classification",
    load_pretrained: bool = True,
    device: Optional[torch.device] = None,
    **kwargs
) -> ModelWrapper:
    """
    Factory function to create and load a model

    Args:
        model_name: Pretrained model name or path
        num_labels: Number of output labels
        task_type: Task type
        load_pretrained: Whether to load pretrained weights
        device: Device to load model on
        **kwargs: Additional model arguments

    Returns:
        Loaded ModelWrapper instance
    """
    model_wrapper = ModelWrapper(
        model_name=model_name,
        num_labels=num_labels,
        task_type=task_type,
        device=device,
        **kwargs
    )

    if load_pretrained:
        model_wrapper.load_model()

    return model_wrapper


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create and load model
    model = create_model(
        model_name="bert-base-uncased",
        num_labels=2,
        task_type="classification"
    )

    # Get model info
    info = model.get_model_info()
    print(f"Model info: {info}")

    # Make predictions
    texts = ["This is great!", "This is terrible!"]
    predictions = model.predict(texts)
    print(f"Predictions: {predictions}")
