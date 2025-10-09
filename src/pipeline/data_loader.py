import logging
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset as HFDataset

from ..utils.exceptions import DataLoadError, DataValidationError

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Custom dataset for text classification
    """

    def __init__(
        self,
        texts: list,
        labels: Optional[list] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 128
    ):
        """
        Initialize text dataset

        Args:
            texts: List of text strings
            labels: List of labels (optional for inference)
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        if labels and len(texts) != len(labels):
            raise DataValidationError(
                f"Number of texts ({len(texts)}) != number of labels ({len(labels)})"
            )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            item = {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze()
            }
        else:
            # Return raw text if no tokenizer provided
            item = {"text": text}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class DataLoaderFactory:
    """
    Factory for creating data loaders from various sources
    """

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Initialize data loader factory

        Args:
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
            batch_size: Batch size for data loaders
            num_workers: Number of data loading workers
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        logger.info(
            f"DataLoaderFactory initialized "
            f"(max_length={max_length}, batch_size={batch_size})"
        )

    def from_local_file(
        self,
        file_path: str,
        text_column: str = "text",
        label_column: str = "label",
        file_format: str = "auto"
    ) -> DataLoader:
        """
        Create data loader from local file

        Args:
            file_path: Path to data file
            text_column: Name of text column
            label_column: Name of label column
            file_format: File format (auto, csv, json, parquet)

        Returns:
            DataLoader instance
        """
        try:
            logger.info(f"Loading data from local file: {file_path}")

            # Load using HuggingFace datasets
            if file_format == "auto":
                ext = Path(file_path).suffix.lower()
                format_map = {".csv": "csv", ".json": "json", ".parquet": "parquet"}
                file_format = format_map.get(ext, "json")

            dataset = load_dataset(file_format, data_files=file_path, split="train")

            return self._create_dataloader_from_hf_dataset(
                dataset, text_column, label_column
            )

        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise DataLoadError(f"Failed to load local file: {e}")

    def from_s3(
        self,
        s3_path: str,
        text_column: str = "text",
        label_column: str = "label"
    ) -> DataLoader:
        """
        Create data loader from S3

        Args:
            s3_path: S3 path (s3://bucket/key)
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            DataLoader instance
        """
        try:
            logger.info(f"Loading data from S3: {s3_path}")

            # Use s3fs for S3 access
            import s3fs
            fs = s3fs.S3FileSystem()

            # Download and load
            ext = Path(s3_path).suffix.lower()
            format_map = {".csv": "csv", ".json": "json", ".parquet": "parquet"}
            file_format = format_map.get(ext, "json")

            with fs.open(s3_path, "rb") as f:
                dataset = load_dataset(file_format, data_files={"train": f}, split="train")

            return self._create_dataloader_from_hf_dataset(
                dataset, text_column, label_column
            )

        except Exception as e:
            logger.error(f"Failed to load data from S3: {e}")
            raise DataLoadError(f"S3 loading failed: {e}")

    def from_gcs(
        self,
        gcs_path: str,
        text_column: str = "text",
        label_column: str = "label"
    ) -> DataLoader:
        """
        Create data loader from Google Cloud Storage

        Args:
            gcs_path: GCS path (gs://bucket/key)
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            DataLoader instance
        """
        try:
            logger.info(f"Loading data from GCS: {gcs_path}")

            # Use gcsfs for GCS access
            import gcsfs
            fs = gcsfs.GCSFileSystem()

            ext = Path(gcs_path).suffix.lower()
            format_map = {".csv": "csv", ".json": "json", ".parquet": "parquet"}
            file_format = format_map.get(ext, "json")

            with fs.open(gcs_path, "rb") as f:
                dataset = load_dataset(file_format, data_files={"train": f}, split="train")

            return self._create_dataloader_from_hf_dataset(
                dataset, text_column, label_column
            )

        except Exception as e:
            logger.error(f"Failed to load data from GCS: {e}")
            raise DataLoadError(f"GCS loading failed: {e}")

    def from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        label_column: str = "label",
        **load_kwargs
    ) -> DataLoader:
        """
        Create data loader from HuggingFace datasets hub

        Args:
            dataset_name: Name of dataset on HuggingFace hub
            split: Dataset split to load
            text_column: Name of text column
            label_column: Name of label column
            **load_kwargs: Additional arguments for load_dataset

        Returns:
            DataLoader instance
        """
        try:
            logger.info(f"Loading dataset from HuggingFace: {dataset_name}")

            dataset = load_dataset(dataset_name, split=split, **load_kwargs)

            return self._create_dataloader_from_hf_dataset(
                dataset, text_column, label_column
            )

        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {e}")
            raise DataLoadError(f"HuggingFace dataset loading failed: {e}")

    def from_lists(
        self,
        texts: list,
        labels: Optional[list] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create data loader from lists

        Args:
            texts: List of text strings
            labels: List of labels (optional)
            shuffle: Whether to shuffle data

        Returns:
            DataLoader instance
        """
        dataset = TextDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def _create_dataloader_from_hf_dataset(
        self,
        dataset: HFDataset,
        text_column: str,
        label_column: str
    ) -> DataLoader:
        """
        Create DataLoader from HuggingFace dataset

        Args:
            dataset: HuggingFace dataset
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            DataLoader instance
        """
        # Extract texts and labels
        if text_column not in dataset.column_names:
            raise DataValidationError(
                f"Text column '{text_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        texts = dataset[text_column]

        labels = None
        if label_column in dataset.column_names:
            labels = dataset[label_column]

        logger.info(
            f"Loaded {len(texts)} examples "
            f"(has_labels={labels is not None})"
        )

        return self.from_lists(texts, labels)

    def create_train_val_loaders(
        self,
        data_source: Union[str, tuple],
        val_split: float = 0.1,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation data loaders

        Args:
            data_source: Data source (file path, S3/GCS path, or (texts, labels) tuple)
            val_split: Validation split ratio
            **kwargs: Additional arguments for data loading

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Determine source type and load full dataset
        if isinstance(data_source, tuple):
            texts, labels = data_source
            dataset = TextDataset(texts, labels, self.tokenizer, self.max_length)
        elif data_source.startswith("s3://"):
            # Load and extract
            temp_loader = self.from_s3(data_source, **kwargs)
            dataset = temp_loader.dataset
        elif data_source.startswith("gs://"):
            temp_loader = self.from_gcs(data_source, **kwargs)
            dataset = temp_loader.dataset
        else:
            temp_loader = self.from_local_file(data_source, **kwargs)
            dataset = temp_loader.dataset

        # Split into train and validation
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        logger.info(
            f"Split dataset: {train_size} train, {val_size} validation"
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader


# Convenience function
def create_data_loader(
    data_source: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    batch_size: int = 32,
    **kwargs
) -> DataLoader:
    """
    Convenience function to create a data loader

    Args:
        data_source: Data source (file path, S3/GCS path, HF dataset name)
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size
        **kwargs: Additional arguments

    Returns:
        DataLoader instance
    """
    factory = DataLoaderFactory(
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size
    )

    # Determine source type
    if data_source.startswith("s3://"):
        return factory.from_s3(data_source, **kwargs)
    elif data_source.startswith("gs://"):
        return factory.from_gcs(data_source, **kwargs)
    elif "/" in data_source and not Path(data_source).exists():
        # Likely a HuggingFace dataset
        return factory.from_huggingface(data_source, **kwargs)
    else:
        # Local file
        return factory.from_local_file(data_source, **kwargs)
