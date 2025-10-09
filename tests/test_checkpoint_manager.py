import pytest
from pathlib import Path
from src.pipeline.checkpoint_manager import CheckpointManager
from src.utils.exceptions import CheckpointNotFoundError


class TestCheckpointManager:
    """Test Checkpoint Manager functionality"""

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Create checkpoint manager instance"""
        return CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            max_checkpoints=3,
            save_best_only=False
        )

    def test_save_checkpoint(self, checkpoint_manager, mock_model, tmp_path):
        """Test saving checkpoint"""
        import torch

        optimizer = torch.optim.Adam(mock_model.parameters())

        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            metrics={"loss": 0.5}
        )

        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()

    def test_load_checkpoint(self, checkpoint_manager, mock_model, tmp_path):
        """Test loading checkpoint"""
        import torch

        optimizer = torch.optim.Adam(mock_model.parameters())

        # Save first
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_model,
            optimizer=optimizer,
            epoch=1,
            step=100
        )

        # Load
        metadata = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=mock_model,
            optimizer=optimizer
        )

        assert metadata["epoch"] == 1
        assert metadata["step"] == 100

    def test_checkpoint_not_found(self, checkpoint_manager):
        """Test loading non-existent checkpoint"""
        with pytest.raises(CheckpointNotFoundError):
            checkpoint_manager.load_checkpoint(
                checkpoint_path="/nonexistent/path"
            )

    def test_max_checkpoints_cleanup(self, checkpoint_manager, mock_model):
        """Test that old checkpoints are cleaned up"""
        import torch

        optimizer = torch.optim.Adam(mock_model.parameters())

        # Save 5 checkpoints (max is 3)
        for i in range(5):
            checkpoint_manager.save_checkpoint(
                model=mock_model,
                optimizer=optimizer,
                epoch=i,
                step=i*100
            )

        # Should only have 3 checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) <= 3

    def test_best_checkpoint_tracking(self, checkpoint_manager, mock_model):
        """Test best checkpoint tracking"""
        import torch

        optimizer = torch.optim.Adam(mock_model.parameters())

        # Save checkpoints with different metrics
        checkpoint_manager.save_checkpoint(
            model=mock_model, optimizer=optimizer,
            epoch=0, step=0, metrics={"loss": 1.0}
        )
        checkpoint_manager.save_checkpoint(
            model=mock_model, optimizer=optimizer,
            epoch=1, step=100, metrics={"loss": 0.5}  # Better
        )
        checkpoint_manager.save_checkpoint(
            model=mock_model, optimizer=optimizer,
            epoch=2, step=200, metrics={"loss": 0.7}  # Worse
        )

        # Best should be step 100
        best_path = checkpoint_manager.get_best_checkpoint_path()
        assert best_path is not None
        assert "step100" in best_path

    def test_has_checkpoint(self, checkpoint_manager, mock_model):
        """Test checking if checkpoints exist"""
        import torch

        assert checkpoint_manager.has_checkpoint() is False

        optimizer = torch.optim.Adam(mock_model.parameters())
        checkpoint_manager.save_checkpoint(
            model=mock_model, optimizer=optimizer,
            epoch=0, step=0
        )

        assert checkpoint_manager.has_checkpoint() is True
