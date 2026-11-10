"""Shared experiment, logging, and checkpoint support for trainers."""

import wandb
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from hw4lib.data.tokenizer import H4Tokenizer
from hw4lib.model import DecoderOnlyTransformer
import os
import random
import shutil
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
from torchinfo import summary


class BaseTrainer(ABC):
    """Base Trainer class that provides common functionality for all trainers.

    This trainer implements:
    1. Experiment tracking and logging (with wandb support)
    2. Checkpoint management
    3. Metric logging and visualization
    4. Directory structure management
    5. Device handling

    Key Components:
    1. Experiment Management:
       - Creates organized directory structure for experiments
       - Handles config file copying and model architecture saving
       - Manages checkpoint saving and loading

    2. Logging and Visualization:
       - Supports both local and wandb logging
       - Saves attention visualizations
       - Tracks training metrics and learning rates
       - Saves generated text outputs

    3. Training Infrastructure:
       - Handles device placement
       - Leaves optimizer creation to each child trainer
       - Supports gradient scaling for mixed precision
       - Stores learning-rate scheduler state

    4. Abstract Methods (to be implemented by child classes):
       - _train_epoch: Single training epoch implementation
       - _validate_epoch: Single validation epoch implementation
       - train: Full training loop implementation
       - evaluate: Evaluation loop implementation

    Args:
        model (nn.Module): The model to train
        tokenizer (H4Tokenizer): Tokenizer for text processing
        config (dict): Configuration dictionary
        run_name (str): Name for the training run
        config_file (str): Path to config file
        device (Optional[str]): Device to run on ('cuda' or 'cpu')

    Directory Structure:
        $HW4P1_OUTPUT_ROOT/ (defaults to ./expts)
        └── {run_name}/
            ├── config.yaml
            ├── model_arch.txt
            ├── checkpoints/
            │   ├── checkpoint-best-metric-model.pth
            │   └── checkpoint-last-epoch-model.pth
            ├── attn/
            │   └── {attention visualizations}
            └── text/
                └── {generated text outputs}
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: H4Tokenizer,
        config: dict,
        run_name: str,
        config_file: str,
        device: Optional[str] = None,
    ) -> None:
        """Initialize shared experiment and training state.

        Args:
            model: Model to train.
            tokenizer: Tokenizer used by the task.
            config: Training configuration.
            run_name: Experiment directory name.
            config_file: Configuration file copied into the experiment directory.
            device: Optional PyTorch device string.
        """
        # If device is not specified, determine it
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device}")
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(device=self.device)
        self.use_wandb = config['training'].get('use_wandb', False)
        self.wandb_run = None
        # Initialize experiment directories
        directories = self._init_experiment(run_name, config_file)
        (
            self.expt_root,
            self.checkpoint_dir,
            self.attn_dir,
            self.text_dir,
            self.best_model_path,
            self.last_model_path,
        ) = directories

        # Training state
        self.current_epoch = 0
        self.global_optimizer_step = 0
        self.best_metric = float('inf')
        self.training_history = []
    
    @abstractmethod
    def _train_epoch(self, dataloader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Train for one epoch.

        Args:
            dataloader: Training DataLoader.

        Returns:
            Training metrics and attention weights.
        """

    @abstractmethod
    def _validate_epoch(self, dataloader) -> Dict[str, float]:
        """Validate for one epoch.

        Args:
            dataloader: Validation DataLoader.

        Returns:
            Validation metrics.
        """

    @abstractmethod
    def train(self, train_dataloader, val_dataloader):
        """Run the training loop.

        Args:
            train_dataloader: Training DataLoader.
            val_dataloader: Validation DataLoader.

        Returns:
            None.
        """

    @abstractmethod
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate a dataset.

        Args:
            dataloader: Evaluation DataLoader.

        Returns:
            Evaluation metrics.
        """

####################################################################################
# provenance helpers
    def _model_for_submission_tracking(self):
        return self.model.module if hasattr(self.model, "module") else self.model


    def _get_active_wandb_ref(self):
        if not self.use_wandb or self.wandb_run is None:
            return None

        return {
            "entity": str(self.wandb_run.entity),
            "project": str(self.wandb_run.project),
            "run_id": str(self.wandb_run.id),
            "run_name": str(self.wandb_run.name),
        }


    def _set_model_wandb_ref(self, ref):
        model = self._model_for_submission_tracking()
        setattr(model, "_submission_wandb_ref", ref)


    def get_model_wandb_ref(self):
        model = self._model_for_submission_tracking()
        return getattr(model, "_submission_wandb_ref", None)


    def _bind_model_to_active_wandb(self):
        if self.use_wandb:
            if (
                self.wandb_run is None
                or wandb.run is not self.wandb_run
            ):
                raise RuntimeError(
                    "This trainer's W&B run is no longer active. "
                    "Resume its saved experiment, or initialize a new trainer and W&B run before continuing training."
                )
        self._set_model_wandb_ref(
            self._get_active_wandb_ref()
        )


    def build_kaggle_submission_message(
        self,
        homework_tag: str,
        manual_wandb_run: Optional[str] = None,
    ) -> str:
        if manual_wandb_run is not None:
            run_path = manual_wandb_run.strip()
        else:
            ref = self.get_model_wandb_ref()

            if ref is None:
                raise RuntimeError(
                    "Unable to determine the W&B run associated with this model. "
                    "If the checkpoint was loaded manually, set "
                    "KAGGLE_WANDB_RUN_OVERRIDE to 'entity/project/run_id'."
                )

            run_path = (
                f"{ref['entity']}/{ref['project']}/{ref['run_id']}"
            )

        parts = run_path.split("/")

        if len(parts) != 3 or not all(parts):
            raise ValueError("W&B run must use the format 'entity/project/run_id'.")

        if manual_wandb_run is not None:
            tracked_ref = self.get_model_wandb_ref()
            if tracked_ref is not None:
                tracked_path = (
                    f"{tracked_ref['entity']}/"
                    f"{tracked_ref['project']}/"
                    f"{tracked_ref['run_id']}"
                )
                if run_path != tracked_path:
                    print(
                        "Note: KAGGLE_WANDB_RUN_OVERRIDE differs from "
                        "the run tracked on the loaded model.\n"
                        f"  Manual override: {run_path}\n"
                        f"  Tracked run:     {tracked_path}\n"
                        "The manual override will be used. "
                        "If you continued training and no longer need "
                        "the override, set it to None and rerun inference."
                    )

        return f"{homework_tag}|WB={run_path}"
####################################################################################

    def _init_experiment(self, run_name: str, config_file: str) -> tuple[Path, ...]:
        """Create experiment directories and save the model summary.

        Args:
            run_name: Experiment directory name.
            config_file: Configuration file copied into the experiment.

        Returns:
            Experiment, checkpoint, attention, text, best-model, and
            last-model paths.
        """
        # Create the experiment under the configured persistent output root.
        # This keeps PSC checkpoints on Jet when HW4P1_OUTPUT_ROOT is exported.
        default_output_root = Path.cwd() / "expts"
        output_root = Path(
            os.environ.get("HW4P1_OUTPUT_ROOT", default_output_root)
        ).expanduser().resolve()
        expt_root = output_root / run_name
        expt_root.mkdir(parents=True, exist_ok=True)

        # Copy config
        shutil.copy2(config_file, expt_root / "config.yaml")

        # Save model architecture with torchinfo summary
        architecture_path = expt_root / "model_arch.txt"
        with architecture_path.open("w", encoding="utf-8") as architecture_file:
            if isinstance(self.model, DecoderOnlyTransformer):
                # Get a sample input shape from your model's expected input
                summary_length = min(self.model.max_len, 32)
                input_data = torch.zeros(
                    (1, summary_length),
                    dtype=torch.long,
                    device=self.device,
                )
                # Generate the summary
                model_summary = summary(
                    self.model,
                    input_data=input_data,
                )
                # Write the summary string to file
                architecture_file.write(str(model_summary))
            else:
                raise NotImplementedError("Model architecture summary not implemented")

        # Create subdirectories
        checkpoint_dir = expt_root / 'checkpoints'
        attn_dir = expt_root / 'attn'
        text_dir = expt_root / 'text'
        
        checkpoint_dir.mkdir(exist_ok=True)
        attn_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)

        # Define checkpoint paths
        best_model_path = checkpoint_dir / 'checkpoint-best-metric-model.pth'
        last_model_path = checkpoint_dir / 'checkpoint-last-epoch-model.pth'

        # Wandb initialization
        if self.use_wandb:
            run_id = self.config['training'].get('wandb_run_id', None)
            entity = self.config['training'].get('wandb_entity') or None
            if run_id and run_id.lower() != "none":
                self.wandb_run = wandb.init(
                    project=self.config['training'].get('wandb_project', 'default-project'),
                    entity=entity,
                    id=run_id,
                    resume="must",
                    config=self.config
                )
            else:
                self.wandb_run = wandb.init(
                    project=self.config['training'].get('wandb_project', 'default-project'),
                    entity=entity,
                    config=self.config,
                    name=run_name
                )

        directories = (
            expt_root,
            checkpoint_dir,
            attn_dir,
            text_dir,
            best_model_path,
            last_model_path,
        )
        return directories

    def _log_metrics(
        self,
        metrics: Dict[str, Dict[str, float]],
        step: int,
    ) -> None:
        """Record metrics locally and optionally in Weights & Biases.

        Args:
            metrics: Metric values grouped by dataset split.
            step: Epoch or evaluation step.

        Returns:
            None.
        """
        self.training_history.append({
            'epoch': step,
            **metrics,
            'lr': self.optimizer.param_groups[0]['lr']
        })
        
        # Log to wandb
        if self.use_wandb and self.wandb_run is not None:
            wandb_metrics = {}
            for split, split_metrics in metrics.items():
                for metric_name, value in split_metrics.items():
                    wandb_metrics[f'{split}/{metric_name}'] = value
            wandb_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            self.wandb_run.log(wandb_metrics, step=step)
        
        # Print metrics with tree structure
        print(f"\n📊 Metrics (Epoch {step}):")

        # Print metrics by split
        splits = sorted(metrics.keys())
        for i, split in enumerate(splits):
            is_last_split = i == len(splits) - 1
            split_prefix = "└──" if is_last_split else "├──"
            print(f"{split_prefix} {split.upper()}:")
            
            # Print metrics within split
            split_metrics = sorted(metrics[split].items())
            for j, (metric_name, value) in enumerate(split_metrics):
                is_last_metric = j == len(split_metrics) - 1
                metric_prefix = "    └──" if is_last_metric else "    ├──"
                if is_last_split:
                    metric_prefix = "    └──" if is_last_metric else "    ├──"
                else:
                    metric_prefix = "│   └──" if is_last_metric else "│   ├──"
                print(f"{metric_prefix} {metric_name}: {value:.4f}")
        
        # Print learning rate
        print("└── TRAINING:")
        print(f"    └── learning_rate: {self.optimizer.param_groups[0]['lr']:.6f}")

    def _save_attention_plot(
        self,
        attn_weights: torch.Tensor,
        epoch: int,
        attn_type: str = "self",
    ) -> None:
        """Save one attention heatmap.

        Args:
            attn_weights: Two-dimensional attention matrix.
            epoch: Epoch shown in the title and filename.
            attn_type: Short label used in the filename.

        Returns:
            None.
        """
        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.cpu().detach().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights, cmap="viridis", cbar=True)
        plt.title(f"Attention Weights - Epoch {epoch}")
        plt.xlabel("Source Sequence")
        plt.ylabel("Target Sequence")
        
        plot_path = os.path.join(
            self.attn_dir,
            f"{attn_type}_attention_epoch{epoch}.png",
        )
        plt.savefig(plot_path)
        plt.close()
        
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.log({f"{attn_type}_attention": wandb.Image(plot_path)}, step=epoch)

    def _save_generated_text(self, text: dict, suffix: str) -> None:
        """Save generated text as JSON.

        Args:
            text: Serializable generation results.
            suffix: Filename suffix identifying the evaluation.

        Returns:
            None.
        """
        text_path = os.path.join(self.text_dir, f"text_{suffix}.json")
        with open(text_path, "w", encoding="utf-8") as text_file:
            json.dump(text, text_file, indent=4)
    
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.save(text_path)

    def save_checkpoint(self, filename: str) -> None:
        """Save model and training state.

        Args:
            filename: Checkpoint filename inside ``self.checkpoint_dir``.

        Returns:
            None.
        """
        checkpoint_path = self.checkpoint_dir / filename
        # Save enough state to resume the next optimizer step reproducibly.
        checkpoint = {
            'format_version': 2,
            'epoch': self.current_epoch,
            'global_optimizer_step': self.global_optimizer_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            'config': self.config,
            'rng_state': self._capture_rng_state(),
            'submission_tracking': {
                'version': 1,
                'wandb_ref': self.get_model_wandb_ref(),
            },
        }
        torch.save(checkpoint, checkpoint_path)
        if (
            self.use_wandb and self.wandb_run is not None
            and self.config['training'].get('wandb_log_checkpoints', False)
        ):
            self.wandb_run.save(str(checkpoint_path))


    def load_checkpoint(self, filename: str) -> None:
        """Load model and training state from one checkpoint.

        Args:
            filename: Checkpoint filename inside ``self.checkpoint_dir``.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            RuntimeError: If the checkpoint or one of its components is invalid.
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint file: {exc}") from exc


        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as exc:
            self._set_model_wandb_ref(None)
            raise RuntimeError(
                "Failed to load model weights. "
                "Check that the checkpoint matches your model architecture."
            ) from exc

        tracking = checkpoint.get("submission_tracking", {})
        ref = tracking.get("wandb_ref") if isinstance(tracking, dict) else None
        self._set_model_wandb_ref(ref)


        try:
            if self.optimizer is None:
                raise RuntimeError("Initialize the optimizer before loading a checkpoint")
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            scheduler_state = checkpoint.get('scheduler_state_dict')
            if scheduler_state is not None:
                if self.scheduler is None:
                    raise RuntimeError(
                        "Initialize the scheduler before loading its checkpoint state"
                    )
                self.scheduler.load_state_dict(scheduler_state)

            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.global_optimizer_step = checkpoint.get(
                'global_optimizer_step',
                0,
            )
            self.best_metric = checkpoint['best_metric']
            self.training_history = checkpoint['training_history']
            self._restore_rng_state(checkpoint['rng_state'])
        except Exception as exc:
            raise RuntimeError(f"Failed to restore checkpoint state: {exc}") from exc

        print(f"Checkpoint loaded; next epoch is {self.current_epoch}")

    @staticmethod
    def _capture_rng_state() -> dict:
        """Capture Python, NumPy, Torch, and CUDA random states.

        Returns:
            Random states containing only safe checkpoint value types.
        """
        numpy_state = np.random.get_state()
        serialized_numpy_state = {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gaussian": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        }
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
        else:
            cuda_states = []
        rng_state = {
            "python": random.getstate(),
            "numpy": serialized_numpy_state,
            "torch": torch.get_rng_state(),
            "cuda": cuda_states,
        }
        return rng_state

    @staticmethod
    def _restore_rng_state(rng_state: dict) -> None:
        """Restore Python, NumPy, Torch, and CUDA random states.

        Args:
            rng_state: State dictionary produced by ``_capture_rng_state``.

        Returns:
            None.

        Raises:
            ValueError: If saved CUDA state count does not match this system.
        """
        random.setstate(rng_state["python"])

        numpy_state = rng_state["numpy"]
        restored_numpy_state = (
            numpy_state["bit_generator"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            numpy_state["position"],
            numpy_state["has_gaussian"],
            numpy_state["cached_gaussian"],
        )
        np.random.set_state(restored_numpy_state)
        torch.set_rng_state(rng_state["torch"].cpu())

        cuda_states = rng_state["cuda"]
        if cuda_states:
            if not torch.cuda.is_available():
                raise ValueError("Checkpoint contains CUDA RNG state on a CPU system")
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError("Checkpoint CUDA RNG state count does not match")
            cpu_cuda_states = [state.cpu() for state in cuda_states]
            torch.cuda.set_rng_state_all(cpu_cuda_states)

    def cleanup(self) -> None:
        """Finish the active Weights & Biases run.

        Returns:
            None.
        """
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None