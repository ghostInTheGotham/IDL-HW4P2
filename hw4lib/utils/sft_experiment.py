"""Provided checkpoint loading and local submission export for HW4P2."""
import json
import zipfile
from pathlib import Path

import torch

def load_base_weights(model, path):
    """Load P1/P2 model weights on CPU."""
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    tracked_model = model.module if hasattr(model, "module") else model
    try:
        model.load_state_dict(payload.get("model_state_dict", payload))
    except Exception:
        tracked_model._submission_wandb_ref = None
        raise

    tracking = payload.get("submission_tracking", {})
    tracked_model._submission_wandb_ref = (
        tracking.get("wandb_ref")
        if isinstance(tracking, dict)
        else None
    )
    model._hw4p2_weight_source = {"path": str(Path(path).resolve()),
        "epoch": payload.get("epoch"), "step": payload.get("global_optimizer_step")}
    return payload.get("config", {}).get("model", {})


def evaluate_and_export(trainer, dataloader, records, *, available_inputs, seed, prefix=""):
    """Evaluate current weights; retain previous outputs when resuming."""
    from time import time_ns
    from .webnlg import score_webnlg, write_predictions
    metrics_path = trainer.expt_root / f"{prefix}metrics.json"
    predictions_path = trainer.expt_root / ("base_predictions.csv" if prefix else "dev_predictions.csv")
    metrics = trainer.evaluate(dataloader)
    predictions = trainer.generate(records)
    metrics.update(score_webnlg(predictions, [row["references"] for row in records]))
    source = getattr(trainer.model, "_hw4p2_weight_source", {})
    metrics.update(split="dev", evaluated_inputs=len(records), available_inputs=available_inputs,
                   seed=seed, checkpoint=source.get("path"))
    previous = [path for path in (metrics_path, predictions_path) if path.is_file()]
    if previous:
        history = (
            trainer.expt_root
            / "evaluation-history"
            / str(time_ns())
        )
        history.mkdir(parents=True)
        for path in previous:
            path.replace(history / path.name)
    with metrics_path.open("x") as file:
        json.dump(metrics, file, indent=2)
    write_predictions(records, predictions, predictions_path)
    return metrics, predictions



def export_submission(project_dir, run_dirs, notebook_path, output_path):
    """Export code, the saved notebook and small evidence files, never weights.

    The submission backend includes this archive in the final Gradescope ZIP.
    Save the executed notebook before exporting.
    """
    project = Path(project_dir).resolve()
    notebook = Path(notebook_path).resolve()
    files = [(notebook, notebook.name)]
    for name in ("mytorch", "hw4lib", "tests", "submission"):
        source = project / name
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("Submission source must not contain symlinks")
            if path.is_file() and path.suffix == ".py" and "__pycache__" not in path.parts:
                files.append((path, str(path.relative_to(project))))
    # Only provided tokenizer assets, not arbitrary JSON files in the project.
    for path in sorted((project / "hw4lib/data/tokenizer_jsons").glob("*.json")):
        if path.is_symlink():
            raise ValueError("Tokenizer assets must not be symlinks")
        files.append((path, str(path.relative_to(project))))
    for name in ("hw1p2.json", "hw2p2.json", "hw3p2.json", "hw4p2.json"):
        backend_config = project / "submission/configs" / name
        if backend_config.is_file():
            files.append((backend_config, f"submission/configs/{name}"))
    for name in ("tests/gqa_reference.json", "requirements.txt", "requirements-colab.txt",
                 "requirements-hw4p2.txt", "requirements-hw4p2-colab.txt"):
        path = project / name
        if path.is_file():
            files.append((path, name))
    for directory in map(Path, run_dirs):
        for name in ("config.yaml", "metrics.json", "base_metrics.json", "history.json",
                     "dev_predictions.csv", "base_predictions.csv", "parameter_counts.json", "wandb_run.json"):
            path = directory / name
            if path.is_symlink():
                raise ValueError("Evidence must not contain symlinks")
            if path.is_file():
                files.append((path, f"evidence/{directory.name}/{name}"))
    with zipfile.ZipFile(output_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
    return Path(output_path)


def save_wandb_run(trainer):
    """Save the run path for resuming this experiment."""
    run = trainer.wandb_run
    if run is None:
        raise RuntimeError("The trainer has no W&B run.")

    path = f"{run.entity}/{run.project}/{run.id}"
    (trainer.expt_root / "wandb_run.json").write_text(
        json.dumps({"path": path}),
        encoding="utf-8",
    )
    return path


def load_selected_model(config, full_dir, lora_dir):
    """Load the prediction model with its W&B provenance."""
    from hw4lib.model import DecoderOnlyTransformer
    from hw4lib.model.lora import inject_lora, load_adapter

    choice = config["submission"]["model"]
    model = DecoderOnlyTransformer(**config["model"])

    if choice == "full":
        load_base_weights(
            model,
            Path(full_dir) / "checkpoints/best.pth",
        )
    elif choice == "lora":
        load_base_weights(model, config["base_checkpoint"])
        inject_lora(model, **config["lora"])
        load_adapter(model, Path(lora_dir) / "adapter.pth")
    else:
        raise ValueError(
            "submission.model must be 'full' or 'lora'"
        )

    return model
