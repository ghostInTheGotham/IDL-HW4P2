"""Final HW4P1 code and training-evidence validation."""

from functools import partial
import json
import math
from pathlib import Path
import runpy
import sys
import tarfile

import yaml

from .testing_framework import TestingFramework
from .test_mytorch_softmax import test_softmax
from .test_mytorch_scaled_dot_product_attention import (
    test_scaled_dot_product_attention,
)
from .test_mytorch_grouped_query_attention import (
    run_with_macos_matmul_warnings_ignored,
    test_attention_specializations_and_optional_masks,
    test_fully_masked_rows_have_zero_probabilities,
    test_grouped_query_attention_forward_and_backward,
    test_grouped_query_attention_validation_and_shapes,
)
from .test_mytorch_linear import test_linear
from .test_mytorch_multi_head_attention import test_multi_head_attention


MAX_CHAR_PERPLEXITY = 3.5
EVIDENCE_SCHEMA_VERSION = 1
REQUIRED_EVIDENCE_FILES = {
    "checkpoint_manifest.json",
    "config.yaml",
    "generated_samples.json",
    "model_arch.txt",
    "selection_metrics.json",
    "test_metrics.json",
    "training_curve.png",
    "training_summary.json",
    "wandb_run.json",
}
GRADED_TEST_MODULES = {
    "LMDataset": (
        "tests.test_gpt2_data",
    ),
    "AdditivePositionalEncoding": (
        "tests.test_positional_encoding",
    ),
    "RotaryPositionalEncoding": (
        "tests.test_rotary_positional_encoding",
    ),
    "CausalSelfAttention": (
        "tests.test_causal_self_attention",
    ),
    "KVCache": (
        "tests.test_kv_cache",
    ),
    "MasksAndFeedForward": (
        "tests.test_mask_padding",
        "tests.test_mask_causal",
        "tests.test_sublayer_feedforward",
    ),
    "DecoderOnlyTransformer": (
        "tests.test_sublayer_selfattention",
        "tests.test_decoderlayer_selfattention",
        "tests.test_transformer_decoder_only",
    ),
    "Decoding": (
        "tests.test_decoding_greedy",
        "tests.test_decoding_sampling",
    ),
    "LMTrainer": (
        "tests.test_lr_scheduler_accumulation",
        "tests.test_lm_trainer_p1",
        "tests.test_hw4p1_pipeline",
    ),
}
RUBRIC_POINTS = {
    "HW3P1Prerequisites": 0,
    "HW3P2Prerequisites": 0,
    "GroupedQueryAttention": 15,
    "LMDataset": 10,
    "CausalSelfAttention": 15,
    "KVCache": 15,
    "DecoderOnlyTransformer": 5,
    "Decoding": 10,
    "LMTrainer": 10,
    "TrainingPPL": 20,
}


def _load_json(path: Path) -> dict:
    """Load one required JSON object."""
    assert path.is_file(), f"{path.name} does not exist"
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    assert isinstance(payload, dict), f"{path.name} must contain a JSON object"
    assert payload.get("schema_version") == EVIDENCE_SCHEMA_VERSION, (
        f"{path.name} has an unsupported schema version"
    )
    return payload


def _submission_paths() -> tuple[Path, Path | None]:
    """Locate notebook output or an extracted Gradescope handin."""
    working_directory = Path.cwd()
    notebook_evidence = working_directory / "submission_evidence"
    if notebook_evidence.is_dir():
        return notebook_evidence, working_directory / "handin.tar.gz"
    if (working_directory / "test_metrics.json").is_file():
        return working_directory, None
    raise AssertionError(
        "Submission evidence was not found. Run the notebook evidence cell first."
    )


def _is_safe_basename(value: object) -> bool:
    """Return whether a manifest value names one file in its own directory."""
    return (
        isinstance(value, str)
        and value not in {"", ".", ".."}
        and "/" not in value
        and "\\" not in value
    )


def _validate_manifest(evidence_dir: Path) -> set[str]:
    """Verify every artifact listed in the submission manifest."""
    manifest = _load_json(evidence_dir / "submission_manifest.json")
    files = manifest.get("files")
    assert isinstance(files, dict) and files, "submission manifest is empty"

    for filename, record in files.items():
        assert _is_safe_basename(filename), (
            f"Unsafe manifest filename: {filename!r}"
        )
        assert isinstance(record, dict), f"Invalid manifest record for {filename}"
        path = evidence_dir / filename
        assert path.is_file(), f"Manifest file is missing: {filename}"

    listed_files = set(files)
    missing = REQUIRED_EVIDENCE_FILES.difference(listed_files)
    assert not missing, f"Submission manifest is missing: {sorted(missing)}"
    return listed_files


def _validate_archive(archive_path: Path | None, listed_files: set[str]) -> None:
    """Verify notebook-created archive structure when the archive is available."""
    if archive_path is None:
        return
    assert archive_path.is_file(), "handin.tar.gz does not exist"
    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())

    assert any(name.startswith("handin/mytorch/") for name in names)
    assert any(name.startswith("handin/hw4lib/") for name in names)
    for filename in listed_files | {"submission_manifest.json"}:
        assert f"handin/{filename}" in names, f"Archive is missing {filename}"
    checkpoint_suffixes = (".pt", ".pth", ".ckpt")
    assert not any(name.lower().endswith(checkpoint_suffixes) for name in names), (
        "Checkpoint weights must not be included in handin.tar.gz"
    )


def _run_test_modules(module_names: tuple[str, ...]) -> None:
    """Run student-visible test modules and propagate their exit status."""
    for module_name in module_names:
        print(f"Running {module_name}")
        sys.modules.pop(module_name, None)
        try:
            runpy.run_module(
                module_name,
                run_name="__main__",
                alter_sys=True,
            )
        except SystemExit as exit_status:
            if exit_status.code not in (None, 0, False):
                raise AssertionError(
                    f"{module_name} failed with status {exit_status.code}"
                ) from exit_status


def check_lm_dataset() -> None:
    _run_test_modules(GRADED_TEST_MODULES["LMDataset"])


def check_additive_positional_encoding() -> None:
    _run_test_modules(GRADED_TEST_MODULES["AdditivePositionalEncoding"])


def check_rotary_positional_encoding() -> None:
    _run_test_modules(GRADED_TEST_MODULES["RotaryPositionalEncoding"])


def check_causal_self_attention() -> None:
    _run_test_modules(GRADED_TEST_MODULES["CausalSelfAttention"])


def check_kv_cache() -> None:
    _run_test_modules(GRADED_TEST_MODULES["KVCache"])


def check_masks_and_feed_forward() -> None:
    _run_test_modules(GRADED_TEST_MODULES["MasksAndFeedForward"])


def check_decoder_only_transformer() -> None:
    _run_test_modules(GRADED_TEST_MODULES["DecoderOnlyTransformer"])


def check_decoding() -> None:
    _run_test_modules(GRADED_TEST_MODULES["Decoding"])


def check_lm_trainer() -> None:
    _run_test_modules(GRADED_TEST_MODULES["LMTrainer"])


def test_lm_perplexity() -> dict:
    """Validate final configuration, evidence integrity, and character PPL."""
    print("Testing Language Modeling ...")
    evidence_dir, archive_path = _submission_paths()
    listed_files = _validate_manifest(evidence_dir)

    test_payload = _load_json(evidence_dir / "test_metrics.json")
    selection_payload = _load_json(evidence_dir / "selection_metrics.json")
    samples_payload = _load_json(evidence_dir / "generated_samples.json")
    summary_payload = _load_json(evidence_dir / "training_summary.json")
    checkpoint_payload = _load_json(evidence_dir / "checkpoint_manifest.json")
    wandb_payload = _load_json(evidence_dir / "wandb_run.json")

    metrics = test_payload.get("metrics")
    assert isinstance(metrics, dict), "test_metrics.json is missing metrics"
    perplexity = float(metrics["perplexity_char"])
    assert math.isfinite(perplexity) and perplexity > 0
    assert perplexity <= MAX_CHAR_PERPLEXITY, (
        "Character-level perplexity is greater than the threshold: "
        f"{perplexity} > {MAX_CHAR_PERPLEXITY}"
    )

    checkpoint_filename = checkpoint_payload.get("filename")
    assert _is_safe_basename(checkpoint_filename), "Invalid checkpoint filename"
    assert test_payload.get("checkpoint_filename") == checkpoint_filename
    assert selection_payload.get("checkpoint_filename") == checkpoint_filename
    assert samples_payload.get("checkpoint_filename") == checkpoint_filename

    assert int(summary_payload.get("epochs_completed", 0)) > 0
    assert int(summary_payload.get("global_optimizer_steps", 0)) > 0
    history = summary_payload.get("training_history")
    assert isinstance(history, list) and history, "Training history is empty"
    samples = samples_payload.get("samples_by_strategy")
    assert isinstance(samples, dict) and samples, "Generated samples are empty"

    config_path = evidence_dir / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_config = config["data"]
    assert data_config["train_partition"] == "train"
    assert data_config["val_partition"] == "val"
    assert data_config["test_partition"] == "test"

    assert config["training"].get("use_wandb") is True, ("W&B logging is required for this assignment.")
    assert wandb_payload.get("enabled") is True, ("W&B run evidence is required.")
    assert all(
        isinstance(wandb_payload.get(key), str)
        and wandb_payload[key].strip()
        for key in ("entity", "project", "run_id")
    ), "W&B run information is incomplete."
    screenshot_path = evidence_dir / "training_screenshot.png"
    assert screenshot_path.is_file(), (
        "Save your W&B run screenshot as training_screenshot.png "
        "in the project folder, then regenerate the submission evidence."
    )
    with screenshot_path.open("rb") as screenshot_file:
        assert screenshot_file.read(8) == b"\x89PNG\r\n\x1a\n"

    _validate_archive(archive_path, listed_files)
    print(
        "Test passed: Character-level perplexity "
        f"({perplexity}) <= Threshold ({MAX_CHAR_PERPLEXITY})"
    )
    return test_payload


def build_testing_framework() -> TestingFramework:
    """Use the same grading cases in the notebook and Gradescope runner."""
    # HW3P1/P2 reuse checks are required but carry no HW4P1 points.
    rubric_dict = RUBRIC_POINTS

    testing_framework = TestingFramework(
        test_categories={key: [] for key in rubric_dict}
    )

    filtered_linear_test = partial(
        run_with_macos_matmul_warnings_ignored,
        test_linear,
    )
    filtered_attention_test = partial(
        run_with_macos_matmul_warnings_ignored,
        test_scaled_dot_product_attention,
    )
    filtered_multi_head_test = partial(
        run_with_macos_matmul_warnings_ignored,
        test_multi_head_attention,
    )
    testing_framework.register_test_case(
        "HW3P1Prerequisites",
        filtered_linear_test,
        "HW3P1 Linear reuse",
    )
    testing_framework.register_test_case(
        "HW3P1Prerequisites",
        test_softmax,
        "HW3P1 Softmax reuse",
    )
    testing_framework.register_test_case(
        "HW3P1Prerequisites",
        filtered_attention_test,
        "HW3P1 ScaledDotProductAttention reuse",
    )
    testing_framework.register_test_case(
        "HW3P1Prerequisites",
        filtered_multi_head_test,
        "HW3P1 MultiHeadAttention reuse",
    )
    testing_framework.register_test_case(
        "GroupedQueryAttention",
        test_grouped_query_attention_forward_and_backward,
        "GQA/MQA Forward and Backward Tests",
    )
    testing_framework.register_test_case(
        "GroupedQueryAttention",
        test_attention_specializations_and_optional_masks,
        "GQA/MQA Mask Tests",
    )
    testing_framework.register_test_case(
        "GroupedQueryAttention",
        test_grouped_query_attention_validation_and_shapes,
        "GQA/MQA Shape Tests",
    )
    testing_framework.register_test_case(
        "GroupedQueryAttention",
        test_fully_masked_rows_have_zero_probabilities,
        "GQA/MQA Fully Masked Row Tests",
    )
    testing_framework.register_test_case(
        "LMDataset",
        check_lm_dataset,
        "Fixed-window language-model dataset",
    )
    testing_framework.register_test_case(
        "HW3P2Prerequisites",
        check_additive_positional_encoding,
        "HW3P2 additive positional encoding reuse",
    )
    testing_framework.register_test_case(
        "HW3P2Prerequisites",
        check_rotary_positional_encoding,
        "HW3P2 RoPE adaptation to the HW4 interface",
    )
    testing_framework.register_test_case(
        "CausalSelfAttention",
        check_causal_self_attention,
        "Causal MHA/GQA/MQA",
    )
    testing_framework.register_test_case(
        "KVCache",
        check_kv_cache,
        "Compact KV cache",
    )
    testing_framework.register_test_case(
        "HW3P2Prerequisites",
        check_masks_and_feed_forward,
        "HW3P2 masks and feed-forward reuse",
    )
    testing_framework.register_test_case(
        "DecoderOnlyTransformer",
        check_decoder_only_transformer,
        "HW3P2 decoder adaptation and HW4 cache integration",
    )
    testing_framework.register_test_case(
        "Decoding",
        check_decoding,
        "HW3P2 greedy adaptation and HW4 sampling implementation",
    )
    testing_framework.register_test_case(
        "LMTrainer",
        check_lm_trainer,
        "HW3P2 training-loop adaptation for language modeling",
    )
    testing_framework.register_test_case(
        "TrainingPPL",
        test_lm_perplexity,
        "Full-training evidence and character PPL",
    )

    return testing_framework


if __name__ == "__main__":
    rubric_dict = RUBRIC_POINTS
    testing_framework = build_testing_framework()
    testing_framework.run_tests()
    testing_framework.summarize_results()
    auto_results = testing_framework.get_autoresults(rubric_dict)
    print(json.dumps(auto_results))

    failed = any(
        result["status"] != "PASSED"
        for category_results in testing_framework.results.values()
        for result in category_results
    )
    if failed:
        raise SystemExit(1)
