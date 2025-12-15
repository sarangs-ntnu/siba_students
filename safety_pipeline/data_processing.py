from __future__ import annotations

from typing import Dict, List

from datasets import Dataset, DatasetDict, load_dataset

from .labels import derive_safety_labels
from .prompts import SYSTEM_PROMPT
from .safety_detection import ViolationDetector


def load_and_validate_dataset(dataset_name: str = "Amod/mental_health_counseling_conversations") -> Dataset:
    """Load the dataset and retain only entries with both Context and Response.

    The function enforces that only the two required fields are present and removes
    null or empty entries without rewriting text content.
    """

    ds = load_dataset(dataset_name, split="train")
    expected_columns = {"Context", "Response"}
    if set(ds.column_names) != expected_columns:
        extra = set(ds.column_names) - expected_columns
        missing = expected_columns - set(ds.column_names)
        raise ValueError(f"Dataset must contain exactly {expected_columns}. Missing={missing}, extra={extra}")

    def _valid(example: Dict[str, str]) -> bool:
        return bool(example["Context"] and example["Context"].strip()) and bool(
            example["Response"] and example["Response"].strip()
        )

    ds = ds.filter(_valid)
    return ds


def derive_labels(ds: Dataset) -> Dataset:
    """Attach derived safety labels as metadata without changing the core text."""

    def _labeler(example: Dict[str, str]) -> Dict[str, Dict[str, bool]]:
        return {"safety_labels": derive_safety_labels(example["Response"])}

    return ds.map(_labeler)


def format_for_instruction_tuning(ds: Dataset) -> Dataset:
    """Convert samples to instruction-response pairs with a fixed system prompt."""

    def _formatter(example: Dict[str, str]) -> Dict[str, str]:
        return {
            "system": SYSTEM_PROMPT,
            "instruction": example["Context"],
            "response": example["Response"],
        }

    return ds.map(_formatter, remove_columns=["Context", "Response"])


def create_sft_dataset(dataset_name: str = "Amod/mental_health_counseling_conversations") -> DatasetDict:
    base = load_and_validate_dataset(dataset_name)
    labeled = derive_labels(base)
    formatted = format_for_instruction_tuning(labeled)
    return DatasetDict({"train": formatted})


def annotate_violation_scores(ds: Dataset, detector: ViolationDetector) -> Dataset:
    """Add violation scores for use in the safety-aware loss."""

    def _score(example: Dict[str, str]) -> Dict[str, float]:
        return {"violation_score": detector.score(example["response"])}

    return ds.map(_score)


def prepare_eval_prompts(ds: Dataset, limit: int = 32) -> List[str]:
    """Collect a deterministic subset of contexts for evaluation generation."""

    return [record["instruction"] for record in ds.select(range(min(limit, len(ds))))]
