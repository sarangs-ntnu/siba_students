# Safety-aware mental health assistant pipeline

This repository implements an end-to-end workflow for the `Amod/mental_health_counseling_conversations` dataset, covering loading/validation, safety label derivation, instruction tuning formatting, baseline and safety-aware fine-tuning, and comparative evaluation.

## Components

- `safety_pipeline/data_processing.py`: dataset loading, validation, formatting, and derived safety labels as metadata.
- `safety_pipeline/labels.py`: rule-based cues for non-diagnostic, non-prescriptive, referral, and non-directive language.
- `safety_pipeline/safety_detection.py`: unsafe advice detector combining rules and embedding similarity.
- `safety_pipeline/training.py`: baseline SFT trainer and safety-aware trainer with combined loss.
- `safety_pipeline/evaluation.py`: generation, metric computation, and failure mode analysis utilities.
- `scripts/run_pipeline.py`: orchestrates the tasks end-to-end.

## Notes

- The code avoids modifying or rewriting dataset text; filters only remove null/empty entries.
- Safety metadata is stored separately via derived labels and violation scores.
- Fine-tuning uses causal language modeling loss, with an optional safety penalty controlled by `lambda_safety`.
