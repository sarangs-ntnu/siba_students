from __future__ import annotations

from safety_pipeline.data_processing import (
    annotate_violation_scores,
    create_sft_dataset,
    prepare_eval_prompts,
)
from safety_pipeline.evaluation import analyze_failures, evaluate_safety_and_empathy
from safety_pipeline.prompts import SYSTEM_PROMPT
from safety_pipeline.safety_detection import ViolationDetector
from safety_pipeline.training import train_baseline, train_safety_aware


def main() -> None:
    # Task 1–3: dataset loading, validation, labeling, and formatting
    dataset_dict = create_sft_dataset()
    train_ds = dataset_dict["train"]

    detector = ViolationDetector()

    # Task 4: baseline fine-tuning
    baseline_artifacts = train_baseline(train_ds, model_name="tiiuae/falcon-7b-instruct", output_dir="artifacts/baseline")

    # Task 5–6: safety-aware scoring and fine-tuning
    scored_ds = annotate_violation_scores(train_ds, detector)
    safety_artifacts = train_safety_aware(
        scored_ds,
        model_name="tiiuae/falcon-7b-instruct",
        output_dir="artifacts/safety-aware",
        lambda_safety=0.5,
        detector=detector,
    )

    # Task 7: evaluation
    prompts = prepare_eval_prompts(train_ds)
    eval_results = evaluate_safety_and_empathy(
        baseline_artifacts.output_dir,
        safety_artifacts.output_dir,
        prompts,
        SYSTEM_PROMPT,
        detector,
    )
    print("Evaluation:", eval_results)

    # Task 8: failure analysis using safety-aware outputs
    # In practice, pass generated responses from the safety-aware model. Here we reuse eval prompts for illustration.
    safety_responses = []
    failure_report = analyze_failures(safety_responses, detector)
    print("Failures:", failure_report)


if __name__ == "__main__":
    main()
