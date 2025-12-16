from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .safety_detection import ViolationDetector


@dataclass
class ModelArtifacts:
    model_name: str
    output_dir: str


def tokenize_dialogue(example: Dict[str, str], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    text = f"<s>\n[SYSTEM]\n{example['system']}\n[USER]\n{example['instruction']}\n[ASSISTANT]\n{example['response']}"  # noqa: E501
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=1024,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def train_baseline(
    dataset: Dataset,
    model_name: str,
    output_dir: str,
    batch_size: int = 2,
    num_epochs: int = 1,
) -> ModelArtifacts:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = dataset.map(lambda ex: tokenize_dialogue(ex, tokenizer))
    model = AutoModelForCausalLM.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return ModelArtifacts(model_name=model_name, output_dir=output_dir)


class SafetyAwareTrainer(Trainer):
    def __init__(self, lambda_safety: float, detector: ViolationDetector, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_safety = lambda_safety
        self.detector = detector

    def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore[override]
        outputs = model(**inputs)
        base_loss = outputs.loss
        responses = inputs.get("responses")
        if responses is None:
            safety_penalty = torch.tensor(0.0, device=base_loss.device)
        else:
            if isinstance(responses, torch.Tensor):
                responses = responses.tolist()
            violation_scores = [self.detector.score(text) for text in responses]
            safety_penalty = torch.tensor(violation_scores, device=base_loss.device).mean()
        total_loss = base_loss + self.lambda_safety * safety_penalty
        return (total_loss, outputs) if return_outputs else total_loss


def train_safety_aware(
    dataset: Dataset,
    model_name: str,
    output_dir: str,
    lambda_safety: float = 0.5,
    batch_size: int = 2,
    num_epochs: int = 1,
    detector: Optional[ViolationDetector] = None,
) -> ModelArtifacts:
    detector = detector or ViolationDetector()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    def _with_response(example: Dict[str, str]) -> Dict[str, str]:
        tokenized = tokenize_dialogue(example, tokenizer)
        tokenized["responses"] = example["response"]
        return tokenized

    tokenized = dataset.map(_with_response)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = SafetyAwareTrainer(
        lambda_safety=lambda_safety,
        detector=detector,
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return ModelArtifacts(model_name=model_name, output_dir=output_dir)
