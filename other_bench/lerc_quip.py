"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or
 * https://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Set, Union

import torch
from qaeval.scoring.scorers.scorer import Scorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict(
    model,
    tokenizer,
    batch_sentences: List[str],
    device: Union[int, str],
) -> List[Dict[str, float]]:
    """Run one forward pass for the Quip scorer."""
    inputs = tokenizer(
        batch_sentences,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    use_cuda = device != "cpu"
    autocast_context = torch.cuda.amp.autocast() if use_cuda else nullcontext()
    with torch.inference_mode():
        with autocast_context:
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
    scores = [value[0] for value in outputs[0].cpu().tolist()]
    return [{"pred_score": score} for score in scores]


class LERCQuipScorer(Scorer):
    """Official Quip scorer with CPU fallback for this benchmark harness."""

    def __init__(self, lerc_quip_path: str, cuda_device: int, batch_size: int = 8) -> None:
        self.device = cuda_device if cuda_device >= 0 else "cpu"
        self.predictor = AutoModelForSequenceClassification.from_pretrained(
            lerc_quip_path
        ).to(self.device)
        self.predictor.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(lerc_quip_path)
        self.batch_size = batch_size

    def keys(self) -> Set[str]:
        """Return the metric keys produced by the scorer."""
        return {"lerc_quip"}

    def offload_to_cpu(self) -> None:
        """Move the Quip model back to CPU before the scorer is discarded."""
        if self.device == "cpu":
            return
        self.predictor.to("cpu")
        self.device = "cpu"

    def _score_single_ref(
        self,
        context: str,
        questions: List[str],
        answers: List[str],
        predictions: List[str],
        probabilities: List[float],
        null_probabilities: List[float],
    ) -> List[Dict[str, float]]:
        """Score one reference list with the official Quip procedure."""
        input_rows = []
        indices = []
        for index, (answer, question, prediction_text, probability, null_probability) in enumerate(
            zip(
                answers,
                questions,
                predictions,
                probabilities,
                null_probabilities,
            )
        ):
            if probability > null_probability:
                input_rows.append(
                    f"{question}\n{answer}\n{prediction_text}\n{context}"
                )
                indices.append(index)

        output_dicts: List[Dict[str, float]] = []
        for start in range(0, len(input_rows), self.batch_size):
            batch = input_rows[start : start + self.batch_size]
            output_dicts.extend(
                predict(self.predictor, self.tokenizer, batch, self.device)
            )

        scores = [0.0] * len(questions)
        for index, output_dict in zip(indices, output_dicts):
            scores[index] = output_dict["pred_score"]
        return [{"lerc_quip": score} for score in scores]
