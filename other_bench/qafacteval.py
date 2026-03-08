"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or
 * https://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import annotations

import gc
import logging
from time import perf_counter
from typing import Any, Dict, List, Union

from qaeval import QAEval
from qaeval.answer_selection import AnswerSelector, NP_CHUNKS_STRATEGY
from qaeval.answering.model import QuestionAnsweringModel
from qaeval.generation.model import QuestionGenerationModel
from qaeval.scoring.scorers import ExactMatchScorer, F1Scorer, IsAnsweredScorer, MetaScorer
from transformers.data.metrics.squad_metrics import compute_f1

from lerc_quip import LERCQuipScorer

MetricsDict = Dict[str, float]
SummaryType = Union[str, List[str]]
LOGGER = logging.getLogger(__name__)
ANSWERING_INPUT_CHUNK_SIZE = 48


def get_filter(
    qa_summ: List[List[object]],
    answers_ref: List[str],
) -> List[bool]:
    """Filter QA pairs with answerability and lexical-overlap checks."""
    answerable = []
    a_orig = []
    for qa_summ_ in qa_summ:
        answerability = qa_summ_[1] > qa_summ_[2]
        answerable.append(answerability)
        a_orig.append(qa_summ_[0])

    f1s = [
        compute_f1(answer, prediction)
        for answer, prediction in zip(answers_ref, a_orig)
    ]
    bool_f1 = [score > 0.60 for score in f1s]
    return [f1_value and answerable_value for f1_value, answerable_value in zip(bool_f1, answerable)]


class QAFactEval(QAEval):
    """Official QAFactEval wrapper on top of QAEval."""

    def __init__(
        self,
        lerc_quip_path: str,
        use_lerc_quip: bool,
        lerc_batch_size: int,
        cuda_device: int,
        *args,
        **kwargs,
    ) -> None:
        del args
        try:
            import spacy

            spacy.load("en_core_web_sm")
        except Exception:
            import spacy.cli

            spacy.cli.download("en_core_web_sm")

        self._cuda_device = cuda_device
        self._generation_model_path = kwargs["generation_model_path"]
        self._answering_model_dir = kwargs["answering_model_dir"]
        self._generation_batch_size = kwargs.get("generation_batch_size", 8)
        self._answering_batch_size = kwargs.get("answering_batch_size", 8)
        self._use_lerc_quip = use_lerc_quip
        self._lerc_quip_path = lerc_quip_path
        self._lerc_batch_size = lerc_batch_size
        self._lerc_quip_scorer: LERCQuipScorer | None = None
        self.answer_selector = AnswerSelector(
            kwargs.get("answer_selection_strategy", NP_CHUNKS_STRATEGY)
        )
        self.question_generator: QuestionGenerationModel | None = None
        self.question_answerer: QuestionAnsweringModel | None = None
        self.verbose = kwargs.get("verbose", False)
        self.scorer = MetaScorer(
            [IsAnsweredScorer(), ExactMatchScorer(), F1Scorer()]
        )

    def score_batch_qafacteval(
        self,
        source: List[SummaryType],
        summaries: List[List[SummaryType]],
        qa_pairs_precomputed: List = None,
        predictions_lists: List = None,
        return_qa_pairs: bool = False,
    ) -> List[List[MetricsDict]]:
        """Score one batch with the official QAFactEval filtering procedure."""
        source = self._flatten_summaries(source)
        summaries = self._flatten_references_list(summaries)
        source, summaries, is_empty_list = self._get_empty_summary_mask(source, summaries)

        if qa_pairs_precomputed:
            qa_pairs_lists = qa_pairs_precomputed
        else:
            generation_started_at = perf_counter()
            self._ensure_question_generator_loaded()
            try:
                qa_pairs_lists = self._generate_qa_pairs(summaries)
            finally:
                self._release_question_generator()
            LOGGER.info(
                "Question generation completed in %.2fs.",
                perf_counter() - generation_started_at,
            )

        summaries_cons = [summary_group[0] for summary_group in summaries]
        answer_started_at = perf_counter()
        self._ensure_question_answerer_loaded()
        try:
            predictions_lists_consistency = self._answer_questions(
                summaries_cons,
                qa_pairs_lists,
            )
            qa_pairs_lists_cons = []
            for predictions_consistency, cur_qa_pair in zip(
                predictions_lists_consistency,
                qa_pairs_lists,
            ):
                qa_summ_new = [
                    [
                        prediction["prediction"],
                        prediction["probability"],
                        prediction["null_probability"],
                    ]
                    for prediction in predictions_consistency[0]
                ]
                answers_ref = [question_pair["answer"] for question_pair in cur_qa_pair[0]]
                bool_total = get_filter(qa_summ_new, answers_ref)
                cur_qa_pair_keep = [
                    qa_pair
                    for count, qa_pair in enumerate(cur_qa_pair[0])
                    if bool_total[count]
                ]
                qa_pairs_lists_cons.append([cur_qa_pair_keep] if cur_qa_pair_keep else [[]])

            if predictions_lists is None:
                predictions_lists = self._answer_questions(source, qa_pairs_lists_cons)
        finally:
            self._release_question_answerer()
        LOGGER.info(
            "Question answering completed in %.2fs.",
            perf_counter() - answer_started_at,
        )

        retained_pair_count = sum(
            len(qa_pairs)
            for qa_pairs_list in qa_pairs_lists_cons
            for qa_pairs in qa_pairs_list
        )
        LOGGER.info(
            "Retained %s QA pairs for final metric scoring.",
            retained_pair_count,
        )

        quip_load_started_at = perf_counter()
        self._ensure_lerc_quip_scorer_loaded()
        LOGGER.info(
            "Loaded the Quip scorer in %.2fs.",
            perf_counter() - quip_load_started_at,
        )
        scoring_started_at = perf_counter()
        try:
            metrics_list, scores_lists = self._score_predictions(
                source,
                qa_pairs_lists_cons,
                predictions_lists,
            )
        finally:
            self._release_lerc_quip_scorer()
        LOGGER.info(
            "Final metric scoring completed in %.2fs.",
            perf_counter() - scoring_started_at,
        )
        if return_qa_pairs:
            output = self._combine_outputs(
                metrics_list,
                qa_pairs_lists_cons,
                predictions_lists,
                scores_lists,
            )
        else:
            output = metrics_list

        output = self._insert_empty_outputs(output, is_empty_list, return_qa_pairs)
        if not return_qa_pairs:
            return output

        output_final = []
        for out, qa_pairs_list, predictions_cons in zip(
            output,
            qa_pairs_lists,
            predictions_lists_consistency,
        ):
            output_final.append((out[0], out[1], qa_pairs_list, predictions_cons))
        return output_final

    def _answer_questions(
        self,
        summaries: List[str],
        qa_pairs_lists: List[List[List[Dict[str, Any]]]],
    ) -> List[List[List[Dict[str, Any]]]]:
        """Answer deduplicated QA inputs in chunks to limit feature-build memory."""
        qa_inputs: list[tuple[str, str]] = []
        context_to_input_index: dict[tuple[str, str], int] = {}
        mapping: dict[tuple[int, int, int], int] = {}

        for i, (summary, qa_pairs_list) in enumerate(zip(summaries, qa_pairs_lists)):
            for j, qa_pairs in enumerate(qa_pairs_list):
                for k, qa in enumerate(qa_pairs):
                    question = qa["question"]
                    key = (question, summary)
                    if key not in context_to_input_index:
                        context_to_input_index[key] = len(qa_inputs)
                        qa_inputs.append(key)
                    mapping[(i, j, k)] = context_to_input_index[key]

        LOGGER.info(
            "Answering %s distinct (question, context) pairs in chunks of %s.",
            len(qa_inputs),
            ANSWERING_INPUT_CHUNK_SIZE,
        )
        predictions: list[tuple[str, float, float, tuple[int, int]]] = []
        for start in range(0, len(qa_inputs), ANSWERING_INPUT_CHUNK_SIZE):
            chunk = qa_inputs[start : start + ANSWERING_INPUT_CHUNK_SIZE]
            predictions.extend(
                self.question_answerer.answer_all(
                    chunk,
                    return_offsets=True,
                )
            )
        LOGGER.info("Finished answering questions")

        predictions_lists = []
        for i, (summary, qa_pairs_list) in enumerate(zip(summaries, qa_pairs_lists)):
            del summary
            predictions_lists.append([])
            for j, qa_pairs in enumerate(qa_pairs_list):
                predictions_lists[-1].append([])
                for k, qa in enumerate(qa_pairs):
                    del qa
                    index = mapping[(i, j, k)]
                    prediction, probability, null_probability, offsets = predictions[index]
                    predictions_lists[-1][-1].append(
                        {
                            "prediction_id": self._get_prediction_id(index),
                            "prediction": prediction,
                            "probability": probability,
                            "null_probability": null_probability,
                            "start": offsets[0],
                            "end": offsets[1],
                        }
                    )
        return predictions_lists

    def _ensure_question_generator_loaded(self) -> None:
        """Load the question-generation model only for the generation phase."""
        if getattr(self, "question_generator", None) is None:
            LOGGER.info("Loading QAEval question-generation model.")
            self.question_generator = QuestionGenerationModel(
                self._generation_model_path,
                cuda_device=self._cuda_device,
                batch_size=self._generation_batch_size,
                silent=not self.verbose,
            )

    def _release_question_generator(self) -> None:
        """Release the generation model before later scoring phases run."""
        if getattr(self, "question_generator", None) is None:
            return
        LOGGER.info("Releasing QAEval question-generation model.")
        self._offload_from_accelerator(self.question_generator)
        del self.question_generator
        self.question_generator = None
        self._collect_model_memory()

    def _ensure_question_answerer_loaded(self) -> None:
        """Load the question-answering model only for answer passes."""
        if getattr(self, "question_answerer", None) is None:
            LOGGER.info("Loading QAEval question-answering model.")
            self.question_answerer = QuestionAnsweringModel(
                self._answering_model_dir,
                cuda_device=self._cuda_device,
                batch_size=self._answering_batch_size,
                silent=not self.verbose,
            )

    def _release_question_answerer(self) -> None:
        """Release the question-answering model before Quip scoring."""
        if getattr(self, "question_answerer", None) is None:
            return
        LOGGER.info("Releasing QAEval question-answering model.")
        self._offload_from_accelerator(self.question_answerer)
        del self.question_answerer
        self.question_answerer = None
        self._collect_model_memory()

    def _ensure_lerc_quip_scorer_loaded(self) -> None:
        """Lazy-load the Quip scorer only for the scoring phase."""
        if not self._use_lerc_quip or self._lerc_quip_scorer is not None:
            return
        LOGGER.info("Loading QAFactEval Quip scorer for the scoring phase.")
        self._lerc_quip_scorer = LERCQuipScorer(
            lerc_quip_path=self._lerc_quip_path,
            cuda_device=self._cuda_device,
            batch_size=self._lerc_batch_size,
        )
        self.scorer.scorers.append(self._lerc_quip_scorer)

    def _release_lerc_quip_scorer(self) -> None:
        """Release the Quip scorer after one batch so later batches can reload QA models."""
        if self._lerc_quip_scorer is None:
            return
        LOGGER.info("Releasing QAFactEval Quip scorer after scoring.")
        self._offload_from_accelerator(self._lerc_quip_scorer)
        self.scorer.scorers = [
            scorer
            for scorer in self.scorer.scorers
            if scorer is not self._lerc_quip_scorer
        ]
        del self._lerc_quip_scorer
        self._lerc_quip_scorer = None
        self._collect_model_memory()

    def _offload_from_accelerator(self, model_holder: object) -> None:
        """Move known model objects back to CPU before releasing them."""
        if self._cuda_device < 0:
            return

        offload_hook = getattr(model_holder, "offload_to_cpu", None)
        if callable(offload_hook):
            offload_hook()
            return

        candidate_modules = []
        direct_model = getattr(model_holder, "model", None)
        if direct_model is not None and hasattr(direct_model, "to"):
            candidate_modules.append(direct_model)

        predictor = getattr(model_holder, "predictor", None)
        predictor_model = getattr(predictor, "_model", None)
        if predictor_model is not None and hasattr(predictor_model, "to"):
            candidate_modules.append(predictor_model)
        elif predictor is not None and hasattr(predictor, "to"):
            candidate_modules.append(predictor)

        for candidate_module in candidate_modules:
            try:
                candidate_module.to("cpu")
            except Exception as error:
                LOGGER.info(
                    "Could not offload %s to CPU before release (%s).",
                    type(model_holder).__name__,
                    error,
                )

    def _collect_model_memory(self) -> None:
        """Prompt Python and CUDA to return memory after large model transitions."""
        gc.collect()
        if self._cuda_device >= 0:
            import torch

            torch.cuda.empty_cache()
