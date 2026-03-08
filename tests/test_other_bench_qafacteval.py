"""Regression tests for the QAFactEval benchmark helpers and wrapper."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from other_bench import quest_eval_qa_fact_eval as benchmark


def create_ready_model_dir(model_dir: Path) -> None:
    """Create the minimal cached model layout expected by the harness."""
    (model_dir / "generation").mkdir(parents=True, exist_ok=True)
    (model_dir / "generation" / "model.tar.gz").write_text("archive", encoding="utf-8")

    answering_dir = model_dir / "answering"
    answering_dir.mkdir(parents=True, exist_ok=True)
    for filename in benchmark.ANSWERING_DIR_MARKERS:
        (answering_dir / filename).write_text(filename, encoding="utf-8")

    lerc_dir = model_dir / "lerc"
    lerc_dir.mkdir(parents=True, exist_ok=True)
    (lerc_dir / "model.tar.gz").write_text("archive", encoding="utf-8")
    (lerc_dir / "pretraining.tar.gz").write_text("archive", encoding="utf-8")

    quip_dir = model_dir / "quip-512-mocha"
    quip_dir.mkdir(parents=True, exist_ok=True)
    for filename in benchmark.QUIP_DIR_MARKERS:
        (quip_dir / filename).write_text(filename, encoding="utf-8")


def load_qafacteval_wrapper(monkeypatch) -> types.ModuleType:
    """Import the repo-local QAFactEval wrapper with lightweight stub deps."""
    sys.modules.pop("other_bench.qafacteval", None)

    qaeval_module = types.ModuleType("qaeval")

    class FakeBaseQAEval:
        """Provide a lightweight base class for import-time wiring."""

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    qaeval_module.QAEval = FakeBaseQAEval

    answer_selection_module = types.ModuleType("qaeval.answer_selection")
    answer_selection_module.AnswerSelector = type("AnswerSelector", (), {})
    answer_selection_module.NP_CHUNKS_STRATEGY = "np-chunks"

    answering_pkg = types.ModuleType("qaeval.answering")
    answering_model_module = types.ModuleType("qaeval.answering.model")
    answering_model_module.QuestionAnsweringModel = type(
        "QuestionAnsweringModel",
        (),
        {},
    )

    generation_pkg = types.ModuleType("qaeval.generation")
    generation_model_module = types.ModuleType("qaeval.generation.model")
    generation_model_module.QuestionGenerationModel = type(
        "QuestionGenerationModel",
        (),
        {},
    )

    scorers_module = types.ModuleType("qaeval.scoring.scorers")
    for scorer_name in ("ExactMatchScorer", "F1Scorer", "IsAnsweredScorer", "MetaScorer"):
        setattr(scorers_module, scorer_name, type(scorer_name, (), {}))

    transformers_module = types.ModuleType("transformers")
    transformers_data_module = types.ModuleType("transformers.data")
    transformers_metrics_module = types.ModuleType("transformers.data.metrics")
    squad_metrics_module = types.ModuleType("transformers.data.metrics.squad_metrics")
    squad_metrics_module.compute_f1 = lambda answer, prediction: 1.0

    lerc_module = types.ModuleType("lerc_quip")
    lerc_module.LERCQuipScorer = type("LERCQuipScorer", (), {})

    monkeypatch.setitem(sys.modules, "qaeval", qaeval_module)
    monkeypatch.setitem(sys.modules, "qaeval.answer_selection", answer_selection_module)
    monkeypatch.setitem(sys.modules, "qaeval.answering", answering_pkg)
    monkeypatch.setitem(sys.modules, "qaeval.answering.model", answering_model_module)
    monkeypatch.setitem(sys.modules, "qaeval.generation", generation_pkg)
    monkeypatch.setitem(sys.modules, "qaeval.generation.model", generation_model_module)
    monkeypatch.setitem(sys.modules, "qaeval.scoring.scorers", scorers_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "transformers.data", transformers_data_module)
    monkeypatch.setitem(sys.modules, "transformers.data.metrics", transformers_metrics_module)
    monkeypatch.setitem(
        sys.modules,
        "transformers.data.metrics.squad_metrics",
        squad_metrics_module,
    )
    monkeypatch.setitem(sys.modules, "lerc_quip", lerc_module)

    return importlib.import_module("other_bench.qafacteval")


def test_qafacteval_assets_are_ready_requires_marker(tmp_path: Path) -> None:
    """Require the ready marker in addition to the downloaded model files."""
    model_dir = tmp_path / "models" / "qafacteval"
    create_ready_model_dir(model_dir)

    assert not benchmark.qafacteval_assets_are_ready(model_dir)

    benchmark.write_qafacteval_ready_marker(model_dir)
    assert benchmark.qafacteval_assets_are_ready(model_dir)

    benchmark.get_qafacteval_ready_marker_path(model_dir).unlink()
    assert not benchmark.qafacteval_assets_are_ready(model_dir)


def test_ensure_wsl_models_downloaded_skips_ready_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Skip the extra WSL prep pass after the local benchmark cache is ready."""
    model_dir = tmp_path / "models" / "qafacteval"
    create_ready_model_dir(model_dir)
    benchmark.write_qafacteval_ready_marker(model_dir)

    called = False

    def fake_run_wsl_python_script(*args, **kwargs) -> None:
        nonlocal called
        del args, kwargs
        called = True

    monkeypatch.setattr(benchmark, "run_wsl_python_script", fake_run_wsl_python_script)

    benchmark.ensure_wsl_models_downloaded("uv", model_dir)

    assert not called


def test_select_qafacteval_lerc_batch_size_uses_gpu_memory_tiers() -> None:
    """Choose larger Quip batches only when the GPU memory budget allows it."""

    class FakeProps:
        """Expose a fake CUDA memory size."""

        def __init__(self, total_memory: int) -> None:
            """Store the reported GPU memory in bytes."""
            self.total_memory = total_memory

    class FakeCuda:
        """Expose deterministic CUDA device properties."""

        def __init__(self, total_memory: int) -> None:
            """Store the fake GPU memory size."""
            self._total_memory = total_memory

        def get_device_properties(self, device: int) -> FakeProps:
            """Return the fake device properties for the requested device."""
            assert device == 0
            return FakeProps(self._total_memory)

    class FakeTorch:
        """Expose the fake CUDA namespace expected by the selector."""

        def __init__(self, total_memory: int) -> None:
            """Store the fake CUDA namespace."""
            self.cuda = FakeCuda(total_memory)

    assert benchmark.select_qafacteval_lerc_batch_size(FakeTorch(12 * 1024**3), 0) == 32
    assert benchmark.select_qafacteval_lerc_batch_size(FakeTorch(8 * 1024**3), 0) == 24
    assert benchmark.select_qafacteval_lerc_batch_size(FakeTorch(6 * 1024**3), 0) == 16
    assert benchmark.select_qafacteval_lerc_batch_size(FakeTorch(4 * 1024**3), 0) == 8
    assert benchmark.select_qafacteval_lerc_batch_size(FakeTorch(12 * 1024**3), -1) == 8


def test_qafacteval_offloads_known_model_holders_before_release(monkeypatch) -> None:
    """Move supported model holders back to CPU before deleting them."""
    module = load_qafacteval_wrapper(monkeypatch)
    metric = object.__new__(module.QAFactEval)
    metric._cuda_device = 0

    calls: list[str] = []

    class FakeTorchModule:
        """Record the devices used in `.to(...)` calls."""

        def __init__(self, name: str) -> None:
            """Store the module name for call tracking."""
            self._name = name

        def to(self, device: str) -> None:
            """Record the requested device transition."""
            calls.append(f"{self._name}:{device}")

    holder_with_model = types.SimpleNamespace(model=FakeTorchModule("model"))
    holder_with_predictor = types.SimpleNamespace(
        predictor=types.SimpleNamespace(_model=FakeTorchModule("predictor_model"))
    )

    metric._offload_from_accelerator(holder_with_model)
    metric._offload_from_accelerator(holder_with_predictor)

    assert calls == ["model:cpu", "predictor_model:cpu"]


def test_qafacteval_scores_with_phase_scoped_model_loading(monkeypatch) -> None:
    """Load and release generator and answerer models in separate phases."""
    module = load_qafacteval_wrapper(monkeypatch)
    metric = object.__new__(module.QAFactEval)
    call_order: list[str] = []

    metric._flatten_summaries = lambda summaries: summaries
    metric._flatten_references_list = lambda references_list: references_list
    metric._get_empty_summary_mask = lambda source, summaries: (source, summaries, [False])
    metric._insert_empty_outputs = lambda output, is_empty_list, return_qa_pairs: output

    def ensure_generator() -> None:
        call_order.append("load_generator")

    def release_generator() -> None:
        call_order.append("release_generator")

    def ensure_answerer() -> None:
        call_order.append("load_answerer")

    def release_answerer() -> None:
        call_order.append("release_answerer")

    def ensure_quip() -> None:
        call_order.append("load_quip")

    def release_quip() -> None:
        call_order.append("release_quip")

    def generate_pairs(summaries: list[list[str]]) -> list[list[list[dict[str, str]]]]:
        del summaries
        call_order.append("generate_pairs")
        return [[[{"question": "question", "answer": "answer"}]]]

    def answer_questions(
        summaries: list[str],
        qa_pairs_lists: list[list[list[dict[str, str]]]],
    ) -> list[list[list[dict[str, float | str]]]]:
        del qa_pairs_lists
        if summaries == ["candidate summary"]:
            call_order.append("answer_consistency")
        else:
            call_order.append("answer_source")
        return [[[{"prediction": "answer", "probability": 1.0, "null_probability": 0.0}]]]

    def score_predictions(
        source: list[str],
        qa_pairs_lists_cons: list[list[list[dict[str, str]]]],
        predictions_lists: list[list[list[dict[str, float | str]]]],
    ) -> tuple[list[tuple[str, str]], list[str]]:
        del source, qa_pairs_lists_cons, predictions_lists
        call_order.append("score_predictions")
        return [("metrics", "scores")], ["score_details"]

    metric._ensure_question_generator_loaded = ensure_generator
    metric._release_question_generator = release_generator
    metric._ensure_question_answerer_loaded = ensure_answerer
    metric._release_question_answerer = release_answerer
    metric._ensure_lerc_quip_scorer_loaded = ensure_quip
    metric._release_lerc_quip_scorer = release_quip
    metric._generate_qa_pairs = generate_pairs
    metric._answer_questions = answer_questions
    metric._score_predictions = score_predictions

    output = metric.score_batch_qafacteval(
        source=["source transcript"],
        summaries=[["candidate summary"]],
        return_qa_pairs=False,
    )

    assert output == [("metrics", "scores")]
    assert call_order == [
        "load_generator",
        "generate_pairs",
        "release_generator",
        "load_answerer",
        "answer_consistency",
        "answer_source",
        "release_answerer",
        "load_quip",
        "score_predictions",
        "release_quip",
    ]
