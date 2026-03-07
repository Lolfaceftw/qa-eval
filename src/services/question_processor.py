"""Compute semantic-similarity signals for generated questions.

References:
    [1] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using
    Siamese BERT-Networks," in Proceedings of the 2019 Conference on Empirical
    Methods in Natural Language Processing and the 9th International Joint
    Conference on Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China,
    2019, pp. 3982-3992, doi: 10.18653/v1/D19-1410.

Current Literature and Gaps:
    ACL Anthology literature was reviewed for sentence-level semantic similarity
    and clustering methods appropriate for duplicate detection. Reference [1]
    supports sentence embeddings with cosine similarity as the semantic scoring
    primitive. This repository still needs repo-specific heuristics for
    canonicalization, category-fit validation, and representative selection
    because the literature does not define this exact transcript-summary
    question-selection workflow.
"""

import os
from collections.abc import Callable, Sequence

# Suppress huggingface_hub progress bars and transformers warnings that break tui.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from src.config.config_manager import ConfigManager


class QuestionProcessor:
    """Load embeddings and compute similarity components for question text."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the embedding model configuration."""
        config = ConfigManager()
        self.model_id = model_id or config.get(
            "embedding.model", "Qwen/Qwen3-Embedding-4B"
        )

        if device is None:
            config_device = config.get("embedding.device")
            self.device = config_device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.log_callback = log_callback
        self.tokenizer = None
        self.model = None

    def _log(self, message: str) -> None:
        """Emit a log line to the optional UI callback."""
        if self.log_callback:
            self.log_callback(f"> {message}\n")

    def _ensure_model_loaded(self) -> None:
        """Load the tokenizer and embedding model on first use."""
        if self.model is None:
            self._log(f"Loading embedding model: {self.model_id} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()

    def get_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        """Generate L2-normalized embeddings for the supplied texts."""
        if not texts:
            return np.empty((0, 0))

        self._ensure_model_loaded()

        with torch.no_grad():
            inputs = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            outputs = self.model(**inputs)

            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            masked_embeddings = token_embeddings * attention_mask
            summed_embeddings = masked_embeddings.sum(dim=1)
            token_counts = attention_mask.sum(dim=1).clamp(min=1)
            # IEEE citation: Adapted sentence-embedding similarity workflow from [1].
            embeddings = summed_embeddings / token_counts
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def get_similarity_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """Compute pairwise cosine similarity for the supplied texts."""
        embeddings = self.get_embeddings(texts)
        if embeddings.size == 0:
            return np.empty((0, 0))
        # IEEE citation: Cosine similarity is used as the semantic comparison
        # primitive in the SBERT workflow described in [1].
        return cosine_similarity(embeddings)

    def find_similar_components(
        self,
        texts: Sequence[str],
        threshold: float | None = None,
    ) -> list[set[int]]:
        """Cluster texts into connected components above the similarity threshold."""
        if not texts:
            return []

        if threshold is None:
            threshold = ConfigManager().get("embedding.threshold", 0.85)

        similarity_matrix = self.get_similarity_matrix(texts)
        parent = list(range(len(texts)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left in range(len(texts)):
            for right in range(left + 1, len(texts)):
                if similarity_matrix[left, right] > threshold:
                    union(left, right)

        components: dict[int, set[int]] = {}
        for index in range(len(texts)):
            root = find(index)
            components.setdefault(root, set()).add(index)

        clustered_components = list(components.values())
        duplicate_clusters = [cluster for cluster in clustered_components if len(cluster) > 1]
        self._log(
            "Found "
            f"{len(duplicate_clusters)} semantic duplicate clusters "
            f"from {len(texts)} candidate questions."
        )
        return clustered_components
