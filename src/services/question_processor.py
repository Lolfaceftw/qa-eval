import os

# Suppress huggingface_hub progress bars and transformers warnings that break tui
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from typing import List, Dict, Any, Callable
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config.config_manager import ConfigManager


class QuestionProcessor:
    """Service for processing, deduplicating, and filtering agent-generated questions."""

    def __init__(
        self,
        model_id: str = None,
        device: str = None,
        log_callback: Callable[[str], None] = None,
    ):
        config = ConfigManager()
        self.model_id = model_id or config.get(
            "embedding.model", "Qwen/Qwen3-Embedding-4B"
        )

        if device is None:
            # Check config first, then auto-detect
            config_device = config.get("embedding.device")
            if config_device:
                self.device = config_device
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.log_callback = log_callback
        self.tokenizer = None
        self.model = None

    def _log(self, message: str):
        if self.log_callback:
            self.log_callback(f"> {message}\n")

    def _ensure_model_loaded(self):
        """Loads the model and tokenizer if not already loaded."""
        if self.model is None:
            self._log(f"Loading embedding model: {self.model_id} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of texts."""
        self._ensure_model_loaded()

        # Qwen-Embedding models typically use a specific instruction prefix or prompt structure
        # for different tasks. For general similarity, we can use them as is or with a prefix.
        # Based on HFM documentation, we'll just tokenize and encode.

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            outputs = self.model(**inputs)
            # Use last hidden state mean pooling or [CLS] as requested by model docs
            # For Qwen3-Embedding, it usually has a specific pooler but let's assume mean pooling for robustness
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return embeddings

    def deduplicate(
        self, questions: List[Dict[str, Any]], threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Removes highly similar questions using cosine similarity.
        'questions' should be a list of dicts like {'question': str, ...}
        """
        if not questions:
            return []

        if threshold is None:
            threshold = ConfigManager().get("embedding.threshold", 0.85)

        texts = [q["question"] for q in questions]
        embeddings = self.get_embeddings(texts)

        # Calculate cosine similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        # TODO: see what this does and implement percentile
        to_remove = set()
        for i in range(len(texts)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(texts)):
                if j in to_remove:
                    continue
                if sim_matrix[i, j] > threshold:
                    # Mark the 'partner' (later one) for removal
                    to_remove.add(j)
                    self._log(
                        f"Removing similar question: '{texts[j]}' (Similarity: {sim_matrix[i, j]:.4f})"
                    )

        filtered_questions = [
            q for idx, q in enumerate(questions) if idx not in to_remove
        ]
        return filtered_questions

    def process_and_limit(
        self, questions_by_category: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Processes questions for multiple categories: deduplicates until all are unique.
        """
        result = {}
        for category, questions in questions_by_category.items():
            self._log(
                f"Processing category '{category}' with {len(questions)} questions"
            )
            # 1. Deduplicate
            final_questions = self.deduplicate(questions)
            result[category] = final_questions
            self._log(
                f"Category '{category}' finished with {len(final_questions)} unique questions (removed {len(questions) - len(final_questions)} duplicates)"
            )

        return result
