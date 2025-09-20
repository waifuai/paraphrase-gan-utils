# src/evaluation.py
import re
import string
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import math

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sentence_transformers import SentenceTransformer
    import numpy as np

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from src.logging_config import get_logger

logger = get_logger("evaluation")

class EvaluationError(Exception):
    """Exception for evaluation-related errors."""
    pass

class ParaphraseEvaluator:
    """Comprehensive evaluator for paraphrase quality."""

    def __init__(self):
        self._setup_nltk()
        self._setup_sentence_transformer()

    def _setup_nltk(self):
        """Setup NLTK components."""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, some metrics will be limited")
            return

        try:
            # Download required NLTK data
            import nltk.data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)

            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)

            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.error("Failed to setup NLTK", error=str(e))
            self.lemmatizer = None
            self.stop_words = set()

    def _setup_sentence_transformer(self):
        """Setup sentence transformer for semantic similarity."""
        if not NLTK_AVAILABLE:
            self.sentence_transformer = None
            return

        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")
        except Exception as e:
            logger.error("Failed to load sentence transformer", error=str(e))
            self.sentence_transformer = None

    def evaluate_paraphrase(
        self,
        original: str,
        paraphrase: str,
        include_semantic: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of a single paraphrase."""
        if not original or not paraphrase:
            return self._empty_evaluation()

        try:
            evaluation = {}

            # Lexical similarity metrics
            evaluation.update(self._lexical_similarity(original, paraphrase))

            # Structural metrics
            evaluation.update(self._structural_similarity(original, paraphrase))

            # Semantic similarity (if available)
            if include_semantic and self.sentence_transformer:
                evaluation.update(self._semantic_similarity(original, paraphrase))

            # Overall quality score
            evaluation["overall_score"] = self._calculate_overall_score(evaluation)

            return evaluation

        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            return self._error_evaluation(str(e))

    def evaluate_paraphrase_batch(
        self,
        originals: List[str],
        paraphrases: List[str],
        include_semantic: bool = True
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of paraphrases."""
        if len(originals) != len(paraphrases):
            raise EvaluationError("Originals and paraphrases lists must have same length")

        results = []
        for original, paraphrase in zip(originals, paraphrases):
            try:
                result = self.evaluate_paraphrase(original, paraphrase, include_semantic)
                results.append(result)
            except Exception as e:
                logger.error("Batch evaluation item failed", error=str(e))
                results.append(self._error_evaluation(str(e)))

        return results

    def _lexical_similarity(self, original: str, paraphrase: str) -> Dict[str, Any]:
        """Calculate lexical similarity metrics."""
        if not NLTK_AVAILABLE:
            return {
                "lexical_similarity": 0.0,
                "word_overlap": 0.0,
                "jaccard_similarity": 0.0
            }

        try:
            # Normalize text
            orig_tokens = self._normalize_text(original)
            para_tokens = self._normalize_text(paraphrase)

            if not orig_tokens or not para_tokens:
                return {
                    "lexical_similarity": 0.0,
                    "word_overlap": 0.0,
                    "jaccard_similarity": 0.0
                }

            # Word overlap
            orig_set = set(orig_tokens)
            para_set = set(para_tokens)
            intersection = orig_set.intersection(para_set)
            union = orig_set.union(para_set)

            word_overlap = len(intersection) / len(orig_set) if orig_set else 0.0
            jaccard_similarity = len(intersection) / len(union) if union else 0.0

            # Cosine similarity using term frequency
            orig_freq = Counter(orig_tokens)
            para_freq = Counter(para_tokens)

            all_words = set(orig_freq.keys()).union(set(para_freq.keys()))
            orig_vector = [orig_freq.get(word, 0) for word in all_words]
            para_vector = [para_freq.get(word, 0) for word in all_words]

            lexical_similarity = self._cosine_similarity(orig_vector, para_vector)

            return {
                "lexical_similarity": lexical_similarity,
                "word_overlap": word_overlap,
                "jaccard_similarity": jaccard_similarity
            }

        except Exception as e:
            logger.error("Lexical similarity calculation failed", error=str(e))
            return {
                "lexical_similarity": 0.0,
                "word_overlap": 0.0,
                "jaccard_similarity": 0.0
            }

    def _structural_similarity(self, original: str, paraphrase: str) -> Dict[str, Any]:
        """Calculate structural similarity metrics."""
        if not NLTK_AVAILABLE:
            return {
                "length_ratio": 0.0,
                "sentence_count_diff": 0,
                "avg_word_length_diff": 0.0
            }

        try:
            # Length metrics
            orig_length = len(original.split())
            para_length = len(paraphrase.split())
            length_ratio = min(orig_length, para_length) / max(orig_length, para_length) if max(orig_length, para_length) > 0 else 0.0

            # Sentence count
            try:
                orig_sentences = sent_tokenize(original)
                para_sentences = sent_tokenize(paraphrase)
                sentence_count_diff = abs(len(orig_sentences) - len(para_sentences))
            except:
                sentence_count_diff = 0

            # Average word length
            orig_words = original.split()
            para_words = paraphrase.split()

            orig_avg_len = sum(len(word) for word in orig_words) / len(orig_words) if orig_words else 0
            para_avg_len = sum(len(word) for word in para_words) / len(para_words) if para_words else 0
            avg_word_length_diff = abs(orig_avg_len - para_avg_len)

            return {
                "length_ratio": length_ratio,
                "sentence_count_diff": sentence_count_diff,
                "avg_word_length_diff": avg_word_length_diff
            }

        except Exception as e:
            logger.error("Structural similarity calculation failed", error=str(e))
            return {
                "length_ratio": 0.0,
                "sentence_count_diff": 0,
                "avg_word_length_diff": 0.0
            }

    def _semantic_similarity(self, original: str, paraphrase: str) -> Dict[str, Any]:
        """Calculate semantic similarity using sentence transformers."""
        if not self.sentence_transformer:
            return {"semantic_similarity": 0.0}

        try:
            # Generate embeddings
            orig_embedding = self.sentence_transformer.encode([original])[0]
            para_embedding = self.sentence_transformer.encode([paraphrase])[0]

            # Calculate cosine similarity
            similarity = self._cosine_similarity(orig_embedding, para_embedding)

            return {"semantic_similarity": float(similarity)}

        except Exception as e:
            logger.error("Semantic similarity calculation failed", error=str(e))
            return {"semantic_similarity": 0.0}

    def _normalize_text(self, text: str) -> List[str]:
        """Normalize text for comparison."""
        if not NLTK_AVAILABLE:
            return text.lower().split()

        try:
            # Remove punctuation and lowercase
            text = text.translate(str.maketrans('', '', string.punctuation)).lower()

            # Tokenize
            tokens = word_tokenize(text)

            # Remove stopwords and lemmatize
            if self.lemmatizer:
                tokens = [
                    self.lemmatizer.lemmatize(token)
                    for token in tokens
                    if token not in self.stop_words and token.isalnum()
                ]
            else:
                tokens = [
                    token for token in tokens
                    if token not in self.stop_words and token.isalnum()
                ]

            return tokens

        except Exception as e:
            logger.error("Text normalization failed", error=str(e))
            return text.lower().split()

    def _cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np

            vec1 = np.array(vector1)
            vec2 = np.array(vector2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            logger.error("Cosine similarity calculation failed", error=str(e))
            return 0.0

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from individual metrics."""
        try:
            scores = []

            # Lexical similarity (weighted)
            if "lexical_similarity" in metrics:
                # We want some similarity but not too much (avoid copying)
                lexical_score = 1.0 - abs(metrics["lexical_similarity"] - 0.3) / 0.7
                scores.append(("lexical", lexical_score, 0.3))

            # Word overlap (weighted)
            if "word_overlap" in metrics:
                overlap_score = 1.0 - metrics["word_overlap"]  # Lower overlap is better for paraphrase
                scores.append(("overlap", overlap_score, 0.2))

            # Length ratio (weighted)
            if "length_ratio" in metrics:
                length_score = metrics["length_ratio"]
                scores.append(("length", length_score, 0.2))

            # Semantic similarity (weighted)
            if "semantic_similarity" in metrics:
                semantic_score = metrics["semantic_similarity"]
                scores.append(("semantic", semantic_score, 0.3))

            if not scores:
                return 0.0

            # Weighted average
            total_score = sum(score * weight for _, score, weight in scores)
            total_weight = sum(weight for _, _, weight in scores)

            return total_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            logger.error("Overall score calculation failed", error=str(e))
            return 0.0

    def _empty_evaluation(self) -> Dict[str, Any]:
        """Return empty evaluation result."""
        return {
            "lexical_similarity": 0.0,
            "word_overlap": 0.0,
            "jaccard_similarity": 0.0,
            "length_ratio": 0.0,
            "sentence_count_diff": 0,
            "avg_word_length_diff": 0.0,
            "semantic_similarity": 0.0,
            "overall_score": 0.0,
            "error": "Empty input"
        }

    def _error_evaluation(self, error_msg: str) -> Dict[str, Any]:
        """Return error evaluation result."""
        return {
            "lexical_similarity": 0.0,
            "word_overlap": 0.0,
            "jaccard_similarity": 0.0,
            "length_ratio": 0.0,
            "sentence_count_diff": 0,
            "avg_word_length_diff": 0.0,
            "semantic_similarity": 0.0,
            "overall_score": 0.0,
            "error": error_msg
        }

    def get_evaluation_summary(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics for multiple evaluations."""
        if not evaluations:
            return {}

        try:
            metrics = [
                "lexical_similarity", "word_overlap", "jaccard_similarity",
                "length_ratio", "semantic_similarity", "overall_score"
            ]

            summary = {}
            for metric in metrics:
                values = [eval.get(metric, 0.0) for eval in evaluations if metric in eval]
                if values:
                    summary[f"{metric}_mean"] = sum(values) / len(values)
                    summary[f"{metric}_min"] = min(values)
                    summary[f"{metric}_max"] = max(values)
                    summary[f"{metric}_std"] = math.sqrt(
                        sum((x - summary[f"{metric}_mean"]) ** 2 for x in values) / len(values)
                    )

            summary["total_evaluations"] = len(evaluations)
            summary["successful_evaluations"] = len([
                eval for eval in evaluations
                if "error" not in eval or not eval.get("error")
            ])

            return summary

        except Exception as e:
            logger.error("Evaluation summary calculation failed", error=str(e))
            return {"error": str(e)}

# Global evaluator instance
_evaluator_instance = None

def get_evaluator() -> ParaphraseEvaluator:
    """Get global evaluator instance."""
    global _evaluator_instance

    if _evaluator_instance is None:
        _evaluator_instance = ParaphraseEvaluator()
        logger.info("Initialized global paraphrase evaluator")

    return _evaluator_instance

def evaluate_paraphrase(
    original: str,
    paraphrase: str,
    include_semantic: bool = True
) -> Dict[str, Any]:
    """Convenience function for single paraphrase evaluation."""
    evaluator = get_evaluator()
    return evaluator.evaluate_paraphrase(original, paraphrase, include_semantic)

def evaluate_paraphrase_batch(
    originals: List[str],
    paraphrases: List[str],
    include_semantic: bool = True
) -> List[Dict[str, Any]]:
    """Convenience function for batch paraphrase evaluation."""
    evaluator = get_evaluator()
    return evaluator.evaluate_paraphrase_batch(originals, paraphrases, include_semantic)