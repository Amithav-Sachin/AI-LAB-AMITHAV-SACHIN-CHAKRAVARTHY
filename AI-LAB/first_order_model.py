"""
First-Order Autoregressive Language Model
AI Laboratory: Bayesian Networks and Autoregressive Language Models

This module implements a first-order Markov autoregressive language model
derived from the chain rule of probability:
    P(X_1, ..., X_T) = P(X_1) * prod_{t=2}^T P(X_t | X_{t-1})
where transitions are learned from frequency counts over tokenized text.
"""

from collections import defaultdict, Counter
import random
from typing import Dict, List, Optional, Tuple


# Default Laboratory Starting Dataset (Part III)
DEFAULT_TRAINING_CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug"
]


def tokenize_sentence(sentence: str, start_token: str = "<START>", end_token: str = "<END>") -> List[str]:
    """
    Tokenizes a sentence by lowercasing, splitting on whitespace,
    and prepending start and appending end tokens.
    """
    cleaned = sentence.strip().lower()
    words = cleaned.split()
    return [start_token] + words + [end_token]


class FirstOrderAutoregressiveLM:
    """
    First-order Autoregressive Language Model.
    
    Probabilistic representation:
        P(X_t | X_1, ..., X_{t-1}) approx P(X_t | X_{t-1})
        
    Transitions:
        P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)
    """

    def __init__(self, start_token: str = "<START>", end_token: str = "<END>"):
        self.start_token = start_token
        self.end_token = end_token
        # Transition counts: C(w_i, w_j) -> self.transition_counts[w_i][w_j]
        self.transition_counts: Dict[str, Counter] = defaultdict(Counter)
        # Conditional probabilities: P(w_j | w_i) -> self.probabilities[w_i][w_j]
        self.probabilities: Dict[str, Dict[str, float]] = defaultdict(dict)
        # Vocabulary of distinct observed tokens
        self.vocabulary: set = set()
        self.is_trained: bool = False

    def train(self, tokenized_sentences: List[List[str]]) -> None:
        """
        1. Count transitions between consecutive tokens.
        2. Construct the conditional distribution P(X_t | X_{t-1}).
        """
        self.transition_counts.clear()
        self.probabilities.clear()
        self.vocabulary.clear()

        # Step 1: Count transitions
        for tokens in tokenized_sentences:
            for w_prev, w_curr in zip(tokens[:-1], tokens[1:]):
                self.transition_counts[w_prev][w_curr] += 1
                self.vocabulary.add(w_prev)
                self.vocabulary.add(w_curr)

        # Step 2: Calculate conditional probabilities P(X_t | X_{t-1})
        for w_prev, next_counts in self.transition_counts.items():
            total_count = sum(next_counts.values())
            for w_curr, count in next_counts.items():
                self.probabilities[w_prev][w_curr] = count / total_count

        self.is_trained = True

    def get_cpt(self, token: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Returns conditional probability distribution for a given token,
        or the full CPT table if token is None.
        """
        if token is not None:
            return self.probabilities.get(token, {})
        return dict(self.probabilities)

    def display_probabilities_for_token(self, token: str) -> None:
        """
        Displays the conditional distribution P(X_t | X_{t-1} = token).
        """
        if token not in self.probabilities:
            print(f"Token '{token}' has no outgoing transitions observed in training data.")
            return

        print(f"Conditional Probability Distribution P(X_t | X_{{t-1}} = '{token}'):")
        dist = self.probabilities[token]
        total_transitions = sum(self.transition_counts[token].values())
        print(f"  Total occurrences as context: {total_transitions}")
        for next_word, prob in sorted(dist.items(), key=lambda x: (-x[1], x[0])):
            count = self.transition_counts[token][next_word]
            print(f"    P({next_word:6s} | {token:6s}) = {count}/{total_transitions} = {prob:.4f} ({prob*100:5.1f}%)")

    def predict_most_probable(self, token: str) -> Tuple[Optional[str], float, List[str]]:
        """
        Predicts the most probable next token:
            argmax_w P(w | token)
        Returns:
            (selected_best_token, max_probability, all_tied_tokens)
        """
        if token not in self.probabilities or not self.probabilities[token]:
            return None, 0.0, []

        dist = self.probabilities[token]
        max_p = max(dist.values())
        candidates = [w for w, p in dist.items() if abs(p - max_p) < 1e-9]
        # In case of tie, return the first alphabetically for deterministic consistency,
        # but also report all tied candidates.
        candidates.sort()
        return candidates[0], max_p, candidates

    def sample_next_token(self, token: str) -> Optional[str]:
        """
        Samples next token from conditional distribution P(X_t | X_{t-1} = token).
        """
        if token not in self.probabilities or not self.probabilities[token]:
            return None

        candidates = list(self.probabilities[token].keys())
        weights = list(self.probabilities[token].values())
        return random.choices(candidates, weights=weights, k=1)[0]

    def generate_sentence(self, mode: str = "sampling", max_tokens: int = 50) -> str:
        """
        Generates a sentence token by token starting from <START> until <END>
        or max_tokens limit is reached.
        
        mode:
          - 'sampling': sample from conditional distribution P(w | w_prev)
          - 'greedy': always choose argmax_w P(w | w_prev)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating sentences.")

        current_token = self.start_token
        generated = [current_token]

        for _ in range(max_tokens):
            if current_token == self.end_token:
                break

            if mode == "greedy":
                next_token, _, _ = self.predict_most_probable(current_token)
            elif mode == "sampling":
                next_token = self.sample_next_token(current_token)
            else:
                raise ValueError(f"Unknown generation mode: {mode}. Choose 'sampling' or 'greedy'.")

            if next_token is None:
                # Dead end (no transitions observed)
                break

            generated.append(next_token)
            current_token = next_token

        return " ".join(generated)

    def test_normalization(self, tolerance: float = 1e-6) -> Dict[str, Tuple[float, bool]]:
        """
        Part VII: Test the Probability Model
        Tests whether sum_v P(v | w) == 1.0 for every conditioning word w.
        """
        results = {}
        for word, next_dist in self.probabilities.items():
            total = sum(next_dist.values())
            is_valid = abs(total - 1.0) < tolerance
            results[word] = (total, is_valid)
        return results


def run_first_order_demonstration():
    print("=" * 70)
    print("AI LABORATORY: FIRST-ORDER AUTOREGRESSIVE LANGUAGE MODEL")
    print("=" * 70)

    # 1. Prepare data
    tokenized_corpus = [tokenize_sentence(s) for s in DEFAULT_TRAINING_CORPUS]
    print(f"\n[1] Training Corpus ({len(tokenized_corpus)} sentences):")
    for s in tokenized_corpus:
        print("   ", " ".join(s))

    # 2. Train model
    model = FirstOrderAutoregressiveLM()
    model.train(tokenized_corpus)
    print(f"\n[2] Model Trained. Vocabulary size: {len(model.vocabulary)} unique tokens.")
    print("    Vocabulary:", sorted(list(model.vocabulary)))

    # 3. Test probability normalization (Part VII)
    print("\n[3] Part VII - Probability Normalization Test: sum_v P(v | w) == 1.0")
    norm_results = model.test_normalization()
    all_passed = True
    for word, (total, passed) in sorted(norm_results.items()):
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"    sum_v P(v | '{word:7s}') = {total:8.6f} -> [{status}]")
    print(f"    Overall Normalization Status: {'ALL PASSED (100% Valid CPT)' if all_passed else 'FAILURE DETECTED'}")

    # 4. Display CPTs for key tokens (Part IV & Part VIII)
    key_words = ["the", "cat", "dog", "sat", "ran", "<START>"]
    print("\n[4] Part IV & VIII - Conditional Distributions for Selected Contexts:")
    for kw in key_words:
        print("-" * 50)
        model.display_probabilities_for_token(kw)
        best, p_val, tied = model.predict_most_probable(kw)
        if len(tied) > 1:
            print(f"    argmax_w P(w | '{kw}') = '{best}' (p={p_val:.4f}, tied with {tied})")
        else:
            print(f"    argmax_w P(w | '{kw}') = '{best}' (p={p_val:.4f})")

    # 5. Text Generation (Part IX & Part X)
    print("\n" + "=" * 70)
    print("[5] Part IX - Text Generation: 20 Sentences using Sampling Mode:")
    print("=" * 70)
    random.seed(42)  # For reproducible demonstration
    sampled_sentences = []
    for i in range(1, 21):
        s = model.generate_sentence(mode="sampling")
        sampled_sentences.append(s)
        print(f"    {i:2d}. {s}")

    print("\n" + "=" * 70)
    print("[6] Part X - Deterministic (Greedy) vs Probabilistic (Sampling):")
    print("=" * 70)
    print("  Mode A: Greedy Generation (5 runs):")
    for i in range(1, 6):
        print(f"    Run {i}: {model.generate_sentence(mode='greedy')}")

    print("\n  Mode B: Sampling Generation (5 runs):")
    for i in range(1, 6):
        print(f"    Run {i}: {model.generate_sentence(mode='sampling')}")


if __name__ == "__main__":
    run_first_order_demonstration()
