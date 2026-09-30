"""
Second-Order Autoregressive Language Model
AI Laboratory: Bayesian Networks and Autoregressive Language Models

This module implements a second-order Markov autoregressive language model
derived from the factorisation:
    P(X_1, ..., X_T) = P(X_1) * P(X_2 | X_1) * prod_{t=3}^T P(X_t | X_{t-2}, X_{t-1})
where transitions are learned from frequency counts of token triples:
    P(X_t | X_{t-2}, X_{t-1}) = C(X_{t-2}, X_{t-1}, X_t) / sum_w C(X_{t-2}, X_{t-1}, w)
"""

from collections import defaultdict, Counter
import random
from typing import Dict, List, Optional, Tuple


DEFAULT_TRAINING_CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug"
]


def tokenize_sentence_second_order(sentence: str, start_token: str = "<START>", end_token: str = "<END>") -> List[str]:
    """
    Tokenizes a sentence by lowercasing, splitting on whitespace,
    and prepending two start tokens (for second-order conditioning) and appending end token.
    """
    cleaned = sentence.strip().lower()
    words = cleaned.split()
    return [start_token, start_token] + words + [end_token]


class SecondOrderAutoregressiveLM:
    """
    Second-order Autoregressive Language Model.
    
    Probabilistic representation:
        P(X_t | X_1, ..., X_{t-1}) approx P(X_t | X_{t-2}, X_{t-1})
        
    Bayesian network structure:
        X_{t-2} -> X_t <- X_{t-1}
        
    Transitions:
        P(w_k | w_i, w_j) = C(w_i, w_j, w_k) / sum_w C(w_i, w_j, w)
    """

    def __init__(self, start_token: str = "<START>", end_token: str = "<END>"):
        self.start_token = start_token
        self.end_token = end_token
        # Transition counts: C(w_{t-2}, w_{t-1}, w_t) -> self.triple_counts[(w1, w2)][w3]
        self.triple_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        # Conditional probabilities: P(w_t | w_{t-2}, w_{t-1}) -> self.probabilities[(w1, w2)][w3]
        self.probabilities: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
        # Vocabulary of distinct observed tokens
        self.vocabulary: set = set()
        self.is_trained: bool = False

    def train(self, tokenized_sentences: List[List[str]]) -> None:
        """
        1. Count transitions between consecutive triples of tokens.
        2. Construct the conditional distribution P(X_t | X_{t-2}, X_{t-1}).
        """
        self.triple_counts.clear()
        self.probabilities.clear()
        self.vocabulary.clear()

        # Step 1: Count triples
        for tokens in tokenized_sentences:
            for w1, w2, w3 in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
                self.triple_counts[(w1, w2)][w3] += 1
                self.vocabulary.add(w1)
                self.vocabulary.add(w2)
                self.vocabulary.add(w3)

        # Step 2: Compute conditional probabilities
        for context, next_counts in self.triple_counts.items():
            total_count = sum(next_counts.values())
            for w_next, count in next_counts.items():
                self.probabilities[context][w_next] = count / total_count

        self.is_trained = True

    def get_cpt(self, context: Optional[Tuple[str, str]] = None) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Returns conditional probability distribution for a given bigram context (w_{t-2}, w_{t-1}),
        or the full CPT table if context is None.
        """
        if context is not None:
            return self.probabilities.get(context, {})
        return dict(self.probabilities)

    def display_probabilities_for_context(self, context: Tuple[str, str]) -> None:
        """
        Displays the conditional distribution P(X_t | X_{t-2}, X_{t-1} = context).
        """
        ctx_str = f"('{context[0]}', '{context[1]}')"
        if context not in self.probabilities:
            print(f"Context {ctx_str} has no outgoing transitions observed in training data.")
            return

        print(f"Conditional Probability Distribution P(X_t | (X_{{t-2}}, X_{{t-1}}) = {ctx_str}):")
        dist = self.probabilities[context]
        total_transitions = sum(self.triple_counts[context].values())
        print(f"  Total occurrences of context: {total_transitions}")
        for next_word, prob in sorted(dist.items(), key=lambda x: (-x[1], x[0])):
            count = self.triple_counts[context][next_word]
            print(f"    P({next_word:6s} | {ctx_str:20s}) = {count}/{total_transitions} = {prob:.4f} ({prob*100:5.1f}%)")

    def predict_most_probable(self, context: Tuple[str, str]) -> Tuple[Optional[str], float, List[str]]:
        """
        Predicts the most probable next token:
            argmax_w P(w | context)
        Returns:
            (selected_best_token, max_probability, all_tied_tokens)
        """
        if context not in self.probabilities or not self.probabilities[context]:
            return None, 0.0, []

        dist = self.probabilities[context]
        max_p = max(dist.values())
        candidates = [w for w, p in dist.items() if abs(p - max_p) < 1e-9]
        candidates.sort()
        return candidates[0], max_p, candidates

    def sample_next_token(self, context: Tuple[str, str]) -> Optional[str]:
        """
        Samples next token from conditional distribution P(X_t | context).
        """
        if context not in self.probabilities or not self.probabilities[context]:
            return None

        candidates = list(self.probabilities[context].keys())
        weights = list(self.probabilities[context].values())
        return random.choices(candidates, weights=weights, k=1)[0]

    def generate_sentence(self, mode: str = "sampling", max_tokens: int = 50) -> str:
        """
        Generates a sentence token by token starting from (<START>, <START>) until <END>
        or max_tokens limit is reached.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating sentences.")

        generated = [self.start_token, self.start_token]

        for _ in range(max_tokens):
            context = (generated[-2], generated[-1])
            if context[-1] == self.end_token:
                break

            if mode == "greedy":
                next_token, _, _ = self.predict_most_probable(context)
            elif mode == "sampling":
                next_token = self.sample_next_token(context)
            else:
                raise ValueError(f"Unknown generation mode: {mode}. Choose 'sampling' or 'greedy'.")

            if next_token is None:
                # Dead end (unseen context)
                break

            generated.append(next_token)
            if next_token == self.end_token:
                break

        # Remove the extra start token for clean display: <START> w1 w2 ... <END>
        return " ".join(generated[1:])

    def test_normalization(self, tolerance: float = 1e-6) -> Dict[Tuple[str, str], Tuple[float, bool]]:
        """
        Tests whether sum_v P(v | w1, w2) == 1.0 for every conditioning context pair (w1, w2).
        """
        results = {}
        for context, next_dist in self.probabilities.items():
            total = sum(next_dist.values())
            is_valid = abs(total - 1.0) < tolerance
            results[context] = (total, is_valid)
        return results


def run_second_order_demonstration():
    print("=" * 70)
    print("AI LABORATORY: SECOND-ORDER AUTOREGRESSIVE LANGUAGE MODEL")
    print("=" * 70)

    # 1. Prepare data
    tokenized_corpus = [tokenize_sentence_second_order(s) for s in DEFAULT_TRAINING_CORPUS]
    print(f"\n[1] Training Corpus ({len(tokenized_corpus)} sentences, padded with double <START>):")
    for s in tokenized_corpus:
        print("   ", " ".join(s))

    # 2. Train model
    model = SecondOrderAutoregressiveLM()
    model.train(tokenized_corpus)
    print(f"\n[2] Model Trained. Vocabulary size: {len(model.vocabulary)} unique tokens.")
    print(f"    Observed bigram contexts: {len(model.probabilities)}")

    # 3. Test probability normalization
    print("\n[3] Part VII/XII - Probability Normalization Test: sum_v P(v | w1, w2) == 1.0")
    norm_results = model.test_normalization()
    all_passed = True
    for ctx, (total, passed) in sorted(norm_results.items()):
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        ctx_str = f"('{ctx[0]}', '{ctx[1]}')"
        print(f"    sum_v P(v | {ctx_str:22s}) = {total:8.6f} -> [{status}]")
    print(f"    Overall Normalization Status: {'ALL PASSED (100% Valid CPT)' if all_passed else 'FAILURE DETECTED'}")

    # 4. Display CPTs for selected key contexts
    key_contexts = [
        ("<START>", "<START>"),
        ("<START>", "the"),
        ("the", "cat"),
        ("the", "dog"),
        ("cat", "sat"),
        ("cat", "ran"),
        ("sat", "on"),
        ("on", "the"),
        ("ran", "to"),
        ("to", "the"),
        ("the", "mat"),
        ("the", "park"),
        ("the", "rug")
    ]
    print("\n[4] Part XII/XIII - Conditional Distributions for Selected Contexts:")
    for ctx in key_contexts:
        print("-" * 55)
        model.display_probabilities_for_context(ctx)
        best, p_val, tied = model.predict_most_probable(ctx)
        if best:
            if len(tied) > 1:
                print(f"    argmax_w P(w | {ctx}) = '{best}' (p={p_val:.4f}, tied with {tied})")
            else:
                print(f"    argmax_w P(w | {ctx}) = '{best}' (p={p_val:.4f})")

    # 5. Text Generation
    print("\n" + "=" * 70)
    print("[5] Part XII/XIII - Text Generation: 20 Sentences using Sampling Mode:")
    print("=" * 70)
    random.seed(42)
    for i in range(1, 21):
        s = model.generate_sentence(mode="sampling")
        print(f"    {i:2d}. {s}")

    print("\n" + "=" * 70)
    print("[6] Part XIII - Greedy vs Sampling Comparison (Second-Order):")
    print("=" * 70)
    print("  Mode A: Greedy Generation (5 runs):")
    for i in range(1, 6):
        print(f"    Run {i}: {model.generate_sentence(mode='greedy')}")

    print("\n  Mode B: Sampling Generation (5 runs):")
    for i in range(1, 6):
        print(f"    Run {i}: {model.generate_sentence(mode='sampling')}")


if __name__ == "__main__":
    run_second_order_demonstration()
