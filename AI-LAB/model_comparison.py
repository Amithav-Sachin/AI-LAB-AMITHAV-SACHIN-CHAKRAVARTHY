"""
Model Comparison: First-Order vs. Second-Order Autoregressive Language Model
AI Laboratory: Bayesian Networks and Autoregressive Language Models (Part XIII)

This module systematically compares the first-order and second-order models across:
1. Number of distinct parameters
2. Number of zero-probability contexts (sparsity)
3. Diversity of generated sentences
4. Qualitative coherence of generated sentences
"""

import random
from collections import Counter
from typing import Dict, Any, List

from first_order_model import (
    FirstOrderAutoregressiveLM,
    tokenize_sentence,
    DEFAULT_TRAINING_CORPUS
)
from second_order_model import (
    SecondOrderAutoregressiveLM,
    tokenize_sentence_second_order
)


def compute_first_order_stats(model: FirstOrderAutoregressiveLM) -> Dict[str, Any]:
    vocab_size = len(model.vocabulary)
    total_possible_transitions = vocab_size * vocab_size
    
    # Non-zero parameters
    nonzero_params = 0
    for w_prev, next_dist in model.probabilities.items():
        nonzero_params += len(next_dist)
        
    # Free parameters: (sum_{w} (num_next(w) - 1)) because sum of probabilities = 1
    free_params = sum(len(next_dist) - 1 for next_dist in model.probabilities.values() if len(next_dist) > 0)

    # Observed contexts vs total possible
    observed_contexts = len(model.probabilities)
    unobserved_contexts = vocab_size - observed_contexts
    zero_prob_transitions = total_possible_transitions - nonzero_params
    sparsity_percent = (zero_prob_transitions / total_possible_transitions) * 100.0

    return {
        "order": "1st-order",
        "vocab_size": vocab_size,
        "total_possible_contexts": vocab_size,
        "observed_contexts": observed_contexts,
        "unobserved_contexts": unobserved_contexts,
        "total_matrix_entries": total_possible_transitions,
        "nonzero_parameters": nonzero_params,
        "free_parameters": free_params,
        "zero_prob_transitions": zero_prob_transitions,
        "sparsity_percent": sparsity_percent,
    }


def compute_second_order_stats(model: SecondOrderAutoregressiveLM) -> Dict[str, Any]:
    vocab_size = len(model.vocabulary)
    total_possible_contexts = vocab_size * vocab_size
    total_possible_parameters = vocab_size * vocab_size * vocab_size

    # Non-zero parameters
    nonzero_params = 0
    for ctx, next_dist in model.probabilities.items():
        nonzero_params += len(next_dist)

    free_params = sum(len(next_dist) - 1 for next_dist in model.probabilities.values() if len(next_dist) > 0)
    observed_contexts = len(model.probabilities)
    unobserved_contexts = total_possible_contexts - observed_contexts
    zero_prob_parameters = total_possible_parameters - nonzero_params
    sparsity_percent = (zero_prob_parameters / total_possible_parameters) * 100.0

    return {
        "order": "2nd-order",
        "vocab_size": vocab_size,
        "total_possible_contexts": total_possible_contexts,
        "observed_contexts": observed_contexts,
        "unobserved_contexts": unobserved_contexts,
        "total_matrix_entries": total_possible_parameters,
        "nonzero_parameters": nonzero_params,
        "free_parameters": free_params,
        "zero_prob_transitions": zero_prob_parameters,
        "sparsity_percent": sparsity_percent,
    }


def evaluate_generation_diversity(model, num_samples: int = 100, seed: int = 42) -> Dict[str, Any]:
    random.seed(seed)
    sentences = [model.generate_sentence(mode="sampling") for _ in range(num_samples)]
    sentence_counts = Counter(sentences)
    unique_count = len(sentence_counts)
    
    # Calculate lengths
    lengths = [len(s.split()) for s in sentences]
    avg_length = sum(lengths) / len(lengths)
    
    # Check looping / max-token hits (sentences of length >= 50)
    looping_count = sum(1 for l in lengths if l >= 50)

    # Check whether generated sentences were in original training corpus
    clean_training = set(DEFAULT_TRAINING_CORPUS)
    valid_training_matches = 0
    novel_sentences = []
    ungrammatical_or_anomalous = []
    
    for s in sentences:
        # Strip <START> and <END>
        clean_s = s.replace("<START>", "").replace("<END>", "").strip()
        if clean_s in clean_training:
            valid_training_matches += 1
        else:
            novel_sentences.append(clean_s)
            # Check for known first-order anomalies
            if clean_s.startswith("the mat") or clean_s.startswith("the park") or clean_s.startswith("the rug") or "sat on the cat" in clean_s or "ran to the rug" in clean_s:
                ungrammatical_or_anomalous.append(clean_s)

    return {
        "num_samples": num_samples,
        "unique_sentences": unique_count,
        "unique_ratio": unique_count / num_samples,
        "avg_length": avg_length,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "looping_sentences": looping_count,
        "exact_training_matches": valid_training_matches,
        "novel_generated_count": len(novel_sentences),
        "anomalous_count": len(ungrammatical_or_anomalous),
        "sample_frequencies": sentence_counts.most_common(5)
    }


def run_comparative_study():
    print("=" * 80)
    print("AI LABORATORY: COMPARATIVE EVALUATION (FIRST-ORDER VS SECOND-ORDER)")
    print("=" * 80)

    # 1. Train First-Order Model
    fo_corpus = [tokenize_sentence(s) for s in DEFAULT_TRAINING_CORPUS]
    fo_model = FirstOrderAutoregressiveLM()
    fo_model.train(fo_corpus)
    fo_stats = compute_first_order_stats(fo_model)

    # 2. Train Second-Order Model
    so_corpus = [tokenize_sentence_second_order(s) for s in DEFAULT_TRAINING_CORPUS]
    so_model = SecondOrderAutoregressiveLM()
    so_model.train(so_corpus)
    so_stats = compute_second_order_stats(so_model)

    # Print Table 1: Model Parameter and Sparsity Comparison
    print("\n[Metric 1 & 2] Model Architecture, Parameters, and Sparsity Comparison:")
    header = f"{'Metric':<35} | {'First-Order LM':<18} | {'Second-Order LM':<18}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'Conditioning Context':<35} | {'X_{t-1}':<18} | {'(X_{t-2}, X_{t-1})':<18}")
    print(f"{'Graph Parent Count':<35} | {'1 parent':<18} | {'2 parents':<18}")
    print(f"{'Vocabulary Size (|V|)':<35} | {fo_stats['vocab_size']:<18} | {so_stats['vocab_size']:<18}")
    print(f"{'Total Theoretical Contexts':<35} | {fo_stats['total_possible_contexts']:<18} | {so_stats['total_possible_contexts']:<18}")
    print(f"{'Observed Contexts':<35} | {fo_stats['observed_contexts']:<18} | {so_stats['observed_contexts']:<18}")
    print(f"{'Zero-Probability Contexts':<35} | {fo_stats['unobserved_contexts']:<18} | {so_stats['unobserved_contexts']:<18}")
    print(f"{'Context Sparsity (%)':<35} | {fo_stats['unobserved_contexts']/fo_stats['total_possible_contexts']*100:<17.1f}% | {so_stats['unobserved_contexts']/so_stats['total_possible_contexts']*100:<17.1f}%")
    print(f"{'Total Parameter Space':<35} | {fo_stats['total_matrix_entries']:<18} | {so_stats['total_matrix_entries']:<18}")
    print(f"{'Non-Zero Learned Parameters':<35} | {fo_stats['nonzero_parameters']:<18} | {so_stats['nonzero_parameters']:<18}")
    print(f"{'Independent Free Parameters':<35} | {fo_stats['free_parameters']:<18} | {so_stats['free_parameters']:<18}")
    print(f"{'Total Sparsity (%)':<35} | {fo_stats['sparsity_percent']:<17.2f}% | {so_stats['sparsity_percent']:<17.2f}%")
    print("-" * len(header))

    # Evaluate Diversity
    fo_div = evaluate_generation_diversity(fo_model, num_samples=100)
    so_div = evaluate_generation_diversity(so_model, num_samples=100)

    print("\n[Metric 3 & 4] Generation Diversity and Coherence (100 Sampled Sentences):")
    print("-" * len(header))
    print(f"{'Metric':<35} | {'First-Order LM':<18} | {'Second-Order LM':<18}")
    print("-" * len(header))
    print(f"{'Unique Sentences (Diversity)':<35} | {fo_div['unique_sentences']}/100 ({fo_div['unique_ratio']*100:.1f}%) | {so_div['unique_sentences']}/100 ({so_div['unique_ratio']*100:.1f}%)")
    print(f"{'Average Sentence Length (tokens)':<35} | {fo_div['avg_length']:<18.2f} | {so_div['avg_length']:<18.2f}")
    print(f"{'Sentence Length Range [min, max]':<35} | [{fo_div['min_length']}, {fo_div['max_length']}]{'':<11} | [{so_div['min_length']}, {so_div['max_length']}]{'':<11}")
    print(f"{'Grammatical & Coherent Matches':<35} | {fo_div['exact_training_matches']}/100{'':<11} | {so_div['exact_training_matches']}/100{'':<11}")
    print(f"{'Anomalous / Fragmented Output':<35} | {fo_div['anomalous_count']}/100{'':<11} | {so_div['anomalous_count']}/100{'':<11}")
    print(f"{'Infinite Loops / Hit Max Limit':<35} | {fo_div['looping_sentences']}/100{'':<11} | {so_div['looping_sentences']}/100{'':<11}")
    print("-" * len(header))

    print("\nMost Frequent Sentences Generated by First-Order LM:")
    for sent, cnt in fo_div["sample_frequencies"]:
        print(f"  [{cnt:2d} times] {sent}")

    print("\nMost Frequent Sentences Generated by Second-Order LM:")
    for sent, cnt in so_div["sample_frequencies"]:
        print(f"  [{cnt:2d} times] {sent}")


if __name__ == "__main__":
    run_comparative_study()
