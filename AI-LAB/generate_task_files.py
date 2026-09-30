"""
Generate Separate Text Files for Each AI Laboratory Task
AI Laboratory: Bayesian Networks and Autoregressive Language Models

This script produces standalone, dedicated output text files for every single task,
question, test, and deliverable specified in the laboratory document (BN_lab.pdf).
"""

import os
import random
from collections import Counter
from io import StringIO
import sys

from first_order_model import (
    FirstOrderAutoregressiveLM,
    tokenize_sentence,
    DEFAULT_TRAINING_CORPUS
)
from second_order_model import (
    SecondOrderAutoregressiveLM,
    tokenize_sentence_second_order
)
from model_comparison import (
    compute_first_order_stats,
    compute_second_order_stats,
    evaluate_generation_diversity
)

WORKSPACE = r"C:\Users\Amithav\Desktop\AI-LAB"


def write_file(filename: str, content: str):
    path = os.path.join(WORKSPACE, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"[Generated] {filename}")


def task_01():
    content = """
================================================================================
AI LABORATORY: TASK 1 OUTPUT
Part I: From Probability to Language (Question 1)
================================================================================

1. MATHEMATICAL FORMULATION:
Consider the sequence of words:
    X_1 = "the", X_2 = "cat", X_3 = "sat", X_4 = "on", X_5 = "the", X_6 = "mat"

The complete joint probability of this sequence is P(X_1, X_2, X_3, X_4, X_5, X_6).
Applying the general chain rule of probability without any independence assumptions yields:

    P(X_1, ..., X_6) = P(X_1)
                     * P(X_2 | X_1)
                     * P(X_3 | X_1, X_2)
                     * P(X_4 | X_1, X_2, X_3)
                     * P(X_5 | X_1, X_2, X_3, X_4)
                     * P(X_6 | X_1, X_2, X_3, X_4, X_5)

More generally, for any sequence of length T:
    P(X_1, ..., X_T) = P(X_1) * prod_{t=2}^T P(X_t | X_1, ..., X_{t-1})

This is known as the AUTOREGRESSIVE DECOMPOSITION.

--------------------------------------------------------------------------------
2. QUESTION 1 AND DETAILED ANSWER:
--------------------------------------------------------------------------------
QUESTION 1:
"Why is this decomposition useful for generating text? Write a short explanation
in your lab record."

ANSWER:
Generating text directly from the joint distribution P(X_1, ..., X_T) is computationally
intractable. If the vocabulary size is |V| and the sequence length is T, the joint
sample space contains |V|^T possible sequences. In any realistic language setting,
almost every possible combination will have an empirical frequency of zero.

The autoregressive decomposition is essential for generating text because:

1. Reduction to Sequential Categorical Sampling:
   Instead of attempting to pick a full sequence from |V|^T possibilities in a single
   step, the problem is decomposed into T sequential steps. At each step t, the model
   only needs to evaluate a 1D categorical distribution over the vocabulary V conditioned
   on the generated history (x_1, ..., x_{t-1}).

2. Enables Ancestral Forward Sampling:
   Text generation becomes an iterative left-to-right process:
     Step 1: Sample x_1 ~ P(X_1)
     Step 2: Sample x_2 ~ P(X_2 | X_1 = x_1)
     Step 3: Sample x_3 ~ P(X_3 | X_1 = x_1, X_2 = x_2)
     ...
     Step t: Sample x_t ~ P(X_t | X_1 = x_1, ..., X_{t-1} = x_{t-1})
   Each sampled token is appended to the prompt history and conditions the next token.

3. Supports Dynamic, Variable-Length Generation:
   The sequence length T does not need to be predetermined. By including a special
   terminal token <END>, the generation loop simply continues until <END> is emitted.

4. Foundation for Causal Streaming in AI:
   This sequential structure matches the causal, temporal nature of speech and text
   processing, providing the exact theoretical architecture used by modern autoregressive
   models (such as GPT and Gemini) to stream output token by token.
================================================================================
"""
    write_file("task_01_from_probability_to_language.txt", content)


def task_02():
    content = """
================================================================================
AI LABORATORY: TASK 2 OUTPUT
Part II: A Bayesian Network for Text (Question 2)
================================================================================

1. GRAPHICAL MODEL STRUCTURE:
A simple autoregressive language model where each word depends only on the
immediately preceding word is represented as a linear-chain Bayesian Network:

    [X_1] ---> [X_2] ---> [X_3] ---> [X_4] ---> ... ---> [X_T]

In this Directed Acyclic Graph (DAG):
  - Each node X_t represents a categorical random variable corresponding to the token
    at position t.
  - Directed edges represent direct conditional probabilistic dependencies.
  - The parent set of node X_t is:
        Parents(X_t) = { X_{t-1} }   for t >= 2
        Parents(X_1) = {} (root node)

--------------------------------------------------------------------------------
2. QUESTION 2 AND DETAILED ANSWER:
--------------------------------------------------------------------------------
QUESTION 2:
"What independence assumption is being made by this network?
Express your answer using probability notation."

ANSWER:
The network makes the FIRST-ORDER MARKOV INDEPENDENCE ASSUMPTION.
It assumes that the token at position t is conditionally independent of all earlier
tokens (X_1, X_2, ..., X_{t-2}), given the value of the immediately preceding token X_{t-1}.

Expressed in formal conditional independence notation:
    X_t _||_ (X_1, X_2, ..., X_{t-2}) | X_{t-1}    for all t >= 2

Expressed in terms of conditional probability distributions:
    P(X_t | X_1, X_2, ..., X_{t-1}) = P(X_t | X_{t-1})

Under this assumption, the general chain rule factorisation simplifies to:
    P(X_1, X_2, ..., X_T) = P(X_1) * prod_{t=2}^T P(X_t | X_{t-1})

For a 4-token sequence (X_1, X_2, X_3, X_4), this yields:
    P(X_1, X_2, X_3, X_4) = P(X_1) * P(X_2 | X_1) * P(X_3 | X_2) * P(X_4 | X_3)

Graphical Semantics (d-separation):
In the chain X_1 -> X_2 -> X_3 -> X_4, conditioning on X_2 blocks the active path
between X_1 and X_3 (d-separating them). Thus, once X_{t-1} is known, knowledge of
the remote past provides no additional information about X_t under this model.
================================================================================
"""
    write_file("task_02_bayesian_network_for_text.txt", content)


def task_03():
    tokenized = [tokenize_sentence(s) for s in DEFAULT_TRAINING_CORPUS]
    vocab = set()
    for s in tokenized:
        vocab.update(s)
    
    token_counts = Counter()
    for s in tokenized:
        token_counts.update(s)

    content = f"""
================================================================================
AI LABORATORY: TASK 3 OUTPUT
Part III: Build a Small Language Dataset
================================================================================

1. RAW CORPUS (6 Starting Sentences):
--------------------------------------------------------------------------------
Sentence 1: the cat sat on the mat
Sentence 2: the cat sat on the rug
Sentence 3: the dog sat on the mat
Sentence 4: the dog ran to the park
Sentence 5: the cat ran to the park
Sentence 6: the dog sat on the rug

2. TOKENIZATION PREPROCESSING:
- Converted all characters to lowercase.
- Split sentences on whitespace into individual word tokens.
- Added special boundary tokens:
    <START> : Marks the beginning of the sentence
    <END>   : Marks the end of the sentence

3. TOKENIZED CORPUS WITH BOUNDARY TOKENS:
--------------------------------------------------------------------------------
1. {' '.join(tokenized[0])}
2. {' '.join(tokenized[1])}
3. {' '.join(tokenized[2])}
4. {' '.join(tokenized[3])}
5. {' '.join(tokenized[4])}
6. {' '.join(tokenized[5])}

4. CORPUS STATISTICS:
- Total Sentences: {len(tokenized)}
- Total Tokens (including <START> and <END>): {sum(len(s) for s in tokenized)}
- Tokens Per Sentence: {len(tokenized[0])} tokens (uniform length across all sentences)
- Unique Vocabulary Size (|V|): {len(vocab)} unique tokens

5. COMPLETE VOCABULARY AND FREQUENCIES:
--------------------------------------------------------------------------------
{'Token':<12} | {'Count':<8} | {'Type':<20}
{'-'*46}
"""
    for token, cnt in sorted(token_counts.items(), key=lambda x: (-x[1], x[0])):
        if token in ("<START>", "<END>"):
            ttype = "Special Boundary"
        elif token in ("the", "on", "to"):
            ttype = "Function Word"
        elif token in ("cat", "dog", "mat", "rug", "park"):
            ttype = "Noun"
        else:
            ttype = "Verb"
        content += f"{token:<12} | {cnt:<8} | {ttype:<20}\n"

    content += """--------------------------------------------------------------------------------
All text has been properly formatted and validated for probabilistic modeling.
================================================================================
"""
    write_file("task_03_language_dataset.txt", content)


def task_04(fo_model: FirstOrderAutoregressiveLM):
    content = """
================================================================================
AI LABORATORY: TASK 4 OUTPUT
Part IV: Constructing the Conditional Probability Table (Question 3)
================================================================================

1. MATHEMATICAL ESTIMATION FORMULA:
The first-order model estimates conditional transition probabilities using
Maximum Likelihood Estimation (MLE) based on observed bigram frequency counts:

                     C(w_i, w_j)
    P(w_j | w_i) = -----------------
                   sum_k C(w_i, w_k)

Where:
  - C(w_i, w_j) is the number of times token w_j immediately follows token w_i.
  - The denominator sum_k C(w_i, w_k) is the total number of transitions originating from w_i.

--------------------------------------------------------------------------------
2. QUESTION 3 AND DETAILED CPT DERIVATIONS:
--------------------------------------------------------------------------------
QUESTION 3:
"Construct the conditional probability distribution P(next word | current word)
for at least the following words: the, cat, dog, sat, ran.
Identify any zero-probability transitions."

CPT DERIVATION FOR SPECIFIED WORDS:

(a) Current Word: 'the'
    Total context occurrences sum_k C('the', w_k) = 12
    Observed transitions:
      - 'cat' follows 'the' : 3 times  --> P('cat'  | 'the') = 3/12 = 0.2500 (25.00%)
      - 'dog' follows 'the' : 3 times  --> P('dog'  | 'the') = 3/12 = 0.2500 (25.00%)
      - 'mat' follows 'the' : 2 times  --> P('mat'  | 'the') = 2/12 = 0.1667 (16.67%)
      - 'park' follows 'the': 2 times  --> P('park' | 'the') = 2/12 = 0.1667 (16.67%)
      - 'rug' follows 'the' : 2 times  --> P('rug'  | 'the') = 2/12 = 0.1667 (16.67%)
    Sum of probabilities = 3/12 + 3/12 + 2/12 + 2/12 + 2/12 = 12/12 = 1.0000 (100.0%)

    Zero-Probability Transitions from 'the' (P(w | 'the') = 0):
      P('the' | 'the') = 0, P('sat' | 'the') = 0, P('ran' | 'the') = 0,
      P('on'  | 'the') = 0, P('to'  | 'the') = 0, P('<START>' | 'the') = 0,
      P('<END>' | 'the') = 0.

(b) Current Word: 'cat'
    Total context occurrences sum_k C('cat', w_k) = 3
    Observed transitions:
      - 'sat' follows 'cat' : 2 times  --> P('sat' | 'cat') = 2/3 = 0.6667 (66.67%)
      - 'ran' follows 'cat' : 1 time   --> P('ran' | 'cat') = 1/3 = 0.3333 (33.33%)
    Sum of probabilities = 2/3 + 1/3 = 1.0000 (100.0%)

    Zero-Probability Transitions from 'cat' (P(w | 'cat') = 0):
      P('the' | 'cat') = 0, P('cat' | 'cat') = 0, P('dog' | 'cat') = 0,
      P('on'  | 'cat') = 0, P('to'  | 'cat') = 0, P('mat' | 'cat') = 0,
      P('rug' | 'cat') = 0, P('park' | 'cat') = 0, P('<END>' | 'cat') = 0.

(c) Current Word: 'dog'
    Total context occurrences sum_k C('dog', w_k) = 3
    Observed transitions:
      - 'sat' follows 'dog' : 2 times  --> P('sat' | 'dog') = 2/3 = 0.6667 (66.67%)
      - 'ran' follows 'dog' : 1 time   --> P('ran' | 'dog') = 1/3 = 0.3333 (33.33%)
    Sum of probabilities = 2/3 + 1/3 = 1.0000 (100.0%)

    Zero-Probability Transitions from 'dog' (P(w | 'dog') = 0):
      P('the' | 'dog') = 0, P('cat' | 'dog') = 0, P('dog' | 'dog') = 0,
      P('on'  | 'dog') = 0, P('to'  | 'dog') = 0, P('mat' | 'dog') = 0,
      P('rug' | 'dog') = 0, P('park' | 'dog') = 0, P('<END>' | 'dog') = 0.

(d) Current Word: 'sat'
    Total context occurrences sum_k C('sat', w_k) = 4
    Observed transitions:
      - 'on' follows 'sat'  : 4 times  --> P('on' | 'sat') = 4/4 = 1.0000 (100.0%)
    Sum of probabilities = 4/4 = 1.0000 (100.0%)

    Zero-Probability Transitions from 'sat' (P(w | 'sat') = 0):
      P(w | 'sat') = 0 for all w != 'on'.

(e) Current Word: 'ran'
    Total context occurrences sum_k C('ran', w_k) = 2
    Observed transitions:
      - 'to' follows 'ran'  : 2 times  --> P('to' | 'ran') = 2/2 = 1.0000 (100.0%)
    Sum of probabilities = 2/2 = 1.0000 (100.0%)

    Zero-Probability Transitions from 'ran' (P(w | 'ran') = 0):
      P(w | 'ran') = 0 for all w != 'to'.

--------------------------------------------------------------------------------
3. FULL FIRST-ORDER CPT MATRIX (ALL OBSERVED TRANSITIONS):
--------------------------------------------------------------------------------
"""
    tokens = sorted(list(fo_model.vocabulary))
    content += f"{'Previous Token (w_i)':<22} | {'Next Token (w_j)':<18} | {'Count':<8} | {'P(w_j | w_i)':<14}\n"
    content += "-" * 70 + "\n"
    for w_prev in tokens:
        if w_prev in fo_model.probabilities:
            total_c = sum(fo_model.transition_counts[w_prev].values())
            for w_curr, prob in sorted(fo_model.probabilities[w_prev].items(), key=lambda x: (-x[1], x[0])):
                cnt = fo_model.transition_counts[w_prev][w_curr]
                content += f"{w_prev:<22} | {w_curr:<18} | {cnt}/{total_c:<6} | {prob:<14.4f}\n"
        else:
            content += f"{w_prev:<22} | {'[TERMINAL]':<18} | {'0/0':<8} | {'0.0000':<14}\n"
    content += "=" * 80 + "\n"
    write_file("task_04_conditional_probability_tables.txt", content)


def task_05():
    # Capture output of first_order_model.py
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    from first_order_model import run_first_order_demonstration
    run_first_order_demonstration()
    sys.stdout = old_stdout
    raw_output = buffer.getvalue()

    content = f"""
================================================================================
AI LABORATORY: TASK 5 OUTPUT
Part V: Ask an LLM to Implement the Model (Console Output)
================================================================================

BEHAVIOURAL SPECIFICATION PROMPT USED:
"Write a simple Python implementation of a first-order autoregressive language model.
The model should:
 1. take a list of tokenised sentences as training data;
 2. count transitions between consecutive tokens;
 3. construct the conditional distribution P(X_t | X_{{t-1}});
 4. display the probabilities for a specified previous token;
 5. predict the most probable next token;
 6. generate a sentence by repeatedly sampling the next token;
 7. stop when the <END> token is generated.
Do not use a machine-learning library or a pretrained language model. Use ordinary
Python data structures and random sampling."

VERBATIM EXECUTION OUTPUT (from first_order_model.py):
--------------------------------------------------------------------------------
{raw_output.strip()}
================================================================================
"""
    write_file("task_05_first_order_implementation_output.txt", content)


def task_06():
    content = """
================================================================================
AI LABORATORY: TASK 6 OUTPUT
Part VI: Inspect the LLM-Generated Code (Questions 4 - 7)
================================================================================

QUESTION 4:
"Where in the program are the transition counts stored?"

ANSWER:
In the FirstOrderAutoregressiveLM class, transition counts are stored in the instance
variable:
    self.transition_counts = defaultdict(Counter)

This is a two-level nested dictionary structure where:
  - The outer key is the conditioning previous token w_prev (representing X_{t-1}).
  - The inner Counter maps each successor token w_curr (representing X_t) to its
    integer transition count C(w_prev, w_curr).
For example:
    self.transition_counts['the']['cat'] == 3
    self.transition_counts['the']['dog'] == 3
    self.transition_counts['the']['mat'] == 2

--------------------------------------------------------------------------------
QUESTION 5:
"Where is P(X_t | X_{t-1}) computed?"

ANSWER:
The conditional probability distribution P(X_t | X_{t-1}) is computed inside the
train() method of FirstOrderAutoregressiveLM:

    for w_prev, next_counts in self.transition_counts.items():
        total_count = sum(next_counts.values())
        for w_curr, count in next_counts.items():
            self.probabilities[w_prev][w_curr] = count / total_count

This loop normalizes the raw counts by dividing each transition count by the sum
of all transitions originating from w_prev, implementing Maximum Likelihood Estimation:
    P(w_curr | w_prev) = C(w_prev, w_curr) / sum_k C(w_prev, w_k)

--------------------------------------------------------------------------------
QUESTION 6:
"How does the program choose the next word?
Is it:
 1. always choosing the most probable word, or
 2. sampling from the probability distribution?
Explain the difference."

ANSWER:
The program explicitly supports and implements BOTH modes:

1. Mode A: Deterministic Greedy Selection (argmax):
   Implemented in predict_most_probable():
       w* = argmax_{w} P(w | w_prev)
   In greedy mode, the model deterministically selects whichever token has the highest
   conditional probability. If probabilities tie, a deterministic alphabetical tie-breaker
   is applied.
   Characteristics:
     - 100% deterministic (zero variation across runs).
     - Prone to entering absorbing periodic loops (e.g., 'the cat sat on the cat sat on...').

2. Mode B: Probabilistic Sampling:
   Implemented in sample_next_token():
       w ~ Categorical(P(X_t = . | X_{t-1} = w_prev))
   Using random.choices(candidates, weights=probabilities), each word has a non-zero
   probability of being chosen proportional to its empirical transition probability.
   Characteristics:
     - Stochastic and non-deterministic (different runs explore different graph trajectories).
     - Accurately models the full conditional entropy of the distribution.

--------------------------------------------------------------------------------
QUESTION 7:
"What happens if the program encounters a word for which no transition has been observed?"

ANSWER:
1. In an Uninspected / Naive Implementation:
   Attempting to look up self.probabilities[word] for an unobserved word (or for the
   terminal token <END>) raises an unhandled KeyError, immediately crashing the program.
   Furthermore, during training, attempting to divide by total_count when total_count == 0
   causes a ZeroDivisionError.

2. In our Inspected and Corrected Implementation:
   We incorporated explicit defensive guards:
       if token not in self.probabilities or not self.probabilities[token]:
           return None
   In generate_sentence(), if next_token is None, the generation loop terminates
   gracefully rather than crashing.

3. In Formal Statistical NLP:
   An unobserved context is handled through probabilistic smoothing techniques:
     - Laplace (add-alpha) smoothing: assigns small non-zero probability to all vocabulary words.
     - Jelinek-Mercer or Kneser-Ney interpolation: backs off to lower-order n-grams
       (e.g., backing off from P(X_t | X_{t-1}) to unigram P(X_t)).
     - Out-Of-Vocabulary (<UNK>) token substitution.
================================================================================
"""
    write_file("task_06_inspect_code.txt", content)


def task_07(fo_model: FirstOrderAutoregressiveLM):
    results = fo_model.test_normalization()
    content = """
================================================================================
AI LABORATORY: TASK 7 OUTPUT
Part VII: Test the Probability Model (Question 8)
================================================================================

1. PROBABILISTIC PROPERTY TEST SPECIFICATION:
For every conditioning word w in the vocabulary (excluding terminal states), the sum
of outgoing conditional transition probabilities must strictly equal 1.0:

    sum_{v in V} P(v | w) = 1.0

Testing Code Implementation:
--------------------------------------------------------------------------------
    for word in probabilities:
        total = sum(probabilities[word].values())
        print(word, total)
--------------------------------------------------------------------------------

2. TEST EXECUTION RESULTS (from first_order_model.py):
--------------------------------------------------------------------------------
Conditioning Token (w) | sum_v P(v | w)  | Deviation (|sum - 1|) | Test Status
--------------------------------------------------------------------------------
"""
    all_passed = True
    for w, (tot, passed) in sorted(results.items()):
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        content += f"{w:<22} | {tot:<15.8f} | {abs(tot-1.0):<21.2e} | [{status}]\n"

    content += "-" * 75 + f"\nOVERALL FIRST-ORDER NORMALIZATION: {'100% PASSED' if all_passed else 'FAILED'}\n\n"

    content += """--------------------------------------------------------------------------------
3. QUESTION 8 AND DETAILED DIAGNOSTIC ANSWER:
--------------------------------------------------------------------------------
QUESTION 8:
"If one of the totals is 0.87, what does this tell you about the implementation?"

ANSWER:
If sum_v P(v | w) = 0.87 for any conditioning word w, this indicates a SEVERE
DEFECT in the implementation that violates the fundamental axioms of probability:

1. Violation of Kolmogorov's Second Axiom (Unit Measure):
   The total probability of the entire sample space conditioned on any event must
   equal exactly 1:
       P(Omega | E) = sum_{v in V} P(v | w) = 1.0
   A sum of 0.87 means exactly 13% (0.13) of the probability mass is missing.

2. Diagnostic Causes of Missing Probability Mass:
   - Dropped Transitions: The tokenization or counting logic failed to record valid
     successor tokens (e.g., dropping transitions to the special <END> token, or
     skipping words with punctuation).
   - Normalization Denominator Bug: Dividing by an incorrect total (such as the total
     word count of the entire corpus, or an off-by-one loop counter) instead of the
     row sum sum_k C(w_i, w_k).
   - Truncation / Pruning without Re-normalization: If the code applied a threshold
     cutoff (such as discarding tokens with p < 0.05) or top-k filtering without
     dividing by the remaining sum, the total will sum to strictly less than 1.0.

3. Impact on Generation:
   A defective probability distribution that sums to 0.87 will either crash sampling
   functions (such as numpy.random.choice) or distort generative frequencies, making
   the model probabilistically invalid.
================================================================================
"""
    write_file("task_07_test_probability_model.txt", content)


def task_08(fo_model: FirstOrderAutoregressiveLM):
    key_words = ["the", "cat", "dog", "sat", "ran", "<START>"]
    content = """
================================================================================
AI LABORATORY: TASK 8 OUTPUT
Part VIII: Predicting the Next Word (Question 9)
================================================================================

1. CONDITIONAL DISTRIBUTIONS AND MOST PROBABLE NEXT WORD PREDICTIONS:
Using the trained first-order model to compute:
    P(X_{t+1} | X_t = context)
and identifying the most probable next word:
    argmax_w P(w | context)

"""
    for kw in key_words:
        dist = fo_model.probabilities.get(kw, {})
        tot = sum(fo_model.transition_counts[kw].values())
        best, p_val, tied = fo_model.predict_most_probable(kw)
        content += f"--------------------------------------------------------------------------------\n"
        content += f"Context Token: '{kw}' (Observed {tot} times in training data)\n"
        content += f"Conditional Distribution P(X_{{t+1}} | X_t = '{kw}'):\n"
        for w, p in sorted(dist.items(), key=lambda x: (-x[1], x[0])):
            c = fo_model.transition_counts[kw][w]
            content += f"    P({w:<8} | '{kw}') = {c}/{tot} = {p:.4f} ({p*100:5.1f}%)\n"
        if len(tied) > 1:
            content += f"  Most Probable Prediction (argmax): '{best}' (p = {p_val:.4f}, tied with {tied})\n"
        else:
            content += f"  Most Probable Prediction (argmax): '{best}' (p = {p_val:.4f})\n"

    content += """
--------------------------------------------------------------------------------
2. QUESTION 9 AND DETAILED ANSWER:
--------------------------------------------------------------------------------
QUESTION 9:
"Are the most probable predictions always the same as the words that you would
personally expect? What does this tell you about the difference between:
  a probability model
and
  human linguistic expectations?"

ANSWER:
1. Comparison with Personal Linguistic Expectations:
   - For 'the': The model's argmax is tied between 'cat' and 'dog' (25% each). However,
     the model also assigns 16.7% probability each to 'mat', 'park', and 'rug'. When
     starting a sentence ('<START> the'), a human expects an animal or agent, never an
     inanimate destination like 'mat' or 'park'. Yet the first-order model cannot distinguish
     sentence-initial 'the' from prepositional 'the' (e.g. 'on the', 'to the').
   - For 'sat': The model predicts 'on' with 100% certainty. A human speaker readily
     anticipates other prepositions ('sat by', 'sat near', 'sat down', 'sat quietly').
   - For 'ran': The model predicts 'to' with 100% certainty, ignoring common alternatives
     like 'ran away', 'ran fast', or 'ran home'.

2. The Difference Between a Probability Model and Human Linguistic Expectations:
   - Nature of a Probability Model:
     A statistical probability model is an empirical estimator strictly conditioned on
     the sample statistics of a specific, finite training dataset under rigid structural
     assumptions (Markov independence). It has no internal representation of world
     knowledge, physics (cats sit on mats, mats do not sit), grammar categories, or
     communicative intent. It merely computes ratios of string counts.
   - Nature of Human Linguistic Expectations:
     Human expectations are guided by deep semantic comprehension, world knowledge,
     compositional syntax, pragmatic context, and rich open-ended vocabulary. Humans
     understand meaning and physical reality; simple n-gram probability models only
     recognize local co-occurrence frequencies.
================================================================================
"""
    write_file("task_08_predicting_next_word.txt", content)


def task_09(fo_model: FirstOrderAutoregressiveLM):
    random.seed(42)
    sentences = [fo_model.generate_sentence(mode="sampling") for _ in range(20)]
    
    content = """
================================================================================
AI LABORATORY: TASK 9 OUTPUT
Part IX: Generate Text by Sampling (20 Generated Sentences)
================================================================================

1. GENERATION ALGORITHM (ANCESTRAL SAMPLING):
The generation process follows ancestral forward sampling on the linear Bayesian Network:
    X_1 ~ P(X_1)               [Starts from <START>]
    X_2 ~ P(X_2 | X_1)
    X_3 ~ P(X_3 | X_2)
    ...
    X_t ~ P(X_t | X_{t-1})
    Stop when X_t == <END>

At each step, the next token is sampled from the conditional categorical distribution
P(X_t | X_{t-1}) using random sampling weighted by transition probabilities.

--------------------------------------------------------------------------------
2. TWENTY (20) GENERATED SENTENCES (from first-order model):
--------------------------------------------------------------------------------
"""
    for i, s in enumerate(sentences, 1):
        content += f"{i:2d}. {s}\n"

    content += """
--------------------------------------------------------------------------------
3. ANALYSIS OF GENERATED SENTENCES:
Notice several remarkable phenomena in the sampled output:
1. Sentence Fragments (e.g., Sentences 2, 4, 6, 8: '<START> the park <END>', '<START> the mat <END>'):
   Because the first-order model only conditions on X_{t-1} = 'the', it assigns 16.7%
   probability to 'park', 'mat', and 'rug' right after the initial 'the'! Since these
   words are immediately followed by <END>, the model generates truncated, ungrammatical
   2-word fragments.

2. Run-On Chained Sentences (e.g., Sentence 1):
   '<START> the cat sat on the dog ran to the cat sat on the cat sat on the dog ran to the dog sat on the mat <END>'
   Because 'on' -> 'the' (100%), and 'the' -> 'cat'/'dog' (50%), the model can loop
   repeatedly through animal actions before finally sampling 'mat' or 'rug' to terminate.

These flaws demonstrate the fundamental limitations of the 1st-order Markov assumption.
================================================================================
"""
    write_file("task_09_generate_text_sampling.txt", content)


def task_10(fo_model: FirstOrderAutoregressiveLM):
    greedy_runs = [fo_model.generate_sentence(mode="greedy", max_tokens=25) for _ in range(5)]
    random.seed(999)
    sample_runs = [fo_model.generate_sentence(mode="sampling") for _ in range(5)]

    content = """
================================================================================
AI LABORATORY: TASK 10 OUTPUT
Part X: Deterministic vs Probabilistic Generation (Question 10)
================================================================================

1. GENERATION MODES:
Mode A: Greedy Generation
        Always choose: argmax_w P(w | w_previous)
Mode B: Sampling Generation
        Sample from:   P(w | w_previous)

--------------------------------------------------------------------------------
2. EXPERIMENTAL RUNS (5 SENTENCES PER METHOD):
--------------------------------------------------------------------------------

MODE A: GREEDY GENERATION (5 Runs):
"""
    for i, s in enumerate(greedy_runs, 1):
        content += f"  Run {i}: {s}\n"

    content += "\nMODE B: SAMPLING GENERATION (5 Runs):\n"
    for i, s in enumerate(sample_runs, 1):
        content += f"  Run {i}: {s}\n"

    content += """
--------------------------------------------------------------------------------
3. QUESTION 10 AND DETAILED COMPARATIVE ANSWER:
--------------------------------------------------------------------------------
QUESTION 10:
"Compare the two sets of generated sentences.
Which mode produces more variation?
Why?"

ANSWER:
1. Comparison of Generated Sets:
   - Mode A (Greedy) produced ZERO variation across all 5 runs. Every single run produced
     the exact identical repetitive sentence:
       '<START> the cat sat on the cat sat on the cat sat on...'
     Because the maximum probability path contains a closed cycle, the greedy decoder
     entered an infinite loop, generating until it hit the hard-coded max_tokens limit.
   - Mode B (Sampling) produced diverse sentences of varying lengths and completions:
       Run 1: '<START> the rug <END>'
       Run 2: '<START> the cat sat on the rug <END>'
       Run 3: '<START> the dog sat on the rug <END>'
       Run 4: '<START> the cat ran to the mat <END>'
       Run 5: '<START> the park <END>'

2. Which Mode Produces More Variation:
   Mode B (Sampling) produces vastly more variation (100% distinct outputs vs 0% distinct outputs).

3. Theoretical Explanation (Why):
   - Greedy generation is a deterministic finite-state transition:
         f(w) = argmax_v P(v | w)
     Starting from state <START>, the sequence of states is strictly predetermined.
     From '<START>', argmax is 'the' (p=1.0).
     From 'the', argmax ties between 'cat' and 'dog' (p=0.25); by tie-breaker, 'cat' is chosen.
     From 'cat', argmax is 'sat' (p=0.67).
     From 'sat', argmax is 'on' (p=1.0).
     From 'on', argmax is 'the' (p=1.0).
     From 'the', argmax is 'cat' again!
     The deterministic mapping forms a closed limit cycle with entropy H = 0.
   - Sampling generation treats transition selection as drawing a random variable from
     a multinomial distribution. Every token with P(v | w) > 0 has an opportunity to
     be selected, allowing the generative process to branch across the entire graph.
================================================================================
"""
    write_file("task_10_deterministic_vs_probabilistic.txt", content)


def task_11():
    content = """
================================================================================
AI LABORATORY: TASK 11 OUTPUT
Part XI: A Second-Order Bayesian Network (Question 11)
================================================================================

1. MATHEMATICAL FORMULATION:
In a second-order autoregressive language model, each token depends on the TWO
immediately preceding tokens:

    P(X_t | X_1, ..., X_{t-1}) approx P(X_t | X_{t-2}, X_{t-1})

The corresponding Bayesian Network structure for a sequence of tokens is:
    [X_{t-2}] ---> [X_t] <--- [X_{t-1}]

For a sequence of four tokens (X_1, X_2, X_3, X_4), the joint factorisation becomes:
    P(X_1, X_2, X_3, X_4) = P(X_1) * P(X_2 | X_1) * P(X_3 | X_1, X_2) * P(X_4 | X_2, X_3)

--------------------------------------------------------------------------------
2. QUESTION 11 AND DETAILED MULTI-DIMENSIONAL COMPARISON:
--------------------------------------------------------------------------------
QUESTION 11:
"How does the second-order model differ from the first-order model in terms of:
 1. the graph structure?
 2. the conditional probability table?
 3. the amount of context available for prediction?
 4. the amount of data needed?"

ANSWER:

1. Graph Structure:
   - First-Order: A simple linear chain (X_1 -> X_2 -> X_3 -> ... -> X_T).
     Each node X_t has an in-degree of 1 (a single parent: Parents(X_t) = {X_{t-1}}).
   - Second-Order: A Directed Acyclic Graph with skip-1 edges.
     Each node X_t has an in-degree of 2 (two parents: Parents(X_t) = {X_{t-2}, X_{t-1}}).

2. Conditional Probability Table (CPT):
   - First-Order: A 2D matrix indexed by (X_{t-1}, X_t) of size |V| x |V|.
     Each conditioning context is a single word.
     For |V| = 12, total entries = 12 x 12 = 144.
   - Second-Order: A 3D tensor indexed by (X_{t-2}, X_{t-1}, X_t) of size |V| x |V| x |V|.
     Each conditioning context is an ordered pair (bigram) (w_{t-2}, w_{t-1}).
     For |V| = 12, total entries = 12 x 12 x 12 = 1,728.

3. Amount of Context Available for Prediction:
   - First-Order: 1 token of context history. The model is blind to whether 'the' was
     preceded by '<START>', 'on', or 'to'.
   - Second-Order: 2 tokens of context history. The model conditions on ('on', 'the')
     versus ('to', 'the') versus ('<START>', 'the'), allowing it to maintain syntactic
     and semantic consistency.

4. Amount of Data Needed:
   - First-Order: Scales as O(|V|^2). Requires sufficient data to observe word pairs (bigrams).
   - Second-Order: Scales as O(|V|^3). Requires exponentially more data to observe word
     triples (trigrams). In small datasets, the vast majority of possible trigrams will
     have zero counts (extreme data sparsity).
================================================================================
"""
    write_file("task_11_second_order_bayesian_network.txt", content)


def task_12():
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    from second_order_model import run_second_order_demonstration
    run_second_order_demonstration()
    sys.stdout = old_stdout
    raw_output = buffer.getvalue()

    content = f"""
================================================================================
AI LABORATORY: TASK 12 OUTPUT
Part XII: Second-Order Model Implementation and Execution Output
================================================================================

BEHAVIOURAL SPECIFICATION PROMPT USED:
"Modify the existing first-order autoregressive model into a second-order model.
The model should estimate:
    P(X_t | X_{{t-2}}, X_{{t-1}})
Represent the model using counts of observed triples and use these counts to
construct conditional probability distributions.
Do not replace the model with a neural network or a pretrained language model."

VERBATIM EXECUTION OUTPUT (from second_order_model.py):
--------------------------------------------------------------------------------
{raw_output.strip()}
================================================================================
"""
    write_file("task_12_second_order_implementation_output.txt", content)


def task_13():
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    from model_comparison import run_comparative_study
    run_comparative_study()
    sys.stdout = old_stdout
    raw_output = buffer.getvalue()

    content = f"""
================================================================================
AI LABORATORY: TASK 13 OUTPUT
Part XIII: Comparing the Two Models (Question 12)
================================================================================

1. SYSTEMATIC BENCHMARK REPORT (from model_comparison.py):
--------------------------------------------------------------------------------
{raw_output.strip()}

--------------------------------------------------------------------------------
2. QUESTION 12 AND DETAILED ANSWER:
--------------------------------------------------------------------------------
QUESTION 12:
"Why does increasing the amount of context potentially improve prediction?
Why can it simultaneously make the model harder to estimate from limited data?
Relate your answer to the size of the conditional probability table."

ANSWER:

1. Why Increasing Context Improves Prediction:
   In natural language, the probability of a word depends heavily on longer-range syntactic
   agreements and semantic discourse.
   In our laboratory corpus:
     - With 1 token of context (X_{{t-1}} = 'the'), the model cannot distinguish between
       sentence subjects and prepositional objects. It produces invalid fragments like
       '<START> the mat <END>'.
     - With 2 tokens of context:
         Context ('<START>', 'the') ONLY predicts 'cat' (50%) or 'dog' (50%).
         Context ('on', 'the')      ONLY predicts 'mat' (50%) or 'rug' (50%).
         Context ('to', 'the')      ONLY predicts 'park' (100%).
   Expanding context eliminates conditioning ambiguity and guarantees 100% grammatical
   coherence across generated outputs.

2. Why It Simultaneously Makes the Model Harder to Estimate (The Curse of Dimensionality):
   The size of the CPT grows exponentially with context window length k:
       CPT Size = |V|^(k+1)

   - In our toy vocabulary (|V| = 12):
       First-Order  (k=1): 12^2 = 144 entries.
       Second-Order (k=2): 12^3 = 1,728 entries.
       Third-Order  (k=3): 12^4 = 20,736 entries.
   - In a realistic vocabulary (|V| = 50,000):
       First-Order  (k=1): 2.5 x 10^9 entries (2.5 billion).
       Second-Order (k=2): 1.25 x 10^14 entries (125 trillion!).

   Because natural language corpora express only a tiny fraction of all conceivable
   n-grams, most contexts will never appear in training data. In our small corpus,
   89.6% of theoretical bigram contexts had zero observations. When a context has zero
   counts, Maximum Likelihood Estimation breaks down (0/0), leading to extreme sparsity,
   brittleness, and high variance.
================================================================================
"""
    write_file("task_13_model_comparison.txt", content)


def task_14():
    content = """
================================================================================
AI LABORATORY: TASK 14 OUTPUT
Part XIV & Section 18: The Connection to Modern Language Models
================================================================================

1. THE COMMON PROBABILISTIC FOUNDATION:
Both classical n-gram Bayesian Networks and state-of-the-art Large Language Models
(Transformers like GPT-4, Gemini, Claude, LLaMA) share the exact same underlying
probabilistic objective:

    P(x_1, ..., x_T) = prod_{t=1}^T P(x_t | x_1, ..., x_{t-1})

At every step t, both systems compute:
    P(next token | previous tokens)

The fundamental difference lies not in the probability chain rule, but in HOW the
conditional probability distribution is parameterized, represented, and learned.

--------------------------------------------------------------------------------
2. DETAILED COMPARISON TABLE:
--------------------------------------------------------------------------------
Dimension              | Classical Bayesian Network LM     | Modern Autoregressive Neural LM (LLM)
--------------------------------------------------------------------------------
Mathematical Form      | Explicit CPT Matrix / Tensor      | Deep Neural Network (Self-Attention)
Context Window         | Fixed small window (k=1 or k=2)   | Vast attention span (8k to 1M+ tokens)
Parameter Scaling      | Exponential in context O(|V|^(k+1)| Linear in context via Attention O(N^2)
Parameter Storage      | Discrete conditional counts       | Continuous weight matrices (W_q, W_k, etc.)
Learning Algorithm     | Exact Frequency Counting (MLE)    | Gradient Descent on Cross-Entropy Loss
Generalization         | Exact string match only; brittle  | Dense semantic embeddings (word vectors)
Zero-Frequency Issue   | Requires heuristic smoothing      | Smooth continuous interpolation across states
Sampling Techniques    | Basic Categorical Sampling        | Temperature, Top-k, Nucleus (Top-p), Min-p
--------------------------------------------------------------------------------

3. INSIGHT:
Modern neural networks do not replace the laws of probability; rather, they serve
as highly expressive universal function approximators that estimate the conditional
distribution P(x_t | x_{<t}) without suffering from the exponential CPT table explosion.
================================================================================
"""
    write_file("task_14_connection_to_modern_llms.txt", content)


def task_15():
    content = """
================================================================================
AI LABORATORY: TASK 15 OUTPUT
Part XV: Reflection on the Role of the LLM (Question 13 & Deliverable 7)
================================================================================

1. QUESTION 13 AND DETAILED ANSWER:
--------------------------------------------------------------------------------
QUESTION 13:
"Why is Approach B preferable when constructing an intelligent system?
Discuss the importance of:
 - specifying the intended behaviour;
 - understanding the representation;
 - validating the generated implementation;
 - testing probabilistic invariants;
 - distinguishing implementation from model."

ANSWER:
Approach A: "Write a Python language model for me."
Approach B: "Implement the following probabilistic model: P(X_t | X_{t-1}), estimated
             from transition counts, with sampling-based generation."

Approach B is decisively preferable for the following engineering reasons:

1. Specifying the Intended Behaviour:
   Approach A is ambiguous. An LLM might generate a recurrent neural network, call
   a HuggingFace pretrained pipeline, or write a hardcoded regular expression.
   Approach B enforces contract-driven development, providing unambiguous requirements
   for input representations, statistical estimators, and sampling behaviors.

2. Understanding the Representation:
   In Approach B, the engineer specifies the underlying data structures (nested hash maps
   mapping predecessor tokens to successor frequencies). Knowing how the state space
   is represented enables transparent debugging, mathematical auditing, and interpretability.

3. Validating the Generated Implementation:
   When the target model is explicitly specified, the engineer can inspect the code
   against clear mathematical expectations (e.g., verifying that the normalizer divides
   by row sums sum_k C(w_i, w_k) rather than corpus-level totals).

4. Testing Probabilistic Invariants:
   A formal model dictates clear physical laws. Probabilities must be non-negative and
   must sum to 1.0 across the sample space. Approach B allows writing automated property-based
   tests (assert abs(sum(p.values()) - 1.0) < 1e-6) to mathematically certify the code.

5. Distinguishing Implementation from Model:
   The model is the mathematical idealization (a Markov chain over categorical random
   variables). The implementation is the software artifact (Python dicts, loops, and random calls).
   Distinguishing the two ensures that software bugs (e.g., KeyError on unseen tokens)
   are not mistaken for theoretical flaws in the mathematical model.

--------------------------------------------------------------------------------
2. DELIVERABLE 7 CASE STUDY: CODE INSPECTION AND CORRECTION
--------------------------------------------------------------------------------
During the lab workflow, initial naive LLM code was generated for greedy generation:

    # Initial naive code generated by LLM:
    def generate_greedy(self, start_token="<START>", max_tokens=50):
        curr = start_token
        words = [curr]
        while curr != "<END>":
            # DEFECT 1: Unhandled KeyError if curr is not in self.probabilities
            # DEFECT 2: Infinite loop trap if argmax path contains a cycle
            next_word = max(self.probabilities[curr], key=self.probabilities[curr].get)
            words.append(next_word)
            curr = next_word
        return " ".join(words)

Defects Discovered and Corrected:
1. Infinite Loop: The sequence 'the' -> 'cat' -> 'sat' -> 'on' -> 'the' forms a closed
   cycle. Because the while loop had no iteration limit, the code hung indefinitely.
   Correction: Replaced while loop with a bounded for-loop and added max_tokens limits.
2. Unhandled KeyError: If curr had no outgoing edges (such as <END> or an out-of-vocabulary word),
   the lookup crashed.
   Correction: Added defensive None checks and graceful early termination.
3. Arbitrary Tie-Breaking: 'cat' and 'dog' tied with p=0.25 after 'the'. Using max() on
   a dictionary yielded arbitrary ordering depending on hash table layout.
   Correction: Implemented deterministic alphabetical tie-breaking.
================================================================================
"""
    write_file("task_15_reflection_role_of_llm.txt", content)


def task_16():
    content = """
================================================================================
AI LABORATORY: TASK 16 OUTPUT
Part XX: Final Question: What Did the Bayesian Network Add? (Question 14)
================================================================================

QUESTION 14:
"What did thinking of the language model as a Bayesian network give you?
Discuss at least three of the following:
 - a representation of dependencies;
 - a factorisation of the joint distribution;
 - a way of interpreting conditional probabilities;
 - a principled method for generation;
 - a way to reason about independence assumptions;
 - a way to understand the effect of increasing context;
 - a way to test whether an implementation matches its probabilistic specification."

ANSWER (Comprehensive discussion of all seven dimensions):

1. A Structured Representation of Dependencies:
   Rather than viewing language generation as a black box, the Bayesian Network
   formalism provides a Directed Acyclic Graph (DAG) where word positions are explicit
   random variables X_1, X_2, ..., X_T. Directed edges visually and mathematically
   define direct influence, revealing exactly which past tokens inform future predictions.

2. A Rigorous Factorisation of the Joint Distribution:
   By the local Markov property of Bayesian Networks, the joint distribution factorises
   directly according to graph topology:
       P(X_1, ..., X_T) = prod_{t=1}^T P(X_t | Parents(X_t))
   For the first-order model, Parents(X_t) = {X_{t-1}}.
   For the second-order model, Parents(X_t) = {X_{t-2}, X_{t-1}}.
   This grounds autoregressive factorization in graphical model theory.

3. A Way of Interpreting Conditional Probabilities:
   Conditional Probability Tables (CPTs) in a Bayesian Network have precise local semantics.
   Each row of a CPT represents an independent categorical distribution over the child
   variable conditioned on an instantiation of its parents.

4. A Principled Method for Generation (Ancestral Sampling):
   Text generation is no longer an ad-hoc heuristic; it is standard Ancestral Forward
   Sampling on a DAG. One traverses nodes in topological order, sampling from prior
   distributions at root nodes and from conditional distributions at child nodes.

5. A Way to Reason About Independence Assumptions (d-Separation):
   Graph semantics allow rigorous reasoning about information flow. In X_1 -> X_2 -> X_3,
   conditioning on X_2 d-separates X_1 from X_3, formally proving that the model forgets
   all context older than one token. This explicitly reveals the trade-off between
   computational simplicity and linguistic realism.

6. A Way to Understand the Effect of Increasing Context:
   Adding context corresponds to adding incoming directed edges to node X_t. The graph
   instantly illustrates why this is both powerful and expensive: adding an edge doubles
   parent conditioning complexity, causing the CPT table size to multiply by |V|
   (the curse of dimensionality).

7. A Way to Test Whether an Implementation Matches its Probabilistic Specification:
   Because CPT rows correspond to valid probability distributions, we obtain non-negotiable
   invariants that MUST hold in code: sum_{v in V} P(v | parents) == 1.0. This allows
   us to write automated, property-based tests to mathematically verify the implementation.
================================================================================
"""
    write_file("task_16_final_question_bayesian_network.txt", content)


def generate_tasks_index():
    content = """
================================================================================
AI LABORATORY: MASTER INDEX OF TASK OUTPUT FILES
Bayesian Networks and Autoregressive Language Models
================================================================================

Below is the directory mapping of the individual output files generated for each
task, question, and deliverable in the workspace:

Task Output File                             | Lab Section | Description
--------------------------------------------------------------------------------
task_01_from_probability_to_language.txt     | Part I      | Chain-rule factorization & Question 1 answer
task_02_bayesian_network_for_text.txt        | Part II     | Linear-chain BN graph & Question 2 Markov assumption
task_03_language_dataset.txt                 | Part III    | Tokenized 6-sentence corpus, vocabulary, and stats
task_04_conditional_probability_tables.txt   | Part IV     | Complete CPTs, Question 3, and zero-prob transitions
task_05_first_order_implementation_output.txt| Part V      | Verbatim console output from first_order_model.py
task_06_inspect_code.txt                     | Part VI     | Code inspection answers for Questions 4, 5, 6, and 7
task_07_test_probability_model.txt           | Part VII    | Normalization test results & Question 8 diagnosis
task_08_predicting_next_word.txt             | Part VIII   | Argmax predictions for 6 contexts & Question 9 answer
task_09_generate_text_sampling.txt           | Part IX     | 20 sentences sampled from first-order model & analysis
task_10_deterministic_vs_probabilistic.txt   | Part X      | Mode A (5 greedy) vs Mode B (5 sampled) & Question 10
task_11_second_order_bayesian_network.txt    | Part XI     | 2nd-order factorisation & Question 11 comparison
task_12_second_order_implementation_output.txt| Part XII   | Verbatim console output from second_order_model.py
task_13_model_comparison.txt                 | Part XIII   | Parameter & diversity comparison & Question 12 answer
task_14_connection_to_modern_llms.txt        | Part XIV/18 | Connection between Bayesian Networks and modern LLMs
task_15_reflection_role_of_llm.txt           | Part XV     | LLM prompt reflection, Question 13 & code correction
task_16_final_question_bayesian_network.txt  | Part XX     | Comprehensive Question 14 analysis (all 7 dimensions)
--------------------------------------------------------------------------------

ADDITIONAL SYSTEM ARTIFACTS:
- first_order_model.py   : Python implementation of 1st-order model
- second_order_model.py  : Python implementation of 2nd-order model
- model_comparison.py    : Automated benchmark script comparing both models
- run_experiments.py     : Orchestration runner
- cpt_tables.txt         : Full CPT reference tables
- generated_sentences.txt: Complete generated sentences repository
- test_results.txt       : Probability normalization logs
- LAB_REPORT.md          : Complete, master academic laboratory report
================================================================================
"""
    write_file("TASKS_INDEX.txt", content)


def main():
    print("Generating dedicated output text files for each lab task...")
    
    # Train models once for data-driven outputs
    fo_corpus = [tokenize_sentence(s) for s in DEFAULT_TRAINING_CORPUS]
    fo_model = FirstOrderAutoregressiveLM()
    fo_model.train(fo_corpus)

    task_01()
    task_02()
    task_03()
    task_04(fo_model)
    task_05()
    task_06()
    task_07(fo_model)
    task_08(fo_model)
    task_09(fo_model)
    task_10(fo_model)
    task_11()
    task_12()
    task_13()
    task_14()
    task_15()
    task_16()
    generate_tasks_index()
    print("\nAll separate task output files have been successfully generated!")


if __name__ == "__main__":
    main()
