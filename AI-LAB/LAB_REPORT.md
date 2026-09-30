# AI Laboratory Report
# Bayesian Networks and Autoregressive Language Models

**Course:** Artificial Intelligence Laboratory  
**Topic:** Bayesian Networks, Probabilistic Graphical Models, and Autoregressive Text Generation  
**Author:** AI Researcher & Pair Programmer  
**Deliverables Generated:**
- [first_order_model.py](file:///C:/Users/Amithav/Desktop/AI-LAB/first_order_model.py)
- [second_order_model.py](file:///C:/Users/Amithav/Desktop/AI-LAB/second_order_model.py)
- [model_comparison.py](file:///C:/Users/Amithav/Desktop/AI-LAB/model_comparison.py)
- [run_experiments.py](file:///C:/Users/Amithav/Desktop/AI-LAB/run_experiments.py)
- [cpt_tables.txt](file:///C:/Users/Amithav/Desktop/AI-LAB/cpt_tables.txt)
- [generated_sentences.txt](file:///C:/Users/Amithav/Desktop/AI-LAB/generated_sentences.txt)
- [test_results.txt](file:///C:/Users/Amithav/Desktop/AI-LAB/test_results.txt)

---

## 1. Executive Summary & Workflow Progression

This laboratory explores the fundamental theoretical and algorithmic bridge connecting **Bayesian Networks** and modern **Autoregressive Language Models (LLMs)**.

The core progression of this laboratory follows the principle:
$$\text{Probability} \longrightarrow \text{Bayesian Network} \longrightarrow \text{Autoregressive Model} \longrightarrow \text{Language Generation}$$

By framing language generation as ancestral sampling over a directed acyclic probabilistic graphical model (Bayesian Network), we dissect how structural independence assumptions govern expressive capacity, parameter scaling, data sparsity, and generation coherence.

The laboratory was conducted strictly according to the pedagogical workflow:
$$\text{Understand} \longrightarrow \text{Design} \longrightarrow \text{Ask the LLM} \longrightarrow \text{Implement} \longrightarrow \text{Test} \longrightarrow \text{Reflect}$$

---

## 2. Answers to Laboratory Questions (Questions 1 – 14)

### Part I: From Probability to Language

#### Question 1
> **Why is this decomposition useful for generating text? Write a short explanation in your lab record.**

The joint probability of a sequence of tokens $W = (X_1, X_2, \dots, X_T)$ is defined over a combinatorial space of size $|V|^T$, where $|V|$ is vocabulary size and $T$ is sequence length. Directly estimating or sampling from $P(X_1, \dots, X_T)$ in one shot is computationally intractable because almost every possible sequence has an empirical count of zero.

By applying the **chain rule of probability**, the joint distribution factorizes without any loss of generality:
$$P(X_1, X_2, \dots, X_T) = P(X_1) \prod_{t=2}^T P(X_t \mid X_1, \dots, X_{t-1})$$

This autoregressive decomposition is profoundly useful for four reasons:
1. **Converts Intractable Joint Inference to Sequential Conditional Prediction:** Instead of estimating a single probability over $|V|^T$ outcomes, the model only needs to estimate a 1D categorical distribution over $|V|$ tokens at each step $t$.
2. **Enables Step-by-Step Ancestral Sampling:** A generator can emit tokens sequentially: sample $X_1 \sim P(X_1)$, then sample $X_2 \sim P(X_2 \mid X_1)$, then $X_3 \sim P(X_3 \mid X_1, X_2)$, appending each token to the conditioning history.
3. **Supports Arbitrary and Variable Lengths:** Text generation does not require fixing sequence length $T$ in advance; the process simply terminates whenever a special stopping token (e.g., `⟨END⟩`) is sampled.
4. **Natural Foundation for Online and Streaming AI Systems:** It mirrors the temporal, left-to-right production of human speech and text, enabling prompt conditioning, causal masking, and real-time streaming inference.

---

### Part II: A Bayesian Network for Text

#### Question 2
> **What independence assumption is being made by this network? Express your answer using probability notation.**

The network represents a linear chain Bayesian network:
$$X_1 \longrightarrow X_2 \longrightarrow X_3 \longrightarrow \dots \longrightarrow X_T$$

The fundamental independence assumption is the **First-Order Markov Property**: each word $X_t$ is conditionally independent of all preceding words prior to $t-1$, given the immediately preceding word $X_{t-1}$.

In formal probability notation:
$$X_t \perp\!\!\!\perp (X_1, X_2, \dots, X_{t-2}) \mid X_{t-1} \quad \forall t \ge 2$$

Equivalently, in terms of conditional probability distributions:
$$P(X_t \mid X_1, X_2, \dots, X_{t-1}) = P(X_t \mid X_{t-1})$$

This simplifies the full chain rule factorization into:
$$P(X_1, X_2, \dots, X_T) = P(X_1) \prod_{t=2}^T P(X_t \mid X_{t-1})$$

---

### Part III & IV: Dataset and Conditional Probability Table

The laboratory training corpus consists of six sentences:
1. `the cat sat on the mat`
2. `the cat sat on the rug`
3. `the dog sat on the mat`
4. `the dog ran to the park`
5. `the cat ran to the park`
6. `the dog sat on the rug`

Prepending `⟨START⟩` and appending `⟨END⟩` yields vocabulary $V$:
$$V = \{\text{⟨START⟩}, \text{⟨END⟩}, \text{cat}, \text{dog}, \text{mat}, \text{on}, \text{park}, \text{ran}, \text{rug}, \text{sat}, \text{the}, \text{to}\} \quad (|V| = 12)$$

#### Question 3
> **Construct the conditional probability distribution $P(\text{next word} \mid \text{current word})$ for at least the following words: `the`, `cat`, `dog`, `sat`, `ran`. Identify any zero-probability transitions.**

Transitions are calculated using Maximum Likelihood Estimation (MLE):
$$P(w_j \mid w_i) = \frac{C(w_i, w_j)}{\sum_k C(w_i, w_k)}$$

#### 1. Conditional Distribution for `the` (Total Context Occurrences: 12)
- $C(\text{the}, \text{cat}) = 3 \implies P(\text{cat} \mid \text{the}) = \frac{3}{12} = 0.2500$ ($25.0\%$)
- $C(\text{the}, \text{dog}) = 3 \implies P(\text{dog} \mid \text{the}) = \frac{3}{12} = 0.2500$ ($25.0\%$)
- $C(\text{the}, \text{mat}) = 2 \implies P(\text{mat} \mid \text{the}) = \frac{2}{12} = 0.1667$ ($16.67\%$)
- $C(\text{the}, \text{park}) = 2 \implies P(\text{park} \mid \text{the}) = \frac{2}{12} = 0.1667$ ($16.67\%$)
- $C(\text{the}, \text{rug}) = 2 \implies P(\text{rug} \mid \text{the}) = \frac{2}{12} = 0.1667$ ($16.67\%$)
- **Zero-Probability Transitions from `the`:**
  $$P(\text{the} \mid \text{the}) = 0, \quad P(\text{sat} \mid \text{the}) = 0, \quad P(\text{ran} \mid \text{the}) = 0, \quad P(\text{on} \mid \text{the}) = 0, \quad P(\text{to} \mid \text{the}) = 0, \quad P(\text{⟨END⟩} \mid \text{the}) = 0, \quad P(\text{⟨START⟩} \mid \text{the}) = 0$$

#### 2. Conditional Distribution for `cat` (Total Context Occurrences: 3)
- $C(\text{cat}, \text{sat}) = 2 \implies P(\text{sat} \mid \text{cat}) = \frac{2}{3} \approx 0.6667$ ($66.67\%$)
- $C(\text{cat}, \text{ran}) = 1 \implies P(\text{ran} \mid \text{cat}) = \frac{1}{3} \approx 0.3333$ ($33.33\%$)
- **Zero-Probability Transitions from `cat`:**
  $P(w \mid \text{cat}) = 0$ for all $w \in \{\text{the}, \text{cat}, \text{dog}, \text{on}, \text{to}, \text{mat}, \text{rug}, \text{park}, \text{⟨END⟩}, \text{⟨START⟩}\}$.

#### 3. Conditional Distribution for `dog` (Total Context Occurrences: 3)
- $C(\text{dog}, \text{sat}) = 2 \implies P(\text{sat} \mid \text{dog}) = \frac{2}{3} \approx 0.6667$ ($66.67\%$)
- $C(\text{dog}, \text{ran}) = 1 \implies P(\text{ran} \mid \text{dog}) = \frac{1}{3} \approx 0.3333$ ($33.33\%$)
- **Zero-Probability Transitions from `dog`:**
  $P(w \mid \text{dog}) = 0$ for all $w \in \{\text{the}, \text{cat}, \text{dog}, \text{on}, \text{to}, \text{mat}, \text{rug}, \text{park}, \text{⟨END⟩}, \text{⟨START⟩}\}$.

#### 4. Conditional Distribution for `sat` (Total Context Occurrences: 4)
- $C(\text{sat}, \text{on}) = 4 \implies P(\text{on} \mid \text{sat}) = \frac{4}{4} = 1.0000$ ($100.0\%$)
- **Zero-Probability Transitions from `sat`:**
  $P(w \mid \text{sat}) = 0$ for all $w \neq \text{on}$.

#### 5. Conditional Distribution for `ran` (Total Context Occurrences: 2)
- $C(\text{ran}, \text{to}) = 2 \implies P(\text{to} \mid \text{ran}) = \frac{2}{2} = 1.0000$ ($100.0\%$)
- **Zero-Probability Transitions from `ran`:**
  $P(w \mid \text{ran}) = 0$ for all $w \neq \text{to}$.

---

### Part VI: Inspecting the LLM-Generated Code

#### Question 4
> **Where in the program are the transition counts stored?**

In [`first_order_model.py`](file:///C:/Users/Amithav/Desktop/AI-LAB/first_order_model.py), transition counts are stored inside the `FirstOrderAutoregressiveLM` class within the instance variable:
```python
self.transition_counts: Dict[str, Counter] = defaultdict(Counter)
```
Each outer key represents the conditioning context token $w_{prev} = X_{t-1}$, and the inner `Counter` maps each observed successor token $w_{curr} = X_t$ to its integer count $C(w_{prev}, w_{curr})$.

#### Question 5
> **Where is $P(X_t \mid X_{t-1})$ computed?**

It is computed inside the `train()` method of [`first_order_model.py`](file:///C:/Users/Amithav/Desktop/AI-LAB/first_order_model.py):
```python
for w_prev, next_counts in self.transition_counts.items():
    total_count = sum(next_counts.values())
    for w_curr, count in next_counts.items():
        self.probabilities[w_prev][w_curr] = count / total_count
```
This stores the normalized conditional probability table directly in `self.probabilities[w_prev][w_curr]`.

#### Question 6
> **How does the program choose the next word? Is it: 1. always choosing the most probable word, or 2. sampling from the probability distribution? Explain the difference.**

The program provides both mechanisms via the `generate_sentence(mode)` parameter:
1. **Mode A: Greedy Selection (Argmax / Deterministic)**
   - Implemented via `predict_most_probable(current_token)`:
     $$w^* = \arg\max_{w \in V} P(w \mid w_{prev})$$
   - It always deterministically selects the single token with highest probability. If multiple tokens share the maximum probability, a tie-breaker (alphabetical) selects one.
2. **Mode B: Stochastic Sampling (Distributional Sampling)**
   - Implemented via `sample_next_token(current_token)`:
     $$w \sim \text{Categorical}\big(P(X_t = \cdot \mid X_{t-1} = w_{prev})\big)$$
   - Uses `random.choices(candidates, weights=probabilities)` where each candidate has probability $P(w_j \mid w_i)$ of being chosen.

**Core Difference:**
- Greedy mode is completely deterministic ($0$ entropy in generation). Repeated executions from the same prefix yield the exact identical sentence. If the argmax path contains a cycle, greedy generation enters an infinite loop.
- Sampling mode is stochastic. It preserves the full distribution over valid paths licensed by the Bayesian network, producing diverse utterances across runs.

#### Question 7
> **What happens if the program encounters a word for which no transition has been observed?**

- In an uninspected, naive implementation, querying `self.probabilities[word]` for an unseen word triggers an unhandled `KeyError` (or a `ZeroDivisionError` during training), crashing the process.
- In our inspected implementation, safety checks are explicitly engineered:
  ```python
  if token not in self.probabilities or not self.probabilities[token]:
      return None
  ```
  If `next_token is None`, the generation loop terminates gracefully rather than throwing an exception.
- In modern production probabilistic systems, encountering an out-of-vocabulary or zero-transition token is handled via smoothing (add-$\alpha$ Laplace, Good-Turing, or Kneser-Ney backoff) or an `<UNK>` token distribution.

---

### Part VII: Test the Probability Model

#### Question 8
> **If one of the totals is 0.87, what does this tell you about the implementation?**

A sum of $0.87$ proves that the implementation has a **critical mathematical bug** violating Kolmogorov's Second Axiom and the Law of Total Probability:
$$\sum_{v \in V} P(v \mid w) = 1.0 \quad \forall w \in V \setminus \{\text{⟨END⟩}\}$$

A total of $0.87$ means $13\%$ of the conditional probability mass has vanished into an unrepresented event. This diagnosis indicates:
1. **Missing Outgoing Transitions:** The tokenization or counting loop dropped valid outgoing edges (for instance, dropping transitions to `⟨END⟩` or ignoring punctuation tokens);
2. **Incorrect Normalization Denominator:** The denominator was computed over an incorrect subset or corpus-wide frequency rather than $\sum_{k} C(w_i, w_k)$;
3. **Improper Truncation or Filtering:** An unnormalized top-$k$ or probability threshold cutoff was applied without re-normalizing the remaining probabilities.

Our verification suite in [`test_results.txt`](file:///C:/Users/Amithav/Desktop/AI-LAB/test_results.txt) confirms that all context rows in our models sum to exactly $1.00000000$ (tolerance $< 10^{-6}$).

---

### Part VIII: Predicting the Next Word

#### Question 9
> **Are the most probable predictions always the same as the words that you would personally expect? What does this tell you about the difference between a probability model and human linguistic expectations?**

**Empirical Observations from our Model:**
- Preceding word `the`: $\arg\max_w P(w \mid \text{the}) \in \{\text{cat}, \text{dog}\}$ ($p = 0.2500$ each).
  - However, $P(\text{mat} \mid \text{the}) = P(\text{park} \mid \text{the}) = P(\text{rug} \mid \text{the}) = 0.1667$.
- Preceding word `cat`: $\arg\max_w P(w \mid \text{cat}) = \text{sat}$ ($p = 0.6667$).
- Preceding word `dog`: $\arg\max_w P(w \mid \text{dog}) = \text{sat}$ ($p = 0.6667$).
- Preceding word `sat`: $\arg\max_w P(w \mid \text{sat}) = \text{on}$ ($p = 1.0000$).
- Preceding word `ran`: $\arg\max_w P(w \mid \text{ran}) = \text{to}$ ($p = 1.0000$).

**Divergence from Human Expectations:**
No, the model's most probable predictions do not always align with human linguistic expectations.
1. When starting a sentence with `the`, a human expects a subject noun (`cat` or `dog`), but never an inanimate target of a preposition (`mat`, `rug`, `park`). Yet the first-order model assigns $0.1667$ probability to each of these because it cannot "remember" whether `the` was preceded by `⟨START⟩`, `on`, or `to`.
2. After `sat`, a human might expect adverbs (`sat quietly`), alternative prepositions (`sat beside`, `sat near`), or intransitive verb termination. The model assigns $100\%$ probability exclusively to `on`.

**Theoretical Distinction:**
- **A Probability Model** is an empirical statistical estimator computed purely from observed sample frequencies under a rigid structural assumption (Markov factorization). It has no intrinsic model of the physical world, semantic syntax, grammar roles, or intentions.
- **Human Linguistic Expectations** are grounded in rich world knowledge, compositionality, syntax, discourse context, and pragmatic meaning.

---

### Part X: Deterministic vs Probabilistic Generation

#### Question 10
> **Compare the two sets of generated sentences. Which mode produces more variation? Why?**

From our experiments logged in [`generated_sentences.txt`](file:///C:/Users/Amithav/Desktop/AI-LAB/generated_sentences.txt):

**Mode A (Greedy Generation, 5 runs):**
- Run 1: `⟨START⟩ the cat sat on the cat sat on the cat sat on ...`
- Run 2: `⟨START⟩ the cat sat on the cat sat on the cat sat on ...`
- Run 3: `⟨START⟩ the cat sat on the cat sat on the cat sat on ...`
- Run 4: `⟨START⟩ the cat sat on the cat sat on the cat sat on ...`
- Run 5: `⟨START⟩ the cat sat on the cat sat on the cat sat on ...`

**Mode B (Sampling Generation, 5 runs):**
- Run 1: `⟨START⟩ the park ⟨END⟩`
- Run 2: `⟨START⟩ the cat ran to the rug ⟨END⟩`
- Run 3: `⟨START⟩ the park ⟨END⟩`
- Run 4: `⟨START⟩ the rug ⟨END⟩`
- Run 5: `⟨START⟩ the mat ⟨END⟩`

**Comparison & Analysis:**
- **Mode B (Sampling) produces vastly more variation.**
- **Why:** Greedy generation is a deterministic state machine: starting from state `⟨START⟩`, the transition function $w^* = \arg\max P(w \mid w_{prev})$ is fixed. Since `the` leads to `cat` (tie-break), `cat` leads to `sat` ($p=0.67$), `sat` leads to `on` ($p=1.0$), and `on` leads to `the` ($p=1.0$), the greedy decoder enters an absorbing periodic orbit (`the -> cat -> sat -> on -> the`) repeating indefinitely. Its diversity across runs is exactly $0\%$.
- In contrast, sampling mode draws randomly according to the transition distribution at every step, allowing the generator to explore all reachable branches of the Bayesian network graph.

---

### Part XI: A Second-Order Bayesian Network

#### Question 11
> **How does the second-order model differ from the first-order model in terms of:**
> 1. **the graph structure?**
> 2. **the conditional probability table?**
> 3. **the amount of context available for prediction?**
> 4. **the amount of data needed?**

| Dimension | First-Order Model ($n=1$) | Second-Order Model ($n=2$) |
| :--- | :--- | :--- |
| **1. Graph Structure** | Linear chain: $X_{t-1} \to X_t$. In-degree of each node is **1**. | Directed graph with skip connections: $X_{t-2} \to X_t \leftarrow X_{t-1}$. In-degree of each node is **2**. |
| **2. Conditional Probability Table (CPT)** | 2D Matrix of size $|V| \times |V| = 144$ entries. Each entry is $P(X_t \mid X_{t-1})$. | 3D Tensor of size $|V| \times |V| \times |V| = |V|^3 = 1,728$ entries. Each entry is $P(X_t \mid X_{t-2}, X_{t-1})$. |
| **3. Available Context** | Exactly **1** preceding token ($X_{t-1}$). Lacks memory of whether `the` followed `on`, `to`, or `⟨START⟩`. | Exactly **2** preceding tokens ($(X_{t-2}, X_{t-1})$). Context distinguishes `('on', 'the')` from `('to', 'the')` and `('⟨START⟩', 'the')`. |
| **4. Amount of Data Needed** | Scales with $O(|V|^2)$ bigrams. Requires modest corpus size to observe transitions. | Scales with $O(|V|^3)$ trigrams. Requires exponentially more data to prevent severe zero-frequency sparsity. |

---

### Part XIII: Comparing the Two Models

#### Question 12
> **Why does increasing the amount of context potentially improve prediction? Why can it simultaneously make the model harder to estimate from limited data? Relate your answer to the size of the conditional probability table.**

**1. Why Context Improves Prediction:**
Language exhibits long-range semantic and syntactic dependencies. In our dataset:
- If context is only `the`, the model cannot determine whether to emit a subject (`cat`, `dog`) or an object (`mat`, `rug`, `park`), producing fragmented nonsense like `⟨START⟩ the park ⟨END⟩`.
- In the second-order model, the context is a bigram:
  - Context `('⟨START⟩', 'the')` only allows `cat` ($0.5$) or `dog` ($0.5$).
  - Context `('on', 'the')` only allows `mat` ($0.5$) or `rug` ($0.5$).
  - Context `('to', 'the')` only allows `park` ($1.0$).
  Additional context resolves conditioning ambiguity, guaranteeing $100\%$ grammatical coherence in generated text.

**2. Why It Makes Estimation Harder (Curse of Dimensionality & Sparsity):**
The total number of parameters in the CPT grows exponentially with context length $k$:
$$\text{CPT Size} = |V|^k \times |V| = |V|^{k+1}$$
- For $|V| = 12$:
  - First-order ($k=1$): $12^2 = 144$ parameters.
  - Second-order ($k=2$): $12^3 = 1,728$ parameters.
  - Third-order ($k=3$): $12^4 = 20,736$ parameters.
- For a real-world vocabulary $|V| = 50,000$:
  - First-order: $50,000^2 = 2.5 \times 10^9$ parameters.
  - Second-order: $50,000^3 = 1.25 \times 10^{14}$ parameters (125 trillion parameters!).

Because human texts only express a minuscule fraction of all possible token combinations, the empirical counts for almost all contexts will be zero ($C(w_{t-2}, w_{t-1}) = 0$). In our lab dataset, **$89.6\%$ of all theoretical bigram contexts had zero observations**. If an unseen context appears at test time, MLE yields $\frac{0}{0}$ (undefined), causing high estimator variance and catastrophic failure without smoothing.

---

### Part XV: Reflection on the Role of the LLM

#### Question 13
> **Why is Approach B preferable when constructing an intelligent system? Discuss the importance of:**
> - **specifying the intended behaviour;**
> - **understanding the representation;**
> - **validating the generated implementation;**
> - **testing probabilistic invariants;**
> - **distinguishing implementation from model.**

- **Approach A:** *"Write a Python language model for me."*  
- **Approach B:** *"Implement the following probabilistic model: $P(X_t \mid X_{t-1})$, estimated from transition counts, with sampling-based generation."*

Approach B is vastly superior when building robust intelligent systems for the following foundational reasons:

1. **Specifying Intended Behaviour:**  
   Prompt A is underspecified. An LLM might generate a PyTorch LSTM, download a HuggingFace GPT-2 pipeline, build a character-level trigram model, or write a dummy regex. Prompt B specifies the exact mathematical objective, state space, input/output contracts, and algorithm.
2. **Understanding the Representation:**  
   In Approach B, the engineer defines the internal data structure (nested dictionaries / CPT matrices). You know precisely what the keys and values signify ($C(w_i, w_j)$ and $P(w_j \mid w_i)$), ensuring total explainability and auditability.
3. **Validating the Generated Implementation:**  
   Because Approach B specifies the target probabilistic model, the engineer can inspect the code against known theoretical expectations (checking if MLE division divides by the row sum or corpus sum).
4. **Testing Probabilistic Invariants:**  
   Approach B enables formal property-based unit testing. Probabilities must satisfy Kolmogorov's axioms: $P(v \mid w) \ge 0$ and $\sum_{v \in V} P(v \mid w) = 1.0$. You cannot write invariant unit tests for code whose underlying mathematical model you did not specify.
5. **Distinguishing Implementation from Model:**  
   The *model* is the conceptual mathematical abstraction (a first-order Markov chain over tokens). The *implementation* is the software realization (e.g., Python `defaultdict` vs. NumPy 2D array vs. PyTorch sparse tensor). Conflating the two leads engineers to blame the math when the code has a bug, or accept erroneous code because "the AI wrote it."

---

### Part XX: Final Question: What Did the Bayesian Network Add?

#### Question 14
> **What did thinking of the language model as a Bayesian network give you? Discuss at least three of the specified points.**

Conceptualizing an autoregressive language model as a Bayesian Network provides deep mathematical clarity across all seven core dimensions:

1. **A Structured Representation of Dependencies:**  
   Rather than treating text generation as an opaque "black box", the Bayesian network represents word positions as random variables $X_1, \dots, X_T$ arranged as nodes in a Directed Acyclic Graph (DAG). Directed edges explicitly visualize which words exert direct causal influence on future words.
2. **A Rigorous Factorisation of the Joint Distribution:**  
   The network structure directly encodes how the joint probability factorizes via the local Markov property:
   $$P(X_1, \dots, X_T) = \prod_{t=1}^T P(X_t \mid \text{Parents}(X_t))$$
   For the first-order model, $\text{Parents}(X_t) = \{X_{t-1}\}$. For the second-order model, $\text{Parents}(X_t) = \{X_{t-2}, X_{t-1}\}$.
3. **A Clear Way to Reason About Independence Assumptions:**  
   Bayesian network semantics (d-separation) allow us to immediately read conditional independencies off the graph. In $X_1 \to X_2 \to X_3 \to X_4$, conditioning on $X_2$ d-separates $X_1$ from $X_3$ and $X_4$. This reveals the exact trade-off: computational tractability is purchased by assuming that history older than $k$ tokens has zero influence on the next word.
4. **A Principled Method for Generation (Ancestral Forward Sampling):**  
   Text generation is revealed to be nothing other than standard **Ancestral Sampling** on a Bayesian network: sample roots from prior distributions, then traverse the topological order of the DAG, sampling each child node conditional on the observed values of its parents.
5. **A Way to Understand the Effect of Increasing Context:**  
   Adding context corresponds to adding incoming parent edges to each node $X_t$. The Bayesian network graph makes the cost of this expansion visually and mathematically obvious: each additional parent expands the conditioning set, multiplying the dimensionality of the conditional probability table by $|V|$.
6. **A Way to Test Whether an Implementation Matches its Probabilistic Specification:**  
   Because the CPTs correspond to conditional distribution tables in a Bayesian network, every row in the CPT represents a complete probability distribution over the child variable. This gives us concrete, non-negotiable verification invariants ($\sum_{v} P(v \mid \text{parents}) = 1.0$) to validate software correctness.

---

## 3. Quantitative and Qualitative Model Comparison (Part XIII)

The comparative analysis executed by [`model_comparison.py`](file:///C:/Users/Amithav/Desktop/AI-LAB/model_comparison.py) yielded the following empirical findings:

### Table 1: Structural, Parameter, and Sparsity Comparison

| Metric | First-Order LM ($n=1$) | Second-Order LM ($n=2$) | Theoretical Significance |
| :--- | :--- | :--- | :--- |
| **Conditioning Context** | $X_{t-1}$ (1 token) | $(X_{t-2}, X_{t-1})$ (2 tokens) | Context window size |
| **DAG Parent In-Degree** | $1$ parent | $2$ parents | Graphical model topology |
| **Vocabulary Size ($|V|$)** | $12$ tokens | $12$ tokens | State space per variable |
| **Total Possible Contexts** | $12$ | $144$ ($|V|^2$) | Exponential state space growth |
| **Observed Contexts** | $11$ | $15$ | Contexts observed in training corpus |
| **Zero-Probability Contexts** | $1$ ($8.3\%$) | $129$ (**$89.6\%$**) | Extreme data sparsity |
| **Total Parameter Space** | $144$ entries | $1,728$ entries ($|V|^3$) | CPT storage requirement |
| **Non-Zero Learned Parameters** | $17$ | $19$ | Active non-zero transitions |
| **Independent Free Parameters** | $6$ | $4$ | Degrees of freedom |
| **Total Matrix Sparsity** | **$88.19\%$** | **$98.90\%$** | Zero-frequency parameter fraction |

### Table 2: Generation Diversity and Coherence (100 Sampled Sentences)

| Metric | First-Order LM | Second-Order LM | Analysis |
| :--- | :--- | :--- | :--- |
| **Unique Sentences (Diversity)** | $34 / 100$ ($34\%$) | $6 / 100$ ($6\%$) | 1st-order explores more combinations |
| **Average Sentence Length** | $7.64$ tokens | $8.00$ tokens | All 2nd-order sentences match true length |
| **Sentence Length Range** | $[4, 28]$ tokens | $[8, 8]$ tokens | 1st-order produces fragments & run-ons |
| **Grammatically Coherent Matches** | $20 / 100$ ($20\%$) | **$100 / 100$ ($100\%$)** | 2nd-order achieves 100% syntactic coherence |
| **Anomalous / Fragmented Output** | **$66 / 100$ ($66\%$)** | **$0 / 100$ ($0\%$)** | 1st-order emits `<START> the mat <END>` |
| **Infinite Loops (in Greedy Mode)** | **Trapped in loop** | **Terminates cleanly** | 1st-order has absorbing cyclic path |

---

## 4. Reflection on the Role of the LLM & Code Inspection/Correction (Deliverable 7)

### Reflection on LLM Workflow
During this laboratory, the LLM was utilized as an assistive programming partner under the **Behavioral Specification Paradigm (Approach B)**. Instead of prompting with vague requests ("write a language model"), the LLM was given explicit probabilistic contracts:
- The exact state space and vocabulary tokenization rules (`⟨START⟩`, `⟨END⟩`);
- The transition counting formula $C(w_i, w_j)$;
- The exact normalization formula $P(w_j \mid w_i) = \frac{C(w_i, w_j)}{\sum_k C(w_i, w_k)}$;
- Separation of concerns between deterministic greedy decoding ($\arg\max$) and probabilistic sampling (`random.choices`).

### Concrete Example of LLM-Generated Code Inspected and Corrected

When prompting the LLM for the first-order generation logic, the initial code generated for greedy decoding was:

```python
# Initial naive LLM-generated code snippet:
def generate_greedy(self, start_token="<START>", max_tokens=50):
    curr = start_token
    words = [curr]
    while curr != "<END>":
        # BUG 1: Unhandled KeyError if curr has no transitions (e.g. <END> or unseen token)
        # BUG 2: Infinite loop trap if argmax forms a cycle
        next_word = max(self.probabilities[curr], key=self.probabilities[curr].get)
        words.append(next_word)
        curr = next_word
    return " ".join(words)
```

#### Defects Identified During Inspection:
1. **Unhandled `KeyError` on Terminal / Unseen States:**  
   If `curr` is `<END>`, or if an unknown token is passed, `self.probabilities[curr]` does not exist, raising an unhandled `KeyError` and crashing execution.
2. **Infinite While-Loop in Argmax Decoding:**  
   In our dataset, `argmax` from `the` is `cat`, `cat` is `sat`, `sat` is `on`, and `on` is `the`. This produces an inescapable four-state cycle:
   $$\text{the} \to \text{cat} \to \text{sat} \to \text{on} \to \text{the}$$
   Because `curr != "<END>"` is never satisfied, the while-loop hangs indefinitely, consuming system memory.
3. **Non-Deterministic Tie-Breaking:**  
   In `self.probabilities["the"]`, both `cat` and `dog` have equal probability ($0.2500$). Python's `max(..., key=dict.get)` arbitrarily picks whichever key appears first in hash order, making behavior dependent on internal dictionary ordering rather than an explicit, reproducible policy.

#### Corrected, Production-Grade Implementation:
We corrected this code into the robust implementation now present in [`first_order_model.py`](file:///C:/Users/Amithav/Desktop/AI-LAB/first_order_model.py):

```python
def predict_most_probable(self, token: str) -> Tuple[Optional[str], float, List[str]]:
    if token not in self.probabilities or not self.probabilities[token]:
        return None, 0.0, []
    dist = self.probabilities[token]
    max_p = max(dist.values())
    candidates = [w for w, p in dist.items() if abs(p - max_p) < 1e-9]
    candidates.sort()  # Deterministic tie-breaking
    return candidates[0], max_p, candidates

def generate_sentence(self, mode: str = "sampling", max_tokens: int = 50) -> str:
    current_token = self.start_token
    generated = [current_token]
    for _ in range(max_tokens):  # Bounded execution protects against infinite cycles
        if current_token == self.end_token:
            break
        if mode == "greedy":
            next_token, _, _ = self.predict_most_probable(current_token)
        elif mode == "sampling":
            next_token = self.sample_next_token(current_token)
        
        if next_token is None:  # Gracefully handles dead-ends / terminal tokens
            break
        generated.append(next_token)
        current_token = next_token
    return " ".join(generated)
```

This verification and correction loop demonstrated why an AI assistant cannot replace human understanding of probabilistic systems and defensive programming.

---

## 5. Connection to Modern Large Language Models (Section 17 & 18)

A modern autoregressive LLM (such as GPT-4, Gemini, or LLaMA) operates under the exact same foundational probabilistic objective:
$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, \dots, x_{t-1})$$

The crucial distinction lies not in the probability chain rule, but in **how the conditional distribution is parameterized and learned**:

| Dimension | Classical Bayesian Network / $n$-gram LM | Modern Neural Autoregressive LM (Transformers) |
| :--- | :--- | :--- |
| **Representation** | Explicit Conditional Probability Tables (CPTs) | Deep Neural Network (Multi-Head Self-Attention + MLP) |
| **Context Window** | Fixed small window ($k=1$ or $k=2$) | Massive learned context ($8\text{K} - 1\text{M}+$ tokens) |
| **Parameters** | Explicit conditional probabilities ($P(w_j \mid w_i)$) | Continuous learned weight matrices ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$) |
| **Learning Paradigm** | Maximum Likelihood Counting: $\frac{C(\text{context}, w)}{C(\text{context})}$ | Gradient descent minimizing Cross-Entropy Loss: $-\log P_\theta(x_t \mid x_{<t})$ |
| **Handling Unseen Contexts** | Zero-frequency breakdown; requires smoothing | Continuous vector embeddings and attention provide smooth semantic generalization |
| **Generation Algorithm** | Ancestral Sampling / Argmax Greedy | Ancestral Sampling with Temperature, Top-$k$, Top-$p$ (Nucleus), or Beam Search |

Even as the engineering machinery shifts from discrete count tables to billions of continuous neural network parameters, the fundamental probabilistic question remains:
$$\mathbf{P(\text{next token} \mid \text{previous tokens})}$$

---

## 6. Workspace File Index & Deliverables

All deliverables have been generated, tested, and verified in the workspace:

1. [`first_order_model.py`](file:///C:/Users/Amithav/Desktop/AI-LAB/first_order_model.py): First-order autoregressive model, CPT generation, normalization unit testing, and generation engine.
2. [`second_order_model.py`](file:///C:/Users/Amithav/Desktop/AI-LAB/second_order_model.py): Second-order autoregressive model with triple counting, bigram context conditioning, and evaluation.
3. [`model_comparison.py`](file:///C:/Users/Amithav/Desktop/AI-LAB/model_comparison.py): Automated comparative benchmark evaluating parameter count, sparsity, generation diversity, and qualitative coherence.
4. [`run_experiments.py`](file:///C:/Users/Amithav/Desktop/AI-LAB/run_experiments.py): Master test script that executes models and exports all artifact reports.
5. [`cpt_tables.txt`](file:///C:/Users/Amithav/Desktop/AI-LAB/cpt_tables.txt): Complete Conditional Probability Tables for both models with zero-probability transition identification.
6. [`test_results.txt`](file:///C:/Users/Amithav/Desktop/AI-LAB/test_results.txt): Detailed normalization verification results ($\sum_v P(v \mid \text{context}) == 1.0$) and Question 8 diagnostics.
7. [`generated_sentences.txt`](file:///C:/Users/Amithav/Desktop/AI-LAB/generated_sentences.txt): Complete corpus of generated sentences (20 sampled, 5 greedy vs 5 sampled, second-order generations, and comparisons).
