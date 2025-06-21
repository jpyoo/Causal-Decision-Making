# Causal-Decision-Making : Causal Reinforcement Learning

## Overview
This repository contains the implementation and analysis for the CS 295 Final Project, conducted by Jay Yoo during Winter 2025 at UCI. The project investigates the integration of causal inference with reinforcement learning (RL) to improve decision-making in medical treatment scenarios. It builds on frameworks by Muller and Pearl for personalized and population-based decision-making, emphasizing the role of observational data and monotonicity conditions in estimating individual treatment effects (ITE).

The project compares a standard Q-learning agent with a causally-informed RL agent, evaluating performance in terms of cumulative regret and reward across different causal response type distributions for male and female patients.

## Key Concepts

### Causal Inference
Causal inference is used to estimate the **probability of benefit** (\(P(\text{benefit})\)) and **probability of harm** (\(P(\text{harm})\)) using observational and experimental data. **Counterfactual logic** models outcomes under hypothetical interventions, denoted as \(Y_x(u) = y\) (“\(Y\) would be \(y\) had \(X\) been \(x\) for unit \(u\)”). The **Individual Treatment Effect (ITE)** is defined as:

\[
\text{ITE} = Y(1, u) - Y(0, u)
\]

The **Conditional Average Treatment Effect (CATE)** extends this to subgroups:

\[
\text{CATE}(u) = \mathbb{E}[Y(do(A=1), u') - Y(do(A=0), u') \mid C(u') = C(u)]
\]

Bounds on \(P(\text{benefit})\) are calculated using:

\[
\max\left\{0, P(y_t) - P(y_c), P(y) - P(y_c), P(y_t) - P(y)\right\} \leq P(\text{benefit}) \leq \min\left\{P(y_t), P(y'_c), P(t, y) + P(c, y)', P(y_t) - P(y_c) + P(t, y') + P(c, y)\right\}
\]

### Monotonicity
**Monotonicity** assumes no harm from treatment (\(P(\text{harm}) = 0\)), simplifying analysis as \(\text{ATE} = P(\text{benefit})\). The project tests scenarios where monotonicity holds (female patients) and where it does not (male patients).

### Reinforcement Learning
Two agents are compared:
- **Q-learning Agent (\(\mathcal{A}_Q\))**: Standard RL without causal knowledge, updated via:

\[
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
\]

- **Causal Agent (\(\mathcal{A}_C\))**: Incorporates causal knowledge via a causal graph (\(\mathcal{G}\)) and updates bounds on \(P(\text{benefit})\) and \(P(\text{harm})\). It operates in four modes:
  1. **Strict P(harm)**: True if upper bound of \(P(\text{harm})\) is zero.
  2. **Optimistic P(harm)**: True if lower bound of \(P(\text{harm})\) is zero.
  3. **Benefit-Harm Comparison**: True if upper bound of \(P(\text{benefit})\) exceeds upper bound of \(P(\text{harm})\).
  4. **Mean Benefit Comparison**: Compares mean bounds of \(P(\text{benefit})\) and \(P(\text{harm})\).

### Causal Response Types
Patients are categorized based on treatment response:
- **Always-takers**: Outcome occurs regardless of treatment (\(Y(do(A=1)) = Y(do(A=0)) = 1\)).
- **Compliers**: Positive response to treatment (\(Y(do(A=1)) = 1\), \(Y(do(A=0)) = 0\)).
- **Defiers**: Negative response to treatment (\(Y(do(A=1)) = 0\), \(Y(do(A=0)) = 1\)).
- **Never-takers**: No outcome regardless of treatment (\(Y(do(A=1)) = Y(do(A=0)) = 0\)).

Distributions vary by gender:

| Gender | Always-takers | Compliers | Defiers | Never-takers |
|--------|---------------|-----------|---------|--------------|
| Female | 0.21          | 0.28      | 0.00    | 0.51         |
| Male   | 0.00          | 0.49      | 0.21    | 0.30         |

## Experimental Setup

### Environment Model
The environment is a **Markov Decision Process (MDP)** defined by \(\langle S, A, R \rangle\), where \(S\) is the state space, \(A\) is the action space, and \(R\) is the reward function. A **Structural Causal Model (SCM)** captures unobserved confounders (e.g., causal response types) influencing decisions and outcomes.

![MDP and SCM](figures/model_representations.png)

*Figure: (Left) MDP showing states, actions, and rewards with partial randomness (dotted arrows) due to unobserved response types. (Right) SCM illustrating unobserved confounder \(U\) affecting decision \(x\) and outcome \(Y\).*

### Datasets
- **Random Sample**: 500 patients per gender, generated via `np.random.choice` to approximate response type distributions.
- **Permutations**: 4,618 permutations of 50 patients, strictly adhering to distributions (e.g., 10 Always-takers, 14 Compliers, 26 Never-takers for females). Note: A bug may limit permutation count (theoretical: ~\(2.3838 \times 10^{20}\)).

### Metrics
- **Cumulative Regret**: Binary (1 if a better action exists, 0 otherwise).
- **Cumulative Reward**: 1 for survival, -1 otherwise.
- Evaluated over 500 episodes.

## Results
- **Random Sample**:
  - The **Optimistic P(harm)** mode excelled for female patients (monotonicity holds), with lower regret and higher reward than Q-learning.
  - For male patients (non-monotonic), Optimistic mode showed mixed results, outperforming in some cases.
  - **Strict P(harm)** underperformed, while **Benefit-Harm** and **Mean Benefit** modes were similar to Q-learning.

![Regret and Reward](figures/acc_reg_rew.png)

*Figure: Accumulated regret and reward for Q-learning and causal agent modes.*

- **Permutations**:
  - Optimistic mode showed the lowest regret and highest reward across 4,618 permutations.
  - Distribution analysis revealed significant performance differences, suggesting causal knowledge improves sample efficiency.

![Permutation Results](figures/combinations.png)

*Figure: Results over all possible combinations of 50 patients.*

![Distribution](figures/distribution.png)

*Figure: Regret and reward distribution for Optimistic mode vs. Q-learning.*

## Limitations
- Permutation dataset may have a bug, generating only 4,618 permutations.
- Permutations assume equal likelihood, ignoring real-world probabilities.
- Causal agent performance needs further formal analysis to quantify improvements.

## Acknowledgments
Thanks to Professor Rina Detcher for guidance on causality in RL.

## References
- Muller, S., & Pearl, J. (2023). Personalized Decision Making with Observational Data. *arXiv preprint arXiv:2303.12692*.
- Muller, S., & Pearl, J. (2023). Monotonicity in Causal Inference. *arXiv preprint arXiv:2303.12693*.
- Li, Y., et al. (2019). Unit Selection in Causal Inference. *Journal of Causal Inference*.
- Tian, J., & Pearl, J. (2000). Probabilities of Causation: Bounds and Identification. *Annals of Mathematics and Artificial Intelligence*.
- Bubeck, S., & Cesa-Bianchi, N. (2012). Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems. *Foundations and Trends in Machine Learning*.

