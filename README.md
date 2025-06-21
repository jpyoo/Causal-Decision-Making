# CS 295 Final Project: Causal Reinforcement Learning

Winter 2025  
February 28, 2025

## Summaries
Decision making is an important field of study in many applications, including medical, political, and financial fields. Muller and Pearl introduced personal and population-based decision-making frameworks to improve the analysis of treatment effects, therefore enhancing the benefits of the decision-making process [mueller2023personalized]. Utilizing the language of causality [li2019unit], the study reveals that experimental results may lack consideration of unobserved confounding features.

Surprisingly, additional observational information enhances individual decision-making, measured as individual treatment effect (ITE), under the absence of a condition called monotonicity. This proves that the average treatment effect (ATE) and conditional treatment effect (CATE) are suboptimal under some conditions.

The monotonicity condition holds when the probability of harm is zero. Through semi-qualitative and numerical examples, the paper shows how ITE and CATE may differ under different monotonicity conditions, and how the lower bound and upper bound can be calculated utilizing observational and experimental results in the language of causality.

The monotonicity, probability of harm, and other necessary terms will be defined in the Analysis section and further explored with a numerical example. In this review, we will only consider the numerical example. Through the analysis, we will find that population data are capable of providing decisive information on individual response, and that nonexperimental data can add information (regarding individual response) beyond that provided by a randomized controlled trial (RCT) alone. We also find that in certain populations, we can guarantee risk-free benefits.

Muller and Pearl further investigate the monotonicity condition and find graphical structures that can benefit from unobserved conditions, which narrow the probability of harm's upper bound and lower bound [mueller2023monotonicity]. Interactive plots are introduced to assist intuitive understanding of the monotonicity condition. These plots visualize the regions of sufficiency and necessity of monotonicity.

Due to the limited scope of this project, the work mainly focuses on the first work [mueller2023personalized].

## Analysis

### Preliminaries

#### Counterfactual Logic
Counterfactual logic provides a framework for reasoning about what would happen under different interventions. The basic counterfactual statement is denoted by $Y_x(u) = y$, and stands for: “`Y` would be `y` had `X` been `x` in unit `U = u`.” The paper, for the purpose of analysis, denotes $y_t$ as recovery among the RCT treatment group and $y_c$ as recovery among the RCT control group. Similarly, it denotes `P(y|t,Gender)` and `P(y|c,Gender)` as recovery among the drug choosers and recovery among the drug avoiders, respectively, from the observational study.

#### Individual Treatment Effect (ITE) and Conditional Average Treatment Effect (CATE)
The ITE measures the causal effect of a treatment on a single individual, defined as the difference in outcomes with and without the treatment. Formally, for an individual `u`:

```math
ITE = Y(1, u) - Y(0, u)
```

The CATE extends this concept to groups defined by covariates, providing an average treatment effect within a specific subgroup:

```math
CATE(u) = E[Y(do(A=1), u') - Y(do(A=0), u') | C(u') = C(u)]
```

#### Probability of Harm, Probability of Benefit
The probability of benefit, `P(benefit)`, quantifies the likelihood that an individual will experience a positive outcome as a result of the treatment. Conversely, the probability of harm, `P(harm)`, represents the likelihood of a negative outcome due to the treatment. Formally,

```math
P(benefit) = Y(y_t, y'_c), \quad \text{and} \quad P(harm) = Y(y'_t, y_c) = 1 - P(benefit)
```

These probabilities can be bounded using observational and experimental data as follows [tian2000probabilities]:

```math
\max\left\{
\begin{array}{l}
0, \\
P(y_t) - P(y_c), \\
P(y) - P(y_c), \\
P(y_t) - P(y)
\end{array}
\right\}
\leq P(\text{benefit}) \leq
\min\left\{
\begin{array}{l}
P(y_t), \\
P(y'_c), \\
P(t, y) + P(c, y)', \\
P(y_t) - P(y_c) + P(t, y') + P(c, y)
\end{array}
\right\}
```

### Monotonicity
Monotonicity is a key concept presented in this paper to demonstrate different conditions in causal inference, stating that the treatment does not cause harm to any individual. This is expressed as `P(harm) = 0`. Under monotonicity, the Average Treatment Effect (ATE) coincides with the probability of benefit, i.e., `ATE = P(benefit)`. This relationship simplifies the estimation of treatment effects, as it eliminates the probability of harm from the analysis.

### Numerical Example
![Observational and Experimental Data Tables](figures/table_2_3.png)

**Figure: Observational and Experimental data tables from the original paper.**  
*Label: fig:data*

Consider a study with both observational and experimental data available for males and females. The observational study revealed that only 70% of men and 70% of women actually chose to take the drug. The bounds on the probability of benefit are calculated, using the above equation and data from the figure, as:

For females:

```math
0.279 \leq P(\text{benefit}|\text{female}) \leq 0.279
```

For males:

```math
0.49 \leq P(\text{benefit}|\text{male}) \leq 0.49
```

With the absence of experimental data, the bounds widen to:

```math
0.0 \leq P(\text{benefit}|\text{female}) \leq 0.279, \quad 0.0 \leq P(\text{benefit}|\text{male}) \leq 0.58
```

Similarly, without the observational data, the bounds widen to:

```math
0.279 \leq P(\text{benefit}|\text{female}) \leq 0.489, \quad 0.28 \leq P(\text{benefit}|\text{male}) \leq 0.49
```

Proving the usefulness of observational data in analysis. Under monotonicity, the CATE serves as a point estimate for `P(benefit)`. For instance:

```math
CATE(\text{female}) = 0.279, \quad CATE(\text{male}) = 0.28
```

The probabilities of harm are then:

```math
P(\text{harm}|\text{female}) = P(\text{benefit}|\text{female}) - CATE(\text{female}) = 0
```

```math
P(\text{harm}|\text{male}) = P(\text{benefit}|\text{male}) - CATE(\text{male}) = 0.21
```

## Further Investigation
I was curious if such causal information can be incorporated with an online reinforcement learning (RL) framework and benefit the learning process. If there exists a strategy that can detect the underlying `P(benefit)` and `P(harm)` from observation and experimental data during training, there must exist a strategy that exploits the extra information that outperforms the conventional RL agent.

The same experimental and observational setup as the original paper were implemented, where monotonicity holds when the patient is a woman, and doesn't hold given `u = men`.

I investigated whether prior observational knowledge can accelerate online learning in reinforcement learning settings while reducing regret by comparing traditional Q-learning against a causally-informed agent to evaluate the benefits of causal knowledge in sample efficiency and learning performance.

### Environment Model
The environment is formalized as a Markov Decision Process (MDP) `M` characterized by the tuple `<S, A, R>` where: `S` is the state space, `A` is the action space, and $R: S \times A \times S \to \mathbb{R}$ is the reward function.

![Model Representations](figures/model_representations.png)

**Figure: Model representations: (a) MDP showing states, actions, and rewards with partial randomness due to unobserved Causal Response Types. (b) SCM illustrating the influence of unobserved confounder U on decision x and outcome Y.**  
*Label: fig:model_representations*

In the Markov Decision Process (MDP) representation, $S_t$ represents the initial state, while $S_c$ denotes the state after taking a corresponding action `a` from the set of possible actions `A`. The squares in the diagram represent the resulting rewards. It is important to note that due to the unobserved Causal Response Types of individuals, the MDP introduces partial randomness, which is represented using dotted line arrows.

We can also represent this environmental model using a Structural Causal Model (SCM). In this representation, the distribution of unobserved Causal Response Types, `U`, acts as an unobserved confounder. This confounder influences both the causal model's decision `x` (which belongs to the set `X`) and the outcome `Y`. The SCM representation provides an intuitive and definitive explanation for calculating `P(benefit)` and `P(harm)` using observational and experimental data. However, in the original submission, this representation was not explicitly introduced because the bounds of these probabilities are formally defined and can be directly calculated from the data using the provided equations.

### Q-Learning Agent
The standard Q-learning agent, denoted as $\mathcal{A}_Q$, learns without observational knowledge. It is characterized by $<S, A, Q, \alpha, \gamma, \epsilon>$ where $\alpha$ is the learning rate, $\gamma$ is a discount factor, and $\epsilon$ is the exploration parameter for the $\epsilon$-greedy policy. The Q-learning update rule is:

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
```

### Causal Agent
The causal agent, denoted as $\mathcal{A}_C$, incorporates prior observational knowledge in the form of causal relationships. It is characterized by $<S, A, \epsilon, \mathcal{G}>$ where:

- $S, A, \epsilon$ are defined as in the Q-learning agent
- $\mathcal{G}$ is a causal graph representing the agent's prior knowledge about causal relationships in the environment

The lower-bound and the upper-bound of the `P(benefit)` are updated at each iteration of the experiment, using the probability bounds equation. The agent operates in four modes: **(0)** Strict P(harm), which returns true if the upper bound of the P(harm) is zero; **(1)** Optimistic P(harm), returning true if the lower bound of the P(harm) is zero; **(2)** Benefit-Harm Comparison, which returns true if the upper bound of P(benefit) exceeds the upper bound of the P(harm); and **(3)** Mean Benefit Comparison, which compares the mean of P(benefit) bounds with the mean of P(harm) bounds.

The causal agent maintains an experience table that tracks state-action-reward transitions, denoted as `exp table`, where:

```math
Y_t = \frac{\text{exp table}[s][a][r]}{\sum_{r} \text{exp table}[s][a][r]}
```

represents the empirical probability of receiving reward `r` after taking action `a` in state `s`.

### Causal Response Types
We define the following response types for patients:

For a binary action $A \in \{0, 1\}$ and its effect on an outcome variable `Y`, we define four mutually exclusive response types:

- **Compliers**: Units where `Y(do(A=1)) = 1` and `Y(do(A=0)) = 0` (outcome responds positively to action).
- **Always-takers**: Units where `Y(do(A=1)) = Y(do(A=0)) = 1` (outcome occurs regardless of action, no causal effect).
- **Never-takers**: Units where `Y(do(A=1)) = Y(do(A=0)) = 0` (outcome never occurs, no causal effect).
- **Defiers**: Units where `Y(do(A=1)) = 0` and `Y(do(A=0)) = 1` (action has opposite effect from expected).

Here, `Y(do(A=a))` represents the potential outcome of `Y` under the intervention that sets `A` to value `a` [li2019unit]. In our experimental setup, we establish different distributions of causal response types based on gender, as introduced in the paper.

| Gender | Always-takers | Compliers | Defiers | Never-takers |
|--------|---------------|-----------|---------|--------------|
| Female | 0.21          | 0.28      | 0.00    | 0.51         |
| Male   | 0.00          | 0.49      | 0.21    | 0.30         |

**Table: Distribution of Causal Response Types by Gender**  
*Label: tab:response_types*

### Experimental Data
We have two distinct sets of experimental data:

1. **Random Sample of 500 Patients**  
   This dataset is generated using a numpy random selection function (`np.random.choice`) to draw 500 patients. This method softly enforces the distribution of Causal Response Types, allowing for some natural variation in the sample.

2. **Multinomial Permutation of 50 Patients**  
   This dataset consists of 4,618 distinct permutations of 50 patients, strictly adhering to the predefined distribution of causal response types. For example, in the female group, it includes exactly 10 Always-takers, 14 Compliers, and 26 Never-takers.

However, two limitations exist in this second dataset:

- There appears to be a discrepancy in the number of permutations. Theoretically, there should be approximately $2.3838 \times 10^{20}$ possible permutations, yet only 4,618 distinct permutations were explored. This suggests a potential bug in the permutation generation process.
- The dataset doesn't account for the probability of experiencing each permutation. It treats all permutations as equally likely, which isn't realistic. For instance, the probability of encountering 26 consecutive Never-takers, followed by 14 Compliers, and then 10 Always-takers is considered the same as the most probable permutation, which doesn't reflect real-world likelihood.

### Experimental Result
To evaluate the impact of observational knowledge, I compared the two types of agents in the same condition. The empirical and numerical results of the Q-learning agent and the four modes of the causal agents are explored.

We compare the agents using the following metrics: (1) Cumulative regret over 500 episodes, (2) cumulative reward over 500 episodes, and (3) empirical result over all possible combinations of 50 patients.

The regret function, in this experiment, is a function with a binary outcome: 1 if there exists another action which could have resulted in better reward, and 0 otherwise [bubeck2012regret]. The reward is set as 1 if the patient survives after the agent's action, and -1 otherwise.

![Accumulated Regrets and Rewards by Agents](figures/acc_reg_rew.png)

**Figure: Accumulated Regrets and Rewards by Agents**  
*Label: fig:acc_reg_rew*

The resulting plot shows accumulated regret and reward for each agent, when response types of patients are drawn following the pre-defined distribution. Interestingly, the strict model (mode 0) had worse performance in all measures, while the Optimistic model (mode 1) had better performance with female subjects, but higher regret and lower reward in male performance. The mean-approached agents (mode 2 and 3) appear to perform in a similar manner as the Q-Learning agent.

Further investigation was made over all possible combinations following the pre-defined response type distribution for 50 patients; a total of 4,618 combinations are explored. The optimistic causal agent has shown the best performance in this experiment, showing an overall lower regret and higher reward in both male and female subjects.

![Result over All Possible Combinations](figures/combinations.png)

**Figure: Result over All Possible Combinations**  
*Label: fig:box-plot*

The empirical results within the design of the experiment didn't show strong improvement from the causal agents. However, the underlying distribution shows a significant difference in performance.

![Mode 1 vs Q Learning Agent Regret and Reward Distribution](figures/distribution.png)

**Figure: Mode 1 vs Q Learning Agent Regret and Reward Distribution**  
*Label: fig:distribution*

From the results, it seems to require a more formal investigation of each strategy's expected reward and regret calculation, or more empirical experiments to explore differences in each agent's performance.
```
