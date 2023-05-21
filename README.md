# Markov decision process using Dynamic Programming

## Introduction
In this project, we will implement dynamic programming algorithms to solve a Markov decision process (MDP). We will use the MDP to model a grid world environment, and we will use dynamic programming to find the optimal policy for the agent. We will implement the value iteration and Q-value iteration algorithms. We will then execute the optimal policy on the environment, and visualize the agent’s trajectory.

The goal of the agent is to find the goal state in the environment. The agent can move in four directions: up, down, left, and right. The agent can hold keys, and can open doors. The agent can only move into a door if it holds the corresponding key. The agent can only hold one key at a time. The agent receives a reward (or penalty) of $−1$ for every transition, except when it reaches the goal: the reward is then equal to 10 times the numeric element at the goal. So, in the example below, the reward at the goal is equal to 30.

## World
The environment is coded in world.py. It contains the class definition of `World(filename)`, which will initialize the environment specified in the text file `filename`. The environment is essentially a 2D grid. A `World` object has the following attributes:
- `states`, a list of all states. When a map is initialized, all possible configurations of agent location and key possession are automatically inferred, and each possible combination is assigned a unique state index.
- `n_states`, the total number of states (a scalar).
- `actions`, a list of all possible actions.
- `n_actions`, the total number of actions (a scalar).
- `terminal`, indicates whether the agent has reached a goal (task terminates).

### prison.txt
You can define the environment in a txt file. An example is provided in prison.txt:

    # # # # # #
    # *   A   # 
    # a   #   #
    # # # #   # 
    #         #
    #   # # B #
    # b # # 3 #
    # # # # # #

We use the following encoding:
|Symbol | Meaning
|-------|--------|
|`⋆`              | Agent location
|`#`             | Wall
|`a` (lower case) | Key
|`A` (upper case) | Door
|`1` (numeric)    | Goal (terminates episode).

## Running the code
Be sure to have the NumPy installed. If you don't have it, you can install it with the following command:

    pip install numpy

To run the code, you can use the following command:
    
    python main.py

# Markov Decision Process 
A Markov decision process (MDP) is a discrete-time stochastic control process. It provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker. MDPs are useful for studying optimization problems solved via dynamic programming and reinforcement learning. The goal in a MDP is to find an optimal policy $\pi$ for the agent.

## Properties
Our implicit MDP model is defined by the following properties:
- State space $S$: The state is represented as an index. For the provided example, there are 64 unique states, since there are 16 free locations, and two keys. In each location, we can hold or not hold either key, which gives rise to $16 \cdot 2 \cdot 2$ possible states. These are simply numbered 0-63.
- Action space $A$: In every state, the agent has four possible actions: `{up, down, left, right}`.
- Dynamics: When the agent moves into a wall, it just remains at the same position. The agent automatically picks up a key when stepping on the specific location, and automatically opens the door when stepping on it while holding the specific key.
- Reward: The reward at every transition is −1, except when we reach a goal: the reward is then equal to 10 times the numeric element at the goal. So, in the example above, the reward at the goal is equal to 30.
- Gamma: We assume $\gamma = 1.0$ throughout the experiments.

## Value Iteration
Value Iteration is an algorithm that computes the optimal value function $V^*$ by iteratively improving the estimate of $V^*$. For this project both value iteration and Q-value iteration are implemented. 

Both approaches are based on the Bellman equation, which is a recursive equation that decomposes the value of a state into the immediate reward and the discounted value of the next state. The Bellman equation is defined as follows:

$$V(s)=\max_{a \in A}\Big[\sum_{s' \in S}T(s'|s,a)[r+\gamma\cdot V(s')\Big]$$

Note that:
- $a \in A$ is action $a$ in the action space $A$.
- $s \in S$ is state $s$ in the state space $S$.
- $V(s)$ is the value of state $s$.
- $V(s)$ is a function of values at next states $s’$ weighted by the probability of reaching those states from $s$.
- $T(s’|s,a)$ is the transition function that tells us how the environment changes from state $s$ to state $s’$ by taking action $a$.
- $r$ is the reward for taking action $a$ in state $s$.
- The value of the next state $s’$, i.e. $V(s')$, is discounted by $\gamma$. 
- The value of the next state is the sum of the rewards and the discounted value of the next state.

# Dynamic Programming

## Dynamic Programming class
A `DynamicProgramming` object has two important attributes:
- `V_s`, a value table. A value table is vector of length n_states. Each element in the vector stores the value estimate for the corresponding state index, i.e. `V (s = 4) = V_s[4]`. If `V_s = None`, then you have not run any method yet to estimate the optimal value table.
- `Q_sa`, a state-action value table. A state-action value matrix of dimensions $n\_states$ x $n\_actions$. Actions are indexed according to `World.actions = {up,down,left,right}`. For example, action up has index 0. Each element in the Q_sa matrix stores the value estimate for the corresponding state-action, i.e., `Q(s = 10,a = 0) = Q_sa[10,0]`. If `Q_sa = None`, then you have not run any method yet to estimate the optimal value table.

An World object has several important methods:
- `value_iteration(self,env,gamma=1.0,theta=0.001)` should run value iteration on the environment env (of class World). You should implement this function your- self. Gamma is the discount factor, which you can leave at the default value of $1.0$. Theta is the threshold for convergence, which you can also leave at the default value of $0.001$.
- `Q_value_iteration(self,env,gamma=1.0,theta=0.001)` runs Q-value iteration on the environment `env` (of class World).
- `execute_policy(self,env)` executes a policy on environment env.

## Algorithms
The `DynamicProgramming` class implements two dynamic programming algorithms, namely `value_iteration()` and `Q_value_iteration()`.

The difference between value iteration and Q-value iteration is that value iteration directly considers the value of the next state, whereas Q-value iteration considers the value of the next state-action pair.

### Value Iteration
The value iteration function $V(s)$ is defined as the expected return of taking an action $a$ in state $s$ and following policy $\pi$.
$$V(s)=\sum_{a \in A}^{} \pi(a|s)\Big[\sum_{s' \in S}T(s'|s,a)[r+\gamma\cdot V(s')\Big]$$

Where $π(a|s)$ is the policy. It gives the conditional probabality of action $a$ given the state $s$.

The value iteration algorithm is implemented in the `value_iteration(self,env,gamma=1.0,theta=0.001)` method of the `DynamicProgramming` class. The algorithm terminates when the difference between the old and new value table is smaller than theta. The algorithm returns the optimal value table. The optimal value table is be stored in the `V_s` attribute of the `DynamicProgramming` object.

### Q-value Iteration
The Q-value function $Q(s,a)$ is defined as the expected return of taking an action $a$ in state $s$ and following policy $\pi$.
$$Q(s,a)=\sum_{s' \in S}T(s'|s,a)[r+\gamma\cdot V(s')]$$

The Q-value iteration algorithm is implemented in the `Q_value_iteration(self,env,gamma=1.0,theta=0.001)` method of the `DynamicProgramming` class. The algorithm terminates when the difference between the old and new Q-value table is smaller than theta. The algorithm returns the optimal Q-value table. The optimal Q-value table is stored in the `Q_sa` attribute of the `DynamicProgramming` object.

### Policy Execution

The `execute_policy(self,env)` method of the `DynamicProgramming` class executes a policy on environment env. The policy is defined by the optimal value table `V_s` or the optimal Q-value table `Q_sa`. The method returns the total reward obtained by executing the policy. The method also prints the sequence of actions taken by the agent.
