# jaisalab

jaisalab is a framework for developing and evaluating reinforcement learning algorithms that take into account constraints that must be satisfied in OpenAI gym environments, that also provides PyTorch implementations for a few state-of-the-art algorithms in Constrained RL. 
Namely, the algorithms implemented are [Constrained Policy Optimization (CPO)](https://github.com/jaimesabalimperial/jaisalab/blob/master/jaisalab/algos/cpo.py), the [Safety Augmented RL (SAUTE)](https://github.com/jaimesabalimperial/jaisalab/blob/master/jaisalab/envs/saute_env.py) wrapper that incorporates safety to any RL algorithm in a plug-n-play manner, and **Distributional Constrained Policy Optimization**, a modification to CPO that exploits uncertainty in the cost value function to better satisfy constraints. 

The framework builds on the [garage](https://github.com/rlworkgroup/garage) toolkit to add functionality to safe RL settings (i.e. where there is a cost associated with each state-action pair on top of the reward) and thus provides an auxiliary set of modular tools for implementing constrained RL algorithms, analogously to how **garage** provides these for regular RL settings. 
