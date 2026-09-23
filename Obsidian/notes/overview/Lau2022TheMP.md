---
title: "The Multi-Agent Pickup and Delivery Problem: MAPF, MARL and Its Warehouse Applications"
author: |
  Tim Tsz-Kit Lau and Biswa Sengupta
year: "2022"
bibtex_id: Lau2022TheMP
---
This work studies two state-of-the-art solution to MAPD: conflict based search (CBS) which claims to be the optimal MAPF algorith and shared experience actor-critic (SEAC) which is a RL algorithm. Overall pretty boring but has an interesting idea of using exact MAPF solutions as expert data to construct imitation learning. Which they didn't do but some have:
- **(Multi-agent) imitation learning** — learn a policy from expert demonstrations:
	- [[PRIMAL Pathfinding via Reinforcement and Imitation Multi-Agent Learning.pdf]]
	- J. Ho and S. Ermon, "Generative adversarial imitation learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2016.
	- A. T. Lin, M. J. Debord, K. Estabridis, G. Hewer, G. Montufar, and S. Osher, "Decentralized multi-agents by imitation of a centralized controller," in *Proceedings of the Annual Conference on Mathematical and Scientific Machine Learning*, 2021.
	- J. Song, H. Ren, D. Sadigh, and S. Ermon, "Multi-agent generative adversarial imitation learning," *arXiv preprint arXiv:1807.09936*, 2018.
	- H. Wang, L. Yu, Z. Cao, and S. Ermon, "Multi-agent imitation learning with copulas," in *Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-KDD)*, 2021.
- **(Multi-agent) inverse reinforcement learning** — learn the reward function from expert demonstrations (when rewards are hard to design or agents can't see each other's rewards/goals):
	- A. Filos, C. Lyle, Y. Gal, S. Levine, N. Jaques, and G. Farquhar, "PsiPhi-learning: Reinforcement learning with demonstrations using successor features and inverse temporal difference learning," in *Proceedings of the International Conference on Machine Learning (ICML)*, 2021.
	- L. Yu, J. Song, and S. Ermon, "Multi-agent adversarial inverse reinforcement learning," in *Proceedings of the International Conference on Machine Learning (ICML)*, 2019.
Another insight is that they used RWARE - I might consider switching to it from POGEMA.
## Types of Lifelong MAPF

1. A lifelong MAPF problem is decomposed into a sequence of single-shot MAPF instances where all agents perform path replanning at every step.
2. A lifelong MAPF problem is decomposed into a sequence of single-shot MAPF instances where path replanning is performed only for agents that have just picked up or delivered their items ([[Ma2017LifelongMP]]).
3. A lifelong MAPF problem is solved as a whole in an offline setting, as reductions to other well-studied combinatorial problems such as an answer set programming problem. (Van Nguyen 2017, *Generalized target assignment and path finding using answer set programming*).

## Shared experience actor-critic

1. Policy gradient algorithm - the model has to learn the optimal policy; usually with a neural network.
2. Actor-critic algorithms - Monte Carlo returns are estimated to reduce variance; an actor estimates a policy and critic judges how good it is.
3. Shared experience actor-critic - experience is shared among agents so that they can learn from one another; they can see past trajectories of other agents.
4. Centralised Training with Decentralised Execution - all agents can access data from all other agents during training but not at execution time.

## Numerical experiments

1. RWARE as the environment!!
2. Authors claim to have 540GB of RAM but still use Intel Xeon
3. CBS claims to be the best but cannot scale over 10 agents - while SEAC can if you own 540GB of RAM and GPU Tesla V100
4. Overall conclusion is that CBS is better but doesn't scale while the authors note that even they aren't rich enough to train SEAC for too long.
5. A crucial and valuable research direction to pursue is to use lifelong MAPF methods to improve the sample efficiency of MARL algorithms for solving the MAPD problem. For instance, the solutions based on lifelong MAPF solvers can be used as expert demonstration data to derive a policy - (multi-agent) imitation learning. Furthermore, when the reward functions are hard to design or each agent has no access to the rewards or goals of other agents, we can use expert demonstration data to learn the rewards - (multi-agent) inverse reinforcement learning.