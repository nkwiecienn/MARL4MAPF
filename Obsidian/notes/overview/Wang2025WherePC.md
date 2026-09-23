---
title: "Where Paths Collide: A Comprehensive Survey of Classic and Learning-Based Multi-Agent Pathfinding"
author: |
  Shiyue Wang and Haozheng Xu and Yuhan Zhang and Jingran Lin and Changhong Lu and Xiangfeng Wang and Wenhao Li
year: "2025"
bibtex_id: Wang2025WherePC
---
Project website: https://wangsh1yue.github.io/Where-Paths-Collide/

This huge paper takes 112 pages to explore over 200 other papers about Multi-Agent Path Finding. It presents a unified framework that encompasses search-based methods (including Conflict-Based Search, Priority-Based Search, and Large Neighbourhood Search), compilation-based approaches (SAT, SMT, CSP, ASP, and MIP formulations), data-driven techniques (reinforcement learning, supervised learning, and hybrid strategies), and even mixed-motive MAPF with game-theoretic considerations, language-grounded planning with large language models, and neural solver architectures that combine the rigour of classical methods with the flexibility of deep learning.

*I would also like to note that the section above was copied directly from paper's abstract and sounds very chat-generated. This paper isn't actually published yet and I viewed it in preprint while it's under review.*

Overall I didn't like this work and I wouldn't recommend anyone reading it all but if I ever need a reference to virtually any algorithm, then it's here. There is a slight chance that I didn't like reading it because I don't believe any sane person would write so much about this subject.

Although it has an entire chapter on RL it doesn't *really* explore it.
## Categories of MAPF Approaches

1. **Search-Based Methods**: Classical graph search and tree-based algorithms that explicitly enumerate or prune the space of collision-free paths. They often guarantee completeness or optimality under certain assumptions but may struggle with large-scale instances.
2. **Compilation-Based Methods**: Formulate MAPF as an Integer Linear Program (ILP), a Satisfiability (SAT) problem, or other well-studied optimization frameworks. These methods exploit powerful generic solvers but may also face scalability issues or long solve times.
3. **Learning-Based Methods**: Leverage diverse machine learning paradigms, such as RL, imitation learning, and evolutionary algorithms. While these can more readily adapt to uncertain or partially observable environments, they frequently handle fewer agents compared to large-scale classical approaches.
4. **Hybrid Methods**: Integrate learning components (e.g., learned heuristics or policies) into a classical MAPF pipeline to balance performance gains from learning with analytical guarantees from traditional solvers.

## Learning-based methods

1. Flowchart for solving MAPF with RL

![[RL flowchart.png]]

2. **Observation spaces** - crucial decision; might include not only current static information but also path guidance, proximity (e.g., Manhattan distance to goal), or more dynamic neighbour information such as velocity.

| Cathegory                | Information examples                                      | Remarks                                                                                                                                                                   |
| ------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| *Local static info*      | Map obstacles, free cells                                 | Provides a local occupancy grid around the agent, enabling immediate collision detection or obstacle avoidance                                                            |
| *Expert path guidance*   | Recommended route from a classi- cal solver               | Helps agents avoid getting lost in large or complex maps by providing explicit routing suggestions; drastically reduces RL sample complexity and training time            |
| *Dynamic neighbour info* | Positions, veloci- ties, or trajectories of nearby agents | Enhances multi-agent collision avoidance by providing situational awareness of other agents in dense or dynamic settings                                                  |
| *Heuristic embedding*    | Distance-to-goal, action feasibility, direction hints     | Gives agents a flexible sense of goal orientation (e.g., whether moving up brings them closer to the goal), combining classical heuristics with RL’s adaptive exploration |
3. **Action space** - the only real decision point here is if collisions are forbidden or allowed but penalised via large negative reward.
4. **Reward design**:
	- *Goal-reaching* - A common approach is to give a large positive reward only when the agent arrives at its target. Some variants also provide a small shaping reward for moving closer to the goal each step, improving learning speed but risking unintended local optima. 
	- *Cooperation* - A purely local or per-agent reward may lead to greedy strategies. By adding a group-oriented term r_team, methods encourage agents to coordinate, reducing deadlocks or cycles. 
	- *Expert-guided* - For difficult or sparse environments, referencing a path from classical planners (e.g., A*) significantly reduces RL training time. Approaches differ in how strictly they guide: some apply partial or decreasing weighting of the expert path. 
	- *Collision penalty* - Typically, collisions incur a large negative reward to override other incentives. Alternatively, collisions may terminate the episode for the colliding agents, which also conveys a strong penalty signal.
5. **Communication protocol** - how partial observations are shared among agents to improve collective decision-making:
	- *Non-communication* - Approaches with no inter-agent communication are typically easier to scale to many agents and ensure faster training, but they may lead to more collisions in dense areas.
	- *Basic communication* - Agents may broadcast local states or partial observations to all neighbors within a certain radius. Social conventions (e.g., “move right if in conflict”) can emerge, but the overhead of repeated broadcasts can be high.
	- *Priority-based communication* - By allowing each agent to communicate only with those neighbors deemed “most critical,” networks avoid saturating communications. Various metrics (e.g., distance, possible collisions in the next steps) can establish these priorities.
	- *Attention-based communication* - Derived from modern deep learning architectures like Transformers, attention weighting helps each agent filter crucial messages. This is particularly helpful in scenarios with many neighbours.
	- *Request-response communication* - Agents solicit updates from others only when critical. This specialised approach can greatly reduce bandwidth consumption while preserving coordination, though it often requires more intricate logic at each agent.
6. **MARL Algorithms for MAPF**:
	- *Independent Learning* - Each agent i runs a single-agent RL algorithm (e.g., DQN, PPO) treating all other agents as part of the environment. Though simple to implement, independent learners may converge slowly or fail to coordinate in dense MAPF scenarios.
	- *Centralised Training, Decentralised Execution* - During training, a centralised critic has access to the global state, the actions of all agents, and possibly their IDs or goals. Once training is done, each agent acts with its own decentralised policy that conditions only on its own observation (e.g., MADDPG), so the critic is not needed at execution. This improves coordination while preserving the decentralised execution that MAPF needs.
	- *Value Decomposition Methods* - When all agents share one team reward, VDN and QMIX factorise the global action-value function into per-agent utilities Q_i, so training is feasible even with a single shared objective and each agent can still pick its action from its own Q_i. VDN simply sums them (Q = ΣQ_i), while QMIX combines them with a monotonic mixing network conditioned on the global state (raising any agent's Q_i never lowers the team value). Relevant when MAPF optimises a global criterion (sum-of-costs or makespan); they can converge faster to coordinated solutions than independent learners.

## Other methods

If for whatever reason I'll need to use classical methods, this work describes them well or at least using many words because I didn't feel like reading it all:
- Conflict Based Search and its variants - state-of-the-art probably when it comes to classical methods;
- Priority-Based Search - described in [[Xu2022MultiGoalMP]] but here has a better picture:
![[Pasted image 20260914141908.png]]
- Some compilation-based methods such as SAT, MIP, Branch-and-Cut-and-Price etc., maybe they could be useful if I decided to do expert based training but during execution they're probably too slow - but it's only my assumption since I didn't read it...
- Then it mentions Learning-Augmented methods but I don't care that much because I would prefer it the other way around (Classical-Methods-Augmented in Learning Methods);
- It comes back to Monte Carlo Tree Search, Supervised Learning, Curriculum Learning, and Evolutionary methods but it only touches the basics;
- Most of the work is about combining classical and learning methods.

## Experiments and evaluation
CBS and CBS variants and PRIMAL are algorithms most compared to in benchmarks