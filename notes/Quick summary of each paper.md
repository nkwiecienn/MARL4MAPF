## (MA)RL for Vehicle Routing

1. [[A Flexible Vehicle Routing Reinforcement Learning Environment for the Reusability of Trained Agents.pdf]]
The general idea is that for RL the learning environment has to be the same as the real one - the problem is that it's not usually the case for CVRP. The authors used masking to solve this issue - the model was taught on a bigger environment and when it encounters a smaller one it masks the actions that would lead to visiting remaining nodes.

The only interesting part was finishing conditions:

a) *Increasing method: While using this condition, a minimum percentage as well as an increasing percentage and rate need to be established. Doing so, at the beginning of the learning process only a fraction of the total nodes will have to be visited, which will depend on the minimum percentage previously mentioned. As the training continues, said percentage will gradually increase, up to a 100%, according to the increase rate.*

b) *Decreasing method: This condition works similarly to the previous one. In this case, the minimum percentage of visited nodes at the beginning of the training will be 100%, which will not decrease for several timesteps. After those steps, if the agent is unable to find a solution, the minimum percentage will decrease, down to a certain limit.*

Tested with PPO.

2. [[A-Multi-Agent-Reinforcement-Learning-Method-With-Route-Recorders-for-Vehicle-Routing-in-Supply.pdf]]
VRP with Time-Windows or MVRPSTW in short; the authors made a really nice architecture image:

![[Route recorder architecture.png]]

which represents even nicer ideas:
- an Encoder is used to learn a representation of the customer nodes in the problem;
- a Route Recorder is used to share the route history of each vehicle; adding route recorders could enable agents make better decisions based on historical information;
- a Decoder generates route sequence for each vehicle sequentially; During the decoding process, instead of giving the next nodes to all vehicles simultaneously, the authors generate the next node in the route for each vehicle one by one to avoid the situation where more than one vehicles select the same node at the same time;
- the policy is learnt with policy gradient method.
Which is all very interesting but it is a VRP problem nonetheless. 

3. [[ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS.pdf]]
4. [[Fair Collaborative Vehicle Routing A Deep Multi-Agent Reinforcement Learning Approach.pdf]]
5. [[Multi-Vehicle Routing Problems with Soft Time Windows A Multi-Agent Reinforcement Learning Approach.pdf]]
6. [[Online Vehicle Routing With Neural Combinatorial Optimization and Deep Reinforcement Learning.pdf]]
7. [[Reinforcement Learning for Solving the Vehicle Routing Problem.pdf]]
## Warehouses
1. [[Anti-conflict AGV path planning in automated container terminals based on multi-agent reinforcement learning.pdf]]
2. [[Learning to Solve the Min-Max Mixed-Shelves Picker-Routing Problem via Hierarchical and Parallel Decoding.pdf]]
3. [[Lifelong Multi-Agent Path Finding for Online Pickup and Delivery Tasks.pdf]]
4. [[Lifelong Multi-Agent Path Finding in Large-Scale Warehouses.pdf]]
5. [[Multi-UAV_Path_Planning_for_Wireless_Data_Harvesting_With_Deep_Reinforcement_Learning.pdf]]
6. [[Order picker routing in warehouses A systematic literature review.pdf]]
7. [[THE MULTI-AGENT PICKUP AND DELIVERY PROBLEM MAPF, MARL AND ITS WAREHOUSE APPLICATIONS.pdf]]
## MARL Reviews
1. [[A review of cooperative multi-agent deep reinforcement learning.pdf]]
2. [[A SURVEY OF PROGRESS ON COOPERATIVE MULTI-AGENT REINFORCEMENT LEARNING IN OPEN ENVIRONMENT.pdf]]
3. [[Actor-Attention-Critic for Multi-Agent Reinforcement Learning.pdf]]
4. [[Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks.pdf]]
5. [[Learning to Communicate with Deep Multi-Agent Reinforcement Learning.pdf]]
6. [[Learning Transferable Cooperative Behavior in Multi-Agent Teams.pdf]]
7. [[PC3D Zero-Shot Cooperation Across Variable Rosters via Personalized Context Distillation.pdf]]
## Others
1. [[CoLight Learning Network-level Cooperation for Traffic Signal Control.pdf]]
2. [[Constrained Policy Optimization.pdf]]
3. [[The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.pdf]]
