# <span id="page-0-0"></span>Learning Transferable Cooperative Behavior in Multi-Agent Teams

**Akshat Agarwal\***  
Carnegie Mellon University  
Pittsburgh, USA

**Sumit Kumar\***  
Carnegie Mellon University  
Pittsburgh, USA

**Katia Sycara**  
Carnegie Mellon University  
Pittsburgh, USA

# Abstract

While multi-agent interactions can be naturally modeled as a graph, the environment has traditionally been considered as a black box. We propose to create a shared agent-entity graph, where agents and environmental entities form vertices, and edges exist between the vertices which can communicate with each other. Agents learn to cooperate by exchanging messages along the edges of this graph. Our proposed multi-agent reinforcement learning framework is invariant to the number of agents or entities present in the system as well as permutation invariance, both of which are desirable properties for any multi-agent system representation. We present state-of-the-art results on coverage, formation and line control tasks for multi-agent teams in a fully decentralized framework and further show that the learned policies quickly transfer to scenarios with different team sizes along with strong zero-shot generalization performance. This is an important step towards developing multi-agent teams which can be realistically deployed in the real world without assuming complete prior knowledge or instantaneous communication at unbounded distances.

# 1 Introduction

Cooperative multiti-agent systems find applications in domains as varied as telecommunications, resource management and robotics, yet the complexity of such systems makes the design of heuristic behavior strategies difficult. While multi-agent reinforcement learning (MARL) enables agents to learn cooperative behavior to maximize some team reward function, it poses significant challenges including the non-stationarity of the environment, combinatorially growing joint action and state spaces of the agents, and the multi-agent credit assignment problem. Practically, most real world environments have partial observability (due to limited range and/or noisy sensors) and limited communication, which means agents have to learn to behave cooperatively conditioned only on local observations and limited communication.

While multi-agent systems have been modeled as graphs in previous works (Sukhbaatar et al. 2016; Hoshen, 2017), the environment has been usually treated as a black box. The agents receive information about other agents and entities in the environment in the form of a single vector or image with everything stacked together, which is a gross under-utilization of the natural structure present in the environment. Here, we propose to incorporate the inherent high-level structure of the environment directly in the learning framework by creating a shared agent-entity graph where both, agents and

\*Equal Contribution. Correspondence to agarwalaks30@gmail.com, sumit.sks4@gmail.com

environmental entities, form vertices and edges exist between those vertices whose occupants can communicate with each other. Agents learn to achieve global consensus important for solving fully cooperative tasks by sending and receiving messages along the edges of this graph [\(Scarselli et al.,](#page-9-1) [2009;](#page-9-1) [Gilmer et al., 2017\)](#page-8-1).

Building on the framework of Graph Neural Networks [\(Vaswani et al., 2017;](#page-9-2) [Jiang et al., 2018\)](#page-8-2), we propose a multi-agent reinforcement learning (MARL) model that is invariant to the number of agents or entities present in the environment, and also invariant to the order or permutation of entities. This facilitates transferring policies trained for one team in a specific environment to a team with different number of agents and/or an environment with a different number of entities. We further show that the team of agents can learn complex cooperative strategies via a curriculum of progressively increasing difficulty. To the best of our knowledge, this is the first work which addresses the problem of multi-agent transfer and curriculum learning of cooperative behaviors in a decentralized framework.

![](_page_1_Diagram_2.jpeg)

Figure 1: The proposed shared agent-entity graph on the right, and a detailed look at the internal architecture of each agent on the left. Messages exchanged between agents are depicted by red edges while those between an entity and an agent are shown by blue edges.

# 2 Related Work

MARL has been a widely studied topic in the machine learning community. One of the earliest works, Independent Q-learning [\(Tan, 1993;](#page-9-3) [Tampuu et al., 2017\)](#page-9-4) trains independent Q-value functions for each agent using regular Q-learning [\(Watkins and Dayan, 1992\)](#page-9-5), while assuming that the other agents are a part of the environment. Since the other agents are also learning, the environment becomes non-stationary and the resulting instability prevents these methods from scaling to more than 2 agents.

Under the paradigm of centralised learning with decentralised execution, a multitude of recent works have trained actor-critic algorithms where the critic is centralised and makes use of global information available during training. During execution, however, agents only use their actor network for selecting actions which enables the entire system to operate in a fully decentralized manner. MADDPG [\(Lowe](#page-8-3) [et al., 2017\)](#page-8-3) learns a centralised critic for each agent by providing the joint state and actions of all agents to the critic, and trains policies for each agent using the DDPG [\(Lillicrap et al., 2015\)](#page-8-4) algorithm. COMA [\(Foerster et al., 2018\)](#page-8-5) also uses a centralised critic but estimates a counterfactual advantage function that helps with multi-agent credit assignment by isolating the effect of each agent's action. VDN [\(Sunehag et al., 2017\)](#page-9-6) decomposes a centralized state-action value function into a sum of individual agent specific functions. The decomposition imposes a strict prior which is not well justified, and limits the complexity of the agents' learned value functions. Q-Mix [\(Rashid et al.,](#page-8-6) [2018\)](#page-8-6) improves upon this by removing the requirement of additive decomposition of the centralised critic, instead imposing a less restrictive monotonicity requirement on agents' individual state-action value functions, and allowing a learnable mixing of the individual functions which does not limit the complexity of functions that could be learned. All of these works, however, do not utilize any structure present in the environment. Instead each agent's observation is simply a concatenation of the states of other agents and various features of the environment. Also, the use of a centralized critic prevents the learned policy to generalize to scenarios with different number of agents in the team than the one during training. Moreover, lack of communication between agents prevents these methods to

be successful in scenarios where explicit coordination is required to solve the task, as we demonstrate in this work.

CommNet [\(Sukhbaatar et al., 2016\)](#page-9-0) is one of the earliest works to learn a differentiable communication protocol between multiple agents in a fully cooperative centralized setting. However, they did not explicitly model interactions between agents, instead each agent receives the averaged states of all its neighbors. VAIN [\(Hoshen, 2017\)](#page-8-0) improves upon the mean aggregation by using an exponential kernel based attention to selectively attend to the messages received from other agents, and showed predictive modeling of multi-agent systems using supervised learning. In this work, we use the scaled dot-product attention mechanism proposed by [Vaswani et al.](#page-9-2) [\(2017\)](#page-9-2) for inter-agent communication, which can be easily substituted with the ones used in CommNet and VAIN.

[Foerster et al.](#page-8-7) [\(2016\)](#page-8-7) demonstrated end-to-end learning of protocols in complex environments inspired by communication riddles and multi-agent computer vision problems with partial observability. [Mordatch and Abbeel](#page-8-8) [\(2018\)](#page-8-8) also demonstrated emergence of compositional language in multi-agent systems in both cooperative and competitive settings. They, however, learned discrete communication using symbols from a limited vocabulary, and made it end-to-end differentiable by using the Gumbelsoftmax estimator. In contrast, our communication module is continuous and fully differentiable.

In work done concurrently to ours, TarMAC [\(Das et al., 2018\)](#page-8-9) uses dot-product attention mechanism for inter-agent communication, however, they do not impose any restrictions on communication, leading to a centralized execution paradigm. DGN [\(Jiang et al., 2018\)](#page-8-2) also uses a similar mechanism for communication but with Q-learning [\(Mnih et al., 2013\)](#page-8-10) for training. They allowed each agent to communicate with its 3 closest neighbors. From a practical consideration, communication between agents is usually restricted by their mutual separation, meaning that an agent can communicate only with neighbors within a certain radius. We would also like to emphasize that being able to communicate with the 3 closest neighbors ensures that the agents' graph is always a single connected component and no agents are ever disconnected from the others, while having a distance-based restriction leads to formation of several different connected components in the agents' graph, none of which can communicate with each other leading to a significantly more difficult learning (to cooperate) problem.

## 3 Method

### 3.1 Agent-Entity Graph

An environment can often be described as a set of different entities with a defined structure. For example, the environment for a self-driving vehicle includes other vehicles, traffic lights, pedestrians, etc. which are interacting with each other. Also, for multi-agent systems or swarms, the environment can be represented as a set of obstacles and/or landmarks. Instead of treating the environment as a black-box, we propose to utilize the inherent high-level structure in the learning process itself.

We define a graph  $\mathcal{G} := (\mathcal{V}, \mathcal{E})$  where each node  $n \in \mathcal{V}$  is either an agent or an environment entity, and there exists an edge  $e \in \mathcal{E}$  between two nodes if the node occupants can communicate with each other. In this work, we consider static entities, i.e., their positions remain same throughout an episode. However, across different episodes, the entities can take random positions in the environment. Also, we assume that the agents have access to the position of all the entities at the beginning of each episode. This means that there always exists an edge between each agent-entity pair. With respect to communication between agents, we consider two variants:

Restricted Communication: Two agents can communicate with each other only if they are separated by a distance less than a pre-defined threshold (or communication bandwidth).

**Unrestricted Communication:** All agents can communicate with each other. In this case,  $\mathcal{G}$  is a fully-connected graph.

Modeling the multi-agent system as a graph provides a strong inductive bias to the learning algorithm. In practice, agents which are closer to each other have a greater impact on each others' behavior, and this crucial information is baked into the graph architecture itself, which greatly aids learning.

### <span id="page-3-1"></span>3.2 Learning to communicate

We now describe the messasure passing mechanism by which agents establish a global consensus among themselves in order to accomplish the given task. Each agent  $i \in \mathcal{V}$  observes only its own local state  $X^i$ . In this work, the state of an agent comprises of its position and velocity. The agent forms its state encoding  $U^i = f_a(X^i)$  by using a learnable differentiable encoder network  $f_a$ .

The agegent then aggregates all the information about the environment into a fixed size embedding  $E^i$  by using a Graph Neural Network (GNN). Specifically, it first forms an embedding  $e_i^l = f_e(X_i^l)$  for each of the entities  $l \in \mathcal{V}$  using an entity encoder function  $f_e$ . Here,  $X_i^l$  is the state of entity  $l$  w.r.t. agent  $i$ . Since entities are static,  $X_i^l$  is simply the position of entity  $l$  w.r.t. agent  $i$ . The agent then uses the dot product attention mechanism proposed by Vaswani et al. (2017) to update the entities' embeddings  $e_i^l$  and finally aggregate them together into a fixed size environment embedding  $E^i$ . We refer to this process as *entity message passing*. Note that, there is no actual message transmission between entities and agents, but, the agents themselves do all the computation with the knowledge of entities' states.

This envinronment information aggregation is an important step as it enables the agents to form a fixed size representation of the environment irrespective of the number of entities. Having a fixed size representation is important for being able to handle scenarios with different number of entities and also for transferring policies learned in one scenario to another. Other ways of forming a fixed size representation is to multi-channel grids or feature maps (Resnick et al., 2018), however, that unnecessarily increases the observation space of the agent and hence the complexity of the problem. Another common method is to take a conservative approach and allocate a vector of size more than required and pad the empty slots with some constant value like 0 or  $-1$  (Foerster et al., 2018). However, that requires a prior knowledge of an upper bound on the number of entities. Moreover, stacking all the observations in a single vector discards the high-level structural information present in the environment resulting in poor performance, as we show later in this work.

It is often the case that a particular representation of the environment is suitable for a given task. For instance, consider a simple task where a team of 3 agents is required to cover a *set* of 3 locations in such a manner that each agent goes to a distinct location. Clearly, in this case, the environment representation should be invariant to the permutation order of entities which is not be possible if one simply stacks all the entities' positions. Instead, the inherent structure of the environment can be incorporated in the learning framework itself by using a graph architecture as done in this work.

Figure 2: Scaled dot-product attention mechanism for message passing.

**Inter-agent communication:** After computing its state encoding  $U^i$  and environment encoding  $E^i$ , agent  $i$  concatenates them together into a joint encoding  $h^i$ . This encoding represents the agent's understanding of its own state and the environment. So far, the agent does not possess any information about its teammates. Now, each agent  $j \in \mathcal{V}$  computes a key  $K^j = W_K h^j$ , query  $Q^j = W_Q h^j$  and value  $V^j = W_V h^j$  vectors where  $W_K, W_Q$  and  $W_V$  are learnable parameters. Agent  $i$ , after receiving query-value pair  $(Q^j, V^j)$  from all of its neighbors  $j \in \mathcal{N}(i)$ , assigns weight  $w^{ij} = \text{softmax}\left(\frac{Q^j K^{i\top}}{d_K}\right)$  to each of the incoming messages. Here,  $d_K$  is the dimensionality of key vector. It then aggregates all the messages by computing a weighted sum of its neighbors' values followed by a linear transformation  $V_f^i = W_{\text{out}} \sum w^{ij} V^j$  where  $W_{\text{out}}$  is another learnable parameter. Finally, the agent updates its encoding by doing a non-linear transformation of its current embedding  $h^i$  concatenated with  $V_f^i$  by using a neural network  $f$ . We summarize our inter-agent communication module in Fig. 2.

The attention mechanism, described above, enables the agents to selectively attend to messages coming from its neighbors. Since the agent network may be sparsely connected with long chains

<span id="page-3-0"></span>![](_page_3_Diagram_7.jpeg)

in each connected component, we use multi-hop communication ( $K$  rounds of message passing) to allow information to propagate between agents that might not be directly connected with each other.

After  $K$  rounds of message passing, each agent has an updated encoding  $h^i$ . It then feeds this encoding into another neural network with value and policy heads to predict an estimate of its state value and a probability distribution over all possible actions respectively. Each agent samples an action from the distribution and acts accordingly, upon which the environment gives a joint reward to the team. In this work, we consider scenarios where the agents form a homogeneous team and share all the learnable parameters including those of agent encoder network, entity encoder network, graph networks, and policy and value networks. Since each agent receives different observations, attends incoming messages from other agents differently and perceives the environment differently (relative state of all the entities), sharing parameters does not preclude them from behaving differently, as is appropriate. The entire model is trained in an end-to-end manner using the actor-critic policy gradient PPO (Schulman et al., 2017) algorithm. A salient feature of our proposed model is that it can be trained and executed both in a completely decentralized manner.

### 3.3 Curriculum Training

Since our model is invariant to the number of agents or entities, sharing network parameters among all the agents enables us to directly use a policy  $\pi$  trained for a task  $\mathcal{T}$  with  $M$  agents and  $L$  entities to a different task  $\mathcal{T}'$  with  $M'$  agents and  $L'$  entities. The policy  $\pi$  can serve as a good initialization for task  $\mathcal{T}'$  which can be improved further by fine-tuning with some experiences collected in  $\mathcal{T}'$ . This facilitates in establishing a curriculum (Bengio et al., 2009) of tasks with increasing difficulty. Agents first learn cooperative behaviors in a small team and with the addition of new members bootstraps their strategies to accomplish the goal for this larger team. In other words, they utilize their previous knowledge in a new scenario and gradually learn complex cooperative strategies in a large team.

Curriculum learning with our proposed shared agent-entity graph enables us to train policies for complex tasks directly training on whom yields poor performance. Using a graph architecture is known to induce a strong inductive bias to the learning algorithm. To analyze this effect, we also evaluate the zero-shot generalization performance of our model.

# 4 Experiments

### 4.1 Task Description

We evaluate our proposed model on three standard swarm robotics tasks (Mesbahi and Egerstedt, 2010; Balch and Arkin, 1998): *coverage control*, *formation control* and *line control*. We have implemented them in the Multi-Agent Particle Environment<sup>2</sup> (Lowe et al., 2017) where the agents can move around in a 2D space following a double integrator dynamics model (Rao and Bernstein, 2001). The action space for each agent is discretized, with the agent being able to control unit acceleration or deceleration in both X and Y directions. We briefly describe the three environments below:

**Coverage Control:** There are  $M$  agents and  $M$  landmarks in the environment (see Figure 3a). The objective is for the agents to deploy themselves in a manner such that every agent reaches a distinct landmark. Note that we do not assign particular landmark to each agent, but instead let the agents communicate with each other and develop a consensus as to who goes where.

**Formation Control:** There are  $M$  agents and 1 landmark in this environment (see Figure 3b). The agents are required to position themselves into an  $M$ -sided regular polygonal formation, with the landmark at its centre.

**Line Control:** There are  $M$  agents and 2 landmarks in the environment (see Figure 3c). The objective is for the agents to position themselves equally spread out in a line between the two landmarks.

<sup>2</sup><https://github.com/openai/multiagent-particle-envs><span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 3: S: Simulation Environments used in this work. Agents are shown in blue circles while the landmarks in grey ones.

#### 4.2 Implementation Specifications

The agegent encoder  $f_a$  and the entity encoder  $f_e$  takes as input the 4-dim agent states and 2-dim entity states respectively and outputs a 128-dim embedding. Both the encoders are a single ReLU fully connected (FC) layer. The communication module uses attention with 128-dim queries, keys and values. The aggregated message is concatenated with the agent's state and passed through a single ReLU FC layer  $f$  containing 128 neurons as the update function. We use  $K = 3$  communication hops between the agents. Both the policy and value heads are 2 ReLU FC layers with 128 neurons. We use orthogonal initialization (Saxe et al., 2013) for all the learnable parameters. We have open-sourced our repository containing all the simulation environments and codes <sup>3</sup>.

All the environments are  $2 \times 2$  2 sq. units in size as is the standard in MAPE. In the restricted communication version, we set the communication distance to be 1 unit. Each episode lasts for a total of 50 timesteps. Evaluation is carried out after every 50 updates on 100 episodes in a newly seeded environment. During evaluation, each agent performs greedy decentralized action selection. Each PPO update is performed after accumulating experience for 128 timesteps on 32 parallel processes, or equivalently, every 4096 total timesteps.

### 4.3 Results

We used 3 metrics to compare different methods: **Success Rate** (S%): In what percentage of episodes does the team achieve its objective? (Higher is better) **Time** (T): How many time steps does the team require to achieve its objective? (Lower is better) **Average Distance** (DIST.): What is the average distance of a landmark from its closest agent? This metric is used in coverage control task only. (Lower is better).

We could not find any prior work on multi-agent reinforcement learning in coverage, formation or line control tasks and hence do not have previously published results to compare with. We used publicly available implementations<sup>4</sup> to compare with Q-Mix (Rashid et al., 2018), VDN (Sunehag et al., 2017), IQL (Tampuu et al., 2017), COMA (Foerster et al., 2018) and MADDPG (Lowe et al., 2017). These methods rely on access to the global state of the system (for example, a centralized view of the entire system) during training instead of inter-agent communication for emergence of cooperative behaviors. For these methods, the agents have full observability, i.e., they know the position and velocity of all the other agents at every time step. In contrast, agents are unaware of the state of other agents in our method. The corresponding results are shown in Table 1.

Even withith full observability, only MADDPG is able to solve the coverage control and formation control tasks for  $M = 3$  agents. It is only partially successful in the line control task. All the other 4 baseline methods report no success in the coverage control task. For the other two tasks too, the

<sup>3</sup>https://github.com/sumitsk/matrl.git<sup>[1]</sup> https://github.com/oxwhirl/pymarl, https://github.com/openai/maddpg<span id="page-6-0"></span>

Table 1 **Table 1: Comparisons with prior works with  $M = 3$  and  $M = 6$  agents.** UC: Unrestricted Communication  
2 RC: Restricted Communication, T: Average Episode Length, S%: success rate, DIST: average agent-landmark  
3 distance.

|   |          |         |     | O       | BSERV  |       |       | M = 3 |     |       | M = 6 |     |
|---|----------|---------|-----|---------|--------|-------|-------|-------|-----|-------|-------|-----|
| T | ASK      | M ETHOD |     | ABILITY |        | C OMM | D IST | T     | S % | D IST | T     | S % |
| C | OVERAGE  | Q-M     | IX  |         | F ULL  | N/A   | 0.46  | 50    | 0   | 0.51  | 50    | 0   |
| C | OVERAGE  | VDN     |     |         | F ULL  | N/A   | 0.44  | 50    | 0   | 0.47  | 50    | 0   |
| C | OVERAGE  | IQL     |     |         | F ULL  | N/A   | 0.51  | 50    | 0   | 0.43  | 50    | 0   |
| C | OVERAGE  | COMA    |     |         | F ULL  | N/A   | 0.41  | 50    | 0   | 0.43  | 50    | 0   |
| C | OVERAGE  | MADDPG  |     |         | F ULL  | N/A   | 0.065 | 17.89 | 95  | 0.52  | 50    | 0   |
| C | OVERAGE  | O       | URS | P       | ARTIAL | UC    | 0.047 | 14.12 | 100 | 0.15  | 20.47 | 93  |
| C | OVERAGE  | O       | URS | P       | ARTIAL | RC    | 0.049 | 14.22 | 98  | 0.17  | 48.32 | 5   |
| F | ORMATION | MADDPG  |     |         | F ULL  | N/A   | –     | 15.66 | 100 | –     | 50    | 0   |
| F | ORMATION | O       | URS | P       | ARTIAL | UC    | –     | 13.56 | 100 | –     | 14.22 | 100 |
| F | ORMATION | O       | URS | P       | ARTIAL | RC    | –     | 12.97 | 100 | –     | 14.26 | 100 |
| L | INE      | MADDPG  |     |         | F ULL  | N/A   | –     | 35.84 | 58  | –     | 50    | 0   |
| L | INE      | O       | URS | P       | ARTIAL | UC    | –     | 15.14 | 98  | –     | 16.31 | 100 |
| L | INE      | O       | URS | P       | ARTIAL | RC    | –     | 15.24 | 97  | –     | 17.07 | 100 |

success rate is 0%, hence we do not report them in the table. On the other hand, our proposed method is able to solve all the given tasks even with partial observability.

We alth also evaluated the models on a more challenging version of these tasks, with  $M = 6$  agents. In this case, all the baseline methods achieve no success at all. In contrast, our method is able to solve the formation control and line control task in both the communication versions. In the coverage control task, although it was not have been able to achieve near-perfect success rate, it still performed better than all the baselines.

### 4.4 Curriculum Training

We o observed from the last set of results that learning cooperative behaviors in a team becomes more challenging with increase in number of agents. Instead of training policies directly from scratch, we deploy a curriculum over the number of agents. A policy is first trained with  $M = 3$  agents. Once the team achieves a desired success rate threshold, the learned policy is then transferred to a team with  $M = 5$  agents. In other words, a team of 5 agents start with the policy trained for 3 agents. The training then begins for this team and on achieving the set threshold, the process is repeated with 7 and finally 10 agents. We set the success rate threshold to be 85%.

In t in this work, we have incorporated entities and agents together in a shared graph and formed a fixed dimensional environment representation using entity message passing (EMP) mechanism as described in Section 3.2. As mentioned before, another commonly used alternative is to stack all the entities’ state in a single vector and pad the vector with some constant value to make it some fixed size. We allocated a size of 20 units, i.e., a maximum of 10 landmarks and filled the slots corresponding to non-existent entities with 0s. We refer to this approach as the one without EMP.

<span id="page-6-1"></span>

Table 2: Curriculum Learning for coverage control task. EMP: Entity Message Passing, N: Number of updates.We compare the performance of curriculum learning on coverage control tasks with and without EMP in the two communication scenarios. The results of which are shown in Table 2. Looking at the performance of the model without EMP for  $M = 3$  agents, it is clear that increasing the size of

observation space (by padding) increases the complexity of the problem and requires more samples to solve the task. Also, in the restricted communication version, the team is unable to accomplish the goal on addition of two members. With no restriction on communication, the team of 5 agents do

learn the desired behavior, however, no success is observed on moving to 7.

In contrast, our proposed model shows fast transfer across increasingly difficult tasks and ultimately, even a team with 10 agents have learned the required cooperative strategies. Note that, the desired level of cooperation, even for a smaller team with 6 agents, did not emerge when training was done directly from scratch in restricted communication setting (see Table [1\)](#page-6-0).

We also evaluated curriculum learning on the other two tasks of formation control and line control. Their results are shown in Table [3.](#page-7-0) In these environments too, our model shows efficient transfer across tasks and is able to instill optimal cooperative behaviors even in large teams. This shows that incorporating the structure of environment in the learning framework induces a strong inductive bias in the multi-agent system that aids in learning transferable skills.

<span id="page-7-0"></span>Table 3: Curriculum Learning on formation control and line control tasks. EMP: Entity Message Passing, N: Number of updates.

|   | T ASK    | C OMM | M S % | = 3 N | M S % | = 5 N | M S % | = 7 N | M S % | = 10 N |
|---|----------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| F | ORMATION | UC    | 94    | 250   | 99    | 300   | 100   | 300   | 100   | 100    |
| F | ORMATION | RC    | 97    | 300   | 100   | 100   | 95    | 200   | 100   | 3100   |
|   | L INE    | UC    | 86    | 1200  | 98    | 700   | 93    | 150   | 82    | 700    |
|   | L INE    | RC    | 93    | 1200  | 93    | 1450  | 88    | 3450  | 87    | 350    |

### 4.5 Zero shot Generalization

We evaluated the policy trained for  $M = 5$  agents directly without any fine-tuning on different team sizes. The corresponding results for all the three tasks are shown in Table 4. In all the three tasks, the trained policy shows impressive zero-shot success rate in both the unrestricted and restricted communication settings. Such results show that our proposed model has been able to capture the inherent structure present in the environment, thanks to shared agent-entity graph architecture, and is ablikely to solve tasks it has never seen before by utilizing its past experiences of related but different tasks.

<span id="page-7-1"></span>

Table 4 **Table 4: Zero Shot Generalization results.** Policy trained for  $M = 5$  agents is evaluated directly for different team sizes without any fine-tuning and the obtained success rates (S%) are reported.

|   | T ASK    | C OMM | M − 3 | M − 2 | M − 1 | M = 5 | M + 1 | M + 2 | M + 3 |
|---|----------|-------|-------|-------|-------|-------|-------|-------|-------|
| C | OVERAGE  | UC    | 89    | 95    | 93    | 98    | 83    | 65    | 41    |
| C | OVERAGE  | RC    | 84    | 92    | 99    | 99    | 99    | 95    | 74    |
| F | ORMATION | UC    | 1     | 9     | 98    | 100   | 91    | 21    | 1     |
| F | ORMATION | RC    | 1     | 68    | 99    | 99    | 34    | 30    | 8     |
|   | L INE    | UC    | 0     | 0     | 57    | 99    | 81    | 45    | 16    |
|   | L INE    | RC    | 0     | 16    | 76    | 99    | 45    | 16    | 8     |

## 5 Conclusion and Future Work

Instead of treating the environment as a black box, we proposed to utilize the inherent structure in a shared agent-entity graph whose vertices are formed by both, the agents and environment entities. The agents learn cooperate behaviors by exchanging messages with each other along the edges of this graph. Our proposed model is invariant to the number of agents or entities present in the environment which enables us to establish a curriculum learning framework in multi-agent systems. We showed state-of-the-art results on coverage and formation control for swarms in a fully decentralized execution framework and demonstrated that the learned policies have strong zero-shot generalization to scenarios with different team sizes. We also showed that complex tasks, which are difficult to solve by directly training policies from scratch, can instead be solved via curriculum. For future work, we want to investigate the performance of the agent-entity graph when there is a team of evolving adversaries also in the environment. Also, developing curriculum learning algorithms for multi-agent teams in the presence of adversaries is another direction of research work from here.

# References

<span id="page-8-14"></span><span id="page-8-13"></span><span id="page-8-12"></span><span id="page-8-11"></span><span id="page-8-10"></span><span id="page-8-9"></span><span id="page-8-8"></span><span id="page-8-7"></span><span id="page-8-6"></span><span id="page-8-5"></span><span id="page-8-4"></span><span id="page-8-3"></span><span id="page-8-2"></span><span id="page-8-1"></span><span id="page-8-0"></span>Tucker Balch and Ronald C Arkin. Behavior-based formation control for multirobot teams. *IEEE transactions on robotics and automation*, 14(6):926–939, 1998. Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In *Proceedings of the 26th annual international conference on machine learning*, pages 41–48. ACM, 2009. Abhishek Das, Théophile Gervet, Joshua Romoff, Dhruv Batra, Devi Parikh, Michael Rabbat, and Joelle Pineau. Tarmac: Targeted multi-agent communication. *arXiv preprint arXiv:1810.11187*, 2018. Jakob Foerster, Ioannis Alexandros Assael, Nando de Freitas, and Shimon Whiteson. Learning to communicate with deep multi-agent reinforcement learning. In *Advances in Neural Information Processing Systems*, pages 2137–2145, 2016. Jakob N Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson. Counterfactual multi-agent policy gradients. In *Thirty-Second AAAI Conference on Artificial Intelligence*, 2018. Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In *Proceedings of the 34th International Conference on Machine Learning-Volume 70*, pages 1263–1272. JMLR. org, 2017. Yedid Hoshen. Vain: Attentional multi-agent predictive modeling. In *Advances in Neural Information Processing Systems*, pages 2701–2711, 2017. Jiechuan Jiang, Chen Dun, and Zongqing Lu. Graph convolutional reinforcement learning for multi-agent cooperation. *arXiv preprint arXiv:1810.09202*, 2018. Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*, 2015. Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. In *Advances in Neural Information Processing Systems*, pages 6379–6390, 2017. Mehran Mesbahi and Magnus Egerstedt. *Graph theoretic methods in multiagent networks*, volume 33. Princeton University Press, 2010. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. *arXiv preprint arXiv:1312.5602*, 2013. Igor Mordatch and Pieter Abbeel. Emergence of grounded compositional language in multi-agent populations. In *Thirty-Second AAAI Conference on Artificial Intelligence*, 2018. Venkatesh G Rao and Dennis S Bernstein. Naive control of the double integrator. *IEEE Control Systems Magazine*, 21(5):86–97, 2001. Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. Qmix: monotonic value function factorisation for deep multi-agent reinforcement learning. *arXiv preprint arXiv:1803.11485*, 2018.

<span id="page-9-9"></span><span id="page-9-8"></span><span id="page-9-7"></span><span id="page-9-6"></span><span id="page-9-5"></span><span id="page-9-4"></span><span id="page-9-3"></span><span id="page-9-2"></span><span id="page-9-1"></span><span id="page-9-0"></span>Cinjon Resnick, Wes Eldridge, David Ha, Denny Britz, Jakob Foerster, Julian Togelius, Kyunghyun Cho, and Joan Bruna. Pommerman: A multi-agent playground. *arXiv preprint arXiv:1809.07124*, 2018. Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *arXiv preprint arXiv:1312.6120*, 2013. Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. *IEEE Transactions on Neural Networks*, 20(1):61–80, 2009. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017. Sainbayar Sukhbaatar, Rob Fergus, et al. Learning multiagent communication with backpropagation. In *Advances in Neural Information Processing Systems*, pages 2244–2252, 2016. Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z Leibo, Karl Tuyls, et al. Value-decomposition networks for cooperative multi-agent learning. *arXiv preprint arXiv:1706.05296*, 2017. Ardi Tampuu, Tambet Matiisen, Dorian Kodelja, Ilya Kuzovkin, Kristjan Korjus, Juhan Aru, Jaan Aru, and Raul Vicente. Multiagent cooperation and competition with deep reinforcement learning. *PloS one*, 12(4):e0172395, 2017. Ming Tan. Multi-agent reinforcement learning: Independent vs. cooperative agents. In *Proceedings of the tenth international conference on machine learning*, pages 330–337, 1993. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, pages 5998–6008, 2017. Christopher JCH Watkins and Peter Dayan. Q-learning. *Machine learning*, 8(3-4):279–292, 1992.