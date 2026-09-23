# Fair Collaborative Vehicle Routing: A Deep Multi-Agent Reinforcement Learning Approach

Stephen Mak<sup>a,\*</sup>, Liming Xu<sup>a</sup>, Tim Pearce<sup>b,1</sup>, Michael Ostroumov<sup>c</sup>, Alexandra Brintrup<sup>a</sup>

a Institute for Manufacturing, Department of Engineering, University of Cambridge

<sup>b</sup>Microsoft Research Cambridge

<sup>c</sup>Value Chain Lab

#### ARTICLE HISTORY

Compiled October 27, 2023

#### ABSTRACT

Collaborative vehicle routing occurs when carriers collaborate through sharing their transportation requests and performing transportation requests on behalf of each other. This achieves economies of scale, thus reducing cost, greenhouse gas emissions and road congestion. But which carrier should partner with whom, and how much should each carrier be compensated? Traditional game theoretic solution concepts are expensive to calculate as the characteristic function scales exponentially with the number of agents. This would require solving the vehicle routing problem (NP-hard) an exponential number of times. We therefore propose to model this problem as a coalitional bargaining game solved using deep multi-agent reinforcement learning, where – crucially – agents are not given access to the characteristic function. Instead, we implicitly reason about the characteristic function; thus, when deployed in production, we only need to evaluate the expensive post-collaboration vehicle routing problem once. Our contribution is that we are the first to consider both the route allocation problem and gain sharing problem simultaneously – without access to the expensive characteristic function. Through decentralised machine learning, our agents bargain with each other and agree to outcomes that correlate well with the Shapley value – a fair profit allocation mechanism. Importantly, we are able to achieve a reduction in run-time of 88%.

#### KEYWORDS

Collaborative Vehicle Routing; Deep Multi-Agent Reinforcement Learning;

Negotiation; Gain Sharing; Multi-Agent Systems; Machine Learning

# 1. Introduction

Heavy goods vehicles (HGVs) in the UK contributed 4.3% of the UK's total greenhouse gas emissions in 2019 [\(UK BEIS](#page-36-0) [2021\)](#page-36-0). HGVs are utilised inefficiently at 61% of their total weight capacity. Moreover, 30% of the distance travelled carries zero freight [\(UK](#page-36-1) [DfT](#page-36-1) [2020,](#page-36-1) RFS0125).

Collaborative vehicle routing (CVR) has been proposed to improve HGV utilisation. Here, carriers collaborate through sharing their delivery information in order to achieve

\*Corresponding author: sm2410@cam.ac.uk, 17 Charles Babbage Road, CB3 0FS, United Kingdom

<sup>1</sup>Previously at Department of Computer Science and Technology, Tsinghua University

<span id="page-1-1"></span>![](_page_1_Figure_0.jpeg)

<span id="page-1-0"></span>Figure 1: Three agents (denoted by colours) before and after collaboration. Squares denote depots. Crosses denote customer locations. Node indices (arbitrary) are denoted in black, with costs given in their respective colours. The collaboration gain is defined as the difference in social welfare (or total cost) before and after collaboration. In [Figure 1b,](#page-1-0) Agents 1, 2 and 3 all decide to collaborate which reduces the system's total cost by 0.88 (or 26%). This results in a collaboration gain per capita (assuming agents split the gain equally) of 0.29. For detailed calculations, see [Section 3.1.](#page-5-0)

economies of scale. If carriers agree to work together, they are said to be in a coalition. As a result of improved utilisation, total travel costs across collaborating carriers can be reduced, resulting in a collaboration gain. The remaining question then is how to allocate this collaboration gain in a fair manner such that carriers are incentivised to form coalitions. An example of CVR is given in [Figure 1.](#page-1-1)

Prior literature suggests that collaborative vehicle routing can reduce costs by around 4–46% and also reduce greenhouse gas emissions and road congestion [\(Cruijssen](#page-34-0) [et al.](#page-34-0) [2007;](#page-34-0) [Zhang et al.](#page-37-0) [2017;](#page-37-0) [Gansterer and Hartl](#page-35-0) [2018b;](#page-35-0) [Pan et al.](#page-36-2) [2019;](#page-36-2) [Gansterer](#page-35-1) [and Hartl](#page-35-1) [2020;](#page-35-1) [Cruijssen](#page-34-1) [2020;](#page-34-1) [Ferrell et al.](#page-34-2) [2020\)](#page-34-2). Sharing resources may also lead to improved resilience to fluctuations in supply and/or demand. Despite these benefits, real-world adoption remains limited, with only a few companies participating [\(Cruijssen](#page-34-3) [et al.](#page-34-3) [2007;](#page-34-3) [Guajardo and Rönnqvist](#page-35-2) [2016;](#page-35-2) [Cruijssen](#page-34-1) [2020\)](#page-34-1). Currently, a key barrier is the computational complexity of calculating a fair gain sharing mechanism that scales with a larger number of companies. [Guajardo and Rönnqvist](#page-35-2) [\(2016\)](#page-35-2) recommends that future work should investigate approximate gain sharing methods. Our paper follows this recommendation.

Our first contribution is modelling the collaborative routing problem as a coalitional bargaining game [\(Okada](#page-36-3) [1996\)](#page-36-3) with intelligent agents obtained through the use of deep multi-agent reinforcement learning (MARL). We provide the theoretical grounding in this paper, tying together the fields of collaborative vehicle routing, coalitional bargaining, and deep multi-agent reinforcement learning in order to obtain a theoretically grounded approach that significantly reduces run-time. Here, agents

attempt to reach agreement on selecting the 'best' carrier(s) to partner with, and rationally share the collaboration gain amongst the coalition. This bargaining process takes place over multiple rounds of bargaining (see [Section 3.3](#page-8-0) for a formal definition). A benefit of this approach is that both the routing problem (who should deliver which requests?) and the gain sharing problem (who receives how much of the added value?) are considered simultaneously, whereas a key limitation of many previous methods consider these sub-problems in isolation from one another [\(Gansterer and](#page-35-0) [Hartl](#page-35-0) [2018b\)](#page-35-0). Moreover, our approach is agnostic to the underlying routing problem – the complexity of the vehicle routing problem (VRP) formulation can be increased with further constraints such as time windows, without further modification to the method.

Our secondond contribution is that agents do not need access to the full characteristic function explicitly. To obtain the full characteristic function, the collaboration gain for all possible coalitions must be calculated. In the three-player setting, there are four possible coalitions  $\{1, 2, 3\}$ ,  $\{1, 2\}$ ,  $\{1, 3\}$  and  $\{2, 3\}$ . Therefore, to obtain the full characteristic function requires solving  $2^{n-1}$  NP-hard post-collaboration VRPs (for a formal introduction, see [Section 2.1.3](#)). As a result, methods that require full access to the characteristic function are intractable for settings with more than 6 carriers ([Cruijssen 2020](#)). Instead, our agents can implicitly reason about the characteristic function through only receiving a high-dimensional graph input of delivery information (for example, latitudes and longitudes), as well as other agents' actions. This eliminates the need to fully evaluate the characteristic function when deployed in production, which involves solving the expensive post-collaboration VRP an exponential number of times. Instead, we only need to solve the post-collaboration VRP once when deployed in real-world settings, thus allowing our approach to achieve a significant run-time reduction. In addition, our approach utilises Centralised Training with Decentralised Execution (CTDE) to obtain decentralised agent policies ([Lowe et al. 2020](#)). Decentralised policies are desirable in real-world applications as each agent does not necessarily require access to the global, underlying state. This helps ensure that companies' sensitive information will not be leaked to competitors. This also aids to stabilise training in multi-agent settings as well as reduce communication costs. Furthermore, our approach is *inductive* as opposed to transductive of prior methods. This enables our agents to generalise to agents never seen bThe remainder of this paper is organised as follows. [Section 2](#page-2-0) positions our work within the wider context of both collaborative vehicle routing and deep multi-agent reinforcement learning. [Section 3](#page-5-1) provides a formal introduction to coalitional games, coalitional bargaining and reinforcement learning. [Section 4](#page-8-1) discusses and justifies various design decisions regarding our agents. [Section 5](#page-20-0) details our experimental setup, results, discussion and future work. Finally, [Section 6](#page-27-0) concludes our findings and provides broader managerial implications as a result of this work.

#### <span id="page-2-0"></span>2. Related Work

#### 2.1. Collaborative vehicle routing

Prior collaborative routing literature tackles the partner selection sub-problem (i.e., who should each carrier work with?) by estimating the collaboration gain between different carriers using heuristics [\(Palhazi Cuervo et al.](#page-36-4) [2016;](#page-36-4) [Adenso-Díaz et al.](#page-34-4) [2014\)](#page-34-4). However, a limitation of this approach is that they do not consider how much each agent should be compensated, nor if agents even agree to join the same coalitions (i.e., if the coalitions are stable). Posing this problem as a coalitional bargaining game not only allows us to tackle the partner selection aspect, but we are also able to consider the gain sharing aspect simultaneously as well.

The majority of the collaborative routing literature is concerned with the exchange of individual transportation requests amongst the carriers. This can be divided into three types of planning approaches: centralised; decentralised without auctions; and decentralised with auctions [\(Gansterer and Hartl](#page-35-0) [2018b,](#page-35-0) [2020\)](#page-35-1).

# 2.1.1. Centralised planning

Centralised planning approaches desire to simply maximise social welfare (the sum of each company's profits). Typically, this goal is achieved by using a form of mixed integer linear programming or (meta)heuristics [\(Cruijssen et al.](#page-34-0) [2007;](#page-34-0) [Gansterer and](#page-35-0) [Hartl](#page-35-0) [2018b;](#page-35-0) [Angelelli et al.](#page-34-5) [2022\)](#page-34-5). This can be viewed as a common-payoff setting, i.e., where all agents are on the same team and receive the same reward. However, assuming a common-payoff setting in practice is unrealistic as companies are self-interested – they mostly care only about their own profits [\(Cruijssen et al.](#page-34-3) [2007\)](#page-34-3). Moreover, there exists fierce competition especially in horizontal collaborations. Therefore, the more realistic setting of decentralised control is needed where agents are modelled to be self-interested.

# 2.1.2. Decentralised planning

There have been few attempts to tackle CVR with decentralised approaches as well. One approach focuses on the problem of partner selection, i.e. "who should work with whom?". [Adenso-Díaz et al.](#page-34-4) [\(2014\)](#page-34-4) proposes an a priori index to estimate the collaboration gain between carriers based on their transportation requests. However, a key limitation is that they do not consider the gain sharing aspect and thus the coalitions formed may not be stable.

A key challenge in decentralised settings is managing the explosion in the number of *bundles*. Consider Figure 1 where Agent 2 may desire to sell delivery node 10 to Agent 1. However, if Agent 2 offers both nodes 10 and 11 as a *bundle*, then Agent 2 may be able to command a higher price. Indeed, the number of possible bundles scales  $\mathcal{O}(2^m)$  where  $m$  is the number of deliveries. To manage this explosion, a heuristic is typically implemented where agents can only submit or request a few bundles (sometimes only one) which would limit optimality (Bo Dai and Chen 2009).

A second challenge is to also elicit other agents' preferences over all bundles. One approach is to invoke structure on the problem in the form of combinatorial auctions which aids optimality [\(Krajewska et al.](#page-35-4) [2008;](#page-35-4) [Gansterer and Hartl](#page-35-0) [2018b;](#page-35-0) [Gansterer](#page-35-5) [et al.](#page-35-5) [2019;](#page-35-5) [Los et al.](#page-35-6) [2022\)](#page-35-6). Auctions are where carriers submit requests they do not wish to fulfil to a common pool. Then, other carriers can submit bids on these requests with various methods of determining the "winners" of said bids. Combinatorial auctions in these settings allow carriers to bid on bundles of transportation requests instead of individual transportation requests which increases its expressivity and optimality. However, this additional structure comes at additional computational complexity. Moreover, in auction mechanism design, there are four desirable properties: efficiency; individual rationality; incentive compatibility; and budget balance. [\(Gansterer et al.](#page-35-5) [2019\)](#page-35-5) proposes two auction-based approaches which may be useful in practice, but would be unable to satisfy all four properties simultaneously: there exists a trade-off instead. [\(Los et al.](#page-35-6) [2022\)](#page-35-6) investigates large-scale carrier collaboration containing 1,000 carriers with decentralised auctions. Whilst impressive in scale, their approach ignores the difficulty of large-scale gain sharing.

Both auction-based and non-auction-based approaches may also be exploited by strategic agent behaviour. Would agents intentionally misreport the costs associated with performing deliveries in order to maximise their own profits? Whilst we do not tackle this problem in our work, we believe MARL could be a useful tool to investigate this strategic behaviour in future work.

#### <span id="page-4-0"></span>2.1.3. Gain sharing

Whilst gain sharing has been studied in collaborative routing using cooperative game theory (Guajardo and Rönnqvist 2016), the solution concepts typically assumes that the characteristic function is given. For a set of  $n$  agents,  $N = \{1, \dots, n\}$ , the characteristic function  $v : \mathbf{2}^N \rightarrow \mathbb{R}_{\geq 0}$  assigns a *value*, or in our case *collaboration gain*, for every possible coalition that could be formed. Note that there exists  $\mathcal{O}(2^n)$  possible coalitions. This is intractable for settings with more than a few agents, because evaluating the collaboration gain of even a single coalition, involves solving a vehicle routing problem which is NP-hard. For detailed calculations of the collaboration gain, see Section 3.1. Guajardo and Rönnqvist (2016) reviews 55 papers from the collaborative transportation literature concerning gain sharing. They recommend that a future research direction should focus on developing approximate gain sharing approaches based on cooperative game theory that scales with the number of agents.

In the wider algorithmic game theory literature, coalition formation has also been extensively studied [\(Chalkiadakis et al.](#page-34-7) [2011\)](#page-34-7). However, much of the existing literature again assumes that the full characteristic function is given. Alternatively, they aim to find more succinct representations of the characteristic function, typically at a cost of increased computational complexity when computing solution concepts [\(Chalkiadakis](#page-34-7) [et al.](#page-34-7) [2011\)](#page-34-7). Examples include Induced Subgraph Games and Marginal Contribution Nets [\(Deng and Papadimitriou](#page-34-8) [1994;](#page-34-8) [Ieong and Shoham](#page-35-7) [2005\)](#page-35-7); however, even these succinct representation schemes require evaluating the value of multiple coalitions and thus solving multiple NP-hard VRPs. We argue that many real-world scenarios consist of the characteristic function being a function of the agents' assets or capabilities. In the collaborative routing setting, this is a function of the transportation requests an agent possesses. We therefore ask: "Can agents form optimal coalitions from the delivery information alone instead of having full access to the characteristic function?". Therefore, our paper can be viewed as using an alternative, succinct representation scheme which approximates a rational outcome by using a function approximator.

# 2.2. Deep multi-agent reinforcement learning

Single agent reinforcement learning has seen increasing adoption in supply chain management. However, supply chains can be naturally modelled as a system comprising multiple self-interested agents [\(Fox et al.](#page-35-8) [2000;](#page-35-8) [Xu et al.](#page-37-1) [2021;](#page-37-1) [Brintrup](#page-34-9) [2021\)](#page-34-9). For a thorough review of reinforcement learning applied towards supply chain management, see [Yan et al.](#page-37-2) [\(2022\)](#page-37-2).

Recently, MARL has seen success in playing board and video games such as Go, StarCraft II and Dota 2 [\(Silver et al.](#page-36-5) [2016;](#page-36-5) [Vinyals et al.](#page-36-6) [2019;](#page-36-6) [OpenAI et al.](#page-36-7) [2019\)](#page-36-7). Whilst these are tremendous feats in the AI space, the underlying games tend to be

Table 1: Characteristics of selected games studied in MARL.<span id="page-5-2"></span>

|                                      |             |              | Known   | Partially  |
|--------------------------------------|-------------|--------------|---------|------------|
| Game                                 | > 2-players | Mixed-Motive | Optimum | Observable |
| Go (Silver et al. 2016)              | ✗           | ✗            | ✗       | ✗          |
| StarCraft II (Vinyals et al. 2019)   | ✗           | ✗            | ✗       | ✓          |
| SMAC a                               |             |              |         |            |
| (Samvelyan et al. 2019)              | ✓           | ✗            | ✗       | ✓          |
| Dota 2 (OpenAI et al. 2019)          | ✓           | ✗            | ✗       | ✓          |
| Gran Turismo (Wurman et al. 2022)    | ✓           | ✗            | ✗       | ✓          |
| Football (Kurach et al. 2020)        | ✓           | ✗            | ✗       | ✓ d        |
| Hide and Seek (Baker et al. 2020)    | ✓           | ✗            | ✗       | ✓          |
| Communication (Foerster et al. 2016) | ✓           | ✗            | ✓       | ✓          |
| GCE b (Mordatch and Abbeel 2018)     | ✓           | ✗            | ✓       | ✓          |
| SSDs c                               |             |              |         |            |
| (Leibo et al. 2017)                  | ✓           | ✓            | ✗       | ✓          |
| Coalitional Bargaining (ours)        | ✓           | ✓            | ✓       | ✗          |

<sup>a</sup>StStarCraft Multi-Agent Challenge; <sup>b</sup>Grounded Communication Environment; <sup>c</sup>Sequential Social Dilemmas; <sup>d</sup>Both fully and partially observable settings supported.

2-player and zero-sum. However, most real-world applications, including supply chain management (Gabel and Riedmiller 2012; Kosasih and Brintrup 2021), are  $n$ -player and mixed-motive (with potential ‘sequential social dilemmas’ [Leibo et al. 2017]). Whilst there is some research in this direction, the majority of MARL research focuses on pure coordination or pure competition settings (see Table 1). Our work is 3-player and mixed-motive which leads to a more challenging joint-policy space, allowing for complex behaviours such as collusion.

The most similar work to ours from a multiti-agent learning perspective is that of Bachrach et al. (2020) and Chalkiadakis and Boutilier (2004). In Bachrach et al. (2020), they apply deep MARL to a spatial and non-spatial Weighted Voting Game, where agents are given full access to the characteristic function. In Chalkiadakis and Boutilier (2004), they apply a Bayesian MARL approach to coalition formation as their problem has uncertainty in the characteristic function. In their problem, each agent knows its own capability, but does not observe other agents' capabilities. As a result, they maintain a belief over other agents' capabilities. However, each agents' capabilities remains constant. In our work, each agents' 'capability' can be thought of as the transportation requests it possesses, which constantly changes between episodes. Thus, our agents must be able to generalise across differing agent capabilities.

# <span id="page-5-1"></span>3. Background

### <span id="page-5-0"></span>3.1. Collaborative Vehicle Routing

We We denote the set of  $n$  agents as  $N = \{1, \dots, n\}$ . A coalition is a subset of  $N$ , i.e.  $C \subseteq N$ . The grand coalition is where all agents are in the coalition, i.e.  $C = N$ .

**Pre-collaboration profit at and social welfare:** The *pre-collaboration profit* of Agent 1 in Figure 2 is calculated as follows: the *Revenue* is 3 (1 for each delivery); the *Cost* is 1.42 (sum of the edge distances); thus the *Profit* is 1.58 (Revenue subtract Cost). Similarly, the pre-collaboration profit of Agents 2 and 3 is 2 and 2.07. The *pre-collaboration social welfare* is the sum of the pre-collaboration profits, thus  $1.58 + 2 + 2.07 = 5.65$ .

**Post-collaboration “profit” and social welfare:** Assuming agents agree to form the grand coalition  $C = \{1, 2, 3\}$ , the post-collaboration “profit” of Agent 1 can be calculated as  $1 - (0.06 + 0.06) = 0.88$ . Note that the post-collaboration “profit” for Agent

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

<span id="page-6-2"></span><span id="page-6-1"></span>

(a) Pre-colobaloration (total cost: 3.35) (b) Post-collaboration with grand coalition {1, 2, 3} (total cost: 2.47) (c) Post-collaboration with grand coalition {1, 2} (Agent 3 is excluded from the coalition, total cost: 2.59)

Figure 2: Three agents, Agents 1, 2 and 3 are denoted by the colours green, orange and purple respectively. Squares denote depots. Crosses denote customer locations. Node indices (arbitrary) are denoted in black, with costs given in their respective colors. The *collaboration gain* is defined as the difference in social welfare before and after collaboration. Figure 2b and Figure 2c refer to two possible post-collaboration scenarios with collaboration gains per capita of 0.29 and 0.38 respectively. Thus, it would be rational for the coalition  $\{1, 2\}$  to form instead of the grand coalition  $\{1, 2, 3\}$ .

1 appears to have decreased from 1.58 to 0.88 as a result of collaboration. This will be accounted for when discussing the characteristic function and thus Agent 1 will not lose out when we calculate its reward. For Agents 2 and 3, the post-collaboration “profit” is 2.19 and 3.46 respectively. Thus a *post-collaboration social welfare* of  $0.88+2.19+3.46 = 6.53$ .

**Collaboration gain:** The *collaboration gain* is defined as the difference in social welfare before and after collaboration for a given coalition, in this case  $6.53 - 5.65 = 0.88$  for the grand coalition. Note that the collaboration gain is always greater than or equal to 0. The *value per capita* is  $\frac{0.88}{3} = 0.29$ . During the bargaining process, agents are able to choose how to divide this collaboration gain amongst themselves. In the unique case where agents agree to divide the collaboration gain equally, i.e. according to the value per capita, we refer to this as *equal gain sharing*. Note that if only Agents 1 and 2 form a coalition (and exclude Agent 3), then the collaboration gain (assuming equal gain sharing) is divided by 2 instead – thus making it rational to object and form the coalition  $\{1, 2\}$  (the value per capita of this coalition is 0.38).

**Characteristic function:** The characteristic function,  $v : \mathbf{2}^N \rightarrow \mathbb{R}$  calculates for every possible coalition the collaboration gain. Importantly, to fully evaluate the characteristic function would require solving a variant of the Vehicle Routing Problem for every possible coalition which scales  $\mathcal{O}(2^n)$ .

Following the example in Figure 2:

| $v(\{1, 2, 3\}) = 0.88$ | Value per Capita = $\frac{0.88}{3} = 0.29$  |
|-------------------------|---------------------------------------------|
| $v(\{1, 2\}) = 0.76$    | Value per Capita = $\frac{0.76}{2} = 0.38$  |
| $v(\{1, 3\}) = 0.24$    | Value per Capita = $\frac{0.24}{2} = 0.12$  |
| $v(\{2, 3\}) = 0.01$    | Value per Capita = $\frac{0.01}{2} = 0.005$ |

<span id="page-7-1"></span>![](_page_7_Diagram_0.jpeg)

Figure 3: Flowchart of the  $n$ -player coalitional bargaining game (Okada 1996). Our proposed approach is therefore to obtain a set of intelligent agents that can bargain with each other in a coalitional bargaining game. To achieve a suitable level of agent intelligence, we train our agents using deep multi-agent reinforcement learning.

It isignificantly important to note that the characteristic function is *0-normalised, essential and super-additive* (see Section 3.2 for a formal definition). This guarantees that agents will not lose profits as a result of collaboration. The final *take-home profit* that each agent (or carrier) receives can then be calculated as the sum of the pre-collaboration profit and its respective allocation of the collaboration gain. For Agents 1, 2 and 3, this would equate to  $1.58 + \frac{0.88}{3} = 1.87$ ,  $2 + \frac{0.88}{3} = 2.29$  and  $2.07 + \frac{0.88}{3} = 2.36$  respectively (assuming equal gain sharing). In reality, carriers will receive this take-home profit (which is always greater than or equal to the pre-collaboration profit) as an incentive to collaborate.

# <span id="page-7-0"></span>3.2. Coalitional games

We conote that the  $n$ -player coalitional game, also called a cooperative game, with a set of agents  $N = \{1, \dots, n\}$ . A *coalition* is defined as a subset of  $N$ , i.e.  $C \subseteq N$ . The set of all coalitions is denoted  $\Sigma$ . The *grand coalition* is where the coalition consists of all agents in  $N$ , i.e.  $C = N$ . A *singleton coalition* is where the coalition consists of only one agent, i.e.  $|C| = 1$ . A *coalition structure*  $CS = \{C^1, \dots, C^k\}$  is a partition of  $N$  into mutually disjoint coalitions,  $C^1 \cup \dots \cup C^k = N$  and  $C^i \cap C^j = \emptyset, \forall i \neq j$ .

A (transferable utility) coalitional game is a pa pair  $G = \langle N, v \rangle$ . The *characteristic function*  $v : \mathbf{2}^N \rightarrow \mathbb{R}_{\geq 0}$  represents the *value* (or collaboration gain in our setting) that a given coalition  $C$  receives. Like Okada (1996), we assume that the characteristic function is *0-normalised*, *essential* and *super-additive*. The characteristic function is *0-normalised* if the value of all singleton coalitions is 0, i.e.  $v(\{i\}) = 0, \forall i \in N$ . It is *essential* if the value of the grand coalition is strictly positive,  $v(N) > 0$ . It is

*super-additive* if  $v(C \cup D) \geq v(C) + v(D)$  for all coalition pairs  $C, D \in \Sigma$  where  $C \cap D = \emptyset$ .

The payoff vector  $\mathbf{x}^C = (x_i^C)_{i \in C}$  denotes the pay-off for player  $i$  in the coalition  $C$ . The payoff vector is *feasible* if  $\sum_{i \in C} x_i^C \leq v(C)$ . The set of all feasible payoff vectors for a given coalition  $C$  is  $X^C$ , and  $X_+^C$  when all the elements of  $X^C$  is non-negative.

# <span id="page-8-0"></span>3.3. Coalitional bargaining

The purpose of this work is to find a partition of the  $N$  carriers ws with an associated payoff vector, i.e.  $(CS, \mathbf{x})$ , which all self-interested, rational carriers agree to. Notice how this does not imply any sequential decision making. However, it was found that certain cooperative solution concepts can be retrieved as the outcome of non-cooperative, extensive form games such as coalitional bargaining (Nash 1953). Therefore, this necessitates sequential decision making in our problem where we propose to obtain intelligent agents through the use of MARL.

Okada (1996) presents the  $n$ -player, random proposers, alternating offers coalitional bargaining game which we adopt. At time-step  $t = 1, 2, \dots$  an agent from  $N$  is selected uniformly at random to be the *proposer*. The proposer, player  $i$ , has two actions – the proposed coalition and proposed pay-off vector. The proposed coalition  $C$  must contain player  $i$  and the value of the coalition  $v(C)$  must be greater than 0. Due to the characteristic function being 0-normalised this implies  $|C| \geq 2$ . The payoff vector  $\mathbf{x}^C$  must be in the set of all feasible, non-negative payoff vectors  $X_+^C$ . After player  $i$  has proposed, the remaining players called the *responders* are uniformly at random selected sequentially to either accept or reject the proposal. If all agents in the proposed coalition  $C$  accepts, then those agents form a coalition with the agreed upon proposal. The remaining players outside of  $C$  continue negotiating from the next time-step. If any responder in  $C$  rejects the proposal, then all players receive an immediate reward of zero and negotiations go on to the next round of bargaining. Then, a new proposer is selected uniformly at random and the time-step incremented by 1. This continues until either agreement is reached, or the maximum time step is reached. When a proposal  $(C, x^C)$  is agreed upon at time  $t$ , every agent  $i$  in  $C$  receives a reward of  $\gamma^{t-1} x_i^C$ , where  $\gamma \in [0, 1]$  is the discount factor. The discount factor decreases the reward received as time passes. This encourages agents to reach agreement within the first time-step in the three-player setting as shown in Okada (1996). The discount factor in this setting is analogous to the *patience* of an agent, or the urgency of the delivery decision. Any agent who is not in a coalition at the end of this process is assumed to have a reward of zero. In the three-player setting, note that if one proposal is accepted, then no more feasible coalitions can form; thus, this denotes the end of the bargaining process as seen in Figure 3.

# <span id="page-8-1"></span>4. Methodology

In summary, analytically calculing cooperative game theory solution concepts is intractable for settings with more than 6 carriers (Cruijssen 2020). Instead, we can recover these cooperative solution concepts through non-cooperative, extensive form games such as coalitional bargaining (Serrano 2004). However, coalitional bargaining requires intelligent, rational agents and it is difficult to manually craft rule-based agents for collaborative routing due to its exponential and NP-hard nature. Instead,

Table 2: Notation Table

| Symbol        | Definition     |                                                                                                                 |
|---------------|----------------|-----------------------------------------------------------------------------------------------------------------|
| Coalitional   | Bargaining     | Game:                                                                                                           |
| n             | Number         | of Agents                                                                                                       |
| N             | Set of         | all n Agents (i.e., grand coalition)                                                                            |
| i             | Agent          | index                                                                                                           |
| C             | A Coalition    |                                                                                                                 |
| Σ             | Set of         | all Coalitions                                                                                                  |
| CS            | A Coalition    | Structure                                                                                                       |
| ∅             | Empty          | set                                                                                                             |
| G             | A coalitional  | game                                                                                                            |
| v ( )         | Characteristic | function                                                                                                        |
| v ( C )       | Value          | of the coalition C , or the collaboration gain of the coalition C in the collaborative vehicle routing setting. |
| C             | Payoff         | vector for a given coalition C                                                                                  |
| X C           | Set of         | all feasible payoff vectors for a given coalition C                                                             |
| X C           |                |                                                                                                                 |
| +             | Set of         | all feasible, non-negative payoff vectors for a given coalition C                                               |
| (Multi-agent) | Reinforcement  | Learning:                                                                                                       |
| γ             | Discount       | factor                                                                                                          |
| M             | A Markov       | decision process (MDP)                                                                                          |
| S             | Set of         | states                                                                                                          |
| s 0           | Initial        | state of an episode                                                                                             |
| A             | Set of         | (joint) actions                                                                                                 |
| T             | Transition     | probability distribution                                                                                        |
| ρ 0           | Distribution   | of the initial state, s 0                                                                                       |
| a             | An action      |                                                                                                                 |
| t             | Time-step      | index                                                                                                           |
| G t           | Return         | following time t                                                                                                |
| T             | Maximum        | time-step (or the horizon length)                                                                               |
| π             | Agent’s        | policy                                                                                                          |
| V π ( s )     | State-value    | function of a state s following a policy π                                                                      |
| V ˆ ( s, θ )  | Policy’s       | (parameterised by θ ) estimate of the state-value function given the state s                                    |
| Q ˆ( s, a, θ  | ) Policy’s     | (parameterised by θ ) estimate of the action-value function given the state s and taking the action a           |
| Q π ( s, a )  | Action-value   | function of a state s taking the action a following a policy π                                                  |
| R             | Set of         | all possible rewards                                                                                            |
| R i,t         | Reward         | at time t for agent i                                                                                           |
| θ i           | Agent          | i’s policy parameters, usually the parameters of a neural network                                               |
| J ( θ )       | Performance    | measure for the policy π θ                                                                                      |
| ∇ J ( θ )     | Column         | vector of partial derivatives of π ( a   s, θ ) with respect to θ                                               |
| g ˆ           | Estimate       | of the policy gradient                                                                                          |
| M             | Number         | of episodes played in parallel                                                                                  |
| α             | Learning       | rate for stochastic gradient descent                                                                            |
| b ( s )       | A baseline     | function for policy gradient methods                                                                            |
| r t ( θ )     | PPO’s          | probability ratio between the new policy (after gradient updates) and the old policy (before gradient updates)  |
| ε             | Threshold      | to clip the probability ratio in PPO                                                                            |
| H             | Entropy        | bonus                                                                                                           |
| Collaborative | Vehicle        | Routing:                                                                                                        |
| D             | Deliveries     | matrix                                                                                                          |
| x             | x-coordinate   | of the location                                                                                                 |
| y             | y-coordinate   | of the location                                                                                                 |
| o             | Agent          | index who owns the location                                                                                     |
| d             | Binary         | variable denoting whether the location is a depot or a customer                                                 |
| c             | Multi-hot      | encoded vector denoting which agents are in the proposed coalition                                              |
| x             | Proposed       | pay-off vector                                                                                                  |
| r             | Responses      | of the agents to the given proposal                                                                             |
| p             | Agent          | index who was selected to propose in the current round of bargaining                                            |
| a             | Binary         | variable denoting whether the current agent is proposing or responding                                          |
| Dir ( α )     | Dirichlet      | distribution with concentration parameters α                                                                    |

we propose to develop intelligent, rational agents through having agents learn through trial-and-error, learning to collaborate in the presence of multiple other self-interested, rational agents (i.e., multi-agent reinforcement learning). A holistic diagram to depict the whole pipeline can be found in [Appendix E](#). The remainder of this section focuses on the reinforcement learning algorithm employed. Pseudo-code of the pipeline can be found in [Appendix D](#).

#### 4.1. Single Agent Reinforcement Learning

Reinforcement Learning (RL) is a subfield of machine learning. Here, the field studies an agent learning what *actions* to take for a given *state* in order to maximise a numerical *reward*. In supervised learning, the ground truth target labels are provided. In RL, we are not told the “correct” actions to take that will maximise (expected) cumulative reward. Instead, the agent must learn through trial-and-error. This leads to an exploration-exploitation dilemma. Should the agent try new actions (explore) in the hope that there is a better sequence of actions that leads to an even higher expected reward? Or, should the agent stick with its current best-known actions (exploit) since the agent believes it is unlikely there will be a better sequence of actions with higher expected reward? (Sutton and Barto 2018). The agent selects actions according to its *policy* based on the current state. The action is sent to the *environment* which calculates the reward and next state which is then returned to the agent. Through the learning process, we aim to obtain a policy that maximises the expected cumulative reward.

In our setting of of collaborative vehicle routing, the environment is the coalitional bargaining game as described in [Section 3.3](#). Each carrier is represented as an individual agent. The state is the locations of depots and customers, as well as auxiliary features to describe the current state of the coalitional bargaining process – see [Section 4.3](#) for further details. There are three actions that an agent can take depending on if it is proposing or responding. When proposing, the agent must decide (a) which other carriers should the agent propose to partner with, and (b) how much should each carrier in the proposal be paid. When responding, the agent must decide (c) if they accept or reject the proposal. The reward is the collaboration gain the agent is allocated as a result of the coalitional bargaining process. Throughout the training process, we train our agents’ policies (or neural network) to maximise expected cumulative reward. See [Section 4](#) for a formal definition of states, actions and rewards in our setting.

We can formalise the problem using Markov decision processes (MDPs) (Puterman 1994). Formally, a finite-horizon, discounted Markov decision process  $\mathcal{M}$  can be defined by the tuple  $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, r, \rho_0, \gamma \rangle$  where  $\mathcal{S}$  is the set of states,  $\mathcal{A}$  is the set of actions,  $\mathcal{T} : \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$  is the transition probability distribution,  $\mathcal{R} : \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$  is the reward function,  $\rho_0 : \mathcal{S} \rightarrow \mathbb{R}$  is the distribution of the initial state  $s_0$ , and  $\gamma \in [0, 1]$  is the discount factor.

An episode begins by first sampling an initial state  $s_0$  from  $\rho_0$ . A *trajectory*  $(s_0, a_0, s_1, a_1, \dots)$  is generated by sampling actions from the agent's policy  $a_t \sim \pi(a_t | s_t)$ . The next states are obtained by sampling the transition dynamics function  $s_{t+1} \sim \mathcal{T}(s_{t+1} | s_t, a_t)$  until reaching a terminal state. At each time step, a reward  $R_t \sim \mathcal{R}(s_t, a_t, s_{t+1})$  is received. At timestep  $t$ , the discounted return,  $G_t$ , is defined as:

$$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^T R_{T+1} = \sum_{k=0}^T \gamma^k R_{t+k+1} \quad (1)$$

where  $T$  is the maximum time-step and  $\gamma \in [0,1]$  is the discount factor. As  $\tau$  approaches 1, the agent will take into account rewards received far into the future. However, as  $\gamma$  approaches 0, the agent will only account for the immediate reward  $R_{t+1}$ , and the agent is often said to be *myopic*.

The *state-value function* of a state  $s$  under a policy  $\pi$  is dentified by  $V_\pi(s)$ . This is the expected return when the agent starts in  $s$  and continues following its policy  $\pi$ . Formally:

$$V_\pi(s) \doteq \mathbb{E}_\pi [G_t | S_t = s] = \mathbb{E}_\pi \left[ \sum_{k=0}^T \gamma^k R_{t+k+1} | S_t = s \right], \quad \forall s \in \mathcal{S} \quad (2)$$

A similar notion is the *action-value function* which is denoted by  $Q_\pi(s, a)$ . This is the expected return when the agent starts from  $s$ , but also takes the action  $a$ , and follows its policy  $\pi$  afterwards. Formally:

$$Q_\pi(s, a) \doteq \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi \left[ \sum_{k=0}^T \gamma^k R_{t+k+1} \mid S_t = s, A_t = a \right] \quad (3)$$

#### 4.2. Multi-agent reinforcement learning

A *stochastic g game* generalises MDPs to involve multiple agents. This can be defined as a tuple  $\langle N, S, A, \mathcal{T}, \mathcal{R}, \gamma, \rangle$  where:

- $N$  denotes the set of  $n$  agents
   $S$  denotes the set of states inc
- $S$  denotes the set of states including the initial state  $s_0$ 
   $A = A_1 \times \dots \times A_n = \{(a_1 \dots a_n) \mid a_i \in A_i \text{ for every } i \in \{1 \dots n\}\}$
- $A = A_i \times \cdots \times A_n = \{(a_1, \dots, a_n) \mid a_i \in A_i \text{ for every } i \in \{1, \dots, n\}\}$  denotes the set of joint actions, where  $A_i$  is player  $i$ 's set of actions and  $\times$  denotes the Cartesian product.
- $\mathcal{T} : S \times A \rightarrow S$  denotes the transition dynamics
   $\mathcal{R} : S \times A \times S \times N \rightarrow \mathbb{R}$  denotes the reward fi
- $\mathcal{R} : S \times A \times S \times N \rightarrow \mathbb{R}$  denotes the reward function
   $\gamma$  denotes the discount factor
- $\gamma$  denotes the discount factor

Forom every time-step  $t$ , an agent  $i \in N$  receives an observation of the global state  $s$  and outputs an action  $a_{i,t}$  sampled from its *policy*  $\pi_i(a_{i,t} \mid s_t)$ . We update the state  $s_t$  to include agent  $i$ 's action before sending this new state to agent  $j \in N, j \neq i$ . Note that the time-step is not yet incremented. We continue this process until all agents in  $N$  have submitted their actions to the environment. This yields the joint action  $\mathbf{a} = (a_1, \dots, a_n)$ . We calculate the reward  $R_{i,t} \sim \mathcal{R}(s_t, \mathbf{a}, s_{t+1}, i)$ . We consider the sparse reward setting, i.e., all rewards are zero until the episode terminates. Upon termination, we calculate the reward for agent  $i$  depending on if agent  $i$  successfully joined a coalition or not. When a proposal  $(C, x^C)$  is agreed upon at time  $t$ , every agent in  $C$  receives a reward of  $\gamma^{t-1} x_i^C v(C)$ . Else, if the agent is not in a coalition  $C$ , it is

assumerted to receive a reward of zero. The return  $G_i$  is discounted by a factor  $\gamma \in [0, 1]$ , given by  $G_i = \sum_{t=1}^T \gamma^{t-1} r_{i,t}$ .

Agent  $i$ 's objective is to find a policy  $\pi_{\theta_i}$  which maximises its expected discounted sum of rewards  $\mathbb{E}[\sum_{t=1}^T \gamma^{t-1} R_{i,t}]$ . It is important to note that this maximisation assumes all opponents' policies  $\pi_{\theta_j} \forall j \neq i$  to be fixed. Thus, one of the key challenges in MARL is the non-stationarity present due to multiple concurrently learning agents.

In our setting, we assume perfect information and thus a agents have full access to the global state. We make this assumption as the aim of our paper is to provide the theoretical grounding between collaborative vehicle routing, coalitional bargaining, and multi-agent reinforcement learning. The imperfect information setting is also a promising research direction, e.g., to investigate the value of information. Future work could study the applicability of *decentralised partially observable* Markov decision processes (dec-POMDPs) (Oliehoek and Amato 2016) to imperfect information settings in collaborative vehicle routing.

A challenge in reinforcement learning is handling the curses (plural) of dimensionality (Powell 2022). With “tabular” methods, the policy is represented by a lookup table. One curse is that the size of the state space grows exponentially with the number of dimensions (even if the state space is discrete). In our setting, our state space is continuous thus further exacerbating the challenge. As a result, we must resort to *function approximation* methods (Sutton et al. 2000). Instead, we aim to replace the lookup table with a parameterised model, with parameters  $\theta \in \mathbb{R}^d$  to map from states to actions. Thus, we can write the policy for agent  $i$  as  $\pi_{\theta_i}(a_{i,t} | s_t)$  instead. Respectively, the state-value function and action-value function can also be re-written  $\hat{V}(s, \theta) \approx V_\pi(s)$  and  $\hat{Q}(s, a, \theta) \approx Q_\pi(s, a)$ . Importantly, the dimensionality  $d$  of the model is typically much less than the number of states. Changing one parameter will effect the estimated value of many other states. Thus, if we can generalise across states, this could greatly accelerate learning. Note that any parameterised model can be used: a linear function, multi-layer perceptron, decision trees etc. Historically, linear functions were favoured due to favourable convergence guarantees. However, deep neural networks have demonstrated significant success due to their high capacity and generalisability (Sutton and Barto 2018; Vinyals et al. 2019; Mnih et al. 2015; OpenAI et al. 2019). Thus, we also opt for deep neural networks as well.

*Policy gradient*-based approaches are a common way to learn a parameterised policy  $\pi_\theta$  which maximises an agent's expected discounted return. It is also performant, for example, it achieved great success in playing Dota 2 (OpenAI et al. 2019) amongst others. Typically, a scalar performance measure  $J(\theta)$  is defined and we maximise their performance using approximate gradient ascent:  $\theta_{t+1} = \theta_t + \alpha \widehat{\nabla J(\theta_t)}$  where  $\widehat{\nabla J(\theta_t)} \in \mathbb{R}^d$  is a stochastic estimate whose expectation approximates the gradient of  $J(\theta_t)$  with respect to  $\theta_t$ . However, a challenge is that the performance depends on both the policy's action selection and also the distribution of states where these actions are selected. Varying  $\theta$  affects both of these distributions and we typically do not know the effect of our policy on the state distribution. The policy gradient theorem (Sutton et al. 2000; Sutton and Barto 2018) shows that we can approximate the gradient of performance with respect to  $\theta$  but without requiring the derivative of the state distribution. Formally:

$$\nabla J(\theta) \propto \sum_s \mu(s) \sum_a Q_\pi(s, a) \nabla \pi(a | s, \theta) \quad (4)$$

Then simplest approach is the REINFORCE algorithm (Williams 1992). Here, an agent plays  $M$  episodes in parallel until termination and remembers all states, actions and rewards it encountered (or trajectory). Next, it estimates the (undiscounted) policy gradient using:

$$\hat{g} g} = \frac{1}{M} \sum_{m=1}^M \left[ \sum_{t=1}^T \hat{A}_t^m \nabla_{\theta} \log \pi_{\theta}(a_t^m | s_t^m) \right] \quad (5)$$

where, for REINFORCE  $\hat{A}_t = \sum_{t'=t}^T \gamma^{t-t} r(s_{t'}^m, a_{t'}^m)$ . The agent updates its policy using stochastic gradient descent, i.e.,  $\theta \leftarrow \theta + \alpha \hat{g}$  where  $\alpha$  is the learning rate. The intuition for this policy update is that for each action the agent took for a given state, it will increase or decrease the (log) probability of taking that same action proportional to the discounted return it received during that episode. However, policy gradient methods are notorious for having high variance in the policy gradient. As a result, we employ multiple variance reduction techniques to mitigate this problem, such as  $M$  parallel environments.

Another variance reduction technique is to subtract a *baseline*. A baseline  $b(s)$  can be any funit function that may or may not depend on the state  $s$ . Importantly, it must not vary with the action  $a$ . We can replace REINFORCE's estimate of  $\hat{A}_t$  by using  $\hat{A}_t = \left[ \left( \sum_{t=t}^T \gamma^{t-t} r(s_t^m, a_t^m) \right) - b(s) \right]$  instead. It can be shown that introducing a baseline does not introduce bias into the policy gradient, but may significantly reduce variance (Williams 1992; Greensmith et al. 2004; Sutton and Barto 2018). An example baseline is the average return an agent received. The term  $\left[ \left( \sum_{t=t}^T \gamma^{t-t} r(s_t^m, a_t^m) \right) - b(s) \right]$  can be thought of as how much better than the baseline an agent performed as a result of choosing its action. A common choice of  $b(s)$  is to estimate the state-value  $\hat{V}_\pi(s_t^m, \theta) = \mathbb{E}_\pi \left[ \sum_{t'=t}^T \gamma^{t'-t} r(s_{t'}^m, a_{t'}^m) \mid S_t = s \right]$ . Selecting a good baseline is crucial. We discuss our proposed baseline functions in Section 4.7.

In REINFFORCE, typically only one gradient update is used per batch of trajectories. As a result, REINFORCE is typically said to be sample inefficient – it requires a lot of episodes to train a performant policy. In addition, REINFORCE can be unstable during training, and sometimes performance collapse may occur as a result of the data distribution changing too drastically.

Proximal Policy Oproportionality (PPO) (Schulman et al. 2017) aims to improve the sample efficiency by performing multiple gradient updates to maximise the use of each gathered data point. However, this risks changing the data distribution too drastically and thus risks performance collapse. To rectify this, the intuition behind PPO is to constrain the policy from deviating too greatly. Let the current policy (before any gradient updates) be denoted  $\pi_{\theta, old}(a_t|s_t)$ . After one round of gradient updates, this would yield new policy parameters, denoted  $\pi_\theta(a_t|s_t)$ . PPO constrains that the probability ratio,  $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta, old}(a_t|s_t)}$ , of taking action  $a_t$  for the same state  $s_t$  under the old policy vs new policy to be no more than a certain percentage  $\varepsilon$ . This should prevent the risk of policy collapse if  $\varepsilon$  is chosen carefully. Moreover, PPO is then able to perform more gradient updates on the same data points, thus greatly improving its sample efficiency. In addition, it is also more stable during training and is less sensitive to chosen hyperparameters. As a result, PPO has been applied to wide range of domains, most notably in OpenAI Five (bots to play Dota 2) (OpenAI et al. 2019)

and also in ChatGPT ([OpenAI 2022](#)).

PPO adjusuts the neural network parameters  $\theta$  to increase or decrease the probability

ratio  $r_t(\theta)$ ) proportional to the advantage the agent received  $\hat{A}$ ,

<sup>[1]</sup> tPPO enforces the  $\varepsilon$ 

threshold by clippining the probability ratio,  $r_t(\theta)$ , to remain within  $\pm \varepsilon$ . We can further encourage exploration by adding an entropy bonus. Thus, the PPO policy gradient can

be estimated as follows:

- Victor Gorandet

$$\hat{g} \approx \frac{1}{M} \sum_{m=1}^M \sum_{t=1}^T \nabla_{\theta} \left[ \min(r_t(\theta) \hat{A}_t, \text{clip(r_t(\theta),1-\varepsilon,1+\varepsilon) \hat{A}_t) + \beta \mathcal{H}[\pi_{\theta}](s_t) \right] \quad (6)$$

where  $\hat{A}_t$  is the baseline,  $\beta$  is the entropy regularisation coefficient and  $\mathcal{H}$  the entropy bonus. An entropy bonus encourages agents to explore rather than exploit. It is important to note that when the advantage is positive, we clip  $r_t(\theta)$  *only* if it is greater than  $1 + \varepsilon$ . If the advantage is negative, we clip  $r_t(\theta)$  *only* if it is less than  $1 - \varepsilon$  (see Figure 1 of (Schulman et al. 2017) for further details). The `clip` function is a function that clips the first argument by the lower and upper bounds denoted by the second and third arguments respectively.

As a result, PPOO has been widely used in a range of applications, most notably in OpenAI Five (for Dota 2) and in ChatGPT ([OpenAI et al. 2019](#); [OpenAI 2022](#)).

# <span id="page-14-0"></span>4.3. State space

The agents receive a variety of inputs from the environment as seen in Figure 4. Let the state at time  $t$  be denoted by  $s_t \in S$  which can be represented by the tuple  $\langle \mathbf{D}, \mathbf{c}, \mathbf{x}, \mathbf{r}, t, p, a \rangle$ . The *deliveries* matrix  $\mathbf{D} \in \mathbb{R}^{12 \times 4}$  describes the features of each of the three depots and nine customers, yielding twelve rows where we refer to each row as a *location*. A location can be represented by the tuple  $\langle x, y, o, d \rangle$  where  $x \in \mathbb{R}$  is the x-coordinate;  $y \in \mathbb{R}$  is the y-coordinate;  $o \in \mathbb{N}$  denotes the agent who owns the location; and  $d \in \{0, 1\}$  denotes whether the location is a depot or a customer. For instance, to represent Agent 2's depot located at  $\langle x = 0.2, y = 0.173 \rangle$ , its corresponding row in  $\mathbf{D}$  would be represented as  $\langle 0.2, 0.173, 2, 1 \rangle$  and the remaining rows in  $\mathbf{D}$  would be comprised of similar entries for the remaining depots and customers, yielding a shape of  $12 \times 4$ . The vector  $\mathbf{c} \in \{0, 1\}^{|N|}$  denotes which agents were selected to be in the proposed coalition. The vector  $\mathbf{x} \in \mathbb{R}^{|N|}$  denotes the proposed pay-off vector, the vector  $\mathbf{r} \in \{0, 1\}^{|N|}$  denotes the reselected responses of the agents. The vectors  $\mathbf{c}$ ,  $\mathbf{x}$  and  $\mathbf{r}$  are initialised to zero if no agent has taken an action in the current round of bargaining. The scalar  $t \in \mathbb{N}_0$  denotes the current round of bargaining,  $p \in \mathbb{N}$  denotes which agent was selected to propose in the current round of bargaining, and  $a \in \{0, 1\}$  denotes whethe

#### 4.4. Action space

The agents have three action heads: *coalitions*, *proposals* and *response*.

The *coalitions* action is denoted by  $\mathbf{c} \in \{0, 1\}^{|N|}$  where  $|N|$  is the total number of agents, in this case, 3. Note that in game theory, typically agents propose a coalition of size  $|C|$  instead of  $|N|$ . However, it is beneficial to output coalitions in this manner as it keeps the output size constant. The coalitions action denotes whether the respective

<span id="page-15-0"></span>![](_page_15_Diagram_0.jpeg)

Figure 4: Actors' neural network design. Grey boxes denote state inputs. Blue boxes denote MLP parameters which come from supervised pre-training (see [Section 4.6](#)). Note that the linear layer to produce coalition logits is learnt and not pre-trained. White boxes denote learnt parameters. Red boxes denote actions. Numbers in brackets denote the output shapes (ignoring batch size as it's shared by all).

agent index is part of the coalition  $C$ . Note that this game assumes that player  $i$  is in the coalition  $\mathbf{c}$ , i.e.,  $c_i = 1$ . The *deliveries* matrix,  $\mathbf{D}$ , is fed through two dense layers with 256 hidden neurons. These parameters come from a supervised pre-training step (see [Section 4.6](#)). The output is fed through a linear layer with  $|N|$  outputs. These outputs are passed into  $|N|$  independent Bernoulli distributions to determine the probability that a given agent is in the coalition  $C$ . A Bernoulli distribution is chosen as the number of outputs required scales linearly with the number of agents. Alternatively, this action can be output auto-regressively, but would be more computationally expensive. It may also be useful to introduce correlation in the agents' actions via more expressive probability distributions which may speed up learning.

The *proposals* action is denoted by  $\mathbf{x} \in \mathbb{R}^{|N|}$  where  $\sum_i x_i = 1$ ,  $x_i \in [0, 1]$ . This vector denotes how much of the collaboration gain is assigned to each respective agent (as a percentage). Note that in game theory, the definition of a feasible pay-off vector is  $\sum_{i \in C} x_i^C \leq v(C)$ . However, agents will never know the value of  $v(C)$  a priori (although it can implicitly reason about it). Thus, to practically implement our neural network, we output a vector that is interpreted as percentages as opposed to absolute values. These percentages are then multiplied by the value of a coalition  $v(C)$  to obtain a feasible pay-off vector.

Note that this is a continuous action space, as opposed to the other actions which are discrete. To parameterise the proposals action head, we use the Dirichlet distribution which is a multivariate generalisation of the Beta distribution. The neural network will output three logits  $\alpha$  which are used as the concentration parameters of the Dirichlet distribution  $\text{Dir}(\alpha)$ . The Dirichlet distribution has support over the probability simplex  $S_K = \{\theta : 0 \leq \theta_k \leq 1, \sum_{k=1}^K \theta_k = 1\}$  (Murphy 2021). Intuitively, agents will propose an equal gain share with high probability if the inputs to the Dirichlet are large and equal. Agents will make proposals uniformly at random within the probability simplex if the inputs to the Dirichlet are small and equal, but greater than 1. If Agent 1 wanted to collaborate with Agent 2 but not 3, the input to the Dirichlet could be  $\langle 10000, 10000, 1.001 \rangle$ . This would result in approximately a 50/50 split between Agents 1 and 2 with high probability.

The Dirichlet distribution is appealing due to two reasons. Firstly, the proposals vector requires that it sums to 1 which matches the form of the Dirichlet distribution. Secondly, the Dirichlet distribution has finite support. In continuous action spaces, a

Gaussian distribution is typically used which has infininite support and can lead to bias (Chou et al. 2017). Chou et al. (2017) overcomes this issue by using a Beta distribution instead as it has finite support and find that their agents learn more efficiently.

To calculate the proposals, the stateate inputs are passed through a variety of dense layers (see Figure 4) to produce an embedding. A linear layer with 3 output neurons is applied to the embedding. As in Chou et al. (2017) we add 1.001 to the output logits to ensure the resultant Dirichlet distribution remains unimodal. As a result, during evaluation the agents can fully exploit by proposing the mode of the distribution, instead of having to sample from the Dirichlet which may involve exploration. The output logits are then masked by the *coalitions* vector, i.e. if a player  $i$  is not in the coalition  $S$ , its corresponding output logit will be 1.001. Finally, to calculate the pay-off vector, we sample from the Dirichlet distribution with the masked output logits.

The  *response* action  $r \in \{0, 1\}$  denotes whether an agent accepts or rejects a given proposal. It takes the resultant embedding followed by a single linear layer with one output neuron. The output is then fed through a Bernoulli distribution.

Whilst we have chosen to use Bernoulli and Dirichlet distributions to parameterise the three action spaces, it may be beneficial to experiment with more expressive probability distributions or e.g. output actions auto-regressively. This may speed up learning and would be an interesting line of future research.

# 4.5. Reward function

Our reward function is sparsese, i.e., at timestep  $t$  the agents will always receive an immediate reward  $R_t$  of zero until the coalitional bargaining game terminates. Upon termination, we calculate a reward for each agent.

If agent  $i$  successfully joins a coalition  $C$  by having all agents in  $C$  accept the proposal, then it receives a reward of  $r_{i,t} = v(C) \cdot x_i$  where  $v(C)$  is the collaboration gain obtained by coalition  $C$ , and  $x_i$  is the  $i$ th element of the pay-off vector  $\mathbf{x}$ . For clarity, if agent  $i$  is the proposer and has its proposal rejected by the responder agents, it will receive an immediate reward of zero. However, there is potential for agent  $i$  to obtain more than zero immediate reward in future rounds of bargaining and thus the discounted return can still be greater than zero.

Else, if agent  $i$  does not successfully join a coalition  $C$  by the end of the episode, then it will receive a terminal reward of zero.

#### <span id="page-16-0"></span>4.6. Transfer learning

A key challenge with policy gradident approaches is its sample inefficiency, even in single agent settings. This is further exacerbated due to the non-stationary learning dynamics imposed by having multiple agents learn concurrently. In typical RL settings, agents learn “tabula rasa”, i.e., without any prior knowledge. Whilst this is mathematically elegant, learning tasks tabula rasa for problems with high complexity, such as in real-world, multi-agent settings, is rare (Agarwal et al. 2022). Instead, it may be preferable to pre-train on some offline dataset in order to learn a good feature extractor. For example, (Silver et al. 2016; Vinyals et al. 2019) pre-train their networks on human gameplay data in a supervised learning setting before using RL. This idea of transfer learning, or recently, *reincarnating RL* (Agarwal et al. 2022) is well accepted in the RL literature and the reader is referred to (Taylor and Stone 2009; Agarwal et al. 2022) for a thorough review. Furthermore, transfer learning is well accepted in supervised

<span id="page-17-1"></span>![](_page_17_Diagram_0.jpeg)

Figure 5: Pre-trained neural network design. Grey boxes denote state inputs. White boxes denote learnt parameters. Red box denotes the output, which predicts the collaboration gain for this given state and coalition . Numbers in brackets denote the output shapes (ignoring batch size as it's shared by all).

learning, especially in the computer vision and natural language processing domains leading to the likes of ChatGPT [\(OpenAI](#page-36-17) [2022\)](#page-36-17). In our case, the pre-training process aids in efficiently initializing the agents' policies and facilitates faster convergence in the MARL framework.

We therefore pre-train our agents to learn a good feature extractor in a supervised learning fashion. We hypothesise that a good feature extractor should be able to predict whether a given coalition for a given state is productive or not. As a result, we create a dataset of one million instances and randomly select a feasible coalition per instance and calculate the social welfare obtained. Next, we train a neural network to predict the social welfare for a given state and coalition. We optimise the neural network to minimise the mean squared error. We split the dataset using an 80/20 train/test split. The neural network design can be seen in [Figure 5.](#page-17-1) We experimented with different neural network architectures but found this architecture performed best. Whilst this is not the exact task agents must perform in the collaborative routing scenario, the intuition is that the neural network should still learn useful patterns which are transferable to the full collaborative routing problem.

# <span id="page-17-0"></span>4.7. Policy gradient baselines

As discussed in [Section 4,](#page-8-1) a useful baseline helps reduce the variance in policy gradient methods. We use two types of baselines: one for the response action (when agents are responding); and one shared for both the coalitions and proposals actions (when agents are proposing). The neural network architectures for the baselines can be found in [Figure 6.](#page-18-0)

The response action is discrete and thus we can easily implement Counterfactual Multi-Agent Policy Gradients (COMA) [\(Foerster et al.](#page-34-16) [2017\)](#page-34-16). They use the following baseline:

<span id="page-18-0"></span>![](_page_18_Diagram_0.jpeg)

(a) Neural Networowk design of the *coalitions* and *proposals* baseline.

![](_page_18_Diagram_2.jpeg)

(b) Neural network design of the *responses* baseline.

Figure 6: Grey boxes denote state inputs. Blue boxes denote MLP p parameters which come from supervised pre-training (see [Section 4.6](#)). White boxes denote learnt parameters. Red boxes denote outputs for the baseline. Numbers in brackets denote the output shapes (ignoring batch size as it's shared by all).

$$A^i(s, \mathbf{a}) = Q(s, \mathbf{a}, i) - \sum_{a'^i} \pi^i(a'^i | \tau^i) \hat{Q}(s, (\mathbf{a}^{-i}, a'^i), i, \phi) \quad (7)$$

where  $Q_\pi(s, \mathbf{a}, i) = \mathbb{E}_\pi \left[ \sum_{t'=t}^T \gamma^{t'-t} r(s_{t'}^m, a_{t'}^m) \mid s, \mathbf{a} \right]$  is the discounted return received if all agents take the joint action  $\mathbf{a}$  in state  $s$ . The estimate comes from a function approximator with parameters  $\phi$ .  $a'^i$  is the other actions agent  $i$  could have taken.  $\tau^i$  is the prior trajectory agent  $i$  has observed.  $\hat{Q}(s, (\mathbf{a}^{-i}, a'^i), i, \phi)$  is the *estimated* discounted return agent  $i$  would receive if it took a different action  $a'^i$  whilst keeping the other agents' actions  $\mathbf{a}^{-i}$  constant. This is estimated through the use of a neural network with parameters  $\phi$ .

Intuitively, the COMA baseline can be thought of as how much better agent *i*'s decision to take action *a* was relative to any other action agent *i* could have taken, *a*<sup>*i*</sup>. In our case, the question we ask is: if an agent has agreed to a given proposal, could it have done better by rejecting instead, assuming other agents' actions remain the same?

In the discrete setting, it is easy to sum over all other actions agent  $i$  could have taken. However, with continuous actions using Dirichlet distributions in the proposals action, this can be difficult. Therefore, we instead estimate the *state-value* which estimates the expected discounted return conditioned on the state  $s$ . We denote this baseline with  $\hat{V}_\pi(s, i, \mathbf{w}) = \mathbb{E}_\pi \left[ \sum_{t'=t}^T \gamma^{t'-t} r(s_{t'}^m, a_{t'}^m) \mid s \right]$  where  $\mathbf{w}$  is the parameters of a function approximator such as a neural network. Thus, our baseline for both the coalitions action and proposals action is given by:

$$A^i(s, \mathbf{a}) = Q_\pi(s, \mathbf{a}, i) - \hat{V}_\pi(s(s, i, \mathbf{w}) \quad (8)$$

Finally, we normalise  $A^i(s, \mathbf{a}\mathbf{a})$  by subtracting the mean and dividing by the standard deviation due to the small magnitude in rewards.

# <span id="page-19-0"></span>4.8. Time limits

It is crucial to deal w with time limits properly in this setting. The full coalitional bargaining game presented in Okada (1996) is infinite horizon, i.e., negotiation could go on indefinitely. Clearly, this is impossible to simulate on a finite computer and we must set a maximum number of rounds. Nevertheless, it is still possible to optimise for the infinite horizon, but care must be taken as shown in Pardo et al. (2018). They argue that if an episode terminates only due to reaching the maximum number of rounds, we should *bootstrap* the discounted estimated value of the next state,  $\hat{v}_\pi(s')$ . Thus, if agents reach agreement *within* the maximum number of rounds, they should receive a reward  $r$  as expected. However, if they *exceed* the maximum number of rounds, they should receive a reward of  $r + \gamma \hat{v}_\pi(s')$ . In our setting with the maximum number of rounds equal to 10, if agents do not reach agreement within 10 rounds, we fictitiously step them into the next state  $s'$ , at round 11 with proposers selected uniformly at random. If player  $i$  is not selected as a proposer, then the selected proposer is asked to propose a coalition and pay-off vector  $(S, \mathbf{x})$  in this fictitious round. We then use a critic to estimate the value of this state,  $\hat{v}_\pi(s')$ .

# 4.9. Skill retention

**Okada (1996)** shows that agents should reach agreement with no delay in agreement. Therefore, as agents learn to collaborate better, they will reach agreement sooner, which is beneficial due to the environment’s discount factor. However, this may lead to agents forgetting how to play the game at later time-steps. To enable retention of skills at later time-steps, we employ a targeted training design. During training, instead of starting all bargaining games at round 1, we uniformly at random start them between round 1 and the last round of bargaining,  $T - 1$ . Therefore, agents will always be exposed to a range of bargaining scenarios even if agents are collaborating optimally.

![](_page_20_Figure_0.jpeg)

Figure 7: A plot of the distribution of depot and customer locations. Depots are denoted by squares. Each depot has three distinct service radii which are selected uniformly at random. Customers may be uniformly at random located within any of corresponding depot's service radius.

#### <span id="page-20-0"></span>5. Experiments

#### 5.1. Problem setting

We base our problem setting on a modified version of (Gansterer and Hartl 2018a). We consider an environment with three companies, each represented by an agent. Each agent has one depot and three customers that it must deliver to. The depot  $(x, y)$  locations for each agent are held fixed at  $\{(-0.2, 0.173), (0.2, 0.173), (0, -0.173)\}$  respectively. The depots' service radius for each instance is selected uniformly at random from the set  $\{0.3, 0.4, 0.6\}$ . The rationale by Gansterer and Hartl (2018a) is that, through varying the depots' service radius, this varies the degree of overlap and thus competition (or collaboration opportunity) between carriers. A high degree of overlap using a radius of 0.6 creates high collaboration opportunity between carriers. A low degree of overlap using a radius of 0.3 has low collaboration opportunity between carriers. With a small radius of 0.3, this can analogously be seen as the scenario when depots do not lie in close proximity to each other. The customers locations are then generated uniformly at random with the depot's service radius.

To calculate the pre-collaboration and post-collaboration gains, the shortest paths are calculated exactly using Gurobi [\(Gurobi Optimization, LLC](#page-35-17) [2021\)](#page-35-17). The pre-collaboration shortest paths can be calculated by solving three (un)Capacitated Vehicle Routing Problems (one for each agent). The post-collaboration shortest paths are calculated by solving a single multi-depot vehicle routing problem. Problem formulations for the capacitated VRP and multi-depot VRP can be found in Appendices [A](#page-29-0) and [B](#page-30-0) respectively. Capacity is effectively removed by setting the capacity of each vehicle to an arbitrarily large number and the weight of each delivery to 1.

Whilst this problem setting is rather simplistic, this is important as it allows us to evaluate our agents rigorously. To calculate optimal solutions (for evaluation purposes only), we must brute force the characteristic function. This is expensive and only possible for small, simple VRPs and 3 agents.

#### 5.2. Experimental design

We perform 10 indendependent runs with different random seeds to train our agents. Agents are trained for 10,000 epochs and evaluated every 100 epochs. Agents are evaluated on instances it has never seen before in training. We train using a batch size of 2048 and evaluate with a batch size of 2048. All agents use a discount factor  $\gamma$  of 0.95. All agents' observations are normalised with a running estimate of the mean and standard deviation. The maximum number of bargaining rounds  $T$  is set to 10. The learning rate was held constant at  $3 \times 10^{-4}$  and we use Adam optimisation. We clip the global norm of gradient updates if they exceed 1. We use  $\varepsilon = 0.05$  to clip the probability ratios in PPO as it seems to help stability in (Yu et al. 2021). All code to generate results is run on the Wilkes 3 high performance computing cluster with AMD EPYC™ 7763 64-Core Processors and NVIDIA A100 GPUs. Note we only use a supercomputer to perform runs in parallel. Training takes approximately 8 hours per run.

# 5.3. Evaluation

#### 5.3.1. Correlation with the Shapley value

The objective of our work is to f find a partition of the  $N$  carriers with an associated fair pay-off vector. We emphasise that certain cooperative solution concepts (e.g. Shapley values) can be retrieved as the outcome of non-cooperative, extensive form games (e.g. coalitional bargaining as in our work). The Shapley value is the most common gain sharing mechanism used in the collaborative vehicle routing setting (Guajardo and Rönnqvist 2016) as it is widely accepted in game theory to be fair – each agent gets paid proportional to their marginal contribution. In addition, it is also guaranteed to be unique. We believe that both of these arguments would help transportation planners to reach agreements better, in line with (Krajewska et al. 2008). Thus, we compare the outcomes that our MARL agents agree to with the Shapley value for each instance by measuring the correlation, mean absolute error, and mean squared error.

#### <span id="page-21-0"></span>5.3.2. Baseline bots

We compare our MARL agents against two rule-based bots as a baseline. The heuristic bot always proposes the grand coalition with equal gain share and always accepts every proposal. The random bot proposes coalitions and gain shares as well as responses all uniformly at random. These two bots help us to understand that (a) our MARL agents are learning interesting, complex behaviours, and (b) our experimental setup is not too easy in design and that simple, intuitive policies are not sufficient for this setting.

#### 5.3.3. Accuracy

A simple evaluation metric is to measure how often the agents propose the correct coalition. For player  $i$ , the correct coalition  $C_i^*$  is defined to be the coalition  $C$  which would maximise player  $i$ 's reward. This involves brute forcing the characteristic function to evaluate the value of each possible coalition which is only possible since we consider 3 agents. We emphasise that brute force is only required to *evaluate* our agents – brute force is not required to train the agents. The reward  $R$  is the collaboration gain from agreeing to coalition  $C$ ,  $v(C)$ , multiplied by the  $i$ th element of the pay-off vector,  $x_i$ .

#### 5.3.4. Optimality gap

We denote the absolute and relative optimality gap of player  $i$  by  $\phi_i$  and  $\eta_i$  respectively. The absolute optimality gap  $\phi_i$  for player  $i$  is defined as  $\phi_i = v(C_i^*) - v(C)$ , where  $C_i^*$  is the correct coalition,  $C_i$  is player  $i$ 's proposed coalition, and  $v(\cdot)$  is the characteristic function (i.e. the collaboration gain of a given coalition). The relative optimality gap  $\eta_i$ , is calculated as:

$$\eta_i = \frac{v(C_i^*) - v(C_i)}{v(C_i^*)} \quad (9)$$

Since the data is randomly generated, there could be scenarios where there is no value in collaborating, i.e. even the value of the grand coalition is 0,  $v(N) = 0$ . Note that we exclude these scenarios when calculating the above evaluation metrics; however, this only occurs 1.9% of the time when brute-forcing 51,200 instances.

# 5.3.5. Other checks

[Okada \(1996\)](#) analyses this coalitional bargaining game in a non-collaborative routing setting, and proves that agents will cooperate by sharing gains equally in our setting. Therefore, in addition to the above metrics, we check that the agents' behaviour agrees with those predicted by [Okada \(1996\)](#). Firstly, we check that agents do converge to an equal gain share. Secondly, all agents should reach agreement in the first time-step in the three-player setting.

# 5.4. Results

We perform ten independent runs comparing our RL bot to the heuristic bot. Ten independent runs in (MA)RL is commonly accepted following the work of [Henderson et al. \(2019\)](#). We also compare to a random bot which simply proposes coalition structures and pay-off vectors as well as responds all uniformly at random.

From Figures 8a and 8b, we conclude that our agents have learnt close to optimal behaviour. Our agents reach an average accuracy of 77% and average optimality gap of 0.01 (or 3.9%). Moreover, we can see from Figure 9a that Agent 1 learns to share gains equally – as expected by game theory (Okada 1996). Whilst we only show the plot for Agent 1, similar plots can be made for Agents 2 and 3 but are omitted due to space constraints. Interestingly, three ‘phases’ of learning are identified as shown in Figure 9b. In Phase 1 (the first approximately 300 epochs), proposers act extremely myopically and propose that they receive the majority of the gain (up to 90%). Occasionally, the responders will accept these sub-optimal proposals and thus the proposer could receive high reward. However, the responders learn to reject more proposals so that they can potentially counter-offer in the next round. This leads to more rounds of bargaining. After about 300 epochs, both proposers and responders reach agreement quickly; however, the gains are not equally shared. In Phase 2, responders realise they can do better by rejecting proposals and potentially proposing counter-proposals. This drives the proposers to propose more equal gain shares. Finally, in Phase 3, we can see that agents have learnt to maximally cooperate with equal gain share and reach agreement within the first time-step as expected by Okada (1996).

<span id="page-23-0"></span>![](_page_23_Figure_0.jpeg)

(a) Average accuracy.

![](_page_23_Figure_2.jpeg)

<span id="page-23-1"></span>

(b) Average optimality gap.

Figure 8: Learning curve of (a) average accuracy (b) average optimality gap across all 3 agents for readability. Solid lines denote mean accuracy across 10 independent runs. Shaded regions denote  $\pm$  two standard deviations. After training for 10,000 epochs, our RL agents reach an average accuracy of 77% with an average optimality gap of 3.9%.

# 5.4.1. Correlation with Shapley Values

In Figure 10, we see that the outcomes from our bargaining procedure correlate well with the calculated Shapley values. The three agents receive an  $R^2$  score of 0.76, mean squared error of 0.08, and mean absolute error of 0.01 (averaged across all three agents). In addition, it is promising that when agent 1 is excluded from the coalition (denoted by orange cross markers), this is usually when Agent 1 has low marginal contribution (as seen by the orange kernel density estimate plot at the top of the x-axis). As a result, we conclude that our agents learn to agree to *fair* outcomes. This is important from a managerial perspective as fairness could be crucial to help incentivise carriers to participate in collaborative vehicle routing (Guajardo and Rönnqvist 2016).

# 5.4.2. Ablations

We further perform two ablations to strengthen the confidence in our findings. Each ablation is carried out with 10 random seeds each. The first ablation changes the maximum number of bargaining rounds from 10 to 30. This ablation is carried out

<span id="page-24-0"></span>![](_page_24_Figure_0.jpeg)

(a) Agent 1's proposed pay-offs.

![](_page_24_Figure_2.jpeg)

<span id="page-24-1"></span>(b) Average number of bargaining rounds.

Figure 9: Learning curve of (a) Agent 1's average proposed pay-offs (b) average number of bargaining rounds across all 3 agents for readability. Solid lines denote mean accuracy across 10 independent runs. Shaded regions denote  $\pm$  two standard deviations. Dashed lines denote the proposed pay-off of an equal gain share agent. In (a), after 10,000 epochs, Agent 1 converges on an approximately equal gain share. In (b), after 10,000 epochs agents reach agreement after an averaged 1.03 rounds of bargaining. Both of these results agree with Okada (1996).

since the underlying coalitional bargaining game is infinite-horizon, yet we must set a maximum number of bargaining rounds. Our ablation shows that increasing the maximum number of time-steps does not significantly change the quality of our agents' solutions. The agents still agree to share gains equally, with an average optimality gap of 4.1% (up from 3.9%) and identifying the correct coalitions 76% of the time (down from 77%). Therefore, we conclude that using a maximum number of time-steps of 10 to be sufficient. This is expected as we deal with time-limits properly as discussed in section [Section 4.8.](#page-19-0) The second ablation changes the agents' discount factor from 0.95 to 1.0. This ablation is carried out as we use a discount factor to reduce variance in the return. We test whether it's possible to use a higher discount factor. We find that using a discount factor of 1.0 decreases performance which we suspect to be due to the increased variance. With a discount factor of 1.0, agents achieve an average optimality gap of 6.07% (up from 3.9%). Agents do still learn to propose an approximately equal

<span id="page-25-0"></span>![](_page_25_Figure_0.jpeg)

Figure 10: The empirical pay-off agent 1 receives as a result of coalitional bargaining vs. the theoretical Shapley values for 2048 test instances. Green circle markers denote when agent 1 was included in the coalition. Orange cross markers denote when agent 1 was excluded from the coalition.  $R^2$  score of 0.76, mean squared error of 0.08 and mean absolute error of 0.01.

gain share but identifies thechnology coalitions only 68% of the time (down from 77%). We conclude that using a discount factor of 0.95 is sufficient to achieve a set of strong agents.

# 5.5. Discussion

In addition, our RL agents are able to reach agreement in 512 parallel instances within an average of 3.0s (or 0.006s per instance). We note that the prior literature assumes full access to the characteristic function, such as [Krajewska et al. \(2008\)](#). Using these prior methods to solve 512 instances takes 24.3s (or 0.047s per instance). Thus, our RL agents achieve a 88% reduction in computational time when compared with prior methods to calculate the Shapley value, such as in ([Krajewska et al. 2008](#)). Whilst 0.047s per instance may seem reasonable even with traditional methods, we stress that this is due to the simplistic VRP setting we consider – prior methods will not scale with the number of agents nor problem complexity via additional constraints such as time-windows. Importantly, our agents agree to outcomes that correlate well with Shapley values and thus we conclude that our method produces *fair* outcomes. This is important to fairly compensate carriers to enable wide-spread industrial adoption of collaborative vehicle routing. Our agents also reach agreement in a decentralised and self-interested manner, which overcomes the limitations of central orchestration methods mentioned in [Section 2](#).

Furthermore, our MARL agents are able to outperform the two baseline bots in both accuracy and optimality. The heuristic bot and random bot has an accuracy of 62% and 25% respectively, and an optimality gap of 8% and 32% respectively. The relatively low performance of both the heuristic bot and random bot suggests that the experimental setup is sufficiently challenging (due to the NP-hard nature of vehicle routing problems), and that simple policies are not performant in this setting. The heuristic bot shows that 38% of the time, it is not desirable to form the grand coalition as some agents may contribute very little. The random bot's high optimality gap shows that, whilst there is symmetry in our problem and depots are equidistant, the choice of partners is still important. This necessitates more intelligent agents and thus complex methods such as MARL. More importantly, we conclude that our MARL agents have learnt interesting behaviours, such as to exclude opponents if they contribute little to the coalition, as seen in [Figure 10.](#page-25-0)

In this work, we make the assumption that each carrier possesses only one truck. We further assume that the same truck driver is assigned to the same truck. This is a reasonable assumption as the road freight industry is highly fragmented: for example, in the UK, there are 60,000 registered carriers [\(Office for National Statistics](#page-36-21) [2022\)](#page-36-21) in 2022, and 1 million registered carriers in the EU in 2020 [\(Eurostat](#page-34-17) [2020\)](#page-34-17). However, if a single carrier possesses multiple trucks and thus multiple drivers, it would be possible to decompose the problem at different levels of granularity. One could consider coalitions of carriers; coalitions of trucks; or even coalitions of truck drivers. Our framework should be applicable to deal with all three types of modelling choices, but clearly the more granular the modelling choice, the more computational power that will be required.

The benefit of studying collaborative routing in a coalitional bargaining game is that game theory describes optimal, rational behaviour in this setting. As a result, we have a measure of the gap to optimality. This is important because of the challenging nature of 3-player, mixed-motive settings for MARL; thus, we can understand if the agents are learning correctly. However, there are three main limitations of this approach. Firstly, collaborative vehicle routing is most fruitful with a large number of participating carriers [\(Cruijssen et al.](#page-34-0) [2007;](#page-34-0) [Los et al.](#page-35-6) [2022\)](#page-35-6). Future work must investigate scaling our MARL approach to a larger number of carriers. We believe this to be possible in a hybrid centralised-decentralised manner. The advantage of our decentralised MARL approach is that it enables us to provide a large volume of high-quality solutions to optimise the central agent. Secondly, future work should investigate the performance of MARL-based approaches on real-world data distributions with real-world constraints. One direction would be to study the effect of data imbalance (such as the locations of depots and customers, as well as the delivery volumes) on the performance of MARL-based methods. Another direction could be to study the effect of partial observability of other carriers' information; we currently consider the perfect information scenario where all delivery information is publicly shared (though, crucially, the characteristic function is still unknown). It would be interesting in future work to explore imperfect information settings, such as the value of information sharing. This could be tackled using decentralised, partially observable Markov decision processes (dec-POMDPs) [\(Oliehoek and Amato](#page-36-13) [2016\)](#page-36-13). Thirdly, our approach currently only incentivises carriers. An independent third-party logistics provider may be required to enable collaborative routing. How should we incentivise third-party logistics providers? How should we incentivise shippers? What role could government play to incentivise collaboration? Moving in these directions with MARL would result in using more complex and flexible games; however, optimal,

rational behaviour would be unknown. Nevertheless, MARL may still be applied to these complex games but in a descriptive manner [\(Shoham et al.](#page-36-22) [2007\)](#page-36-22), i.e. to analyse the emergent behaviour of agents assuming a given MARL algorithm. We believe this

to be an exciting line of future research.

#### <span id="page-27-0"></span>6. Conclusions and Managerial Implications

Collaborative Vehicle Routing has promised cost savings between 4 - 46% in the last two decades. Yet industrial adoption remains limited. A key remaining barrier is the design of a gain sharing mechanism that is fair and scalable such that carriers are incentivsed to collaborate. Orchestration of truck sharing is usually proposed via a central optimiser, where an intermediary would receive information from each carrier and allocate trucks to each route. Subscription to intermediaries do not necessarily outweigh costs, and carriers typically do not obtain any benefits from sharing their trucks. In this paper, we propose an automated, decentralised approach, where software agents representing carriers find optimal routes through a coalitional bargaining game, and any gain obtained via improved truck utilisation is shared between the carriers. Manual orchestration costs are also avoided as the approach is automated.

To facilitate decentralised optimisation and fair gain sharing we utilised deep multi-agent reinforcement learning. The main challenge of our setting is the inability of extant methods to fully evaluate the characteristic function due to high computational complexity. The characteristic function calculates the collaboration gain for every possible coalition, which requires solving an exponential number of NP-hard VRPs. The autonomous agents designed in this work are able to correctly reason over a high-dimensional graph input to implicitly reason about the characteristic function instead. This eliminates the need to evaluate the expensive post-collaboration vehicle routing problem an exponential number of times and increases its practicability as we only need to evaluate this once. Furthermore, applying MARL to mixed-motive games is highly non-trivial and applying out-of-the-box MARL algorithms to this problem does not work. We show that we are able to achieve strong performance through careful design decisions, such as transfer learning, a targeted training design and COMA, and provide intuition for why these approaches help.

Moreover, the multi-agent reinforcement learning approach designed in our work is applicable to any coalitional bargaining game. Thus, our work may be suitable to problems in the broader collaborative logistics literature such as warehouse sharing. Another important point is that collaboration is not centrally orchestrated but facilitated using decentralised decision making. This marks an important step towards real-world adoption which might encourage transportation planners to consider more profitable and fair collaboration scenarios. Whilst we initially envisage this system operating as a decision support system, as transportation planners gain trust in the agents' decisions, we ultimately envisage this system to operate fully autonomously. This would enable even faster decision making that is traceable and consistent, potentially enabling a more responsive supply chain [\(Brintrup et al.](#page-34-18) [2009\)](#page-34-18). We urge transport planners and software system providers to consider potential adoption scenarios and integration into information systems.

Our work has limitations which provide avenues for future research. The current focus of this work is to obtain strong autonomous agents that maximally cooperate in the challenging mixed-motive setting of collaborative vehicle routing. Whilst we have achieved this, we have focused on a setting with 3 carriers as the focus of our work was to provide the theoretical link between collaborative vehicle routing, coalitional bargaining, and deep multi-agent reinforcement learning. Future work should investigate the scalability of a MARL approach to a larger number of agents. Furthermore, CVR problems typically include various additional considerations such as axle weights, goods compatibility, and packing orders, which have not yet been incorporated to the framework proposed here. Our approach is agnostic to the underlying optimisation design, and being so, we do not envisage the incorporation of additional problem features to hinder its function.

# Acknowledgement(s)

This work was performed using resources provided by the Cambridge Service for Data Driven Discovery (CSD3) operated by the University of Cambridge Research Computing Service (<www.csd3.cam.ac.uk>), provided by Dell EMC and Intel using Tier-2 funding from the Engineering and Physical Sciences Research Council (capital grant EP/T022159/1), and DiRAC funding from the Science and Technology Facilities Council (<www.dirac.ac.uk>).

We thank the three anonymous reviewers for their support and insightful comments during the review process which has greatly enhanced this paper. We also thank the Supply Chain Artificial Intelligence Lab (SCAIL) for their insightful discussions regarding early drafts of this paper.

# Disclosure statement

The authors report no conflict of interest.

# Funding

This work was supported by the UK Engineering and Physical Sciences Research Council (EPSRC) grant on "Intelligent Systems for Supply Chain Automation" under Grant Number 2275316, as well as by the UK EPSRC Connected Everything Network Plus under Grant EP/S036113/1.

#### <span id="page-29-0"></span>Appendix A. Capacitated vehicle routing problem

In on our paper, the pre-collaboration social welfare can be calculated by first solving three independent Capacitated Vehicle Routing Problems, where we assume an arbitrarily high capacity for each vehicle.

The capacitated vehicle routind problem (CVRP) and their variants have been studied for over 60 years (Toth and Vigo 2014). Here we show the *three-index (vehicle-flow) formulation*.

The CVRP considers the setting where goods are distributed to  $n$  customers. The goods are initially located at the *depot*, denoted by nodes (or vertices)  $o$  and  $d$ . Node  $o$  refers to the starting point of a route, and node  $d$  the end point of a route. The customers are denoted by the set of nodes  $N = \{1, 2, \dots, n\}$ . Each customer  $i \in N$  has a *demand*  $q_i \geq 0$ . In our setting, we consider  $q_i = 1$  for all customers. A *fleet* of  $|K|$  vehicles  $K = \{1, 2, \dots, |K|\}$  are said to be *homogeneous* if they all have the same capacity  $Q > 0$ . In our setting, we consider only one vehicle and set its capacity  $Q$  to an arbitrarily high number to remove the capacity constraint. A vehicle must start at the depot, and can deliver to a set of customers  $S \subseteq N$  before returning to the depot. The *travel cost*  $c_{i,j}$  is associated for a vehicle travelling between nodes  $i$  and  $j$  which we assume to be the Euclidean distance.

This problem can be modelled as a complete directed graph  $G = (V, A)$ , where the vertex set  $V := N \cup \{o, d\}$  and the arc set  $A := (V \setminus \{d\}) \times (V \setminus \{o\})$ . We define the *in-arcs* of  $S$  as  $\delta^-(S) = \{(i, j) \in A : i \notin S, j \in S\}$ . The *out-arcs* of  $S$  is  $\delta^+(S) = \{(i, j) \in A : i \in S, j \notin S\}$ .

The binary decision variables  $x_{ijk}$  denotes whether a vehicle  $k \in K$  travels over the arc  $(i, j) \in A$ . The binary decision variables  $y_{ik}$  denotes whether a vehicle  $k \in K$  visits node  $i \in V$ .  $u_{ik}$  denotes the load in vehicle  $k$  before visiting node  $i$ . We define the demand at the depot nodes  $o$  and  $d$  to be 0, i.e.  $q_o = q_d = 0$ . This yields:

$$\text{minimize} \quad \sum_{k\in K} c^T x_k \quad (1a)$$

$$\text{subject to } \sum_{k \in K} y_{ik} = 1, \quad \forall i \in N, \quad (1b)$$

<span id="page-29-3"></span><span id="page-29-2"></span><span id="page-29-1"></span>
$$x_k(\delta^+(i(i)) - x_k(\delta^-(i)) = \begin{cases} 1, & i = 0, \\ 0, & i \in N, \end{cases} \quad \forall i \in V \setminus \{d\}, k \in K, \quad (1c)$$

<span id="page-29-4"></span>
$$y_{ik} = x_k(\delta^+(i)) \quad \forall i \in V \setminus \{d\}, k \in K, \quad (1d)$$

<span id="page-29-5"></span>
$$y_{dk} = x_k(\delta^-(d)) \quad \forall k \in K, \quad (1e)$$

$$\text{and } y_{dk} + \text{Or}_{dk} \leq Q_{dk} \leq y_{dk} + \text{Or}_{dk} \quad \forall (i,j) \in A, k \in K. \quad (1f)$$

<span id="page-29-6"></span>
$$\begin{aligned} u_{}^{u_{ik}} - u_{jk} + Qx_{ijk} &\leq Q - q_j & \forall (i,j) \in A, k \in K, & (1f) \\ q_i &\leq q_j & \forall i,j \in V, k \in K & (1e) \end{aligned}$$

<span id="page-29-7"></span>
$$q_i \leq u_{ik} \leq Q \quad \forall i \in V, k \in K, \quad (21)$$

$$x = (x_k) \in \{0, 1\}^{K \times A}, \quad (1h)$$

$$y = (y_k) \in\{0,1\}^{K\times V}. \quad (\text{ii})$$

- The objective function (1a) minimises the Euclidean distance travelled by the vehicle.
- Constraint (1b) ) ensures the vehicle only visits each customer once.
  Constraint (1c ) ensures that the sum of vehicles entering node *d* and
- Constraint (1c) ensures that the sum of vehicles entering node  $d$  and exiting node  $d$  is  $-1$ . This ensures that a vehicle  $k$  performs a route starting at  $o$  and ending

at  $d$ .

- Constraintint (1d and 1e) couples variables  $x_{ijk}$  and  $y_{ik}$ .
  Constraint (1f) is the Miller-Tucker-Zemlin constraint
- Constraint (1f) is the Miller-Tucker-Zemlin constraint which helps eliminate subtours.
- Constraint (1g) is the capacity constraint.

#### <span id="page-30-0"></span>Appendix B. Multi-depot vehicle routing problem

In our paper, the post-collaboration social welfare can be calculated by solving the multi-depot vehicle routing problem (MDVRP) once. The number of depots corresponds to the number of agents within the accepted coalition. Again, we remove capacity constraints by setting the capacity of each vehicle to an arbitrarily large number. However, we add the additional constraint that each vehicle has to visit at least one customer.

Then MDVRP is a simple extension of the CVRP formulation provided in Appendix A. Instead of having the depot simply represented by nodes  $o$  and  $d$ , the depots are extended to belong to a specific vehicle  $k$  through nodes  $o_k$  and  $d_k$ . Doing so yields:

$$\text{minimize} \quad \sum_{k \in K} c^T x_k \quad (2a)$$

$$\text{subeta} = \sum_{k \in K} y_{ik} = 1, \quad \forall i \in V, \quad (2b)$$

<span id="page-30-3"></span><span id="page-30-2"></span><span id="page-30-1"></span>
$$x_k(\delta^+(i)) - x_k(\delta^-(i)) = \begin{cases} 1, & i = o_k, \\ 0, & i \in N, \end{cases} \quad \forall i \in V \setminus \{d_k\}, k \in K, \quad (2c)$$

<span id="page-30-4"></span>
$$y_{ik} = x_k(\delta^+(i)) \quad \forall i \in \{1, \dots, K\}, k \in K, \quad (2.1)$$

<span id="page-30-5"></span>
$$\begin{aligned} y_{d_{k,k}} &= x_k(\delta^-(d_k)) & \forall k \in K, & (2e) \\ &= 1 & \forall k \in K, & (2f) \end{aligned}$$

<span id="page-30-6"></span>
$$y_{d_k k} = 1 \quad \forall k \in K, \quad (2\text{f)$$

$$y_{d_k k+1} = O_{\text{rel}} \leq O_{\text{rel}} \quad \forall (i,j) \in A \cup b \in K, \quad (2n)$$

<span id="page-30-7"></span>
$$\begin{aligned} u_{ik} -u_{jk} + Qx_{ijk} &\leq Q - q_j & \forall (i,j) \in A, k \in K, & (2g) \\ a &\leq a - Q & \forall i \in V, k \in K & (2h) \end{aligned}$$

<span id="page-30-8"></span>
$$q_i \leq u_{ik} \leq Q \quad \forall i \in V, k \in K, \quad (2a)$$

$$x = (x_k) \in \{0, 1\}^{K \times A}, \quad (2.1)$$

$$y = (y_k) \in \{0, 1\}^{K \times V}. \quad (2j)$$

- The objective function (2a) minimises the Euclidean distance travelled by all vehicles.
- Constraint (2b) ensures that each vehicle only visits each customer once.
  Constraint (2c) ensures that the sum of vehicles entering node  $d_t$  and
- Constraint (2c) ensures that theach vehicle entering node  $d_k$  and exiting node  $d_k$  is  $-1$ . This ensures that a vehicle  $k$  performs a route starting at  $o_k$  and ending at  $d_k$ .
- Constraint (2d and 2e) couples variables  $x_{ijk}$  and  $y_{ik}$ .
  Constraint 2f ensures that each vehicle performs at least one constraint.
- Constraint **2f** e ensures that each vehicle performs at least one delivery.
  Constraint (**2e**) is the Miller-Tucker-Zemlin constraint which helps
- Constraint (2g) is the Miller-Tucker-Zemlin constraint which helps eliminate subtours.
- Constraint (2h) is the capacity constraint.

# Appendix C. Expected Number of Bargaining Rounds by a Random bot

Let  $X be a discrete random variable denoting the number of bargaining rounds. Let's assume we have a random agent as discussed in [Section 5.3.2](#) which proposes coalitions, pay-off vectors and responses uniformly at random. We wish to calculate the expected number of bargaining rounds achieved by three random bots,  $\mathbb{E}[X]$ . The maximum number of bargaining rounds is 10 in our experiments (although our ablations show that increasing this to 30 has no meaningful difference).$ 

$$\mathbb{E}[X] = \sum_{k=1}^{10} x \cdot P(X = x) \quad (\text{C1})$$

$$= 1 \cdot P(X = 1) + 2 \cdot P(X = 2) + \dots + 10 \cdot P(X = 10) \quad (\text{C2})$$

To obtain  $P(X = 1)$ , note that e.g. for Player 2, the random bot can propose four coalitions,  $C = \{1, 2, 3\}$ ,  $\{1, 2\}$ ,  $\{2, 3\}$  or  $\{2\}$  since Player 2 must be in the coalition  $C$ . If the coalition  $C = \{1, 2, 3\}$  is proposed, then both Players 1 and 3 must accept for the bargaining process to terminate, which yields a probability of acceptance (and thus termination) of  $\frac{1}{2}^2$ .

Therefore,  $P(X = 1)$  can be re-written as follows:

$$P(X = 1) = \left[ [ P(|C| = 3) \times \frac{1}{2} \right ] + \left[ P(|C| = 2) \times \frac{1}{2} \right ] + [P(|C| = 1)] \quad (\text{C3})$$

$$= \left[ 0.25 \times \frac{1}{2}^2 \right] + \left[ (0.25 + 0.25) \times \frac{1}{2} \right] + [0.25] \quad (\text{C4})$$

Repeating a similar logic to calculate  $\mathbb{E}[X]$  yields an expected number of bargaining rounds of 1.775.

$$\mathbb{E}[X] = (1 \cdot 0.5625) + (2 \cdot 0.2461) + (3 \cdot 0.1077) + (4 \cdot 0.0471) + \dots + (10 \cdot 0.0003) \quad (\text{C5})$$

= 1.775 (C6)

Emay be averaged over 10 rounds, and then averaged over 10 rounds again. The resulting average is the average of the rounds averaged over 10 rounds.

#### <span id="page-32-0"></span>Appendix D. Pseudo-code of the entire pipeline Algorithm 1 Pseudo-code of MARL pipeline 2: 4: for θ<sup>i</sup> in θ do 5: (∆ˆy<sup>θ</sup><sup>i</sup> ) <sup>2</sup> = (v(C) − <sup>y</sup>ˆ<sup>θ</sup><sup>i</sup> ) 6: <sup>∆</sup>θ<sup>i</sup> <sup>=</sup> ∇<sup>θ</sup><sup>i</sup> (∆ˆy<sup>θ</sup><sup>i</sup> ) 8: 9: // MARL Training 10: for each training epoch e do 13: while <sup>s</sup><sup>t</sup> ̸<sup>=</sup> terminal and t < T do 14: t += 1 15: // Calculate joint actions a 16: for i in N do 17: <sup>a</sup>i,t ∼ <sup>π</sup><sup>θ</sup><sup>i</sup> 18: <sup>s</sup>t+1 ∼ T (s<sup>t</sup> 19: <sup>R</sup>i,t ∼ R(s<sup>t</sup> , A<sup>t</sup> 20: Store each ⟨<sup>s</sup>i,t, ai,t, log(π<sup>θ</sup><sup>i</sup> 21: 22: // Here, all M episodes will be finished 23: for t = 1 to T do 24: Gi,t = P<sup>T</sup> t ′=t γ t 25: for t=T down to 1 do 26: (∆Qi,t) <sup>2</sup> = h <sup>G</sup>i,t − <sup>Q</sup><sup>ˆ</sup> <sup>θ</sup>critic (si,t, a) i2

1. 1: Initialise  $\theta = \theta_1, \theta_2, \dots, \theta_n, \theta_{critic}$  //  $n$  actors' (neural network) policy and critic
2. 2:
3. 3: // Supervised Pre-training (regression, minimise mean-squared error)
4. 4: **for**  $\theta_i$  in  $\theta$  **do**
5. 5:  $(\Delta\hat{y}_{\theta_i})^2 = (v(C) - \hat{y}_{\theta_i})^2$  // Calculate loss
6. 6:  $\Delta\theta_i = \nabla_{\theta_i}(\Delta\hat{y}_{\theta_i})^2$  // Calculate gradients
7. 7:  $\theta_i = \theta_i + \alpha\Delta\theta_i$  // Update parameters
8. 8:
9. 9: // MARL Training
10. 10: **for** each training epoch  $e$  **do**
11. 11: Initialise  $M = 2048$  parallel environments // Coalitional bargaining envs.
12. 12:  $s_1 \sim \rho_1, t = 0$  // Sample the initial state  $s_1$  from  $\rho_1$
13. 13: **while**  $s_t \neq \text{terminal and } t < T$  **do**
14. 14:  $t += 1$
15. 15: // Calculate joint actions **a**
16. 16: **for** i in **N do**
17. 17:  $a_{i,t} \sim \pi_{\theta_i}(a_{i,t}|s_{i,t})$  // Select actions stochastically for exploration
18. 18:  $s_{t+1} \sim \mathcal{T}(s_t, A_t)$  // Sample next state from transition dynamics
19. 19:  $R_{i,t} \sim \mathcal{R}(s_t, A_t, s_{t+1})$   $\forall i \in N$  // Calculate reward
20. 20: Store each  $\langle s_{i,t}, a_{i,t}, \log(\pi_{\theta_i}(a_{i,t}|s_{i,t})), s_{i,t+1}, R_{i,t} \rangle \forall i \in N$  in agent  $i$ 's buffer
21. 21:
22. 22: // Here, all  $M$  episodes will be finished
23. 23: **for**  $t = 1$  **to T do**
24. 24:  $G_{i,t} = \sum_{t'=t}^T \gamma^{t'-t} R_{i,t}$   $\forall i \in N$  // Calculate discounted returns
25. 25: **for**  $t=T$  **down to** 1 **do**
26. 26:  $(\Delta Q_{i,t})^2 = \left[ G_{i,t} - \hat{Q}_{\theta_{critic}}(s_{i,t}, \mathbf{a}) \right]^2$  // Calculate critic loss
27. 27:  $\Delta\theta_{critic} = \nabla_{\theta_{critic}}(\Delta Q_{i,t})^2$  // Calculate critic gradients
28. 28:  $\theta_{critic} = \theta_{critic} + \alpha\Delta\theta_{critic}$  // Update critic parameters
29. 29: **for**  $t=T$  **down to** 1 **do**
30. 30: // Calculate proposal baseline
31. 31:  $A_{t,prop.}^i = G_{i,t} - \hat{V}(s, \theta_{critic})$   $\forall i \in N$
32. 32: // Calculate response baseline
33. 33:  $A_{t,resp.}^i = G_{i,t} - \sum_a \hat{Q}(s_{i,t}, a, \mathbf{a}^{-a}, \theta_{critic}) \pi_{\theta_i}(a_{i,t}|s_{i,t})$   $\forall i \in N$
34. 34: // Accumulate actor proposal gradients
35. 35:  $\Delta\theta_i += \nabla_{\theta_i} \left[ \min(r_t(\theta_i) A_{t,prop.}^i, \text{clip}(r_t(\theta_i), 1 - \varepsilon, 1 + \varepsilon) A_{t,prop.}^i) \right] \forall i \in N$
36. 36: // Accumulate actor response gradients
37. 37:  $\Delta\theta_i += \nabla_{\theta_i} \left[ \min(r_t(\theta_i) A_{t,resp.}^i, \text{clip}(r_t(\theta_i), 1 - \varepsilon, 1 + \varepsilon) A_{t,resp.}^i) \right] \forall i \in N$
38. 38:  $\theta_i = \theta_i + \alpha\Delta\theta_i$   $\forall i \in N$  // Update actors' policy parameters
39. 39:  $\Delta\theta_i = \mathbf{0}$  // Reset gradients

# <span id="page-33-0"></span>Appendix E. Holistic Diagram of our Pipeline

![](_page_33_Diagram_1.jpeg)

Figure E1: A holistic diagram of our proposed approach. We depict a sample trajectory where Agent 2 is selected to propose and that Agent 2 proposes to form the grand coalition with an equal payoff vector. Agents 1 and 3 then agree to the proposal. Finally, the actors' parameters and critic's parameters are updated accordingly.

#### References

- <span id="page-34-15"></span><span id="page-34-4"></span>Adenso-Díaz, B., S. Lozano, and P. Moreno (2014). Analysis of the synergies of merging multi-company transportation needs. Transportmetrica A: Transport Science 10 (6), 533–547. Publisher: Informa UK Limited. Agarwal, R., M. Schwarzer, P. S. Castro, A. Courville, and M. G. Bellemare (2022, October). Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress. arXiv:2206.01626 [cs, stat]. Angelelli, E., V. Morandi, and M. G. Speranza (2022, July). Optimization models for fair horizontal collaboration in demand-responsive transportation. Transportation Research Part C: Emerging Technologies 140, 103725. Bachrach, Y., R. Everett, E. Hughes, A. Lazaridou, J. Z. Leibo, M. Lanctot, M. Johanson,
- <span id="page-34-18"></span><span id="page-34-17"></span><span id="page-34-16"></span><span id="page-34-14"></span><span id="page-34-13"></span><span id="page-34-12"></span><span id="page-34-11"></span><span id="page-34-10"></span><span id="page-34-9"></span><span id="page-34-8"></span><span id="page-34-7"></span><span id="page-34-6"></span><span id="page-34-5"></span><span id="page-34-3"></span><span id="page-34-2"></span><span id="page-34-1"></span><span id="page-34-0"></span>W. M. Czarnecki, and T. Graepel (2020, November). Negotiating team formation using deep reinforcement learning. Artificial Intelligence 288, 103356. Baker, B., I. Kanitscheider, T. Markov, Y. Wu, G. Powell, B. McGrew, and I. Mordatch (2020, February). Emergent Tool Use From Multi-Agent Autocurricula. arXiv:1909.07528 [cs, stat]. arXiv: 1909.07528. Bo Dai and H. Chen (2009). Mathematical model and solution approach for collaborative logistics in less than truckload (LTL) transportation. In 2009 International Conference on Computers & Industrial Engineering. IEEE. Brintrup, A. (2021, January). AI in the Supply Chain: a classification framework and critical analysis of current state. In Oxford Handbook of Supply Chain Management. Oxford University Press. Brintrup, A., D. Ranasinghe, S. Kwan, A. K. Parlikad, and K. Owens (2009, January). Roadmap to Self-Serving Assets in Civil Aerospace. Proceedings of the 1st CIRP Industrial Product-Service Systems (IPS2) Conference. Chalkiadakis, G. and C. Boutilier (2004, July). Bayesian reinforcement learning for coalition formation under uncertainty. In Proceedings of the Third International Joint Conference on Autonomous Agents and Multiagent Systems, 2004. AAMAS 2004., pp. 1090–1097. Chalkiadakis, G., E. Elkind, and M. Wooldridge (2011). Computational Aspects of Cooperative Game Theory (Synthesis Lectures on Artificial Inetlligence and Machine Learning) (1st ed.). Morgan & Claypool Publishers. Chou, P.-W., D. Maturana, and S. Scherer (2017, July). Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution. In Proceedings of the 34th International Conference on Machine Learning, pp. 834–843. PMLR. ISSN: 2640-3498. Cruijssen, F. (2020, January). Cross-Chain Collaboration in Logistics: Looking Back and Ahead. International Series in Operations Research and Management Science. Springer. Cruijssen, F., O. Bräysy, W. Dullaert, H. Fleuren, and M. Salomon (2007). Joint route planning under varying market conditions. International Journal of Physical Distribution & Logistics Management 37 (4), 287–304. Publisher: Emerald. Cruijssen, F., M. Cools, and W. Dullaert (2007). Horizontal cooperation in logistics: Opportunities and impediments. Transportation Research Part E: Logistics and Transportation Review 43 (2), 129–142. Publisher: Elsevier BV. Deng, X. and C. H. Papadimitriou (1994). On the Complexity of Cooperative Solution Concepts. Mathematics of Operations Research 19 (2), 257–266. Publisher: INFORMS. Eurostat (2020). Annual detailed enterprise statistics for services (NACE Rev. 2 H-N and S95): SBS\_na\_1a\_se\_r2. Ferrell, W., K. Ellis, P. Kaminsky, and C. Rainwater (2020). Horizontal collaboration: opportunities for improved logistics planning. International Journal of Production Research 58 (14), 4267–4284. Publisher: Informa UK Limited. Foerster, J., G. Farquhar, T. Afouras, N. Nardelli, and S. Whiteson (2017, December). Counterfactual Multi-Agent Policy Gradients. arXiv:1705.08926 [cs]. arXiv: 1705.08926. Foerster, J. N., Y. M. Assael, N. de Freitas, and S. Whiteson (2016, May). Learning to

- <span id="page-35-16"></span><span id="page-35-15"></span><span id="page-35-12"></span><span id="page-35-8"></span><span id="page-35-5"></span><span id="page-35-1"></span><span id="page-35-0"></span>Communicate with Deep Multi-Agent Reinforcement Learning. arXiv:1605.06676 [cs]. Fox, M. S., M. Barbuceanu, and R. Teigen (2000). Agent-Oriented Supply-Chain Management. In Information-Based Manufacturing, pp. 81–104. Boston, MA: Springer US. Gabel, T. and M. Riedmiller (2012). Distributed policy search reinforcement learning for job-shop scheduling tasks. International Journal of Production Research 50 (1), 41–61. Publisher: Informa UK Limited. Gansterer, M. and R. F. Hartl (2018a). Centralized bundle generation in auction-based collaborative transportation. OR Spectrum 40 (3), 613–635. Publisher: Springer Science and Business Media LLC. Gansterer, M. and R. F. Hartl (2018b). Collaborative vehicle routing: A survey. European Journal of Operational Research 268 (1), 1–12. Publisher: Elsevier BV. Gansterer, M. and R. F. Hartl (2020). Shared resources in collaborative vehicle routing. TOP 28 (1), 1–20. Publisher: Springer Science and Business Media LLC. Gansterer, M., R. F. Hartl, and R. Vetschera (2019). The cost of incentive compatibility in auction-based mechanisms for carrier collaboration. Networks 73 (4), 490–514. \_eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/net.21828. Greensmith, E., P. L. Bartlett, and J. Baxter (2004, December). Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning. J. Mach. Learn. Res. 5, 1471–1530. Publisher: JMLR.org. Guajardo, M. and M. Rönnqvist (2016). A review on cost allocation methods in collaborative transportation. International Transactions in Operational Research 23 (3), 371–392. Publisher: Wiley. Gurobi Optimization, LLC (2021). Gurobi Optimizer Reference Manual. Henderson, P., R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger (2019, January). Deep Reinforcement Learning that Matters. arXiv:1709.06560 [cs, stat]. arXiv: 1709.06560. Ieong, S. and Y. Shoham (2005). Marginal contribution nets: a compact representation scheme for coalitional games. In Proceedings of the 6th ACM conference on Electronic commerce - EC '05, Vancouver, BC, Canada, pp. 193–202. ACM Press. Kosasih, E. E. and A. Brintrup (2021, July). Reinforcement Learning Provides a Flexible Approach for Realistic Supply Chain Safety Stock Optimisation. arXiv. arXiv:2107.00913 [cs]. Krajewska, M. A., H. Kopfer, G. Laporte, S. Ropke, and G. Zaccour (2008). Horizontal cooperation among freight carriers: request allocation and profit sharing. Journal of the Operational Research Society 59 (11), 1483–1491. Publisher: Informa UK Limited. Kurach, K., A. Raichuk, P. Stańczyk, M. Zając, O. Bachem, L. Espeholt, C. Riquelme,
- <span id="page-35-18"></span><span id="page-35-17"></span><span id="page-35-13"></span><span id="page-35-11"></span><span id="page-35-9"></span><span id="page-35-7"></span><span id="page-35-6"></span><span id="page-35-4"></span><span id="page-35-2"></span>D. Vincent, M. Michalski, O. Bousquet, and S. Gelly (2020, April). Google Research Football: A Novel Reinforcement Learning Environment. Technical Report arXiv:1907.11180, arXiv. arXiv:1907.11180 [cs, stat] type: article. Leibo, J. Z., V. Zambaldi, M. Lanctot, J. Marecki, and T. Graepel (2017, February). Multi-agent Reinforcement Learning in Sequential Social Dilemmas. arXiv:1702.03037 [cs]. arXiv: 1702.03037. Los, J., F. Schulte, M. Gansterer, R. F. Hartl, M. T. J. Spaan, and R. R. Negenborn (2022). Large-scale collaborative vehicle routing. Annals of Operations Research. Publisher: Springer Science and Business Media LLC. Lowe, R., Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch (2020, March). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. arXiv:1706.02275 [cs]. arXiv: 1706.02275. Mnih, V., K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves,
  - M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik,
- <span id="page-35-14"></span><span id="page-35-10"></span><span id="page-35-3"></span>I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis (2015, February). Human-level control through deep reinforcement learning. Nature 518 (7540), 529–533. Mordatch, I. and P. Abbeel (2018, July). Emergence of Grounded Compositional Language in Multi-Agent Populations. Technical Report arXiv:1703.04908, arXiv. arXiv:1703.04908 [cs]

type: article.

- <span id="page-36-21"></span><span id="page-36-18"></span><span id="page-36-13"></span><span id="page-36-9"></span><span id="page-36-3"></span>Murphy, K. P. (2021). Probabilistic Machine Learning: An introduction. MIT Press. Nash, J. (1953). Two-Person Cooperative Games. Econometrica 21 (1), 128. Publisher: JSTOR. Office for National Statistics (2022, September). UK business: activity, size and location - Office for National Statistics. Okada, A. (1996). A Noncooperative Coalitional Bargaining Game with Random Proposers. Games and Economic Behavior 16 (1), 97–108. Publisher: Elsevier BV. Oliehoek, F. A. and C. Amato (2016). A Concise Introduction to Decentralized POMDPs. SpringerBriefs in Intelligent Systems. Cham: Springer International Publishing. OpenAI (2022). ChatGPT. OpenAI, C. Berner, G. Brockman, B. Chan, V. Cheung, P. Dębiak, C. Dennison, D. Farhi,
  - Q. Fischer, S. Hashme, C. Hesse, R. Józefowicz, S. Gray, C. Olsson, J. Pachocki, M. Petrov,
  - H. P. d. O. Pinto, J. Raiman, T. Salimans, J. Schlatter, J. Schneider, S. Sidor, I. Sutskever,
- <span id="page-36-20"></span><span id="page-36-17"></span><span id="page-36-14"></span><span id="page-36-12"></span><span id="page-36-7"></span><span id="page-36-4"></span><span id="page-36-2"></span>J. Tang, F. Wolski, and S. Zhang (2019, December). Dota 2 with Large Scale Deep Reinforcement Learning. arXiv:1912.06680 [cs, stat]. arXiv: 1912.06680. Palhazi Cuervo, D., C. Vanovermeire, and K. Sörensen (2016). Determining collaborative profits in coalitions formed by two partners with varying characteristics. Transportation Research Part C: Emerging Technologies 70, 171–184. Publisher: Elsevier BV. Pan, S., D. Trentesaux, E. Ballot, and G. Q. Huang (2019). Horizontal collaborative transport: survey of solutions and practical implementation issues. International Journal of Production Research 57 (15-16), 5340–5361. Publisher: Informa UK Limited. Pardo, F., A. Tavakoli, V. Levdik, and P. Kormushev (2018, July). Time Limits in Reinforcement Learning. arXiv:1712.00378 [cs]. arXiv: 1712.00378. Powell, W. (2022). Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions. Wiley. Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming (1st ed.). USA: John Wiley & Sons, Inc. Samvelyan, M., T. Rashid, C. S. de Witt, G. Farquhar, N. Nardelli, T. G. J. Rudner, C.-M. Hung, P. H. S. Torr, J. Foerster, and S. Whiteson (2019, December). The StarCraft Multi-Agent Challenge. Technical Report arXiv:1902.04043, arXiv. arXiv:1902.04043 [cs, stat] type: article. Schulman, J., F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017, August). Proximal Policy Optimization Algorithms. arXiv:1707.06347 [cs]. arXiv: 1707.06347. Serrano, R. (2004). Fifty Years of the Nash Program, 1953-2003. SSRN Electronic Journal. Shoham, Y., R. Powers, and T. Grenager (2007). If multi-agent learning is the answer, what is the question? Artificial Intelligence 171 (7), 365–377. Publisher: Elsevier BV. Silver, D., A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser,
  - I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham,
  - N. Kalchbrenner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and
- <span id="page-36-23"></span><span id="page-36-22"></span><span id="page-36-19"></span><span id="page-36-16"></span><span id="page-36-15"></span><span id="page-36-11"></span><span id="page-36-10"></span><span id="page-36-8"></span><span id="page-36-6"></span><span id="page-36-5"></span><span id="page-36-1"></span><span id="page-36-0"></span>D. Hassabis (2016, January). Mastering the game of Go with deep neural networks and tree search. Nature 529 (7587), 484–489. Sutton, R. S. and A. G. Barto (2018). Reinforcement learning: an introduction (Second edition ed.). Adaptive computation and machine learning series. Cambridge, Massachusetts: The MIT Press. Sutton, R. S., D. Mcallester, S. Singh, and Y. Mansour (2000). Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems 12, Volume 12, pp. 1057–1063. MIT Press. Taylor, M. E. and P. Stone (2009, December). Transfer Learning for Reinforcement Learning Domains: A Survey. J. Mach. Learn. Res. 10, 1633–1685. Publisher: JMLR.org. Toth, P. and D. Vigo (Eds.) (2014). Vehicle Routing: Problems, Methods, and Applications, Second Edition. Number 18 in MOS-SIAM Series on Optimization. SIAM. UK BEIS (2021, February). Final UK greenhouse gas emissions national statistics. UK DfT (2020, July). Road freight statistics: 2019. Vinyals, O., I. Babuschkin, W. M. Czarnecki, M. Mathieu, A. Dudzik, J. Chung, D. H. Choi,

- R. Powell, T. Ewalds, P. Georgiev, J. Oh, D. Horgan, M. Kroiss, I. Danihelka, A. Huang,
- L. Sifre, T. Cai, J. P. Agapiou, M. Jaderberg, A. S. Vezhnevets, R. Leblond, T. Pohlen,
- V. Dalibard, D. Budden, Y. Sulsky, J. Molloy, T. L. Paine, C. Gulcehre, Z. Wang, T. Pfaff,
- Y. Wu, R. Ring, D. Yogatama, D. Wünsch, K. Mckinney, O. Smith, T. Schaul, T. Lillicrap,
- <span id="page-37-4"></span>K. Kavukcuoglu, D. Hassabis, C. Apps, and D. Silver (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature 575 (7782), 350–354. Publisher: Springer Science and Business Media LLC. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning 8 (3-4), 229–256. Publisher: Springer Science and Business Media LLC. Wurman, P. R., S. Barrett, K. Kawamoto, J. MacGlashan, K. Subramanian, T. J. Walsh,
  - R. Capobianco, A. Devlic, F. Eckert, F. Fuchs, L. Gilpin, P. Khandelwal, V. Kompella,
  - H. Lin, P. MacAlpine, D. Oller, T. Seno, C. Sherstan, M. D. Thomure, H. Aghabozorgi,
- <span id="page-37-5"></span><span id="page-37-3"></span><span id="page-37-2"></span><span id="page-37-1"></span><span id="page-37-0"></span>L. Barrett, R. Douglas, D. Whitehead, P. Dürr, P. Stone, M. Spranger, and H. Kitano (2022, February). Outracing champion Gran Turismo drivers with deep reinforcement learning. Nature 602 (7896), 223–228. Xu, L., S. Mak, and A. Brintrup (2021). Will bots take over the supply chain? Revisiting agent-based supply chain automation. International Journal of Production Economics 241, 108279. Publisher: Elsevier BV. Yan, Y., A. H. F. Chow, C. P. Ho, Y.-H. Kuo, Q. Wu, and C. Ying (2022, June). Reinforcement learning for logistics and supply chain management: Methodologies, state of the art, and future opportunities. Transportation Research Part E: Logistics and Transportation Review 162, 102712. Yu, C., A. Velu, E. Vinitsky, Y. Wang, A. Bayen, and Y. Wu (2021, March). The Surprising Effectiveness of MAPPO in Cooperative, Multi-Agent Games. arXiv:2103.01955 [cs]. arXiv: 2103.01955. Zhang, M., S. Pratap, G. Q. Huang, and Z. Zhao (2017). Optimal collaborative transportation service trading in B2B e-commerce logistics. International Journal of Production Research 55 (18), 5485–5501. Publisher: Informa UK Limited.