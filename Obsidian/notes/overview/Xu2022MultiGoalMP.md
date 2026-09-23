---
title: |
  Multi-Goal Multi-Agent Pickup and Delivery
author: |
  Qinghong Xu and Jiaoyang Li and Sven Koenig and Hang Ma
year: "2022"
bibtex_id: "Xu2022MultiGoalMP"
---
This article introduces a problem of Pickup and Delivery with online tasks. To execute a task, an agent needs to visit a pair of goal locations, consisting of a pickup location and a delivery location with a possible Multi-Goal extensions where a single task consists of different numbers of goal locations. Each agent receives a sequence of tasks it must complete - the assignment being done with Large Neighbourhood Search. Then a MAPF algorithm (Priority-Based Search) plans paths with an entire sequence in mind. Who decides the order remains unknown, however sequences are usually order so I assume it's LNS. Authors also include the possibility of online tasks hence this problem might be very similar to mine. Overall not that interesting since it doesn't use RL.

## Related works
- G. Sharon, R. Stern, A. Felner, and N. R. Sturtevant, "Conflict-based search for optimal multi-agent pathfinding," in *AAAI Conference on Artificial Intelligence*, 2012, pp. 563–569.
- E. Boyarski, A. Felner, R. Stern, G. Sharon, D. Tolpin, O. Betzalel, and E. Shimony, "ICBS: Improved conflict-based search algorithm for multi-agent pathfinding," in *International Joint Conference on Artificial Intelligence*, 2015, pp. 740–746.
Both of those works explore the (Improved) Conflict-Based Search which claim to be the complete and optimal MAPF algorithms.


## Problem definition
1. At each time-step, the system can release new tasks;
2. A task is an ordered sequenced of goal locations - with two special locations being the first and last "depot";
3. Goal locations are called task endpoints and start locations non-task endpoints;
4. The problem is to assign tasks to agents and plan collision-free paths to execute all tasks assigned to them;
5. The effectiveness is measured by the average service time - the time that task spends in the system;
6. The efficiency is measured by the average time per time-step;
7. Each agent maintains a dummy endpoint, i.e., an endpoint that it can move to and stay indefinitely at without collisions (initially, this dummy endpoint is its start location).

## Proposed algorithm
1. **Large Neighbourhood Search** assigns tasks - starts with Hungarian-based insertion and improves it with Shaw removal and regret-based re-insertion;
2. Each agent is assigned a dummy endpoint ("depot");
3. **Priority-Based Search** is the MAPF algorithm used in this work:
	- It constructs the shortest path (with A*) for each agent ignoring collisions;
	- Then it starts building a Priority Tree and performs depth-first search on it to construct a priority ordering of the agents;
	- The PT starts empty, only with a root; when a collision is detected it adds two child nodes: one child node that the first agent involved in the collision has a higher priority than the second one and vice versa for the other child node;
	- in each child node, the agent that got the lower priority has its path re-planned with A*;
	- if no such path exists, the child node is pruned (dead end);
	- the depth-first search continues from one of the children - if the new paths still contain a collision, it is resolved the same way, adding one more priority pair at each level of the tree;
	- the search stops at the first node with no collisions - the priority pairs collected along the way form the priority ordering, and that node's paths are the solution;
4. **LNS-wPBS** is a variant of LNS-PBS that, unlike LNS-PBS, uses windowed PBS (wPBS) for planning collision free paths for only the first *w* time-steps and then plan path again once the agents have moved for *w* time-steps. This makes LNS-wPBS more efficient than LNS-PBS.
## Experiment setting
- **Maps** (4-connected grids, warehouse "small" taken from Liu et al. 2019):
	- *small* 35×21 — 2×5 shelf strips in the middle, task endpoints above/below each strip, agent start locations (non-task endpoints) in columns on the left and right sides;
	- *medium* 101×81 and *large* 187×153 — the same layout scaled up (8×40 and 15×76 shelf strips).
- **Tasks**: 500 (small), 1,000 (medium), 1,000–5,000 (large); goal locations drawn uniformly at random from task endpoints; 2 goals per task for MAPD, 1–5 (random) for MG-MAPD.
- **Number of agents** *M*: 10–50 (small), 100–500 (medium), 1,000 (large).
- **Baselines**: CENTRAL, RMCA, HBH+MLA\* (TA-Hybrid is offline-only; RHCR and (SMT-)HCBS need an external task assigner, so they're excluded).
- **Metrics**: average service time per task (*st*), average runtime per timestep in ms (*rt*), and runtime over time (stability), split into task-assignment and path-finding time.
