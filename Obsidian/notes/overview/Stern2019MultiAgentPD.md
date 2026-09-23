---
title: |
  Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks
author: |
  Roni Stern and Nathan R Sturtevant and Ariel Felner and Sven Koenig and Hang Ma and Thayne T. Walker and Jiaoyang Li and Dor Atzmon and Liron Cohen and T. K. Satish Kumar and Eli Boyarski and Roman Bart{\'a}k
year: "2019"
bibtex_id: "Stern2019MultiAgentPD"
---
This article describes the taxonomy of MAPF problems: conflicts, objectives and the benchmark conventions.

## Types of conflicts

![](_page_1_Diagram_5.jpeg)

An illustration of common types of conflicts. From left to right: an edge conflict, a vertex conflict, a following conflict, a cycle conflict, and a swapping conflict.

## Objective functions

- **Makespan.** The number of time steps required for all agents to reach their target. For a MAPF solution $\pi = \{\pi_1, \dots, \pi_k\}$, the makespan of $\pi$ is defined as $\max_{1 \leq i \leq k} |\pi_i|$.
- **Sum of costs.** The sum of time steps required by each agent to reach its target. The sum of costs of $\pi$ is defined as $\sum_{1 \leq i \leq k} |\pi_i|$. Sum of costs is also known as *flowtime*.

## Problem variations

In classical MAPF each agent has only one target - even in Online or Lifelong MAPF each agent has only one target at a time; in other words there exists a set sequence for each agent and the order of vising is outside its scope. It is important to note if analysed problem holds this assumption. 

## Benchmarks

- Warehouse examples: Ma et al. 2017; Cohen et al. 2018a
- Benchmarking idea: *For a chosen MAPF algorithm, map type, and scenario, try to solve as many agents as possible in each scenario, adding them in consecutive order. That is, start by creating a MAPF problem of two agents, using the first two source-target pairs associated with the chosen scenario, and run the MAPF algorithm of choice to solve this problem. If the algorithm of choice successfully solves this MAPF problem in reasonable time, create a new MAPF problem with 3 agents by using the first three source-target pairs of that scenario and try to solve it with the MAPF algorithm of choice. This continues iteratively until the algorithm of choice cannot solve the created MAPF problem in reasonable time. An evaluated algorithm can then report, for every scenario, the maximal number of agents it was able to solve in reasonable time*

