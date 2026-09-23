---
title: |
  Lifelong Multi-Agent Path Finding for Online Pickup and Delivery Tasks
author: |
  Hang Ma and Jiaoyang Li and T. K. Satish Kumar and Sven Koenig
year: "2017"
bibtex_id: "Ma2017LifelongMP"
---
This work explores the Lifelong variant of MAPD and introduces Token Passing with Task Swaps. Tasks can enter the system at any time. Therefore, assigning agents to tasks and path planning cannot be done in advance but rather need to be done during execution in real-time. Overall not that interesting but the idea of tokens is worth noting.

The idea of a "well-formed" problem is described here in detail and might be usefull.

## Token passing

The token is a synchronised shared block of memory that contains the current paths of all agents, task set, and agent assignments. It works like a lock: agents take the token one after another, and only the agent holding it can pick a task and write its path into it. Agents plan one after the other.

Collisions are never resolved after the fact. Whoever gets the token first keeps its path, and later agents plan around it. It's greedy and first come, first served, which is why it's fast. But I doubt it's effective.

## Token Passing with Task Swaps (task stealing)

Same loop, but an agent may also take a task already assigned to another agent if that agent hasn't picked it up yet and the new agent would reach the pickup location sooner. The agent that lost the task then gets the token and tries to find a new task in the same way, which can cause a chain of swaps. If the chain fails, every change is undone. This gives shorter service times than TP (up to about 42% in their experiments) but more computation (under 200 ms per timestep) and not always a better result.

## CENTRAL

The algorithm CENTRAL (also mentioned in [[Xu2022MultiGoalMP]]) is the centralized baseline the authors compare TP and TPTS against. Instead of agents taking turns, one central planner decides everything:
1. Give every agent a target endpoint.
	- An agent standing on the pickup location of a waiting task gets that task, provided no other agent is already heading to the task's delivery location. Its target becomes the delivery location, and it is now occupied.
	- The free agents are matched to these endpoints with the Hungarian algorithm. The costs are scaled so that (a) a pickup always beats parking, and (b) giving one agent a closer pickup matters more than any parking distances.
2. Plan paths with CBS (optimal MAPF) for all agents at once, from their current cells to their targets. To make this faster, it's split into two smaller CBS runs: first the agents that just became occupied, then the free agents. Each run treats the other agents' latest paths as obstacles.
3. Everyone moves one step, and the whole thing repeats.