# Reading list

Ordered for the concrete problem: **vehicles receive a set of orders online, start and end at a depot, and must both sequence the orders well and avoid each other** — i.e. lifelong / online **multi-goal MAPD**, to be solved with MARL. Assignment of orders into sets is out of scope, so papers about task allocation are deliberately ranked low.

**Importance legend**

| Mark | Meaning |
|---|---|
| ⭐⭐⭐ | Core — read properly, take notes, expect to cite |
| ⭐⭐ | Important — read fully, but once |
| ⭐ | Useful — read selectively (method section / results) |
| ○ | Context — skim abstract + figures, park it |

**If you only read ten:** 1, 2, 3, 5, 11, 12, 13, 15, 18, 25.

---

## Phase 0 — Pin down the problem (start here)

Goal: get the vocabulary and the exact formal name of what you are solving, before reading any method.

- [x] ⭐⭐⭐ 1. [[Multi-Agent Pathfinding Definitions, Variants, and Benchmarks.pdf]] — Stern et al. 2019. Read first and keep open as a reference. Gives you the conflict taxonomy (vertex, edge, following, swapping, cycle) and the objective taxonomy (makespan vs. sum-of-costs), plus the standard benchmark conventions. Every later paper assumes this vocabulary; your env's collision rules and reward should be stated in these terms.
- [x] ⭐⭐⭐ 2. [[Multi-Goal Multi-Agent Pickup and Delivery.pdf]] — Xu, Li, Koenig, Ma 2022. The closest formal match to your setting: each agent has a *sequence* of goals rather than one. This is the paper that gives your problem a name — anchor your problem statement to it.
- [x] ⭐⭐⭐ 3. [[Lifelong Multi-Agent Path Finding for Online Pickup and Delivery Tasks.pdf]] — Ma et al. 2017. Defines MAPD and the online/lifelong setting; introduces Token Passing. Read for the problem model and for well-formed instance conditions (why a depot/parking structure matters for guaranteed liveness) — that constraint directly affects your depot design.
- [x] ⭐⭐ 4. [[THE MULTI-AGENT PICKUP AND DELIVERY PROBLEM MAPF, MARL AND ITS WAREHOUSE APPLICATIONS.pdf]] — Lau & Sengupta 2022. Short bridge between the MAPF/MAPD literature and MARL; useful precisely because it frames both halves at once. Read it to check your framing, not for a method.

## Phase 1 — Map the field

- [x] ⭐⭐⭐ 5. [[Where Paths Collide A Comprehensive Survey of Classic and Learning-Based Multi-Agent Pathfinding.pdf]] — Wang et al. 2025. The one survey worth reading end to end: covers classic search and learning-based MAPF in the same taxonomy, so it tells you where a MARL approach actually sits relative to CBS-family solvers. *Note: no markdown version in the vault yet, PDF only.*
- [ ] ⭐⭐ 6. [[Research Challenges and Opportunities in Multi-Agent Path Finding and Multi-Agent Pickup and Delivery Problems.pdf]] — Salzman & Stern 2020. Short position paper. Read it to harvest open problems — useful for motivating your contribution, and for checking that "optimal routing + collision avoidance jointly" is genuinely open rather than solved.

## Phase 2 — Classical solvers: your baselines and your expert

You need these even though you are doing MARL — as baselines to beat, and as a source of demonstrations if you go the imitation-learning route.

- [ ] ⭐⭐⭐ 7. [[Conflict-based search for optimal multi-agent pathfinding.pdf]] — Sharon et al. 2012. CBS. The foundational optimal solver; understand the two-level structure (constraint tree over conflicts, single-agent search below) because most of the field is described as a delta from it.
- [ ] ⭐⭐ 8. [[EECBS A Bounded-Suboptimal Search for Multi-Agent Path Finding.pdf]] — Li, Ruml, Koenig 2020. The practical bounded-suboptimal version. This is the realistic baseline to compare against on your maps.
- [ ] ⭐⭐⭐ 9. [[Lifelong Multi-Agent Path Finding in Large-Scale Warehouses.pdf]] — Li et al. 2020. RHCR: replan in a receding horizon for lifelong warehouse MAPF. This is the strongest non-learning baseline for *exactly* your setting and the one a reviewer will ask about. Read the horizon/replanning-frequency trade-off carefully — it is the argument for why a learned decentralised policy could win.
- [ ] ⭐ 10. [[LaCAM Search-Based Algorithm for Quick Multi-Agent Pathfinding.pdf]] — Okumura 2022. Very fast suboptimal search; relevant if you need a cheap expert to generate lots of training data, or a scale baseline.
- [ ] ○ 11. [[CBS Improved Conflict-Based Search Algorithm for Multi-Agent Pathfinding.pdf]] — Boyarski et al. 2015 (ICBS). Only if you end up implementing a CBS variant yourself; otherwise the improvements are implementation detail.

## Phase 3 — Learning-based MAPF (the heart of your project)

Read these in order — the line of work is genuinely cumulative.

- [ ] ⭐⭐⭐ 12. [[PRIMAL Pathfinding via Reinforcement and Imitation Multi-Agent Learning.pdf]] — Sartoretti et al. 2018. The reference point for decentralised learned MAPF: partial observation, RL + imitation from an expert, reward shaping for collisions. Pay attention to the observation encoding and the reward table — you will be reusing both shapes.
- [ ] ⭐⭐⭐ 13. [[PRIMAL2 Pathfinding via Reinforcement and Imitation Multi-Agent Learning - Lifelong.pdf]] — Damani et al. 2020. PRIMAL extended to the *lifelong* setting, which is your setting. Read for how they handle a new goal arriving mid-episode and for corridor/deadlock handling in structured warehouse maps.
- [ ] ⭐⭐ 14. [[Distributed Heuristic Multi-Agent Path Finding with Communication.pdf]] — Ma, Luo, Ma 2021. DHC: feeds a single-agent heuristic into a communicating decentralised policy. The "planner as a feature, network for the conflict resolution" pattern is probably the right division of labour for you too.
- [ ] ⭐⭐⭐ 15. [[Learn to Follow Decentralized Lifelong Multi-Agent Pathfinding via Planning and Learning.pdf]] — Skrynnik et al. 2023. Decentralised lifelong MAPF as *plan a route, learn to follow it without colliding*. This hybrid decomposition maps almost one-to-one onto your problem (sequence the order set; learn the collision-free following), so treat it as a candidate architecture rather than just related work.
- [ ] ⭐⭐ 16. [[POGEMA A BENCHMARK PLATFORM FOR COOPERATIVE MULTI-AGENT PATHFINDING.pdf]] — Skrynnik et al. 2024. Benchmark platform and metric set for learnable MAPF. Read it against your own env — either adopt it or be able to say why yours differs. Their metrics (throughput, CSR, SoC ratio, cooperation) are a ready-made evaluation protocol.
- [ ] ⭐ 17. [[MAPF-GPT Imitation Learning for Multi-Agent Pathfinding at Scale.pdf]] — Andreychuk et al. 2024. Pure imitation at scale with a transformer, no communication and no RL. Read for the tokenised observation design and as evidence about how far behaviour cloning alone gets you.

## Phase 4 — MARL algorithms and training machinery

Read this phase while implementing, not before.

- [ ] ⭐⭐⭐ 18. [[The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.pdf]] — Yu et al. 2021. MAPPO. Read for the practical recipe (centralised critic, parameter sharing, value normalisation, clipping choices) — this is the default algorithm you should start from and the implementation details matter more than the theory.
- [ ] ⭐⭐ 19. [[Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks.pdf]] — Papoudakis et al. 2020. Honest comparison across cooperative baselines with tuning details. Use it to justify your algorithm choice and to set expectations on variance/seeds.
- [ ] ⭐ 20. [[Learning Transferable Cooperative Behavior in Multi-Agent Teams.pdf]] — Agarwal, Kumar, Sycara 2019. Graph/attention-based policies that transfer across team sizes. Directly relevant because your number of active vehicles and orders varies, and you want one trained policy to handle that.
- [ ] ⭐ 21. [[Actor-Attention-Critic for Multi-Agent Reinforcement Learning.pdf]] — Iqbal & Sha 2018. MAAC: attention over other agents in the critic, so it scales with a variable number of neighbours. Read the critic construction only.
- [ ] ⭐ 22. [[Constrained Policy Optimization.pdf]] — Achiam et al. 2017. Read this *if* you decide collisions should be a hard constraint with a safety budget rather than a reward penalty — a defensible design choice given that collision avoidance is a hard requirement in your problem.
- [ ] ○ 23. [[Learning to Communicate with Deep Multi-Agent Reinforcement Learning.pdf]] — Foerster et al. 2016. Historical foundation for learned communication (DIAL/RIAL). Skim unless you build an explicit communication channel between vehicles.
- [ ] ○ 24. [[PC3D Zero-Shot Cooperation Across Variable Rosters via Personalized Context Distillation.pdf]] — Akman & Kucharski 2026. Zero-shot cooperation when the set of teammates changes. Park it — relevant only once a policy works and you want robustness to fleet composition.

## Phase 5 — The routing half: sequencing the assigned order set

Your agents must decide *which order next*, which is a learned-routing problem. This literature is mature; borrow its decoders rather than reinventing them.

- [ ] ⭐⭐⭐ 25. [[ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS.pdf]] — Kool, van Hoof, Welling 2018. The standard attention encoder–decoder for routing, with masking of visited nodes and a rollout baseline. This is the backbone almost every later routing paper modifies, and the masking mechanism is what lets one policy handle variable order-set sizes.
- [ ] ⭐⭐⭐ 26. [[Learning to Solve the Min-Max Mixed-Shelves Picker-Routing Problem via Hierarchical and Parallel Decoding.pdf]] — Luttmann & Xie 2025. The best match on the routing side: multiple pickers, multiple items per tour, warehouse topology, and a **min-max** (makespan) objective, solved with hierarchical + parallel decoding. Read the decoder design closely — deciding for several vehicles within one step without them choosing the same node is your problem too.
- [ ] ⭐⭐ 27. [[A-Multi-Agent-Reinforcement-Learning-Method-With-Route-Recorders-for-Vehicle-Routing-in-Supply.pdf]] — Ren et al. 2022. Route recorders to share per-vehicle route history, and sequential (not simultaneous) decoding to avoid two vehicles picking the same node. You have already noted this in [[Quick summary of each paper]] — revisit it after Kool so the architecture reads as a delta.
- [ ] ⭐ 28. [[Reinforcement Learning for Solving the Vehicle Routing Problem.pdf]] — Nazari et al. 2018. Earlier pointer-network-style approach, notable for handling dynamically changing inputs (demand updating during the tour), which is the closest thing here to your online arrivals.
- [ ] ⭐ 29. [[Online Vehicle Routing With Neural Combinatorial Optimization and Deep Reinforcement Learning.pdf]] — Yu, Yu, Gu 2019. Explicitly online routing. Read for how requests arriving mid-episode are folded into the state.
- [ ] ⭐ 30. [[Order picker routing in warehouses A systematic literature review.pdf]] — Masae, Glock, Grosse 2020. Not RL at all: the classical warehouse picker-routing heuristics (S-shape, largest gap, combined). Worth an hour to get cheap, strong, non-learned sequencing baselines and the standard warehouse layout terminology.
- [ ] ○ 31. [[Multi-Vehicle Routing Problems with Soft Time Windows A Multi-Agent Reinforcement Learning Approach.pdf]] — Zhang et al. 2020. MARL for multi-vehicle routing with soft time windows. Skim — time windows are not part of your problem.
- [ ] ○ 32. [[Fair Collaborative Vehicle Routing A Deep Multi-Agent Reinforcement Learning Approach.pdf]] — Mak et al. 2023. Self-interested carriers and profit sharing. Off-target: your vehicles are one cooperative fleet. Skim the abstract and park it.

## Phase 6 — Applications and framing context

Skim these; they are for the related-work section and for design ideas, not for your core method.

- [ ] ⭐⭐ 33. [[Dynamic Multi-Agent Pickup and Delivery in Robotic Cellular Warehousing Systems.pdf]] — Ren et al. 2026. The most recent dynamic-MAPD-in-a-warehouse paper here; read to check nothing in your contribution has just been published.
- [ ] ⭐ 34. [[Anti-conflict AGV path planning in automated container terminals based on multi-agent reinforcement learning.pdf]] — Hu et al. 2021. MARL for conflict-free AGV routing in a real structured facility. Read the conflict-handling and reward design; the container-terminal layout is a close cousin of warehouse aisles.
- [ ] ⭐ 35. [[A Flexible Vehicle Routing Reinforcement Learning Environment for the Reusability of Trained Agents.pdf]] — Díaz et al. 2024. You already extracted the useful part (action masking + the increasing/decreasing episode-termination curriculum) in [[Quick summary of each paper]]. Keep it only as the citation for that curriculum trick.
- [ ] ○ 36. [[Multi-UAV_Path_Planning_for_Wireless_Data_Harvesting_With_Deep_Reinforcement_Learning.pdf]] — Bayerlein et al. 2020. Different domain, but a clean example of map-as-image observations and multi-agent path planning with a shared policy. Skim the observation encoding.
- [ ] ○ 37. [[Multi-Agent Coordination across Diverse Applications A Survey.pdf]] — Sun et al. 2025. Broad coordination survey; use as a citation source, not a read.
- [ ] ○ 38. [[A review of cooperative multi-agent deep reinforcement learning.pdf]] — Oroojlooyjadid & Hajinezhad 2019. General cooperative-MARL review, now partly dated. Use the taxonomy section for background framing.
- [ ] ○ 39. [[A SURVEY OF PROGRESS ON COOPERATIVE MULTI-AGENT REINFORCEMENT LEARNING IN OPEN ENVIRONMENT.pdf]] — Yuan et al. 2023. Open-environment MARL (changing agents, states, objectives). Relevant only to the extent that order sets arriving online make your env non-stationary — read the openness taxonomy, skip the rest.

---

## Notes on the ordering

- **Phases 0–1 before any code.** They fix the problem name, the objective, and the conflict definitions, which determine your env's step semantics.
- **Phase 2 in parallel with implementation.** You need a classical solver available anyway: as an evaluation baseline, and as an expert if you follow PRIMAL-style imitation.
- **Phases 3 and 5 are the two halves of your contribution.** Phase 3 is collision avoidance, Phase 5 is order sequencing. The specific gap you are working in is that most Phase 3 papers assume goals are given one at a time, and most Phase 5 papers ignore collisions entirely. Papers 2, 15 and 26 are the three that straddle the gap — they deserve the closest reading.
- **Phase 4 is reference material** for when training misbehaves; do not front-load it.
- **Phase 6 is skimmable** and mostly serves the related-work section.
