"""warehouse_marl

A thin, online-orders layer on top of POGEMA's Lifelong-MAPF environment,
built for warehouse VRP-style problems where:

  * vehicles do not share a common depot,
  * each vehicle has a set of nodes (orders) it must visit,
  * that set is *not* fully known up front -- new orders arrive online
    from an upstream order-assignment system, and
  * collision avoidance between vehicles is handled by the environment.
"""

__version__ = "0.1.0"
