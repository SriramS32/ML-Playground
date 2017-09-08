# Implement fastmap
# Then implement a* that uses FastMap heuristics
import networkx as nx


G = nx.Graph()
for i in xrange(10):
	G.add_node(i)


# Example graph
"""
Given complete weighted undirected graph G(V, E)
Select arbitrary node Oa.
Get shortest path tree from Oa and find farthest node Ob. Repeat the same
for Ob and find new farthest node. Iterate tau times. O(E + V log V)

Define points using shortest path trees from Oa and Ob in new dimension. Update weights. Use L1 norm.
Repeat process.
"""