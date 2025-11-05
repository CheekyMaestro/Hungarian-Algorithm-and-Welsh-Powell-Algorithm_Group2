import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
import matplotlib.pyplot as plt

cost_matrix = np.array([
    [82, 83, 69, 92],  # Worker 1
    [77, 37, 49, 92],  # Worker 2
    [11, 69,  5, 86],  # Worker 3
    [ 8,  9, 98, 23]   # Worker 4
])

workers = ['Worker 1', 'Worker 2', 'Worker 3', 'Worker 4']
jobs    = ['Job 1',    'Job 2',    'Job 3',    'Job 4']

# Run Hungarian
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Print the assignment clearly (1-based for human reading)
print("Assignment (human-friendly):")
total = 0
solution_edges = []
for r, c in zip(row_ind, col_ind):
    cost = cost_matrix[r, c]
    total += cost
    print(f" - {workers[r]} -> {jobs[c]}  (original cost = {cost})")
    solution_edges.append((workers[r], jobs[c]))
print("Total cost =", total)

# Build bipartite graph
B = nx.Graph()
B.add_nodes_from(workers, bipartite=0)
B.add_nodes_from(jobs, bipartite=1)

# Add all edges with weight attribute
for i in range(len(workers)):
    for j in range(len(jobs)):
        B.add_edge(workers[i], jobs[j], weight=int(cost_matrix[i, j]))

# Prepare edge visuals
edge_colors = []
edge_widths = []
for u, v in B.edges():
    if (u, v) in solution_edges or (v, u) in solution_edges:
        edge_colors.append('red'); edge_widths.append(2.5)
    else:
        edge_colors.append('gray'); edge_widths.append(0.5)

# Draw graph
plt.figure(figsize=(10,7))
# Use bipartite_layout but give the worker list explicitly so positions are stable
pos = nx.bipartite_layout(B, workers)

nx.draw_networkx_nodes(B, pos, nodelist=workers, node_color='skyblue', node_size=1800)
nx.draw_networkx_nodes(B, pos, nodelist=jobs, node_color='lightgreen', node_size=1800)
nx.draw_networkx_edges(B, pos, edge_color=edge_colors, width=edge_widths)
nx.draw_networkx_labels(B, pos, font_size=10)

# Draw edge labels only for solution edges using the edge weights stored in the graph.
solution_edge_labels = {}
for u, v in solution_edges:
    # confirm the graph has the edge and its stored weight
    solution_edge_labels[(u, v)] = B[u][v]['weight']

nx.draw_networkx_edge_labels(B, pos, edge_labels=solution_edge_labels, font_color='red', font_size=9)

plt.title("Hungarian Algorithm - Optimal Assignment Visualization")
plt.axis('off')
plt.show()
