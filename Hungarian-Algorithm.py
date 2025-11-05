# Cell 1: Install & Import Libraries

# We need scipy for the algorithm and networkx for visualization
!pip install scipy numpy networkx matplotlib

import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

# Cell 2: Define Input Data (Cost Matrix)

# Let's imagine 4 Workers and 4 Jobs.
# The cost_matrix[i][j] is the cost of assigning Worker i to Job j.
cost_matrix = np.array([
    [82, 83, 69, 92],  # Costs for Worker 1 to do Jobs 1, 2, 3, 4
    [77, 37, 49, 92],  # Costs for Worker 2
    [11, 69,  5, 86],  # Costs for Worker 3
    [ 8,  9, 98, 23]   # Costs for Worker 4
])

# Define the labels for our nodes
workers = ['Worker 1', 'Worker 2', 'Worker 3', 'Worker 4']
jobs = ['Job 1', 'Job 2', 'Job 3', 'Job 4']

print("Cost Matrix:")
print(cost_matrix)

# Cell 3: Run the Hungarian Algorithm

# This function returns two arrays:
# 1. row_ind: The row indices (our 'workers') of the optimal assignment
# 2. col_ind: The corresponding column indices (our 'jobs')
row_ind, col_ind = linear_sum_assignment(cost_matrix)

print(f"Row indices (Workers): {row_ind}")
print(f"Column indices (Jobs): {col_ind}")

# Cell 4: Display the Output (Pairs and Total Cost)

print("Optimal Assignment (Output):")
print("------------------------------")

total_cost = 0
for i in range(len(row_ind)):
    row = row_ind[i]
    col = col_ind[i]
    cost = cost_matrix[row, col]
    total_cost += cost

    print(f"* Pair: {workers[row]} -> {jobs[col]} (Cost: {cost})")

print("------------------------------")
print(f"Total Minimum Cost: {total_cost}")

# Cell 5: Visualize the Result

# 1. Create a Bipartite Graph
B = nx.Graph()

# 2. Add nodes with bipartite attribute
B.add_nodes_from(workers, bipartite=0)
B.add_nodes_from(jobs, bipartite=1)

# 3. Add all edges from the cost matrix
for i in range(len(workers)):
    for j in range(len(jobs)):
        B.add_edge(workers[i], jobs[j], weight=cost_matrix[i][j])

# 4. Get the optimal solution edges
solution_edges = []
for i in range(len(row_ind)):
    solution_edges.append((workers[row_ind[i]], jobs[col_ind[i]]))

# 5. Set up colors and styles
# Make solution edges red and thick, others gray and thin
edge_colors = []
edge_widths = []
for u, v in B.edges():
    if (u, v) in solution_edges or (v, u) in solution_edges:
        edge_colors.append('red')
        edge_widths.append(2.5)
    else:
        edge_colors.append('gray')
        edge_widths.append(0.5)

# 6. Draw the graph
plt.figure(figsize=(10, 7))
# Use a bipartite layout to separate the two sets
pos = nx.bipartite_layout(B, workers)

# Draw nodes
nx.draw_networkx_nodes(B, pos, nodelist=workers, node_color='skyblue', node_size=2000, label='Workers')
nx.draw_networkx_nodes(B, pos, nodelist=jobs, node_color='lightgreen', node_size=2000, label='Jobs')

# Draw edges
nx.draw_networkx_edges(B, pos, edge_color=edge_colors, width=edge_widths)

# Draw labels
nx.draw_networkx_labels(B, pos, font_size=10)

# Draw edge labels (weights) for the solution
solution_edge_labels = {}
for u, v in solution_edges:
    solution_edge_labels[(u,v)] = B[u][v]['weight']

nx.draw_networkx_edge_labels(B, pos, edge_labels=solution_edge_labels, font_color='red')


plt.title("Hungarian Algorithm - Optimal Assignment Visualization")
plt.axis('off') # Turn off the axes
plt.show()
