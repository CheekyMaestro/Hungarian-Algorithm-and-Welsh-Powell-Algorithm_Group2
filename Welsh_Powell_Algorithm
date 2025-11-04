# Segment 1: Setup and Graph Data for Welsh-Powell
import networkx as nx
import matplotlib.pyplot as plt

# Define the graph from the slide (Nodes A-K)
G_wp = nx.Graph()
edges_wp = [
    ('A', 'H'), ('A', 'B'), ('B', 'D'), ('C', 'D'), ('D', 'I'),
    ('D', 'K'), ('E', 'F'), ('E', 'K'), ('F', 'G'), ('G', 'H'),
    ('G', 'K'), ('H', 'I'), ('H', 'J'), ('H', 'K'), ('I', 'J'), ('J', 'K')
]
G_wp.add_edges_from(edges_wp)

print("--- Step 1: Find Degree of Each Vertex ---")
print(f"Nodes: {list(G_wp.nodes)}")
print(f"Edges: {list(G_wp.edges)}")

# Output the degree of each vertex (like slide 12)
degree_map = dict(G_wp.degree())
print("\nVertex Degrees:")
for v, degree in degree_map.items():
    print(f"Vertex {v}: {degree}")

# Segment 2: Sort Vertices
sorted_vertices = sorted(G_wp.nodes(), key=lambda v: G_wp.degree(v), reverse=True)

print("\n--- Step 2: List Vertices in Order of Descending Degrees ---")
print(f"Coloring Order (Highest Degree First): {sorted_vertices}")

# Segment 3: Coloring Execution
def welsh_powell_coloring(graph, sorted_nodes):
    node_colors = {}
    color_count = 0

    print("\n--- Step 3, 4, 5: Iterative Coloring ---")

    # Repeat until all nodes are colored
    while any(node not in node_colors for node in graph.nodes()):
        color_count += 1
        current_color_name = f"Color {color_count}"

        # Select the next uncolored vertex with the highest degree
        start_node = next((v for v in sorted_nodes if v not in node_colors), None)

        if start_node is not None:
            # Color the starting vertex (Step 3)
            node_colors[start_node] = color_count
            print(f"\nASSIGNING {current_color_name} (Starting with {start_node}):")
            print(f"-> {start_node} colored {current_color_name}.")

            # Color all uncolored non-adjacent vertices (Step 4)
            for v in sorted_nodes:
                if v not in node_colors:
                    # Check if 'v' is adjacent to any vertex already colored 'current_color'
                    is_adjacent_to_same_color = False
                    for neighbor in graph.neighbors(v):
                        if neighbor in node_colors and node_colors[neighbor] == color_count:
                            is_adjacent_to_same_color = True
                            break

                    if not is_adjacent_to_same_color:
                        node_colors[v] = color_count
                        print(f"-> {v} colored {current_color_name} (Not connected to other {current_color_name} nodes).")
                    else:
                        print(f"-> {v} SKIPPED: It is adjacent to a node already colored {current_color_name}.")

    return node_colors, color_count

# Run the algorithm and get the detailed output
coloring_result, num_colors = welsh_powell_coloring(G_wp, sorted_vertices)

print("\n--- Final Result ---")
print("Coloring Result (Node: Color_ID):")
print(coloring_result)
print(f"Minimum Colors Used (Chromatic Number estimate): {num_colors}")

# Segment 4: Visualization of Welsh-Powell Result

color_map = plt.cm.get_cmap('Spectral', num_colors)
colors = [color_map(coloring_result[node] - 1) for node in G_wp.nodes()]

plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G_wp, seed=42)
nx.draw(G_wp, pos,
        with_labels=True,
        node_color=colors,
        node_size=1000,
        font_weight='bold',
        edge_color='gray')

# Legend for colors
color_patches = [plt.Rectangle((0, 0), 1, 1, fc=color_map(i - 1))
                 for i in range(1, num_colors + 1)]
plt.legend(color_patches, [f'Color {i}' for i in range(1, num_colors + 1)],
           title="Color Mapping", loc='lower right')

plt.title(f"Welsh-Powell Graph Coloring (Used {num_colors} Colors)")
plt.show()
