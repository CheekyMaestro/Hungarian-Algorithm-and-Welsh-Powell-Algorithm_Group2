# Hungarian-Algorithm-and-Welsh-Powell-Algorithm_Group2

#  Welsh-Powell Graph Coloring — Code Walkthrough & Function Reference

This report explains **every function and segment** in your `Welsh_Powell_Algorithm.py` with clear intent, inputs/outputs, and algorithmic notes. It also includes complexity, edge cases, and extension tips so you can adapt the code for other graphs.

---

## 1) Overview

- **Goal:** Color the vertices of an undirected graph so that **no two adjacent vertices share the same color**.
- **Heuristic used:** **Welsh–Powell** — sort vertices by degree (highest first), then greedily assign the lowest possible color to as many non‑adjacent vertices as you can, introduce a new color only when necessary.
- **Libraries:**
  - `networkx` — graph structure and utilities
  - `matplotlib` — visualization of the final coloring

> Result: A valid coloring and a *small* (but not guaranteed minimal) number of colors.  

---

## 2) Environment & Dependencies

```python
import networkx as nx
import matplotlib.pyplot as plt
```

- `networkx` is used to create/manage the undirected graph (`nx.Graph()`), query degrees, and iterate neighbors.
- `matplotlib.pyplot` renders a colorized drawing of the graph.

---

## 3) Graph Construction

```python
G_wp = nx.Graph()
edges_wp = [
    ('A', 'H'), ('A', 'B'), ('B', 'D'), ('C', 'D'), ('D', 'I'),
    ('D', 'K'), ('E', 'F'), ('E', 'K'), ('F', 'G'), ('G', 'H'),
    ('G', 'K'), ('H', 'I'), ('H', 'J'), ('H', 'K'), ('I', 'J'), ('J', 'K')
]
G_wp.add_edges_from(edges_wp)
```

**What it does**
- Creates an **undirected** graph `G_wp`.
- Adds all edges in a batch from the `edges_wp` list (nodes are created implicitly).

**Why it matters**
- Welsh–Powell depends on **vertex degrees** — edges define degrees, which define the coloring order.

---

## 4) Inspecting Nodes/Edges and Degrees

```python
print("--- Step 1: Find Degree of Each Vertex ---")
print(f"Nodes: {list(G_wp.nodes)}")
print(f"Edges: {list(G_wp.edges)}")

degree_map = dict(G_wp.degree())
print("\nVertex Degrees:")
for v, degree in degree_map.items():
    print(f"Vertex {v}: {degree}")
```

**What it does**
- Prints the node and edge sets for debugging/reproducibility.
- Builds `degree_map` as `{node: degree}` for all vertices.

**Complexity**
- Building degrees is `O(V + E)` in NetworkX (each edge contributes to two vertex degrees).

**Why it matters**
- The next step sorts vertices by degree; seeing degrees helps validate correctness and compare with slides/expected examples.

---

## 5) Sorting Vertices by Descending Degree

```python
sorted_vertices = sorted(G_wp.nodes(), key=lambda v: G_wp.degree(v), reverse=True)

print("\n--- Step 2: List Vertices in Order of Descending Degrees ---")
print(f"Coloring Order (Highest Degree First): {sorted_vertices}")
```

**What it does**
- Produces the **Welsh–Powell order**: highest‑degree vertices first.

**Why it matters**
- Coloring higher‑degree vertices early reduces conflicts and tends to **lower the total number of colors** used by a greedy assignment.

**Complexity**
- Sorting is `O(V log V)`; each `G_wp.degree(v)` is `O(1)` average in NetworkX.

---

## 6) Core Function: `welsh_powell_coloring`

```python
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
            print(f\"\nASSIGNING {current_color_name} (Starting with {start_node}):\")
            print(f\"-> {start_node} colored {current_color_name}.")

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
                        print(f\"-> {v} colored {current_color_name} (Not connected to other {current_color_name} nodes).")
                    else:
                        print(f\"-> {v} SKIPPED: It is adjacent to a node already colored {current_color_name}.")

    return node_colors, color_count
```

### Purpose
Implements the **Welsh–Powell greedy assignment** over a pre‑sorted node order.

### Inputs
- `graph`: a `networkx.Graph` (undirected) with your vertices and edges.
- `sorted_nodes`: list of nodes in **descending degree** order (precomputed once).

### Outputs
- `node_colors`: `dict[node] -> int` mapping each node to its color index (1..`color_count`).
- `color_count`: total number of distinct colors used.

### Step-by-step behavior
1. **Loop until all nodes are colored**:  
   Checks with `any(node not in node_colors for node in graph.nodes())`.
2. **Introduce a new color** (`color_count += 1`) and name it.
3. **Pick the next uncolored highest‑degree node** as the **seed** for this color.
4. **Greedy pass over the sorted list**: for every *uncolored* node `v`,
   - Inspect all `graph.neighbors(v)`;
   - If **none** of those neighbors already has the *current* color, **assign** the current color to `v`;
   - Otherwise **skip** `v` for this round.
5. **Repeat** with a new color if any node remains uncolored.

### Correctness intuition
- In each round, you build a **maximal independent set** (with respect to the current set of uncolored vertices) and color it with a single color.
- Sorting by degree first tends to reduce the number of colors in practice.

### Complexity
- Outer loop runs at most `chromatic_number ≤ V` times.
- Inner pass scans all `V` nodes; neighbor checks sum to `O(E)` across each pass.  
- **Overall worst‑case:** ~`O(V^2 + V·E)` (often closer to `O(V^2 + E)` in sparse graphs).  
  For typical sparse graphs, this is efficient for classroom‑/assignment‑sized inputs.

### Edge cases handled
- **Disconnected graphs**: Works naturally (each component is colored during the passes).
- **Complete graph**: Uses `V` colors (as expected).
- **Already independent set**: Colors all nodes in the first pass with one color.

### Customization tips
- Replace the sorting key to try other heuristics (e.g., tie‑break by name, degree‑then‑clustering).
- To enforce a **fixed palette**, map `1..color_count` to a list of color names (e.g., `["tab:blue", ...]`).

---

## 7) Executing the Algorithm

```python
coloring_result, num_colors = welsh_powell_coloring(G_wp, sorted_vertices)

print("\n--- Final Result ---")
print("Coloring Result (Node: Color_ID):")
print(coloring_result)
print(f"Minimum Colors Used (Chromatic Number estimate): {num_colors}")
```

**What it does**
- Runs the function and prints a reusable dictionary of assignments plus the total number of colors.

**Why it matters**
- You can **assert** properties in tests (e.g., no edge `(u, v)` with `color[u] == color[v]`), or export `coloring_result` for other pipelines.

---

## 8) Visualization

```python
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
```

**Key details**
- `plt.cm.get_cmap('Spectral', num_colors)`: asks Matplotlib for a **discrete** colormap with exactly `num_colors` colors.
- `spring_layout(..., seed=42)`: deterministic layout for reproducible figures in reports.
- The **legend** is built from invisible rectangles colored by the colormap to label “Color 1 … Color N”.

**Troubleshooting**
- If colors look too similar, try another colormap (e.g., `'tab20'`) or increase figure size.
- If labels overlap, increase `node_size` or adjust layout parameters (e.g., `k` in `spring_layout`).

---

## 9) How to Extend

- **Swap the graph**: Just change `edges_wp` or read from a file (CSV/edge list) and rebuild `G_wp`.
- **Alternate orderings**: Replace the sorting rule; e.g., break ties lexicographically to make outputs stable.
- **Compare heuristics**: Implement DSATUR or basic greedy and measure `num_colors` vs. runtime.

---

## 10) Quick Sanity Checks (optional)

- **No conflict**: For every edge `(u, v)`, verify `color[u] != color[v]`.
- **Upper bound**: `num_colors ≤ max_degree + 1` (Brooks’ theorem has exceptions but this is a safe yardstick).

---

## 11) Summary

- **Function implemented**: `welsh_powell_coloring(graph, sorted_nodes)` — returns a valid coloring and the number of colors used.
- **Pipeline**: build graph → compute degrees → sort by degree → iterative coloring → visualize.
- **Strength**: simple, fast, produces compact colorings on many real graphs.
- **Limitation**: heuristic; not guaranteed minimal colors.

> You can drop this Markdown directly into your repo (e.g., `README.md`) or keep it as a separate `Welsh_Powell_Report.md` alongside the code.
