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



---

## 1) Goal at a Glance
The script solves the **assignment problem** (assign *n* workers to *n* jobs) with **minimum total cost** using the **Hungarian Algorithm** via SciPy’s `linear_sum_assignment`. The optimal pairs are then **printed** (pair list & total cost) and **visualized** as a bipartite graph (NetworkX + Matplotlib).

> **Main input:** a 2D `cost_matrix` of shape `n × n`.  
> **Main outputs:** the optimal pairs `(worker_i → job_j)` and the `Total Minimum Cost`.

---

## 2) Code Flow (block by block)
1. **Import libraries:** `numpy`, `scipy.optimize.linear_sum_assignment`, `networkx`, `matplotlib.pyplot`  
2. **Define data:** `cost_matrix`, `workers`, `jobs`  
3. **Run Hungarian:** `row_ind, col_ind = linear_sum_assignment(cost_matrix)`  
4. **Print results & sum cost:** iterate over indices → print pairs + accumulate cost  
5. **Visualize bipartite graph:** build `nx.Graph()`, add edges with weights, mark the optimal edges, draw with a bipartite layout

---

## 3) Function Reference (purpose, parameters, return values)

> Below are the **functions invoked by the script**. For each, we outline its purpose, key parameters, and what it returns.

### 3.1. SciPy — Hungarian Algorithm
- **`scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)`**  
  - **Purpose:** Solve the *linear assignment problem* (Hungarian method).  
  - **Key parameters:**
    - `cost_matrix`: 2D `array_like` (square or rectangular). Smaller values are preferred (minimization).  
    - `maximize` *(optional)*: defaults to `False` (minimization). Set `True` for **maximization** (larger is better).
  - **Returns:** `(row_ind, col_ind)` — two 1D arrays of length `k` (usually `n` if square) describing the optimal pairing: worker `row_ind[i]` is paired with job `col_ind[i]`.
  - **Note:** Time complexity is **O(n³)** for an `n × n` matrix (Hungarian algorithm).

### 3.2. NumPy
- **`np.array(list_of_lists)`**  
  - Builds a 2D array for the `cost_matrix`.  
  - Ensures consistent and fast indexing via `[i, j]`.

### 3.3. NetworkX — Graph Building & Layout
- **`nx.Graph()`**  
  - Creates an **undirected** graph to model the worker–job bipartite relation.
- **`G.add_nodes_from(iterable, bipartite=0|1)`**  
  - Adds multiple nodes at once and tags a `bipartite` attribute (to separate the two partitions: workers vs. jobs).
- **`G.add_edge(u, v, weight=value)`**  
  - Adds an edge with a `weight` attribute taken from `cost_matrix[i][j]`.
- **`nx.bipartite_layout(G, nodes)`**  
  - Produces 2D coordinates that neatly separate the **two partitions** (`nodes` is the list of one side, e.g., `workers`).

### 3.4. NetworkX — Drawing the Graph
- **`nx.draw_networkx_nodes(G, pos, nodelist=..., node_color=..., node_size=..., label=...)`**  
  - Draws nodes (workers and jobs) with different colors for clarity.
- **`nx.draw_networkx_edges(G, pos, edge_color=..., width=...)`**  
  - Draws all edges. In this script, optimal edges are **red & thick**, others are **gray & thin**.
- **`nx.draw_networkx_labels(G, pos, font_size=...)`**  
  - Renders node labels at positions `pos`.
- **`nx.draw_networkx_edge_labels(G, pos, edge_labels=..., font_color=...)`**  
  - Renders **edge weights** (costs) **only for the optimal edges** (set in `solution_edge_labels`).

### 3.5. Matplotlib
- **`plt.figure(figsize=(w, h))`**, **`plt.title(str)`**, **`plt.axis('off')`**, **`plt.show()`**  
  - Sets up a plotting canvas, titles the figure, hides axes, and displays the final image.

### 3.6. Python Built-ins
- **`range`, `len`, `print`, `for` loops**  
  - Iterate through the result pairs, index `cost_matrix[row, col]`, and print a summary.

---

## 4) Code Walkthrough (by structure)

### (A) Imports
```python
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
import matplotlib.pyplot as plt
```
- Loads all required packages.  
- **Note:** Any `!pip install ...` lines are notebook-only. For a `.py` script, use `requirements.txt` or a virtual environment.

### (B) Data Definition — Cost Matrix & Labels
```python
cost_matrix = np.array([
    [82, 83, 69, 92],
    [77, 37, 49, 92],
    [11, 69,  5, 86],
    [ 8,  9, 98, 23]
])
workers = ['Worker 1', 'Worker 2', 'Worker 3', 'Worker 4']
jobs    = ['Job 1',    'Job 2',    'Job 3',    'Job 4']
```
- `cost_matrix[i][j]` = cost if worker `i` performs job `j`.  
- Labels are used for printing pairs and naming nodes in the visualization.

### (C) Run Hungarian
```python
row_ind, col_ind = linear_sum_assignment(cost_matrix)
```
- Produces two **index arrays** of optimal pairs.  
- Example: if `row_ind=[0,1,2,3]` and `col_ind=[2,1,0,3]`, then:  
  `(Worker 1→Job 3), (Worker 2→Job 2), (Worker 3→Job 1), (Worker 4→Job 4)`.

### (D) Print Pairs & Compute Total Cost
```python
total_cost = 0
for i in range(len(row_ind)):
    row = row_ind[i]
    col = col_ind[i]
    cost = cost_matrix[row, col]
    total_cost += cost
    print(f"* Pair: {workers[row]} -> {jobs[col]} (Cost: {cost})")
print(f"Total Minimum Cost: {total_cost}")
```
- Iterates through the optimal pairs, retrieves the cost from `cost_matrix[row, col]`, and accumulates it.

### (E) Bipartite Graph Visualization
```python
B = nx.Graph()
B.add_nodes_from(workers, bipartite=0)
B.add_nodes_from(jobs,    bipartite=1)

for i in range(len(workers)):
    for j in range(len(jobs)):
        B.add_edge(workers[i], jobs[j], weight=cost_matrix[i][j])

solution_edges = [(workers[row_ind[i]], jobs[col_ind[i]]) for i in range(len(row_ind))]

edge_colors, edge_widths = [], []
for u, v in B.edges():
    if (u, v) in solution_edges or (v, u) in solution_edges:
        edge_colors.append('red');  edge_widths.append(2.5)
    else:
        edge_colors.append('gray'); edge_widths.append(0.5)

pos = nx.bipartite_layout(B, workers)

nx.draw_networkx_nodes(B, pos, nodelist=workers, node_color='skyblue',  node_size=2000, label='Workers')
nx.draw_networkx_nodes(B, pos, nodelist=jobs,    node_color='lightgreen', node_size=2000, label='Jobs')
nx.draw_networkx_edges(B, pos, edge_color=edge_colors, width=edge_widths)
nx.draw_networkx_labels(B, pos, font_size=10)

solution_edge_labels = {(u, v): B[u][v]['weight'] for (u, v) in solution_edges}
nx.draw_networkx_edge_labels(B, pos, edge_labels=solution_edge_labels, font_color='red')

plt.title("Hungarian Algorithm - Optimal Assignment Visualization")
plt.axis('off'); plt.show()
```
- **Builds a bipartite graph** with two partitions (workers & jobs).  
- **Optimal assignment edges** are styled distinctly (red & thick).  
- **Edge labels** show weights only for the optimal edges to keep the figure clean.

---

## 5) Complexity & Properties
- **Hungarian (`linear_sum_assignment`):** `O(n³)` for `n × n`.  
- **Graph build:** `O(V + E) ≈ O(n + n²)` for a full bipartite graph.  
- **Drawing:** depends on layout and number of edges; trivial for `n=4`.

**Correctness:**  
- Produces a *matching* that **covers all rows** (and all columns if square), with **no conflicts** (one job per worker).  
- The total minimum cost is the sum over the optimal pairs returned.

---

## 6) Common Customizations
- **Matrix size:** `cost_matrix` can be `m × n` (rectangular). The solver still works; the result length is `k = min(m, n)` pairs.
- **Maximization** (higher is better):  
  ```python
  row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
  ```
  *Or* convert benefits to costs: `cost_matrix = max_val - benefit_matrix`.
- **Forbidden pairs:** use a very large number or `np.inf` as the cost to forbid an assignment.
- **Dynamic labels:** if `len(workers) != len(jobs)`, the solver will pair up to `min(len(workers), len(jobs))`.
- **Visual:** tweak node/edge colors, `node_size`, and `font_size` to match your presentation needs.

---

## 7) Quick Validation (optional)
Add the following checks after computing `row_ind`, `col_ind`:
```python
# No duplicate columns
assert len(set(col_ind)) == len(col_ind)
# Length matches the smaller dimension
assert len(row_ind) == min(len(workers), len(jobs))
# Total cost consistency
calc = sum(cost_matrix[r, c] for r, c in zip(row_ind, col_ind))
assert calc == total_cost
```


---





