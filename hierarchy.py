import re
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx

import os

from sentence_transformers import SentenceTransformer, util

MODEL = None


from sentence_transformers import SentenceTransformer, models
from main import get_model
model = get_model()


def semantic_similarity(a, b):
    if not a or not b:
        return 0.0
    model = get_model()
    emb = model.encode([a, b], normalize_embeddings=True)
    return float(util.cos_sim(emb[0], emb[1]))


def _add_edge(hierarchy, parent, child):
    parent = parent.strip()
    child = child.strip()
    if not parent or not child:
        return
    hierarchy.setdefault(parent, [])
    if child not in hierarchy[parent]:
        hierarchy[parent].append(child)
    hierarchy.setdefault(child, [])


def _split_top_level(text):
    """Split *text* by ';' that are NOT inside parentheses.

    Example:
      "groupe A (x ; y) ; groupe B (a ; b)"  ->  ["groupe A (x ; y)", "groupe B (a ; b)"]
    """
    segments = []
    depth = 0
    current = []
    for ch in text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == ";" and depth == 0:
            seg = "".join(current).strip()
            if seg:
                segments.append(seg)
            current = []
        else:
            current.append(ch)
    seg = "".join(current).strip()
    if seg:
        segments.append(seg)
    return segments


def _parse_group(hierarchy, main_group, inner, similarity_threshold):
    """Parse the inside of one (...) block and populate hierarchy under main_group.

    Depth rules:
      "A : B"     -> main_group -> A (depth 1) -> B (depth 2)
      "A : B : C" -> main_group -> A (depth 1) -> B (depth 2) -> C (depth 3)
      standalone  -> depth 2 under current_parent if similarity >= threshold,
                     else depth 1 under main_group.
                     Standalone tokens NEVER reach depth 3.
    """
    tokens = [t.strip() for t in inner.split(";") if t.strip()]
    current_parent = None

    for token in tokens:
        parts = [p.strip() for p in token.split(":")]

        if len(parts) == 3:
            # A : B : C  ->  main_group -> A -> B -> C
            a, b, c = parts
            _add_edge(hierarchy, main_group, a)
            _add_edge(hierarchy, a, b)
            _add_edge(hierarchy, b, c)
            current_parent = b  # most recent explicit parent for following standalones

        elif len(parts) == 2:
            # A : B  ->  main_group -> A -> B
            a, b = parts
            _add_edge(hierarchy, main_group, a)
            _add_edge(hierarchy, a, b)
            current_parent = a

        else:
            # Standalone: attach to current_parent if similar, else main_group
            if current_parent:
                score = semantic_similarity(token, current_parent)
                if score >= similarity_threshold:
                    _add_edge(hierarchy, current_parent, token)
                else:
                    _add_edge(hierarchy, main_group, token)
            else:
                _add_edge(hierarchy, main_group, token)


def creation_hierarchy(subject_matter, similarity_threshold=0.3):
    # Builds a hierarchy for ALL main groups found in subjectTerms.
    #
    # The RMFAB field may contain several groups separated by ";" at the top
    # level (outside parentheses), e.g.:
    #   "groupe A (x ; y : z) ; groupe B (a ; b : c)"
    #
    # Each group becomes an independent root in the returned hierarchy dict.
    # Within each group, depth rules are (see _parse_group):
    #   depth 0 : main_group
    #   depth 1 : direct children  (explicit "A" or standalone fallback)
    #   depth 2 : children via "A : B" or similar standalone
    #   depth 3 : grandchildren via "A : B : C" only (never via standalone)

    hierarchy = {}
    subject_terms = (subject_matter or {}).get("subjectTerms", "")
    if not subject_terms:
        return hierarchy

    text = re.sub(r"\s+", " ", subject_terms).strip()

    # Fast path: no parentheses -> single bare node
    if "(" not in text:
        return {text: []}

    for segment in _split_top_level(text):
        
        if "(" in segment:
            main_group = segment.split("(", 1)[0].strip()
            inner      = segment.split("(", 1)[1].rsplit(")", 1)[0].strip()
            hierarchy.setdefault(main_group, [])
            if inner:
                _parse_group(hierarchy, main_group, inner, similarity_threshold)
        else:
            # Bare segment with no parentheses: top-level node, no children
            if segment:
                hierarchy.setdefault(segment, [])

    return hierarchy


def hierarchy_to_edges(hierarchy):
    return [(p, c) for p, children in hierarchy.items() for c in children]


def compute_levels(hierarchy, root):
    levels = {root: 0}
    queue = [root]

    while queue:
        parent = queue.pop(0)
        for child in hierarchy.get(parent, []):
            if child not in levels:
                levels[child] = levels[parent] + 1
                queue.append(child)
    return levels


def _get_roots(hierarchy):
    """Return all nodes that are not a child of any other node (i.e. top-level roots)."""
    all_children = {child for children in hierarchy.values() for child in children}
    return [node for node in hierarchy if node not in all_children]


def hierarchical_layout(hierarchy):
    """Compute a left-to-right hierarchical layout supporting multiple roots.

    Each root starts its own subtree. Subtrees are stacked vertically and
    separated by a blank row so they stay visually distinct.
    """
    roots = _get_roots(hierarchy)
    pos = {}
    y_offset = 0

    for root in roots:
        levels = compute_levels(hierarchy, root)
        nodes_by_level = {}
        for node, lvl in levels.items():
            nodes_by_level.setdefault(lvl, []).append(node)

        # Height of this subtree (number of rows)
        subtree_height = max(len(nodes) for nodes in nodes_by_level.values())

        for lvl, nodes in nodes_by_level.items():
            for i, node in enumerate(nodes):
                pos[node] = (lvl, -(y_offset + i))

        y_offset += subtree_height + 1  # +1 blank row between subtrees

    return pos


def plot_hierarchy(hierarchy, title="Subject Terms Hierarchy", save_path=None):
    graph = nx.DiGraph()
    graph.add_edges_from(hierarchy_to_edges(hierarchy))

    if not graph.nodes:
        return graph

    roots = set(_get_roots(hierarchy))
    pos = hierarchical_layout(hierarchy)

    node_colors = []
    for node in graph.nodes():
        if node in roots:
            node_colors.append("#4C72B0")   # blue  — root
        elif hierarchy.get(node):
            node_colors.append("#55A868")   # green — internal node
        else:
            node_colors.append("#EEDC82")   # yellow — leaf

    plt.figure(figsize=(14, 8))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color="#444444",
        node_size=5000,
        font_size=25,
        arrows=True,
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return graph


if __name__ == "__main__":
    import sys
    from fabritius_extract.fab_sel_workid_v2 import process

    if len(sys.argv) != 3:
        print("Usage: python best_hierarchy.py <json_input> <work_id>")
        sys.exit(1)

    json_input = sys.argv[1]
    work_id    = sys.argv[2]

    print("Loading model...")
    get_model()
    print("Model loaded.")

    result         = process(json_input, work_id)
    subject_matter = result.get("subjectMatter", {})

    hierarchy = creation_hierarchy(subject_matter)
    print(f"Hierarchy: {hierarchy}")

    if hierarchy:
        plot_hierarchy(hierarchy, title=f"Hierarchy for Work ID: {work_id}")
    else:
        print("Empty hierarchy — nothing to plot.")

    for parent, children in hierarchy.items():
        for child in children:
            score = semantic_similarity(parent, child)
            print(f"Similarity between '{parent}' and '{child}': {score:.4f}")
