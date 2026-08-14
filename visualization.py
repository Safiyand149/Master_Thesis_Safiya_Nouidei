"""
Generates 6 frames illustrating the progressive construction of the multimodal
graph and shortest-path retrieval (Dijkstra), with a concrete example.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import math

# ---------------------------------------------------------------------------
# Colour palette per node type (consistent with the thesis)
# ---------------------------------------------------------------------------
COL_IMAGE      = "#e63946"   # red     : image nodes (artworks)
COL_CAPTION    = "#2a9d8f"   # green   : caption node / query text
COL_CAPLABEL   = "#f4a020"   # orange  : caption labels
COL_HIERLABEL  = "#457b9d"   # blue    : hierarchical labels
COL_EMOTION    = "#9d4edd"   # purple  : emotion
COL_QUERY      = "#264653"   # dark    : query (TEXT story) in the final phase
COL_INACTIVE   = "#d9d9d9"   # grey    : node not involved
COL_PATH       = "#1d3557"   # navy    : shortest-path edge
COL_COOC       = "#1565c0"   # blue    : learned co-occurrence edge

# ---------------------------------------------------------------------------
# Concrete example (a family-memory story + 2 candidate artworks)
# Story: "a joyful family gathering, faces, the memory of childhood"
# ---------------------------------------------------------------------------

# Central query / text node
TEXT = "TEXT\n(story)"

# Query text labels (orange caption labels)
QUERY_LABELS = ["joy", "family", "memory", "childhood"]

# Artwork A: family scene (strong candidate)
IMG_A = "IMAGE_A"
A_CAPLABELS = ["faces", "indoor"]     # caption labels of A
A_HIER      = [("scene", 1), ("family", 2)]  # hier labels (term, depth)
A_EMOTION   = "admiration"

# Artwork B: dark landscape (weaker candidate)
IMG_B = "IMAGE_B"
B_CAPLABELS = ["dark", "outdoor"]
B_HIER      = [("landscape", 1)]
B_EMOTION   = "sadness"

# ---------------------------------------------------------------------------
# FIXED position of every node (stable layout across the 6 frames)
# ---------------------------------------------------------------------------
POS = {
    TEXT:          (0.50, 0.92),
    # query labels (top)
    "joy":        (0.18, 0.70),
    "family":     (0.50, 0.60),
    "memory":    (0.82, 0.70),
    "childhood":     (0.38, 0.74),
    # A side (left/bottom)
    "faces":     (0.42, 0.42),
    "indoor":   (0.20, 0.40),
    "scene":       (0.30, 0.54),
    IMG_A:         (0.28, 0.12),
    "admiration":  (0.10, 0.24),
    # B side (right/bottom)
    "dark":      (0.80, 0.42),
    "outdoor":   (0.92, 0.50),
    "landscape":     (0.70, 0.50),
    IMG_B:         (0.78, 0.12),
    "sadness":   (0.94, 0.24),
}

# Each node's type -> base colour
NODE_TYPE = {
    TEXT: "query",
    "joy": "caplabel", "family": "caplabel", "memory": "caplabel", "childhood": "caplabel",
    "faces": "caplabel", "indoor": "caplabel",
    "dark": "caplabel", "outdoor": "caplabel",
    "scene": "hier", "landscape": "hier",
    IMG_A: "image", IMG_B: "image",
    "admiration": "emotion", "sadness": "emotion",
}

COLOR_BY_TYPE = {
    "query":    COL_CAPTION,
    "caplabel": COL_CAPLABEL,
    "hier":     COL_HIERLABEL,
    "emotion":  COL_EMOTION,
    "image":    COL_IMAGE,
}

def hier_weight(d):
    """hier_label->image edge weight: 0.5 + 0.5*log(1+d)."""
    return 0.5 + 0.5 * math.log(1 + d)

# ---------------------------------------------------------------------------
# Generic frame-drawing function
# ---------------------------------------------------------------------------
def draw_frame(step_title, active_nodes, edges, fname,
               highlight_path_edges=None, dimmed_nodes=None,
               edge_labels=None, legend=True, subtitle=None):
    """
    active_nodes : set of visible nodes (coloured by their type)
    edges        : list of (u, v, style) where style in {'struct','cooc'}
    highlight_path_edges : list of (u,v) drawn in bold navy (Dijkstra)
    dimmed_nodes : set of nodes that are visible but greyed out
    edge_labels  : dict {(u,v): "0.9"} of displayed weights
    """
    dimmed_nodes = dimmed_nodes or set()
    highlight_path_edges = highlight_path_edges or []
    edge_labels = edge_labels or {}

    fig, ax = plt.subplots(figsize=(12.5, 11))
    # Leave space at the top for the title; node layout stays within [0, 0.88].
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.12)
    ax.axis("off")
    ax.set_aspect("equal")  # keep circles perfectly round

    # Title (above the node area).
    ax.text(0.5, 1.10, step_title, ha="center", va="top",
            fontsize=21, fontweight="bold")
    if subtitle:
        ax.text(0.5, 1.055, subtitle, ha="center", va="top",
                fontsize=12.5, style="italic", color="#555555")

    hp_set = {frozenset(e) for e in highlight_path_edges}

    # --- edges ---
    for u, v, style in edges:
        if u not in active_nodes or v not in active_nodes:
            continue
        x1, y1 = POS[u]; x2, y2 = POS[v]
        is_path = frozenset((u, v)) in hp_set
        if is_path:
            ax.plot([x1, x2], [y1, y2], color=COL_PATH, lw=5,
                    zorder=2, solid_capstyle="round")
        elif style == "cooc":
            ax.plot([x1, x2], [y1, y2], color=COL_COOC, lw=3,
                    zorder=1, alpha=0.9)
        else:  # struct
            grey = (u in dimmed_nodes or v in dimmed_nodes)
            ax.plot([x1, x2], [y1, y2],
                    color="#cccccc" if grey else "#b0b0b0",
                    lw=1.5, zorder=1, alpha=0.8)
        # weight label
        key = (u, v) if (u, v) in edge_labels else (v, u)
        if key in edge_labels:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, edge_labels[key], fontsize=9, ha="center",
                    va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.85))

    # --- nodes ---
    for n in active_nodes:
        x, y = POS[n]
        ntype = NODE_TYPE[n]
        if n in dimmed_nodes:
            color = COL_INACTIVE
        elif n == TEXT and step_title.startswith("Step 6"):
            color = COL_CAPLABEL  # the query becomes an orange source label
        else:
            color = COLOR_BY_TYPE[ntype]
        # radius depends on the node type
        if ntype == "image":
            r = 0.052
        elif ntype == "query":
            r = 0.050
        else:
            r = 0.040
        circ = plt.Circle((x, y), r, color=color, zorder=3,
                          ec="black", lw=1.2)
        ax.add_patch(circ)
        # text
        txt_color = "white" if ntype in ("image", "query") and n not in dimmed_nodes else "black"
        ax.text(x, y, n, ha="center", va="center", fontsize=9.5,
                fontweight="bold" if ntype in ("image", "query") else "normal",
                color="black", zorder=5)

    # --- legend ---
    if legend:
        handles = [
            mpatches.Patch(color=COL_IMAGE,    label="Image (artwork)"),
            mpatches.Patch(color=COL_CAPLABEL, label="Caption label / query token"),
            mpatches.Patch(color=COL_HIERLABEL,label="Hierarchical label"),
            mpatches.Patch(color=COL_EMOTION,  label="Emotion"),
            mpatches.Patch(color=COL_CAPTION,  label="Caption / query text"),
        ]
        leg = ax.legend(handles=handles, loc="lower center", ncol=3,
                        fontsize=9.5, frameon=True, bbox_to_anchor=(0.5, -0.04))
        leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  {fname}")


# ===========================================================================
# EDGE DEFINITIONS (final complete state of the graph)
# ===========================================================================
# Structural edges: (u, v, 'struct')
STRUCT_EDGES = [
    # query caption labels linked to TEXT
    (TEXT, "joy", "struct"),
    (TEXT, "family", "struct"),
    (TEXT, "memory", "struct"),
    (TEXT, "childhood", "struct"),
    # IMAGE_A: its labels
    (IMG_A, "faces", "struct"),
    (IMG_A, "indoor", "struct"),
    (IMG_A, "scene", "struct"),
    ("scene", "family", "struct"),     # hierarchy: family is a child of scene
    # IMAGE_B: its labels
    (IMG_B, "dark", "struct"),
    (IMG_B, "outdoor", "struct"),
    (IMG_B, "landscape", "struct"),
    # emotions
    (IMG_A, "admiration", "struct"),
    (IMG_B, "sadness", "struct"),
]

# Learned co-occurrence edges (Step 4): (u, v, 'cooc')
COOC_EDGES = [
    ("family", "faces", "cooc"),   # family & faces co-occur strongly
    ("joy", "admiration", "cooc"),   # joy ~ admiration (positive valence)
    ("memory", "family", "cooc"),  # memory & family
    ("childhood", "faces", "cooc"),   # childhood & faces
]

ALL_EDGES = STRUCT_EDGES + COOC_EDGES

# Displayed weights (for Step 4 and 6)
EDGE_W = {
    (TEXT, "joy"): "1.0", (TEXT, "family"): "1.0",
    (TEXT, "memory"): "1.0", (TEXT, "childhood"): "1.0",
    (IMG_A, "faces"): "0.8", (IMG_A, "indoor"): "0.8",
    (IMG_A, "scene"): "0.69", ("scene", "family"): "0.85",
    (IMG_A, "admiration"): "0.8",
    (IMG_B, "dark"): "0.8", (IMG_B, "outdoor"): "0.8",
    (IMG_B, "landscape"): "0.69", (IMG_B, "sadness"): "0.8",
    ("family", "faces"): "0.9", ("joy", "admiration"): "0.6",
    ("memory", "family"): "0.7", ("childhood", "faces"): "0.5",
}


# ===========================================================================
# GENERATING THE 6 FRAMES
# ===========================================================================
def generate():
    A_labels = {"faces", "indoor", "scene", "admiration"}
    B_labels = {"dark", "outdoor", "landscape", "sadness"}
    query_labels = {"joy", "family", "memory", "childhood"}

    # ---- STEP 1: Raw multimodal input ----
    # Just the raw inputs: the text + 2 images (not connected yet).
    draw_frame(
        "Step 1 — Raw multimodal input",
        active_nodes={TEXT, IMG_A, IMG_B},
        edges=[],
        fname="frame_0.png",
        subtitle="A story (text) and the candidate artworks, still isolated",
        legend=True,
    )

    # ---- STEP 2: Text -> tokens (caption labels) ----
    # The text is split into labels; images receive their caption labels.
    nodes2 = {TEXT, IMG_A, IMG_B} | query_labels | {"faces", "indoor", "dark", "outdoor"}
    edges2 = [e for e in STRUCT_EDGES
              if e[0] in nodes2 and e[1] in nodes2
              and e[1] not in {"scene", "landscape", "admiration", "sadness"}]
    draw_frame(
        "Step 2 — Tokenization & caption labels",
        active_nodes=nodes2,
        edges=edges2,
        fname="frame_1.png",
        subtitle="Text and descriptions split into tokens (lowercased, stopwords removed)",
        legend=True,
    )

    # ---- STEP 3: Hierarchical labels + emotions ----
    nodes3 = set(POS.keys())
    edges3 = list(STRUCT_EDGES)
    draw_frame(
        "Step 3 — Hierarchical labels & emotions",
        active_nodes=nodes3,
        edges=edges3,
        fname="frame_2.png",
        subtitle="Adding hierarchical labels (depth) and one dominant emotion per artwork",
        legend=True,
    )

    # ---- STEP 4: Learning semantic co-occurrences ----
    nodes4 = set(POS.keys())
    edges4 = list(STRUCT_EDGES) + list(COOC_EDGES)
    draw_frame(
        "Step 4 — Learning semantic co-occurrences",
        active_nodes=nodes4,
        edges=edges4,
        fname="frame_3.png",
        subtitle="Co-occurring label pairs are linked (weighted blue edges)",
        edge_labels={k: v for k, v in EDGE_W.items()
                     if k in {("family","faces"), ("joy","admiration"),
                              ("memory","family"), ("scene","family")}},
        legend=True,
    )

    # ---- STEP 5: Full multimodal graph ----
    nodes5 = set(POS.keys())
    edges5 = list(ALL_EDGES)
    draw_frame(
        "Step 5 — Full multimodal graph",
        active_nodes=nodes5,
        edges=edges5,
        fname="frame_4.png",
        subtitle="Full heterogeneous graph: images, captions, labels, hierarchy, emotions",
        edge_labels=EDGE_W,
        legend=True,
    )

    # ---- STEP 6: Shortest-path reasoning (Dijkstra) ----
    # Winning path: TEXT -> family -> faces -> IMAGE_A
    # (the "family/memory" query leads to IMAGE_A via a strong co-occurrence)
    nodes6 = set(POS.keys())
    edges6 = list(ALL_EDGES)
    path_edges = [
        (TEXT, "family"),
        ("family", "faces"),
        ("faces", IMG_A),
    ]
    # Nodes NOT on the path -> greyed out.
    on_path = {TEXT, "family", "faces", IMG_A}
    # Keep the relevant query labels and IMG_A coloured too.
    keep_colored = on_path | {"memory", "joy", "childhood"}
    dimmed = set(POS.keys()) - keep_colored - {IMG_B}
    # IMG_B stays red (discarded candidate) but its branches are greyed out.
    dimmed |= {"scene"}  # scene is not on the selected direct path
    draw_frame(
        "Step 6 — Shortest-path retrieval (Dijkstra)",
        active_nodes=nodes6,
        edges=edges6,
        fname="frame_5.png",
        subtitle="The weighted shortest path links the query to IMAGE_A: the returned artwork",
        highlight_path_edges=path_edges,
        dimmed_nodes=dimmed,
        edge_labels={(TEXT,"family"): "1.0",
                     ("family","faces"): "0.9",
                     (IMG_A,"faces"): "0.8"},
        legend=True,
    )


if __name__ == "__main__":
    print("Generating the 6 frames of the progressive graph...")
    generate()
    print("\nDone: frame_0.png to frame_5.png")
