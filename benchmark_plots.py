from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def apply_style() -> None:
    plt.rcParams.update({
        "font.family"        : "DejaVu Sans",
        "font.size"          : 11,
        "axes.titlesize"     : 13,
        "axes.titleweight"   : "bold",
        "axes.titlepad"      : 14,
        "axes.labelsize"     : 11,
        "xtick.labelsize"    : 10,
        "ytick.labelsize"    : 10,
        "legend.fontsize"    : 10,
        "figure.titlesize"   : 14,
        "figure.titleweight" : "bold",
        "figure.facecolor"   : "white",
        "axes.facecolor"     : "#F7F8FA",
        "axes.grid"          : True,
        "grid.color"         : "white",
        "grid.linewidth"     : 1.2,
        "axes.axisbelow"     : True,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.edgecolor"     : "#CCCED2",
        "axes.linewidth"     : 1.0,
        "xtick.direction"    : "out",
        "ytick.direction"    : "out",
        "legend.framealpha"  : 0.92,
        "legend.edgecolor"   : "#CCCED2",
        "legend.fancybox"    : False,
        "savefig.dpi"        : 200,
        "savefig.bbox"       : "tight",
        "savefig.facecolor"  : "white",
    })


# Blue marks the proposed method, orange marks the baseline.
COLORS = {
    "Graph": "#2E6BE6",   # proposed method
    "CLIP" : "#E8702A",   # baseline
}
ALPHA_BAR = 0.90
CAPSIZE   = 6
ERROR_KW  = dict(elinewidth=1.4, capthick=1.4, ecolor="#444444")


def _save(fig, name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {path}")
    return path


def _short(q: str, n: int = 22) -> str:
    # Truncate long query strings with an ellipsis so axis labels stay readable.
    return (q[: n - 1] + "\u2026") if len(q) > n else q


def _bar_value_labels(ax, bars, values, fmt="{:.1f}", unit="",
                      fontsize=10, rotation=0, dy_frac=0.02, log=False):
    # Write each bar's value just above its top edge.
    y0, y1 = ax.get_ylim()
    for bar, v in zip(bars, values):
        h = bar.get_height()
        if h <= 0:
            continue
        if log:
            y = h * 1.10
        else:
            y = h + (y1 - y0) * dy_frac
        ax.text(bar.get_x() + bar.get_width() / 2.0, y,
                fmt.format(v) + unit, ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold",
                rotation=rotation, color="#333333")


def plot_training_time(graph_train: dict, clip_train: dict, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))

    methods = ["Graph", "CLIP"]
    vals    = [graph_train["avg_time"], clip_train["avg_time"]]
    stds    = [graph_train["std_time"], clip_train["std_time"]]
    colors  = [COLORS[m] for m in methods]

    bars = ax.bar(methods, vals, yerr=stds, color=colors, width=0.5,
                  alpha=ALPHA_BAR, capsize=CAPSIZE, error_kw=ERROR_KW,
                  edgecolor="white", linewidth=0.8, zorder=3)

    _bar_value_labels(ax, bars, vals, fmt="{:.1f}", unit=" s", fontsize=11)

    ax.set_title("Indexing Time (one-off)")
    ax.set_ylabel("Average duration (s)")
    ax.set_ylim(0, max(v + s for v, s in zip(vals, stds)) * 1.30)
    ax.tick_params(axis="x", bottom=False)

    # Show the coefficient of variation (std as a percentage of the mean) for each method.
    g_cv = graph_train["std_time"] / max(graph_train["avg_time"], 1e-9) * 100
    c_cv = clip_train["std_time"]  / max(clip_train["avg_time"],  1e-9) * 100
    ax.annotate(
        f"Relative variability (std/mean):\nGraph {g_cv:.0f}%   \u00b7   CLIP {c_cv:.0f}%",
        xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top",
        fontsize=8.5, color="#555555",
        bbox=dict(boxstyle="round,pad=0.35", fc="#EEEFF3", ec="#CCCED2", lw=0.8),
    )

    fig.tight_layout()
    return _save(fig, "fig1_training_time.png", out_dir)


def plot_inference_time(graph_infer: list, clip_infer: list,
                        queries: list, out_dir: str) -> str:
    labels = [_short(q, 22) for q in queries]
    x      = np.arange(len(queries))
    width  = 0.38

    g_t = np.array([r["avg_time"] for r in graph_infer])
    g_s = np.array([r["std_time"] for r in graph_infer])
    c_t = np.array([r["avg_time"] for r in clip_infer])
    c_s = np.array([r["std_time"] for r in clip_infer])

    fig, ax = plt.subplots(figsize=(10, 5.5))

    b1 = ax.bar(x - width / 2, g_t, width, yerr=g_s, label="Graph (proposed)",
                color=COLORS["Graph"], alpha=ALPHA_BAR, capsize=4,
                error_kw=ERROR_KW, edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + width / 2, c_t, width, yerr=c_s, label="CLIP",
                color=COLORS["CLIP"], alpha=ALPHA_BAR, capsize=4,
                error_kw=ERROR_KW, edgecolor="white", linewidth=0.8, zorder=3)

    # Log scale because the two methods differ by orders of magnitude.
    ax.set_yscale("log")
    _bar_value_labels(ax, b1, g_t, fmt="{:.3f}", unit=" s", fontsize=8.5,
                      rotation=0, log=True)
    _bar_value_labels(ax, b2, c_t, fmt="{:.3f}", unit=" s", fontsize=8.5,
                      rotation=0, log=True)

    ax.set_title("Inference Time per Query\n(emotion filter excluded \u00b7 log scale)")
    ax.set_ylabel("Average duration per query (s, log)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(min(c_t.min(), g_t.min()) * 0.3, g_t.max() * 3.0)
    ax.legend(loc="upper left")
    ax.grid(axis="y", which="both", color="white")
    fig.tight_layout()
    return _save(fig, "fig2_inference_time.png", out_dir)


def plot_training_memory(graph_train: dict, clip_train: dict, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))

    methods = ["Graph", "CLIP"]
    vals = [max(0.0, graph_train["avg_mem"]), max(0.0, clip_train["avg_mem"])]
    stds = [graph_train["std_mem"], clip_train["std_mem"]]
    colors = [COLORS[m] for m in methods]

    bars = ax.bar(methods, vals, yerr=stds, color=colors, width=0.5,
                  alpha=ALPHA_BAR, capsize=CAPSIZE, error_kw=ERROR_KW,
                  edgecolor="white", linewidth=0.8, zorder=3)

    _bar_value_labels(ax, bars, vals, fmt="{:.0f}", unit=" MB", fontsize=11)

    model_mem = clip_train.get("model_mem_mb", 0.0)
    if model_mem > 0:
        clip_bar = bars[1]
        cx = clip_bar.get_x() + clip_bar.get_width() / 2

    ax.set_title("Indexing Memory Footprint (\u0394 RSS)")
    ax.set_ylabel("RSS increase (MB)")
    ax.set_ylim(0, max(v + s for v, s in zip(vals, stds)) * 1.30)
    ax.tick_params(axis="x", bottom=False)
    fig.tight_layout()
    return _save(fig, "fig3_training_memory.png", out_dir)


def plot_inference_ratio(graph_infer: list, clip_infer: list,
                         queries: list, out_dir: str) -> str:
    labels = [_short(q, 22) for q in queries]
    ratios = [g["avg_time"] / max(c["avg_time"], 1e-9)
              for g, c in zip(graph_infer, clip_infer)]
    x = np.arange(len(queries))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(x, ratios, width=0.55, color=COLORS["Graph"],
                  alpha=ALPHA_BAR, edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_yscale("log")
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, r * 1.08,
                f"{r:.0f}\u00d7", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333333")

    # Dashed red line marks parity, where both methods would be equally fast.
    ax.axhline(1, color="#CC2222", linestyle="--", linewidth=1.4, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_title("Inference Cost Ratio \u2014 Graph vs CLIP\n"
                 "(how many times slower the graph method is \u00b7 log scale)")
    ax.set_ylabel("Time ratio  Graph / CLIP  (\u00d7, log)")
    ax.set_ylim(0.5, max(ratios) * 3.5)

    legend = [
        Line2D([0], [0], color="#CC2222", lw=1.4, ls="--", label="Parity (1\u00d7)"),
        mpatches.Patch(color=COLORS["Graph"], alpha=ALPHA_BAR,
                       label="Graph slower than CLIP"),
    ]
    ax.legend(handles=legend, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", which="both", color="white")
    fig.tight_layout()
    return _save(fig, "fig4_inference_ratio.png", out_dir)


# ---------------------------------------------------------------------------
# Figure 5 — Qualitative comparison (top-k thumbnails), one figure per query
# ---------------------------------------------------------------------------
def plot_qualitative_comparison(queries, graph_infer, clip_infer,
                                id_to_item, find_image_fn, out_dir,
                                top_k=5, load_img_fn=None, draw_on_ax_fn=None):
    """
    For each query, render a 2 x top_k grid of thumbnails:
      row 0 = Graph (proposed)   row 1 = CLIP

    Which IDs are shown
    -------------------
    Prefers result["display_ids"] when present — these are the IDs benchmark.py
    has already filtered through main.filter_ids_with_images, i.e. guaranteed to
    have an image file on disk. Falls back to result["top_ids"] otherwise. This
    is what keeps the CLIP row from collapsing into "image not found": its raw
    top_ids ranking is dominated by description-only items.

    How thumbnails are drawn
    ------------------------
    If `draw_on_ax_fn(ax, wid)` is provided (e.g. main._load_img_or_placeholder),
    it is used directly so both methods share the EXACT same image-loading and
    placeholder code path as the rest of the app. Otherwise we fall back to
    `load_img_fn(path)` / find_image_fn(wid), or a default mpimg reader.
    """
    import matplotlib.image as mpimg
    if load_img_fn is None:
        def load_img_fn(p):
            try:
                return mpimg.imread(p)
            except Exception:
                return None

    def _ids_for(res):
        # Prefer the pre-filtered display IDs; pad with None so every row has top_k slots.
        r = res.get("result", {}) or {}
        ids = r.get("display_ids")
        if not ids:
            ids = r.get("top_ids", [])
        ids = list(ids[:top_k])
        ids += [None] * (top_k - len(ids))
        return ids

    paths = []
    for q_idx, (query, g_res, c_res) in enumerate(
        zip(queries, graph_infer, clip_infer), 1
    ):
        g_top = _ids_for(g_res)
        c_top = _ids_for(c_res)

        fig = plt.figure(figsize=(top_k * 3.0, 7.2), facecolor="white")
        fig.suptitle(f'Qualitative Comparison \u2014 Query {q_idx}: "{query}"',
                     fontsize=13, fontweight="bold", y=1.02)

        # Colored banner labelling each method's row.
        for y_pos, (text, color) in zip(
            [0.95, 0.49],
            [("Graph (proposed): semantic graph + dense retrieval", COLORS["Graph"]),
             ("CLIP (ViT-B/32, dot-product)", COLORS["CLIP"])],
        ):
            fig.text(0.5, y_pos, text, ha="center", va="center",
                     fontsize=10.5, fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.45", fc=color, ec="none",
                               alpha=0.92))

        for row, (top_ids, method, res) in enumerate(
            [(g_top, "Graph", g_res), (c_top, "CLIP", c_res)]
        ):
            color = COLORS[method]
            r = res.get("result", {}) or {}
            score_map = (r.get("fused_scores", {}) if method == "Graph"
                         else r.get("all_scores", {}))
            for col, wid in enumerate(top_ids):
                ax = fig.add_subplot(2, top_k, row * top_k + col + 1)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(2.5)

                # Empty slot: no result at this rank.
                if wid is None:
                    ax.set_facecolor("#EEEEEE")
                    ax.text(0.5, 0.5, "no\nresult", ha="center", va="center",
                            fontsize=8, color="#AAAAAA", transform=ax.transAxes)
                    ax.set_title(f"#{col+1}", fontsize=8, color="#AAAAAA")
                    continue

                # Use the app's own draw function when given, otherwise load the image ourselves.
                if draw_on_ax_fn is not None:
                    draw_on_ax_fn(ax, wid)
                else:
                    path = find_image_fn(wid)
                    arr = load_img_fn(path) if path else None
                    if arr is not None:
                        ax.imshow(arr, aspect="auto")
                    else:
                        ax.set_facecolor("#E8E8E8")
                        ax.text(0.5, 0.5, f"image not found\n(ID {wid})",
                                ha="center", va="center", fontsize=7,
                                color="#888888", transform=ax.transAxes)

                # Caption each thumbnail with its rank, title, and retrieval score.
                item = id_to_item.get(wid, {}) or {}
                title = (item.get("objectWork") or {}).get("titleText", "") or "\u2014"
                short = (title[:22] + "\u2026") if len(title) > 22 else title
                score = score_map.get(wid)
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "\u2014"
                ax.set_title(f"#{col+1} \u00b7 {short}\nscore: {score_str}",
                             fontsize=7.5, pad=4, linespacing=1.45, color="#222222")

        plt.subplots_adjust(hspace=0.55, wspace=0.08, top=0.86, bottom=0.04)
        paths.append(_save(fig, f"fig5{chr(96 + q_idx)}_qualitative_q{q_idx}.png",
                           out_dir))
    return paths


# ---------------------------------------------------------------------------
# Convenience: render every performance figure at once
# ---------------------------------------------------------------------------
def render_all_performance_figures(graph_train, clip_train,
                                   graph_infer, clip_infer, queries, out_dir):
    apply_style()
    plot_training_time(graph_train, clip_train, out_dir)
    plot_inference_time(graph_infer, clip_infer, queries, out_dir)
    plot_training_memory(graph_train, clip_train, out_dir)
    plot_inference_ratio(graph_infer, clip_infer, queries, out_dir)


# ---------------------------------------------------------------------------
# Demo with synthetic numbers (mirrors the orders of magnitude in the thesis)
# ---------------------------------------------------------------------------
def _demo():
    apply_style()
    out = "benchmark_demo"

    graph_train = {"avg_time": 13.4, "std_time": 12.0, "avg_mem": 750.0, "std_mem": 180.0}
    clip_train  = {"avg_time": 14.7, "std_time": 0.55, "avg_mem": 261.0, "std_mem": 40.0,
                   "model_mem_mb": 150.0}

    queries = [
        "a portrait of a woman with a dog",
        "a historical battle scene",
        "a landscape with mountains and a lake",
        "a still life with fruits",
    ]
    graph_infer = [
        {"avg_time": 1.05, "std_time": 0.10, "result": {"top_ids": [], "fused_scores": {}}},
        {"avg_time": 2.13, "std_time": 0.22, "result": {"top_ids": [], "fused_scores": {}}},
        {"avg_time": 0.89, "std_time": 0.08, "result": {"top_ids": [], "fused_scores": {}}},
        {"avg_time": 1.42, "std_time": 0.15, "result": {"top_ids": [], "fused_scores": {}}},
    ]
    clip_infer = [
        {"avg_time": 0.012, "std_time": 0.001, "result": {"top_ids": [], "all_scores": {}}},
        {"avg_time": 0.012, "std_time": 0.001, "result": {"top_ids": [], "all_scores": {}}},
        {"avg_time": 0.011, "std_time": 0.001, "result": {"top_ids": [], "all_scores": {}}},
        {"avg_time": 0.012, "std_time": 0.001, "result": {"top_ids": [], "all_scores": {}}},
    ]

    render_all_performance_figures(graph_train, clip_train,
                                   graph_infer, clip_infer, queries, out)
    print(f"\nDemo figures written to ./{out}/")


if __name__ == "__main__":
    _demo()