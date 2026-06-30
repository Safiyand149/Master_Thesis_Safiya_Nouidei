import os
import gc
import time
import numpy as np
import torch
import random
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

# --- Imports from main.py ---
import main as _main_module          # to access the module globals

from main import (
    load_dataset, extract_caption_labels, extract_captions_moods,
    dense_retrieval, fusion_scores, phase2_emotion_filter,
    _compute_similarity, expand_query_semantically,
    construct_graph, train_model, extract_hierarchical_labels,
    extract_captions_moods_batch, extract_image_features_batch,
    caption_preprocessing, _embed, _get_graph_index,
    find_image, build_image_index, _wid_has_image,
    filter_ids_with_images, _load_img_or_placeholder,
    STOPWORDS, EMOTION_LABELS, EMOTION_VALENCE,
    USE_IMAGE_FEATURES, DEVICE, IMAGE_ROOT,
)

# --- PIL ---
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available - CLIP images will be skipped.")

# --- CLIP ---
try:
    import clip as openai_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. Install: pip install git+https://github.com/openai/CLIP.git")

# --- Image lookup ---
# IMPORTANT: CLIP must use the EXACT same image index and lookup as the graph
# method, otherwise the two pipelines disagree on which workIDs have an image
# (different glob order, casing, cached index, etc.) and CLIP's thumbnails come
# up empty. We therefore alias directly onto main.py's functions instead of
# maintaining a second, separate CLIP image index.
import glob
import re

IMAGE_ROOT_CLIP = IMAGE_ROOT  # same path as main.py

# Same index, same regex, same disk layout as the graph method.
build_image_index_clip = build_image_index
find_image_clip        = find_image

# --- psutil ---
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available - RSS memory tracking disabled.")


# ══════════════════════════════════════════════════════════════════════════════
# Professional matplotlib style (applied globally)
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    # Font
    "font.family"        : "DejaVu Sans",
    "font.size"          : 11,
    "axes.titlesize"     : 13,
    "axes.titleweight"   : "bold",
    "axes.titlepad"      : 14,
    "axes.labelsize"     : 11,
    "axes.labelweight"   : "regular",
    "xtick.labelsize"    : 10,
    "ytick.labelsize"    : 10,
    "legend.fontsize"    : 10,
    "figure.titlesize"   : 14,
    "figure.titleweight" : "bold",
    # Background & grid
    "figure.facecolor"   : "white",
    "axes.facecolor"     : "#F7F8FA",
    "axes.grid"          : True,
    "grid.color"         : "white",
    "grid.linewidth"     : 1.2,
    "grid.alpha"         : 1.0,
    "axes.axisbelow"     : True,
    # Spines
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.spines.left"   : True,
    "axes.spines.bottom" : True,
    "axes.edgecolor"     : "#CCCED2",
    "axes.linewidth"     : 1.0,
    # Ticks
    "xtick.direction"    : "out",
    "ytick.direction"    : "out",
    "xtick.major.pad"    : 6,
    "ytick.major.pad"    : 6,
    # Legend
    "legend.framealpha"  : 0.92,
    "legend.edgecolor"   : "#CCCED2",
    "legend.fancybox"    : False,
    # Saving
    "savefig.dpi"        : 180,
    "savefig.bbox"       : "tight",
    "savefig.facecolor"  : "white",
})

# Consistent palette
COLORS = {
    "Graph": "#2E6BE6",   # deep blue (proposed method)
    "CLIP" : "#E8702A",   # warm orange (baseline)
}
ALPHA_BAR  = 0.88
CAPSIZE    = 6
ERROR_KW   = dict(elinewidth=1.4, capthick=1.4)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def get_rss_mb() -> float:
    if not PSUTIL_AVAILABLE:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def truncate_for_clip(text: str, char_limit: int = 300) -> str:
    """~77 CLIP tokens ~= 300 characters."""
    return text[:char_limit] + "…" if len(text) > char_limit else text


def infer_emotion_from_query(query: str) -> str:
    # Truncate for fairness with CLIP
    query = truncate_for_clip(query)
    embs = _embed([query] + EMOTION_LABELS)
    sims = embs[0] @ embs[1:].T
    return EMOTION_LABELS[int(np.argmax(sims))]


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 - Reset main.py's _GRAPH_INDEX cache between runs
# ══════════════════════════════════════════════════════════════════════════════

def _invalidate_graph_cache():
    """
    Resets the global variables _GRAPH_INDEX and _GRAPH_INDEX_HASH of the
    main.py module, forcing a full rebuild of the vector index on the next
    call to _get_graph_index().

    The _get_graph_index cache is controlled by the MODULE globals, not by
    function attributes, which is why we modify the imported module's
    attributes directly.
    """
    _main_module._GRAPH_INDEX      = None
    _main_module._GRAPH_INDEX_HASH = None


# ══════════════════════════════════════════════════════════════════════════════
# --- PROPOSED GRAPH-BASED METHOD ---
# ══════════════════════════════════════════════════════════════════════════════

def train_original(dataset, raw):
    """Indexing (training) phase of the proposed graph-based method."""
    # Truncate descriptions to 300 chars for fairness with CLIP
    truncated_dataset = [
        {**item, "iconographicInterpretation": truncate_for_clip(item["iconographicInterpretation"])}
        for item in dataset
    ]
    truncated_raw = [
        {**item, "iconographicInterpretation": truncate_for_clip(item["iconographicInterpretation"])}
        for item in raw
    ]

    global_freq = Counter()
    doc_freq    = Counter()
    for item in truncated_raw:
        tokens = caption_preprocessing(item["iconographicInterpretation"])
        global_freq.update(tokens)
        doc_freq.update(set(tokens))
    n_docs = len(truncated_raw)
    common = {w for w, c in global_freq.items() if c > 50}

    cap_labels = {
        item["workID"]: [
            t for t in caption_preprocessing(item["iconographicInterpretation"])
            if len(t) > 2 and t not in STOPWORDS and t not in common
        ]
        for item in truncated_dataset
        if item["iconographicInterpretation"].strip()
    }

    hier_labels = {
        item["workID"]: extract_hierarchical_labels(
            item["subjectTerms"], item["iconographicTerms"], item["conceptualTerms"]
        )
        for item in truncated_dataset
    }

    moods           = extract_captions_moods_batch(truncated_dataset)
    image_feat_dict = extract_image_features_batch(
        [item["workID"] for item in truncated_dataset]
    ) if USE_IMAGE_FEATURES else {}

    trained = train_model(cap_labels, hier_labels)
    G       = construct_graph(cap_labels, hier_labels, moods, trained, image_feat_dict)
    _get_graph_index(G)  # construction et mise en cache de l'index vectoriel

    return {
        "G"          : G,
        "moods"      : moods,
        "global_freq": global_freq,
        "common"     : common,
        "doc_freq"   : doc_freq,
        "n_docs"     : n_docs,
        "dataset"    : truncated_dataset,  # use the truncated dataset for consistency
    }


def _run_original_retrieval_only(query: str, ctx: dict):
    """
    Runs only the retrieval + fusion steps of the graph-based method.
    Phase 2 (emotion filter) is excluded - used for pure timing.
    """
    # Truncate the query to 300 chars for fairness with CLIP
    query = truncate_for_clip(query)

    G           = ctx["G"]
    global_freq = ctx["global_freq"]
    common      = ctx["common"]
    doc_freq    = ctx["doc_freq"]
    n_docs      = ctx["n_docs"]
    dataset     = ctx["dataset"]

    plain_labels = extract_caption_labels(query, global_freq=global_freq, common=common)
    mood_labels  = [f"mood_{m}" for m in extract_captions_moods(query)]
    q_labels     = plain_labels + mood_labels

    graph_nodes     = list(G.nodes())
    expanded_labels = expand_query_semantically(q_labels, graph_nodes, top_k=3)
    q_labels        = expanded_labels

    dense_results = dense_retrieval(query, dataset, top_k=50)

    _, graph_scores = _compute_similarity(
        G, q_labels, top_k=50, doc_freq=doc_freq, n_docs=n_docs,
    )

    fused_scores = fusion_scores(graph_scores, dense_results, alpha=0.5)
    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    top_ids      = [wid for wid, _ in sorted_fused[:50]]

    return top_ids, graph_scores, dense_results, fused_scores


def _run_original_timed(query: str, ctx: dict):
    """Full run (warm-up), unmeasured."""
    top_ids, _, _, _ = _run_original_retrieval_only(query, ctx)
    desired_emotion = infer_emotion_from_query(query)
    phase2_emotion_filter(top_ids, desired_emotion, ctx["moods"])


def benchmark_original_training(dataset, raw, n_runs: int = 3) -> dict:
    """
    Measures the time and memory of the graph-based indexing phase.

    FIX 1: the _GRAPH_INDEX cache is invalidated before each run.

    Memory is reported as the *peak* cold-build delta-RSS rather than the mean,
    for the same reason as CLIP: once CamemBERT embeddings are cached, later
    runs allocate far less and the mean would under-report the true footprint.
    """
    times, mems = [], []
    ctx = None

    for run in range(n_runs):
        ctx = None
        _invalidate_graph_cache()   # FIX 1: forced rebuild
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        mem_before = get_rss_mb()
        t0 = time.perf_counter()

        ctx = train_original(dataset, raw)

        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        mems.append(get_rss_mb() - mem_before)
        print(f"    Run {run+1}/{n_runs}: {elapsed:.2f}s | d-RSS {mems[-1]:.0f} MB")

    return {
        "avg_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "avg_mem" : float(max(0.0, max(mems))),   # peak cold-build delta-RSS
        "std_mem" : float(np.std(mems)),
        "ctx"     : ctx,
    }


def benchmark_original_inference(query: str, ctx: dict, n_runs: int = 5,
                                  warmup: int = 1) -> dict:
    """
    Measures the time and memory of graph-based inference.
    Phase 2 is excluded from timing for both methods (fairness).
    """
    for _ in range(warmup):
        _run_original_timed(query, ctx)

    times, mems = [], []
    result = None
    for _ in range(n_runs):
        gc.collect()
        mem_before = get_rss_mb()

        t0 = time.perf_counter()
        top_ids, _, _, fused_scores = _run_original_retrieval_only(query, ctx)
        elapsed = time.perf_counter() - t0

        times.append(elapsed)
        mems.append(get_rss_mb() - mem_before)

        # Phase 2 outside timing - for qualitative result only
        desired_emotion = infer_emotion_from_query(query)
        final_wid, emotion, score = phase2_emotion_filter(top_ids, desired_emotion, ctx["moods"])
        result = {
            "wid"        : final_wid,
            "emotion"    : emotion,
            "score"      : score,
            "top_ids"    : top_ids,
            "fused_scores": fused_scores,
        }

    return {
        "avg_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "avg_mem" : float(np.mean(mems)),
        "std_mem" : float(np.std(mems)),
        "result"  : result,
    }


# ══════════════════════════════════════════════════════════════════════════════
# --- CLIP METHOD ---
# ══════════════════════════════════════════════════════════════════════════════

class CLIPRetriever:
    """
    CLIP wrapper.
    encode_text_nocache: a real forward pass on every call (no cache).
    encode_text_batch_cached: internal cache reserved for the indexing phase.
    """

    CLIP_MODEL = "ViT-B/32"

    def __init__(self, measure_model_mem: bool = True):
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not available.")
        if measure_model_mem:
            mem_before = get_rss_mb()
            self.model, self.preprocess = openai_clip.load(self.CLIP_MODEL, device=DEVICE)
            self.model_mem_mb = get_rss_mb() - mem_before
        else:
            self.model, self.preprocess = openai_clip.load(self.CLIP_MODEL, device=DEVICE)
            self.model_mem_mb = 0.0

        self._index_cache: dict = {}
        self._image_cache: dict = {}

    def encode_text_nocache(self, text: str) -> np.ndarray:
        """Cache-free encoding - measures a real forward pass."""
        truncated = truncate_for_clip(text)
        for attempt in [truncated, truncated[:150] + "…"]:
            try:
                with torch.no_grad():
                    tok  = openai_clip.tokenize([attempt]).to(DEVICE)
                    feat = self.model.encode_text(tok)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    return feat.cpu().numpy().flatten()
            except RuntimeError:
                continue
        return np.zeros(512, dtype=np.float32)

    def encode_text_batch_cached(self, texts: list) -> np.ndarray:
        """Cached encoding - reserved for indexing."""
        for text in texts:
            if text not in self._index_cache:
                self._index_cache[text] = self.encode_text_nocache(text)
        return np.vstack([self._index_cache[t] for t in texts])

    def encode_image(self, path: str) -> np.ndarray:
        if path in self._image_cache:
            return self._image_cache[path]
        try:
            img  = self.preprocess(PILImage.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = self.model.encode_image(img)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                vec  = feat.cpu().numpy().flatten()
        except Exception:
            vec = np.zeros(512, dtype=np.float32)
        self._image_cache[path] = vec
        return vec


def train_clip(dataset: list, measure_model_mem: bool = True) -> dict:
    """CLIP training phase: model loading + dataset encoding."""
    retriever = CLIPRetriever(measure_model_mem=measure_model_mem)

    descriptions, wids_desc = [], []
    for item in dataset:
        desc = item["iconographicInterpretation"].strip()
        if desc:
            descriptions.append(desc)
            wids_desc.append(item["workID"])

    print(f"  [CLIP training] Encoding {len(descriptions)} descriptions...")
    desc_embs = None
    if descriptions:
        try:
            desc_embs = retriever.encode_text_batch_cached(descriptions)
        except Exception as e:
            print(f"  Warning while encoding descriptions: {e}")

    image_paths, wids_img = [], []
    if PIL_AVAILABLE:
        print(f"  [CLIP training] Looking up images for {len(dataset)} items "
              f"(via main.find_image)...")
        missing = 0
        for item in dataset:
            path = find_image_clip(item["workID"])   # == main.find_image
            if path:
                image_paths.append(path)
                wids_img.append(item["workID"])
            else:
                missing += 1
        print(f"  [CLIP training] {len(image_paths)} images found, "
              f"{missing} missing, out of {len(dataset)} items.")
    else:
        print("  [CLIP training] PIL not available - images skipped.")

    print(f"  [CLIP training] Encoding {len(image_paths)} images...")

    img_embs = None
    if image_paths:
        vecs = [retriever.encode_image(p) for p in image_paths]
        img_embs = np.vstack(vecs) if vecs else None

    emo_embs = retriever.encode_text_batch_cached(EMOTION_LABELS)

    return {
        "retriever": retriever,
        "desc_embs": desc_embs,
        "wids_desc": wids_desc,
        "img_embs" : img_embs,
        "wids_img" : wids_img,
        "emo_embs" : emo_embs,
        "dataset"  : dataset,
    }


def infer_clip(query: str, ctx: dict) -> dict:
    """
    CLIP inference for one query.
    FIX 3: encode_text_nocache -> a real forward pass on every call.
    Emotion detection (a pre-computed 9x512 dot-product) is included in the
    timed block because its cost is negligible (< 0.1 ms).
    """
    retriever = ctx["retriever"]
    desc_embs = ctx["desc_embs"]
    wids_desc = ctx["wids_desc"]
    img_embs  = ctx["img_embs"]
    wids_img  = ctx["wids_img"]
    emo_embs  = ctx["emo_embs"]

    query_emb = retriever.encode_text_nocache(query)   # FIX 3

    scores     = {}      # combined description+image score (used for the metric)
    img_scores = {}      # CLIP text->image score only (guaranteed displayable)
    desc_count = 0
    img_count  = 0
    if desc_embs is not None:
        for wid, sim in zip(wids_desc, desc_embs @ query_emb):
            scores[wid] = float(sim)
            desc_count += 1

    if img_embs is not None:
        for wid, sim in zip(wids_img, img_embs @ query_emb):
            img_count += 1
            img_scores[wid] = float(sim)        # keep the pure text->image score
            if wid in scores:
                scores[wid] = (scores[wid] + float(sim)) / 2.0
            else:
                scores[wid] = float(sim)

    print(f"  [CLIP inference] Scores based on {desc_count} descriptions and {img_count} images.")

    if not scores:
        return {"wid": None, "emotion": "autre", "score": 0.0,
                "top_ids": [], "img_top_ids": [], "all_scores": {}}

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Full ranking (every scored workID, best first) — kept for the quantitative
    # metric and timing parity with the graph method.
    top_ids = [wid for wid, _ in sorted_scores]

    # Ranking restricted to artworks that CLIP actually encoded as IMAGES. These
    # workIDs are, by construction, the ones whose image file find_image() could
    # locate at indexing time, so they are guaranteed to be displayable. This is
    # what the qualitative grid should show for CLIP — otherwise the top of the
    # combined ranking is dominated by description-only items (text->text CLIP
    # similarity is systematically higher than text->image), and the grid ends
    # up empty ("image not found").
    img_top_ids = [wid for wid, _ in
                   sorted(img_scores.items(), key=lambda x: x[1], reverse=True)]

    final_wid       = top_ids[0] if top_ids else None
    query_emo_sim   = emo_embs @ query_emb
    desired_emotion = EMOTION_LABELS[int(np.argmax(query_emo_sim))]

    return {
        "wid"        : final_wid,
        "emotion"    : desired_emotion,
        "score"      : scores.get(final_wid, 0.0),
        "top_ids"    : top_ids,
        "img_top_ids": img_top_ids,
        "all_scores" : scores,
    }


def benchmark_clip_training(dataset: list, n_runs: int = 3) -> dict:
    """
    Measures the time and memory of the CLIP training phase.

    FIX 4: model_mem_mb measured only on run 0 (cold load).
    FIX 2: avg_mem is the *peak* cold-load delta-RSS, not the mean.

    Why peak and not mean: each run rebuilds a fresh CLIPRetriever, but on
    runs 2+ the previous model and the OS page-cache keep the weights resident,
    so the measured delta-RSS collapses toward zero and the *mean* would
    under-report the true footprint (sometimes below model_mem_mb, which is
    impossible). We therefore (a) explicitly free the previous context before
    each run so memory is comparable, and (b) report the maximum delta across
    runs, which corresponds to the genuine one-off cost of loading + indexing.
    """
    times, mems = [], []
    model_mem_mb = 0.0
    ctx = None

    for run in range(n_runs):
        # Free the previous context BEFORE measuring, so each run starts from a
        # comparable baseline rather than inheriting the prior model in RAM.
        ctx = None
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        mem_before = get_rss_mb()
        t0 = time.perf_counter()

        ctx = train_clip(dataset, measure_model_mem=(run == 0))

        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        mems.append(get_rss_mb() - mem_before)
        if run == 0:
            model_mem_mb = ctx["retriever"].model_mem_mb
        print(f"    Run {run+1}/{n_runs}: {elapsed:.2f}s | d-RSS {mems[-1]:.0f} MB")

    # Peak cold-load delta-RSS (clipped at 0). Guaranteed >= model_mem_mb on the
    # cold run, so the figure annotation stays consistent.
    peak_mem = max(0.0, max(mems))
    avg_mem  = max(peak_mem, model_mem_mb)  # never report less than the weights

    return {
        "avg_time"    : float(np.mean(times)),
        "std_time"    : float(np.std(times)),
        "avg_mem"     : float(avg_mem),        # FIX 2: peak, not mean
        "std_mem"     : float(np.std(mems)),
        "model_mem_mb": float(model_mem_mb),
        "ctx"         : ctx,
    }


def benchmark_clip_inference(query: str, ctx: dict, n_runs: int = 5,
                              warmup: int = 1) -> dict:
    """
    Measures the time and memory of CLIP inference.
    FIX 3: encode_text_nocache -> a real forward pass on every run.
    """
    for _ in range(warmup):
        infer_clip(query, ctx)

    times, mems = [], []
    result = None
    for _ in range(n_runs):
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        mem_before = get_rss_mb()
        t0 = time.perf_counter()

        result = infer_clip(query, ctx)

        times.append(time.perf_counter() - t0)
        mems.append(get_rss_mb() - mem_before)

    return {
        "avg_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "avg_mem" : float(np.mean(mems)),
        "std_mem" : float(np.std(mems)),
        "result"  : result,
    }


# ══════════════════════════════════════════════════════════════════════════════
# --- REPORT & FIGURES ---
# ══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, name: str, out_dir: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {path}")


def print_summary(orig_train, orig_infer_list,
                  clip_train, clip_infer_list, queries):
    sep = "-" * 80
    print(f"\n{sep}")
    print("  BENCHMARK SUMMARY - Graph-based Method vs CLIP")
    print(sep)
    print(f"\n  Training phase (complete indexing, once):")
    print(f"  {'Method':<12} {'Avg. Time (s)':>15} {'+/-std':>8}  {'Avg. d-RSS (MB)':>16} {'+/-std':>8}")
    print(f"  {'Graph':<12} {orig_train['avg_time']:>15.3f} {orig_train['std_time']:>8.3f}  "
          f"{orig_train['avg_mem']:>16.1f} {orig_train['std_mem']:>8.1f}")
    print(f"  {'CLIP':<12} {clip_train['avg_time']:>15.3f} {clip_train['std_time']:>8.3f}  "
          f"{clip_train['avg_mem']:>16.1f} {clip_train['std_mem']:>8.1f}  "
          f"(incl. ~{clip_train['model_mem_mb']:.0f} MB model weights)")

    print(f"\n  Inference phase (per query — Phase 2 emotion filter excluded):")
    print(f"  {'Query':<44} {'Original (s)':>12} {'CLIP (s)':>10} {'Speedup':>8}")
    for q, o, c in zip(queries, orig_infer_list, clip_infer_list):
        short   = (q[:42] + "…") if len(q) > 42 else q
        speedup = o["avg_time"] / max(c["avg_time"], 1e-9)
        print(f"  {short:<44} {o['avg_time']:>12.4f} {c['avg_time']:>10.4f} {speedup:>7.1f}x")

    print(f"\n  [!] Architectural note:")
    print(f"     Graph: semantic graph + dense retrieval + score fusion.")
    print(f"     CLIP: direct dot-product on full vector index.")
    print(f"     Speedup reflects architectural difference as much as raw speed.")
    print(f"     Phase 2 (CamemBERT) is excluded from timing for both methods.")
    print(f"\n{sep}\n")


def save_csv(orig_train, orig_infer_list, clip_train, clip_infer_list,
             queries, out_dir: str):
    path = os.path.join(out_dir, "benchmark_results.csv")
    rows = []
    for method, data in [("Graph", orig_train), ("CLIP", clip_train)]:
        rows.append({"phase": "training", "method": method, "query": "N/A",
                     "avg_time_s": data["avg_time"], "std_time_s": data["std_time"],
                     "avg_delta_rss_mb": data["avg_mem"], "std_delta_rss_mb": data["std_mem"]})
    for q, o, c in zip(queries, orig_infer_list, clip_infer_list):
        for method, data in [("Graph", o), ("CLIP", c)]:
            rows.append({"phase": "inference (Phase 2 excluded)", "method": method, "query": q,
                         "avg_time_s": data["avg_time"], "std_time_s": data["std_time"],
                         "avg_delta_rss_mb": data["avg_mem"], "std_delta_rss_mb": data["std_mem"]})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved: {path}")



# --- FIGURES: delegated to benchmark_plots.py (English, publication-quality) ---
# The proposed method is labelled "Graph" in all figures; the result dicts keep
# their original structure so nothing else in the pipeline changes.

import benchmark_plots as bplots
bplots.apply_style()


def _save_fig(fig, name: str, out_dir: str):
    """Kept for backward compatibility (qualitative grid still uses it)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_training_time(graph_train, clip_train, out_dir):
    bplots.plot_training_time(graph_train, clip_train, out_dir)


def plot_inference_time(graph_infer, clip_infer, queries, out_dir):
    bplots.plot_inference_time(graph_infer, clip_infer, queries, out_dir)


def plot_training_memory(graph_train, clip_train, out_dir):
    bplots.plot_training_memory(graph_train, clip_train, out_dir)


def plot_inference_ratio(graph_infer, clip_infer, queries, out_dir):
    bplots.plot_inference_ratio(graph_infer, clip_infer, queries, out_dir)


def plot_qualitative_comparison(queries, graph_infer, clip_infer,
                                graph_ctx, out_dir, top_k=5):
    """
    Renders the 2 x top_k thumbnail grid.

    Image selection is done with the SAME function the graph algorithm uses in
    main.py - main.filter_ids_with_images(sorted_ids, scores, target_k) - applied
    identically to both methods. Image drawing is done with the SAME routine -
    main._load_img_or_placeholder(ax, wid). So CLIP and the graph method share
    the exact same image retrieval and display code path.
    """
    dataset    = graph_ctx["dataset"]
    id_to_item = {item["workID"]: item for item in dataset}

    # For each query, pick the top_k displayable IDs of each method using the
    # graph algorithm's own filter, and store them under "display_ids".
    #
    # rank_key selects WHICH ranking we feed to filter_ids_with_images:
    #   - Graph : "top_ids"      (fused graph+dense ranking)
    #   - CLIP  : "img_top_ids"  (ranking restricted to artworks CLIP encoded as
    #                             images, hence guaranteed to have a file on disk)
    # Using "top_ids" for CLIP is what caused the empty grid: that combined
    # ranking is dominated by description-only items (no local image), so the
    # filter exhausts the candidates before reaching target_k and every cell
    # falls back to "image not found".
    for label, infer_list, score_key, rank_key in [
        ("Graph", graph_infer, "fused_scores", "top_ids"),
        ("CLIP",  clip_infer,  "all_scores",   "img_top_ids"),
    ]:
        for qi, res in enumerate(infer_list, 1):
            r       = res.get("result", {})
            primary = r.get(rank_key) or []
            # Fall back to / top up with the combined ranking so we never show
            # fewer than top_k thumbnails when more image-backed candidates exist.
            backup  = r.get("top_ids", [])
            seen    = set(primary)
            ranked  = list(primary) + [w for w in backup if w not in seen]
            scores  = r.get(score_key, {})
            # SAME call as main.py: keep the first target_k IDs that have an image
            display = filter_ids_with_images(ranked, scores, target_k=top_k)
            r["display_ids"] = display
            res["result"]    = r
            print(f"    [{label} q{qi}] displayable top-{top_k}: "
                  f"{len(display)}/{top_k} (from {len(ranked)} ranked ids)")

    bplots.plot_qualitative_comparison(
        queries, graph_infer, clip_infer,
        id_to_item=id_to_item,
        find_image_fn=find_image,                 # main.find_image
        out_dir=out_dir,
        top_k=top_k,
        draw_on_ax_fn=_load_img_or_placeholder,   # main._load_img_or_placeholder
    )



# ══════════════════════════════════════════════════════════════════════════════
# ─── MAIN ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # --- Configuration ---
    JSON_PATH    = "fabritius_data_base/fabritius_export.json.xz"
    N_SAMPLES    = 1000
    N_RUNS_TRAIN = 3
    N_RUNS_INFER = 5
    WARMUP       = 1
    OUT_DIR      = "benchmark"
    os.makedirs(OUT_DIR, exist_ok=True)

    TEST_QUERIES = [
        "un portrait d'une femme assise",
        "une scène de bataille historique",
        "un paysage avec des montagnes et un lac",
        "une nature morte avec des fruits",
    ]

    # --- Dataset loading ---
    print("\n[1/6] Loading dataset...")
    raw      = load_dataset(JSON_PATH)
    build_image_index()  # warm the shared image index used by both methods

    # Eligible = has a description. We additionally split by image availability,
    # because the qualitative comparison can only show artworks whose image file
    # main.find_image() can locate. We therefore PRIORITISE description+image
    # items, then top up with description-only items if needed. This guarantees
    # both CLIP and the graph method have image-backed candidates to display
    # (otherwise CLIP's grid is all 'image not found').
    eligible = [item for item in raw if item["iconographicInterpretation"]]
    random.seed(42)
    random.shuffle(eligible)

    with_img, without_img = [], []
    for item in eligible:
        (with_img if _wid_has_image(item["workID"]) else without_img).append(item)
        # Stop early once we clearly have enough image-backed items.
        if len(with_img) >= N_SAMPLES:
            break

    if len(with_img) >= N_SAMPLES:
        dataset = with_img[:N_SAMPLES]
    else:
        # Not enough image-backed items: take all of them, then top up.
        need    = N_SAMPLES - len(with_img)
        dataset = with_img + without_img[:need]

    n_with_img = sum(1 for it in dataset if _wid_has_image(it["workID"]))
    print(f"  Dataset: {len(dataset)} items with description "
          f"({n_with_img} of them have an image on disk)")
    if n_with_img == 0:
        print("  [WARNING] No sampled artwork has a locatable image - check "
              "IMAGE_ROOT in main.py and the find_image regex; the qualitative "
              "figure cannot show thumbnails without matching image files.")

    # --- Training: Graph-based method ---
    print(f"\n[2/6] TRAINING - Graph-based method ({N_RUNS_TRAIN} run(s))...")
    graph_train = benchmark_original_training(dataset, raw, n_runs=N_RUNS_TRAIN)
    print(f"  -> avg {graph_train['avg_time']:.2f}s +/- {graph_train['std_time']:.2f}s  |  "
          f"d-RSS {graph_train['avg_mem']:.0f} MB +/- {graph_train['std_mem']:.0f} MB")

    # --- Training: CLIP ---
    clip_train = None
    clip_ctx   = None
    if CLIP_AVAILABLE:
        print(f"\n[3/6] TRAINING - CLIP ({N_RUNS_TRAIN} run(s))...")
        clip_train = benchmark_clip_training(dataset, n_runs=N_RUNS_TRAIN)
        clip_ctx   = clip_train["ctx"]
        print(f"  -> avg {clip_train['avg_time']:.2f}s +/- {clip_train['std_time']:.2f}s  |  "
              f"d-RSS {clip_train['avg_mem']:.0f} MB +/- {clip_train['std_mem']:.0f} MB  "
              f"(model only: {clip_train['model_mem_mb']:.0f} MB)")
    else:
        print("\n[3/6] CLIP not available - skipped.")

    # --- Inference: Graph-based method ---
    graph_infer_list = []
    print(f"\n[4/6] INFERENCE - Graph-based method "
          f"({N_RUNS_INFER} runs + {WARMUP} warm-up, Phase 2 excluded)...")
    for query in TEST_QUERIES:
        print(f"  Query: {query}")
        res = benchmark_original_inference(query, graph_train["ctx"],
                                           n_runs=N_RUNS_INFER, warmup=WARMUP)
        graph_infer_list.append(res)
        print(f"    -> {res['avg_time']:.4f}s +/- {res['std_time']:.4f}s")

    # --- Inference: CLIP ---
    clip_infer_list = []
    if CLIP_AVAILABLE and clip_ctx:
        print(f"\n[5/6] INFERENCE - CLIP "
              f"({N_RUNS_INFER} runs + {WARMUP} warm-up)...")
        for query in TEST_QUERIES:
            print(f"  Query: {query}")
            res = benchmark_clip_inference(query, clip_ctx,
                                           n_runs=N_RUNS_INFER, warmup=WARMUP)
            clip_infer_list.append(res)
            print(f"    -> {res['avg_time']:.4f}s +/- {res['std_time']:.4f}s")
    else:
        clip_infer_list = [
            {"avg_time": 0.0, "std_time": 0.0, "avg_mem": 0.0, "std_mem": 0.0,
             "result": {"wid": None, "emotion": "autre", "score": 0.0,
                        "top_ids": [], "img_top_ids": [], "all_scores": {}}}
        ] * len(TEST_QUERIES)

    # --- Report & figures ---
    print("\n[6/6] Generating report...")
    if clip_train:
        print_summary(graph_train, graph_infer_list,
                      clip_train, clip_infer_list, TEST_QUERIES)
        save_csv(graph_train, graph_infer_list,
                 clip_train, clip_infer_list, TEST_QUERIES, OUT_DIR)

        print("\n  Generating performance figures (fig1-4)...")
        plot_training_time   (graph_train, clip_train, OUT_DIR)
        plot_inference_time  (graph_infer_list, clip_infer_list, TEST_QUERIES, OUT_DIR)
        plot_training_memory (graph_train, clip_train, OUT_DIR)
        plot_inference_ratio (graph_infer_list, clip_infer_list, TEST_QUERIES, OUT_DIR)

        print("\n  Generating qualitative figures (fig5a-d)...")
        plot_qualitative_comparison(
            TEST_QUERIES,
            graph_infer_list,
            clip_infer_list,
            graph_train["ctx"],
            OUT_DIR,
            top_k=5,
        )

        print(f"\n  All figures are in the '{OUT_DIR}/' folder.")
    else:
        print("  CLIP not available - partial report (Graph only).")