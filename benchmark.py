"""
benchmark.py — Benchmark amélioré : Original vs CLIP
======================================================

Corrections majeures par rapport à la version précédente :
  1. Séparation claire TRAINING (indexation) vs INFERENCE (requête).
  2. Comparaison équitable : l'Original utilise ses descriptions COMPLÈTES
     (pas tronquées à 300 chars), CLIP utilise ses descriptions tronquées.
  3. Émotion inférée dynamiquement depuis la requête dans les deux méthodes.
  4. Mémoire du modèle CLIP mesurée explicitement au moment du chargement.
  5. Warm-up run séparé (non compté) pour purger les effets de cold-start.
  6. Phase d'indexation CLIP (encode_text + encode_image) incluse dans le
     temps "training" de CLIP.
  7. n_runs configurable (défaut 5) pour des moyennes stables.
  8. Rapport complet : training time, inference time, memory (indexation + runtime).
"""

import os
import gc
import time
import tracemalloc
import numpy as np
import torch
import random
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# ── Import depuis main.py ──────────────────────────────────────────────────────
from main import (
    load_dataset, extract_caption_labels, extract_captions_moods,
    dense_retrieval, fusion_scores, phase2_emotion_filter,
    _compute_similarity, expand_query_semantically,
    construct_graph, train_model, extract_hierarchical_labels,
    extract_captions_moods_batch, extract_image_features_batch,
    caption_preprocessing, _embed, _get_graph_index,
    find_image, build_image_index, _wid_has_image,
    STOPWORDS, EMOTION_LABELS, EMOTION_VALENCE,
    USE_IMAGE_FEATURES, DEVICE,
)

# ── PIL ────────────────────────────────────────────────────────────────────────
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL non disponible — les images CLIP seront ignorées.")

# ── CLIP ───────────────────────────────────────────────────────────────────────
try:
    import clip as openai_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP non disponible. Installer : pip install git+https://github.com/openai/CLIP.git")

# ── psutil ─────────────────────────────────────────────────────────────────────
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil non disponible — mémoire RSS désactivée.")


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires
# ══════════════════════════════════════════════════════════════════════════════

def get_rss_mb() -> float:
    """Mémoire RSS du process courant en MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def truncate_for_clip(text: str, char_limit: int = 300) -> str:
    """
    Troncature CLIP : ~77 tokens ≈ 300 caractères.
    Uniquement utilisée pour les entrées de CLIP, pas de l'Original.
    """
    if len(text) <= char_limit:
        return text
    return text[:char_limit] + "…"


def infer_emotion_from_query(query: str) -> str:
    """
    Infère l'émotion souhaitée depuis la requête par similarité cosinus.
    Même logique que dans CLIP et identique à phase2_emotion_filter.
    """
    embs = _embed([query] + EMOTION_LABELS)
    sims = embs[0] @ embs[1:].T
    return EMOTION_LABELS[int(np.argmax(sims))]


# ══════════════════════════════════════════════════════════════════════════════
# ─── MÉTHODE ORIGINALE ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def train_original(dataset, raw):
    """
    Phase TRAINING de l'Original.
    Tout ce qui est fait une seule fois avant les requêtes.
    Retourne le contexte complet nécessaire à l'inférence.
    """
    global_freq = Counter()
    doc_freq    = Counter()
    for item in raw:
        tokens = caption_preprocessing(item["iconographicInterpretation"])
        global_freq.update(tokens)
        doc_freq.update(set(tokens))
    n_docs = len(raw)
    common = {w for w, c in global_freq.items() if c > 50}

    cap_labels = {
        item["workID"]: [
            t for t in caption_preprocessing(item["iconographicInterpretation"])
            if len(t) > 2 and t not in STOPWORDS and t not in common
        ]
        for item in dataset
        if item["iconographicInterpretation"].strip()
    }

    hier_labels = {
        item["workID"]: extract_hierarchical_labels(
            item["subjectTerms"], item["iconographicTerms"], item["conceptualTerms"]
        )
        for item in dataset
    }

    moods           = extract_captions_moods_batch(dataset)
    image_feat_dict = extract_image_features_batch(
        [item["workID"] for item in dataset]
    ) if USE_IMAGE_FEATURES else {}

    trained = train_model(cap_labels, hier_labels)
    G       = construct_graph(cap_labels, hier_labels, moods, trained, image_feat_dict)
    _get_graph_index(G)   # construit et cache l'index vectoriel

    return {
        "G": G,
        "moods": moods,
        "global_freq": global_freq,
        "common": common,
        "doc_freq": doc_freq,
        "n_docs": n_docs,
        "dataset": dataset,
    }


def infer_original(query: str, ctx: dict) -> dict:
    """
    Phase INFERENCE de l'Original pour une requête.
    Utilise les descriptions COMPLÈTES (pas de troncature).
    L'émotion est inférée depuis la requête (cohérent avec CLIP).
    """
    G          = ctx["G"]
    moods      = ctx["moods"]
    global_freq= ctx["global_freq"]
    common     = ctx["common"]
    doc_freq   = ctx["doc_freq"]
    n_docs     = ctx["n_docs"]
    dataset    = ctx["dataset"]

    plain_labels = extract_caption_labels(query, global_freq=global_freq, common=common)
    mood_labels  = [f"mood_{m}" for m in extract_captions_moods(query)]
    q_labels     = plain_labels + mood_labels

    graph_nodes    = list(G.nodes())
    expanded_labels= expand_query_semantically(q_labels, graph_nodes, top_k=3)
    q_labels       = expanded_labels

    dense_results = dense_retrieval(query, dataset, top_k=50)

    top_ids_raw, graph_scores = _compute_similarity(
        G, q_labels, top_k=50, doc_freq=doc_freq, n_docs=n_docs,
    )

    fused_scores = fusion_scores(graph_scores, dense_results, alpha=0.5)
    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    top_ids      = [wid for wid, _ in sorted_fused[:50]]

    # Émotion inférée dynamiquement (pas hardcodée)
    desired_emotion = infer_emotion_from_query(query)
    final_wid, emotion, score = phase2_emotion_filter(top_ids, desired_emotion, moods)

    return {"wid": final_wid, "emotion": emotion, "score": score}


def benchmark_original_training(dataset, raw, n_runs: int = 3) -> dict:
    """Mesure le temps et la mémoire de la phase training de l'Original."""
    times, mems = [], []
    ctx = None

    for run in range(n_runs):
        gc.collect()
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
        mem_before = get_rss_mb()
        t0 = time.perf_counter()

        ctx = train_original(dataset, raw)

        elapsed = time.perf_counter() - t0
        mem_after = get_rss_mb()
        times.append(elapsed)
        mems.append(mem_after - mem_before)

        if run < n_runs - 1:
            # Re-init pour que le prochain run re-construise tout
            # (on garde le dernier ctx pour l'inférence)
            pass

    return {
        "avg_time" : float(np.mean(times)),
        "std_time" : float(np.std(times)),
        "avg_mem"  : float(np.mean(mems)),
        "std_mem"  : float(np.std(mems)),
        "ctx"      : ctx,          # dernier contexte, prêt pour l'inférence
    }


def benchmark_original_inference(query: str, ctx: dict, n_runs: int = 5,
                                  warmup: int = 1) -> dict:
    """Mesure le temps et la mémoire de l'inférence de l'Original."""
    # Warm-up (non compté)
    for _ in range(warmup):
        infer_original(query, ctx)

    times, mems = [], []
    for _ in range(n_runs):
        gc.collect()
        mem_before = get_rss_mb()
        t0 = time.perf_counter()

        result = infer_original(query, ctx)

        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        mems.append(get_rss_mb() - mem_before)

    return {
        "avg_time" : float(np.mean(times)),
        "std_time" : float(np.std(times)),
        "avg_mem"  : float(np.mean(mems)),
        "std_mem"  : float(np.std(mems)),
        "result"   : result,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ─── MÉTHODE CLIP ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class CLIPRetriever:
    """Wrapper CLIP avec cache pour textes et images."""

    CLIP_MODEL = "ViT-B/32"

    def __init__(self):
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP non disponible.")
        mem_before = get_rss_mb()
        self.model, self.preprocess = openai_clip.load(self.CLIP_MODEL, device=DEVICE)
        self.model_mem_mb = get_rss_mb() - mem_before   # mémoire du modèle lui-même
        self.text_cache  = {}
        self.image_cache = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _safe_tokenize_and_encode(self, texts: list) -> np.ndarray:
        """Encode une liste de textes (avec fallback si trop longs)."""
        results = {}
        for text in texts:
            if text in self.text_cache:
                results[text] = self.text_cache[text]
                continue
            truncated = truncate_for_clip(text)
            try:
                with torch.no_grad():
                    tok  = openai_clip.tokenize([truncated]).to(DEVICE)
                    feat = self.model.encode_text(tok)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    vec  = feat.cpu().numpy().flatten()
            except RuntimeError:
                # Second fallback : encore plus court
                shorter = truncated[:150] + "…"
                try:
                    with torch.no_grad():
                        tok  = openai_clip.tokenize([shorter]).to(DEVICE)
                        feat = self.model.encode_text(tok)
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                        vec  = feat.cpu().numpy().flatten()
                except Exception:
                    vec = np.zeros(512, dtype=np.float32)
            self.text_cache[text] = vec
            results[text] = vec
        return np.vstack([results[t] for t in texts])

    def encode_text_batch(self, texts: list) -> np.ndarray:
        """Encode en batch (regroupe les textes non cachés)."""
        uncached = [t for t in texts if t not in self.text_cache]
        if uncached:
            # On encode texte par texte pour gérer les erreurs individuellement
            for text in uncached:
                self._safe_tokenize_and_encode([text])
        return np.vstack([self.text_cache[t] for t in texts])

    def encode_image(self, path: str) -> np.ndarray:
        if path in self.image_cache:
            return self.image_cache[path]
        try:
            img  = self.preprocess(PILImage.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = self.model.encode_image(img)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                vec  = feat.cpu().numpy().flatten()
        except Exception:
            vec = np.zeros(512, dtype=np.float32)
        self.image_cache[path] = vec
        return vec


def train_clip(dataset: list) -> dict:
    """
    Phase TRAINING de CLIP.
    Charge le modèle + encode toutes les descriptions et images du dataset.
    C'est l'équivalent de la construction du graphe + index pour l'Original.
    """
    retriever = CLIPRetriever()

    descriptions, wids_desc = [], []
    for item in dataset:
        desc = item["iconographicInterpretation"].strip()
        if desc:
            descriptions.append(desc)     # CLIP tronquera lui-même
            wids_desc.append(item["workID"])

    print(f"  [CLIP training] Encodage de {len(descriptions)} descriptions…")
    desc_embs = None
    if descriptions:
        try:
            desc_embs = retriever.encode_text_batch(descriptions)
        except Exception as e:
            print(f"  Warning encode descriptions: {e}")

    image_paths, wids_img = [], []
    if PIL_AVAILABLE:
        for item in dataset:
            path = find_image(item["workID"])
            if path:
                image_paths.append(path)
                wids_img.append(item["workID"])

        print(f"  [CLIP training] Encodage de {len(image_paths)} images…")

    img_embs = None
    if image_paths:
        vecs = []
        for path in image_paths:
            vecs.append(retriever.encode_image(path))
        img_embs = np.vstack(vecs) if vecs else None

    # Pré-encodage des labels d'émotion
    emo_embs = retriever.encode_text_batch(EMOTION_LABELS)

    return {
        "retriever" : retriever,
        "desc_embs" : desc_embs,
        "wids_desc" : wids_desc,
        "img_embs"  : img_embs,
        "wids_img"  : wids_img,
        "emo_embs"  : emo_embs,
    }


def infer_clip(query: str, ctx: dict) -> dict:
    """
    Phase INFERENCE de CLIP pour une requête.
    Le modèle et les index de descriptions/images sont déjà chargés dans ctx.
    """
    retriever = ctx["retriever"]
    desc_embs = ctx["desc_embs"]
    wids_desc = ctx["wids_desc"]
    img_embs  = ctx["img_embs"]
    wids_img  = ctx["wids_img"]
    emo_embs  = ctx["emo_embs"]

    # Encode la requête (potentiellement depuis le cache)
    query_emb = retriever.encode_text_batch([query])[0]

    scores = {}

    # Similarité avec descriptions
    if desc_embs is not None:
        sims = desc_embs @ query_emb
        for wid, sim in zip(wids_desc, sims):
            scores[wid] = float(sim)

    # Similarité avec images (moyenne avec le score texte)
    if img_embs is not None:
        img_sims = img_embs @ query_emb
        for wid, sim in zip(wids_img, img_sims):
            if wid in scores:
                scores[wid] = (scores[wid] + float(sim)) / 2.0
            else:
                scores[wid] = float(sim)

    if not scores:
        return {"wid": None, "emotion": "autre", "score": 0.0}

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids       = [wid for wid, _ in sorted_scores[:50]]
    final_wid     = top_ids[0] if top_ids else None

    # Émotion via similarité CLIP (cohérent avec l'Original)
    query_emo_sim  = emo_embs @ query_emb
    desired_emotion= EMOTION_LABELS[int(np.argmax(query_emo_sim))]

    return {
        "wid"    : final_wid,
        "emotion": desired_emotion,
        "score"  : sorted_scores[0][1] if sorted_scores else 0.0,
    }


def benchmark_clip_training(dataset: list, n_runs: int = 3) -> dict:
    """Mesure le temps et la mémoire de la phase training de CLIP."""
    times, mems = [], []
    model_mem   = 0.0
    ctx = None

    for run in range(n_runs):
        gc.collect()
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
        mem_before = get_rss_mb()
        t0 = time.perf_counter()

        ctx = train_clip(dataset)

        elapsed = time.perf_counter() - t0
        mem_after = get_rss_mb()
        times.append(elapsed)
        mems.append(mem_after - mem_before)
        model_mem = ctx["retriever"].model_mem_mb

    return {
        "avg_time"      : float(np.mean(times)),
        "std_time"      : float(np.std(times)),
        "avg_mem"       : float(np.mean(mems)),
        "std_mem"       : float(np.std(mems)),
        "model_mem_mb"  : model_mem,   # mémoire statique du modèle CLIP
        "ctx"           : ctx,
    }


def benchmark_clip_inference(query: str, ctx: dict, n_runs: int = 5,
                              warmup: int = 1) -> dict:
    """Mesure le temps et la mémoire de l'inférence de CLIP."""
    for _ in range(warmup):
        infer_clip(query, ctx)

    times, mems = [], []
    for _ in range(n_runs):
        gc.collect()
        mem_before = get_rss_mb()
        t0 = time.perf_counter()

        result = infer_clip(query, ctx)

        times.append(time.perf_counter() - t0)
        mems.append(get_rss_mb() - mem_before)

    return {
        "avg_time" : float(np.mean(times)),
        "std_time" : float(np.std(times)),
        "avg_mem"  : float(np.mean(mems)),
        "std_mem"  : float(np.std(mems)),
        "result"   : result,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ─── RAPPORT & VISUALISATIONS ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(orig_train, orig_infer_list,
                  clip_train, clip_infer_list,
                  queries):
    """Affiche un résumé lisible dans le terminal."""
    sep = "─" * 72
    print(f"\n{sep}")
    print("  RÉSUMÉ BENCHMARK — Original vs CLIP")
    print(sep)

    print(f"\n{'Phase TRAINING (une seule fois)':}")
    print(f"  {'Méthode':<12} {'Temps moy (s)':>14} {'±':>2} {'std':>8}  {'Mémoire (MB)':>13} {'±':>2} {'std':>8}")
    print(f"  {'Original':<12} {orig_train['avg_time']:>14.3f} {'':>2} {orig_train['std_time']:>8.3f}  {orig_train['avg_mem']:>13.1f} {'':>2} {orig_train['std_mem']:>8.1f}")
    clip_mem_note = f"{clip_train['avg_mem']:>13.1f} (+{clip_train['model_mem_mb']:.0f}MB modèle)"
    print(f"  {'CLIP':<12} {clip_train['avg_time']:>14.3f} {'':>2} {clip_train['std_time']:>8.3f}  {clip_mem_note}")

    print(f"\n{'Phase INFERENCE (par requête)':}")
    print(f"  {'Requête':<42} {'Orig (s)':>9} {'CLIP (s)':>9} {'Speedup':>8}")
    for q, o, c in zip(queries, orig_infer_list, clip_infer_list):
        short = q[:40] + "…" if len(q) > 40 else q
        speedup = o["avg_time"] / max(c["avg_time"], 1e-6)
        print(f"  {short:<42} {o['avg_time']:>9.4f} {c['avg_time']:>9.4f} {speedup:>7.1f}×")

    print(f"\n{sep}\n")


def save_csv(orig_train, orig_infer_list, clip_train, clip_infer_list,
             queries, out_dir: str):
    path = os.path.join(out_dir, "benchmark_results.csv")
    rows = []

    # Training rows
    for method, data in [("Original", orig_train), ("CLIP", clip_train)]:
        rows.append({
            "phase"     : "training",
            "method"    : method,
            "query"     : "N/A",
            "avg_time"  : data["avg_time"],
            "std_time"  : data["std_time"],
            "avg_mem_mb": data["avg_mem"],
            "std_mem_mb": data["std_mem"],
        })

    # Inference rows
    for q, o, c in zip(queries, orig_infer_list, clip_infer_list):
        for method, data in [("Original", o), ("CLIP", c)]:
            rows.append({
                "phase"     : "inference",
                "method"    : method,
                "query"     : q,
                "avg_time"  : data["avg_time"],
                "std_time"  : data["std_time"],
                "avg_mem_mb": data["avg_mem"],
                "std_mem_mb": data["std_mem"],
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV sauvegardé : {path}")


def plot_results(orig_train, orig_infer_list, clip_train, clip_infer_list,
                 queries, out_dir: str):
    short_queries = [q[:22] + "…" if len(q) > 22 else q for q in queries]
    x = np.arange(len(queries))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Benchmark Original vs CLIP — Time & Memory", fontsize=13, fontweight="bold")

    # ── 1. Temps training ────────────────────────────────────────────────────
    ax = axes[0, 0]
    methods = ["Original", "CLIP"]
    t_vals  = [orig_train["avg_time"], clip_train["avg_time"]]
    t_stds  = [orig_train["std_time"], clip_train["std_time"]]
    colors  = ["#4C72B0", "#DD8452"]
    bars = ax.bar(methods, t_vals, yerr=t_stds, color=colors, alpha=0.85,
                  capsize=6, width=0.4)
    ax.bar_label(bars, fmt="%.1fs", padding=4, fontsize=9)
    ax.set_title("Time — TRAINING Phase (indexing)")
    ax.set_ylabel("Seconds")
    ax.set_ylim(0, max(t_vals) * 1.35)
    ax.grid(axis="y", alpha=0.3)

    # ── 2. Temps inférence ───────────────────────────────────────────────────
    ax = axes[0, 1]
    o_times = [r["avg_time"] for r in orig_infer_list]
    o_stds  = [r["std_time"] for r in orig_infer_list]
    c_times = [r["avg_time"] for r in clip_infer_list]
    c_stds  = [r["std_time"] for r in clip_infer_list]
    ax.bar(x - width/2, o_times, width, yerr=o_stds, label="Original",
           color="#4C72B0", alpha=0.85, capsize=5)
    ax.bar(x + width/2, c_times, width, yerr=c_stds, label="CLIP",
           color="#DD8452", alpha=0.85, capsize=5)
    ax.set_title("Time — INFERENCE Phase (per query)")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(short_queries, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── 3. Mémoire training ──────────────────────────────────────────────────
    ax = axes[1, 0]
    m_vals = [orig_train["avg_mem"], clip_train["avg_mem"] + clip_train["model_mem_mb"]]
    m_stds = [orig_train["std_mem"], clip_train["std_mem"]]
    labels_mem = ["Original", f"CLIP\n(incl. ~{clip_train['model_mem_mb']:.0f}MB model)"]
    bars = ax.bar(labels_mem, m_vals, yerr=m_stds, color=colors, alpha=0.85,
                  capsize=6, width=0.4)
    ax.bar_label(bars, fmt="%.0fMB", padding=4, fontsize=9)
    ax.set_title("Memory — TRAINING Phase")
    ax.set_ylabel("MB")
    ax.set_ylim(0, max(m_vals) * 1.35)
    ax.grid(axis="y", alpha=0.3)

    # ── 4. Speedup inférence ─────────────────────────────────────────────────
    ax = axes[1, 1]
    speedups = [o["avg_time"] / max(c["avg_time"], 1e-9)
                for o, c in zip(orig_infer_list, clip_infer_list)]
    bars = ax.bar(short_queries, speedups, color="#55A868", alpha=0.85)
    ax.bar_label(bars, fmt="%.0f×", padding=4, fontsize=9)
    ax.axhline(1, color="red", linestyle="--", linewidth=1, label="No speedup")
    ax.set_title("Speedup CLIP vs Original (inference)")
    ax.set_ylabel("Acceleration factor")
    ax.set_xticklabels(short_queries, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "benchmark_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ─── MAIN ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Config ────────────────────────────────────────────────────────────────
    JSON_PATH   = "fabritius_data_base/fabritius_export.json.xz"
    N_SAMPLES   = 1000        # taille du dataset pour le benchmark
    N_RUNS_TRAIN= 2           # répétitions pour les phases training (lentes)
    N_RUNS_INFER= 5           # répétitions pour les phases inférence
    WARMUP      = 1           # runs de warm-up non comptabilisés
    OUT_DIR     = "benchmark"
    os.makedirs(OUT_DIR, exist_ok=True)

    TEST_QUERIES = [
        "un portrait d'une femme avec un chien",
        "une scène de bataille historique",
        "un paysage avec des montagnes et un lac",
        "une nature morte avec des fruits",
    ]

    # ── Chargement du dataset ─────────────────────────────────────────────────
    print("\n[1/6] Chargement du dataset…")
    raw = load_dataset(JSON_PATH)
    eligible = [item for item in raw if item["iconographicInterpretation"]]
    random.seed(42)
    dataset = random.sample(eligible, min(N_SAMPLES, len(eligible)))
    print(f"  Dataset : {len(dataset)} items avec description")

    # ── Training Original ─────────────────────────────────────────────────────
    print(f"\n[2/6] Benchmark TRAINING — Original ({N_RUNS_TRAIN} run(s))…")
    orig_train = benchmark_original_training(dataset, raw, n_runs=N_RUNS_TRAIN)
    print(f"  → {orig_train['avg_time']:.2f}s ± {orig_train['std_time']:.2f}s  |  "
          f"{orig_train['avg_mem']:.0f}MB ± {orig_train['std_mem']:.0f}MB")

    # ── Training CLIP ─────────────────────────────────────────────────────────
    if CLIP_AVAILABLE:
        print(f"\n[3/6] Benchmark TRAINING — CLIP ({N_RUNS_TRAIN} run(s))…")
        clip_train = benchmark_clip_training(dataset, n_runs=N_RUNS_TRAIN)
        print(f"  → {clip_train['avg_time']:.2f}s ± {clip_train['std_time']:.2f}s  |  "
              f"{clip_train['avg_mem']:.0f}MB ± {clip_train['std_mem']:.0f}MB  "
              f"(modèle seul : {clip_train['model_mem_mb']:.0f}MB)")
    else:
        print("\n[3/6] CLIP non disponible — skip.")
        clip_train = None

    # ── Inférence ─────────────────────────────────────────────────────────────
    orig_infer_list = []
    clip_infer_list = []

    print(f"\n[4/6] Benchmark INFERENCE — Original ({N_RUNS_INFER} runs + {WARMUP} warm-up)…")
    for query in TEST_QUERIES:
        print(f"  Requête : {query}")
        res = benchmark_original_inference(query, orig_train["ctx"],
                                           n_runs=N_RUNS_INFER, warmup=WARMUP)
        orig_infer_list.append(res)
        print(f"    → {res['avg_time']:.4f}s ± {res['std_time']:.4f}s")

    if CLIP_AVAILABLE and clip_train:
        print(f"\n[5/6] Benchmark INFERENCE — CLIP ({N_RUNS_INFER} runs + {WARMUP} warm-up)…")
        for query in TEST_QUERIES:
            print(f"  Requête : {query}")
            res = benchmark_clip_inference(query, clip_train["ctx"],
                                           n_runs=N_RUNS_INFER, warmup=WARMUP)
            clip_infer_list.append(res)
            print(f"    → {res['avg_time']:.4f}s ± {res['std_time']:.4f}s")
    else:
        clip_infer_list = [
            {"avg_time": 0, "std_time": 0, "avg_mem": 0, "std_mem": 0, "result": {}}
        ] * len(TEST_QUERIES)

    # ── Rapport ───────────────────────────────────────────────────────────────
    print("\n[6/6] Génération du rapport…")
    if clip_train:
        print_summary(orig_train, orig_infer_list,
                      clip_train, clip_infer_list,
                      TEST_QUERIES)
        save_csv(orig_train, orig_infer_list,
                 clip_train, clip_infer_list,
                 TEST_QUERIES, OUT_DIR)
        plot_results(orig_train, orig_infer_list,
                     clip_train, clip_infer_list,
                     TEST_QUERIES, OUT_DIR)
    else:
        print("  CLIP non disponible — rapport partiel.")