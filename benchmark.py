
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

# ── Imports depuis main.py ────────────────────────────────────────────────────
import main as _main_module          # pour accéder aux globals du module

from main import (
    load_dataset, extract_caption_labels, extract_captions_moods,
    dense_retrieval, fusion_scores, phase2_emotion_filter,
    _compute_similarity, expand_query_semantically,
    construct_graph, train_model, extract_hierarchical_labels,
    extract_captions_moods_batch, extract_image_features_batch,
    caption_preprocessing, _embed, _get_graph_index,
    find_image, build_image_index, _wid_has_image,
    STOPWORDS, EMOTION_LABELS, EMOTION_VALENCE,
    USE_IMAGE_FEATURES, DEVICE, IMAGE_ROOT,
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
    print("CLIP non disponible. Installation : pip install git+https://github.com/openai/CLIP.git")

# ── Fonction pour trouver les images (copie de main.py) ────────────────────────
import glob
import re

IMAGE_ROOT_CLIP = IMAGE_ROOT  # Utilise le même chemin que main.py  
IMAGE_INDEX_CLIP = None

def build_image_index_clip():
    global IMAGE_INDEX_CLIP
    if IMAGE_INDEX_CLIP is None:
        print("Indexation des images pour CLIP...")
        pattern = os.path.join(IMAGE_ROOT_CLIP, "**", "*.jpg")
        files = glob.glob(pattern, recursive=True)
        IMAGE_INDEX_CLIP = {os.path.basename(f): f for f in files}
    return IMAGE_INDEX_CLIP

def find_image_clip(work_id: str):
    index = build_image_index_clip()
    id_pattern = re.compile(
        rf"-{re.escape(str(work_id))}[a-z]*[-.]",
        re.IGNORECASE
    )
    for name, path in index.items():
        if id_pattern.search(name):
            return path
    return None

# ── psutil ─────────────────────────────────────────────────────────────────────
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil non disponible — suivi mémoire RSS désactivé.")


# ══════════════════════════════════════════════════════════════════════════════
# Style matplotlib professionnel (appliqué globalement)
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    # Police
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
    # Fond & grille
    "figure.facecolor"   : "white",
    "axes.facecolor"     : "#F7F8FA",
    "axes.grid"          : True,
    "grid.color"         : "white",
    "grid.linewidth"     : 1.2,
    "grid.alpha"         : 1.0,
    "axes.axisbelow"     : True,
    # Bordures
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
    # Légende
    "legend.framealpha"  : 0.92,
    "legend.edgecolor"   : "#CCCED2",
    "legend.fancybox"    : False,
    # Sauvegarde
    "savefig.dpi"        : 180,
    "savefig.bbox"       : "tight",
    "savefig.facecolor"  : "white",
})

# Palette cohérente
COLORS = {
    "Original": "#2E6BE6",   # bleu profond
    "CLIP"    : "#E8702A",   # orange chaud
}
ALPHA_BAR  = 0.88
CAPSIZE    = 6
ERROR_KW   = dict(elinewidth=1.4, capthick=1.4)


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires
# ══════════════════════════════════════════════════════════════════════════════

def get_rss_mb() -> float:
    if not PSUTIL_AVAILABLE:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def truncate_for_clip(text: str, char_limit: int = 300) -> str:
    """~77 tokens CLIP ≈ 300 caractères."""
    return text[:char_limit] + "…" if len(text) > char_limit else text


def infer_emotion_from_query(query: str) -> str:
    # Troncature pour équité avec CLIP
    query = truncate_for_clip(query)
    embs = _embed([query] + EMOTION_LABELS)
    sims = embs[0] @ embs[1:].T
    return EMOTION_LABELS[int(np.argmax(sims))]


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 — Réinitialisation du cache _GRAPH_INDEX de main.py entre les runs
# ══════════════════════════════════════════════════════════════════════════════

def _invalidate_graph_cache():
    """
    Remet à zéro les variables globales _GRAPH_INDEX et _GRAPH_INDEX_HASH
    du module main.py, forçant une reconstruction complète du graphe vectoriel
    au prochain appel de _get_graph_index().

    Le cache de _get_graph_index est contrôlé par les globals du MODULE, pas par
    des attributs de la fonction — c'est pourquoi on modifie directement les
    attributs du module importé.
    """
    _main_module._GRAPH_INDEX      = None
    _main_module._GRAPH_INDEX_HASH = None


# ══════════════════════════════════════════════════════════════════════════════
# ─── MÉTHODE ORIGINALE ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def train_original(dataset, raw):
    """Phase d'entraînement (indexation) de la méthode Originale."""
    # Troncature des descriptions à 300 caractères pour équité avec CLIP
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
        "dataset"    : truncated_dataset,  # utiliser le dataset tronqué pour cohérence
    }


def _run_original_retrieval_only(query: str, ctx: dict):
    """
    Exécute uniquement les étapes récupération + fusion de la méthode Originale.
    La Phase 2 (filtre émotionnel) est exclue — utilisé pour le chronométrage pur.
    """
    # Troncature de la requête à 300 caractères pour équité avec CLIP
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
    """Exécution complète (warm-up) sans mesure."""
    top_ids, _, _, _ = _run_original_retrieval_only(query, ctx)
    desired_emotion = infer_emotion_from_query(query)
    phase2_emotion_filter(top_ids, desired_emotion, ctx["moods"])


def benchmark_original_training(dataset, raw, n_runs: int = 3) -> dict:
    """
    Mesure le temps et la mémoire de la phase d'entraînement Originale.
    FIX 1 : le cache _GRAPH_INDEX est invalidé avant chaque run.
    """
    times, mems = [], []
    ctx = None

    for run in range(n_runs):
        _invalidate_graph_cache()   # FIX 1 : reconstruction forcée
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        mem_before = get_rss_mb()
        t0 = time.perf_counter()

        ctx = train_original(dataset, raw)

        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        mems.append(get_rss_mb() - mem_before)
        print(f"    Run {run+1}/{n_runs}: {elapsed:.2f}s | ΔRSS {mems[-1]:.0f} MB")

    return {
        "avg_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "avg_mem" : float(np.mean(mems)),
        "std_mem" : float(np.std(mems)),
        "ctx"     : ctx,
    }


def benchmark_original_inference(query: str, ctx: dict, n_runs: int = 5,
                                  warmup: int = 1) -> dict:
    """
    Mesure le temps et la mémoire de l'inférence Originale.
    Phase 2 exclue du chronométrage pour les deux méthodes (équité).
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

        # Phase 2 hors chrono — pour résultat qualitatif uniquement
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
# ─── MÉTHODE CLIP ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class CLIPRetriever:
    """
    Wrapper CLIP.
    encode_text_nocache : forward pass réel à chaque appel (pas de cache).
    encode_text_batch_cached : cache interne réservé à la phase d'indexation.
    """

    CLIP_MODEL = "ViT-B/32"

    def __init__(self, measure_model_mem: bool = True):
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP non disponible.")
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
        """Encodage sans cache — mesure un vrai forward pass."""
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
        """Encodage avec cache — réservé à l'indexation."""
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
    """Phase d'entraînement CLIP : chargement modèle + encodage du dataset."""
    retriever = CLIPRetriever(measure_model_mem=measure_model_mem)

    descriptions, wids_desc = [], []
    for item in dataset:
        desc = item["iconographicInterpretation"].strip()
        if desc:
            descriptions.append(desc)
            wids_desc.append(item["workID"])

    print(f"  [CLIP training] Encodage de {len(descriptions)} descriptions…")
    desc_embs = None
    if descriptions:
        try:
            desc_embs = retriever.encode_text_batch_cached(descriptions)
        except Exception as e:
            print(f"  Avertissement encodage descriptions : {e}")

    image_paths, wids_img = [], []
    if PIL_AVAILABLE:
        print(f"  [CLIP training] Recherche d'images pour {len(dataset)} items…")
        for item in dataset:
            path = find_image_clip(item["workID"])
            if path:
                image_paths.append(path)
                wids_img.append(item["workID"])
            else:
                print(f"    Image non trouvée pour workID {item['workID']}")
        print(f"  [CLIP training] {len(image_paths)} images trouvées sur {len(dataset)} items.")
    else:
        print("  [CLIP training] PIL non disponible — images ignorées.")

    print(f"  [CLIP training] Encodage de {len(image_paths)} images…")

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
    Inférence CLIP pour une requête.
    FIX 3 : encode_text_nocache → forward pass réel à chaque appel.
    La détection d'émotion (produit scalaire 9×512 pré-calculé) est incluse
    dans le bloc chronométré car son coût est négligeable (< 0.1 ms).
    """
    retriever = ctx["retriever"]
    desc_embs = ctx["desc_embs"]
    wids_desc = ctx["wids_desc"]
    img_embs  = ctx["img_embs"]
    wids_img  = ctx["wids_img"]
    emo_embs  = ctx["emo_embs"]

    query_emb = retriever.encode_text_nocache(query)   # FIX 3

    scores = {}
    desc_count = 0
    img_count = 0
    if desc_embs is not None:
        for wid, sim in zip(wids_desc, desc_embs @ query_emb):
            scores[wid] = float(sim)
            desc_count += 1

    if img_embs is not None:
        for wid, sim in zip(wids_img, img_embs @ query_emb):
            img_count += 1
            if wid in scores:
                scores[wid] = (scores[wid] + float(sim)) / 2.0
            else:
                scores[wid] = float(sim)

    print(f"  [CLIP inference] Scores basés sur {desc_count} descriptions et {img_count} images.")

    if not scores:
        return {"wid": None, "emotion": "autre", "score": 0.0, "top_ids": [], "all_scores": {}}

    sorted_scores   = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids         = [wid for wid, _ in sorted_scores[:50]]
    final_wid       = top_ids[0] if top_ids else None
    query_emo_sim   = emo_embs @ query_emb
    desired_emotion = EMOTION_LABELS[int(np.argmax(query_emo_sim))]

    return {
        "wid"       : final_wid,
        "emotion"   : desired_emotion,
        "score"     : sorted_scores[0][1] if sorted_scores else 0.0,
        "top_ids"   : top_ids,
        "all_scores": scores,
    }


def benchmark_clip_training(dataset: list, n_runs: int = 3) -> dict:
    """
    Mesure le temps et la mémoire de la phase d'entraînement CLIP.
    FIX 4 : model_mem_mb mesuré uniquement au run 0.
    FIX 2 : avg_mem = delta RSS total (poids modèle inclus) — pas de double-comptage.
    """
    times, mems = [], []
    model_mem_mb = 0.0
    ctx = None

    for run in range(n_runs):
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
        print(f"    Run {run+1}/{n_runs}: {elapsed:.2f}s | ΔRSS {mems[-1]:.0f} MB")

    return {
        "avg_time"    : float(np.mean(times)),
        "std_time"    : float(np.std(times)),
        "avg_mem"     : float(np.mean(mems)),   # FIX 2
        "std_mem"     : float(np.std(mems)),
        "model_mem_mb": model_mem_mb,
        "ctx"         : ctx,
    }


def benchmark_clip_inference(query: str, ctx: dict, n_runs: int = 5,
                              warmup: int = 1) -> dict:
    """
    Mesure le temps et la mémoire de l'inférence CLIP.
    FIX 3 : encode_text_nocache → forward pass réel à chaque run.
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
# ─── RAPPORT & VISUALISATIONS ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, name: str, out_dir: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure sauvegardée : {path}")


def print_summary(orig_train, orig_infer_list,
                  clip_train, clip_infer_list, queries):
    sep = "─" * 80
    print(f"\n{sep}")
    print("  BENCHMARK SUMMARY — Original Method vs CLIP")
    print(sep)
    print(f"\n  Training phase (complete indexing, once):")
    print(f"  {'Method':<12} {'Avg. Time (s)':>15} {'± std':>8}  {'Avg. Δ RSS (MB)':>16} {'± std':>8}")
    print(f"  {'Original':<12} {orig_train['avg_time']:>15.3f} {orig_train['std_time']:>8.3f}  "
          f"{orig_train['avg_mem']:>16.1f} {orig_train['std_mem']:>8.1f}")
    print(f"  {'CLIP':<12} {clip_train['avg_time']:>15.3f} {clip_train['std_time']:>8.3f}  "
          f"{clip_train['avg_mem']:>16.1f} {clip_train['std_mem']:>8.1f}  "
          f"(incl. ~{clip_train['model_mem_mb']:.0f} MB model weights)")

    print(f"\n  Inference phase (per query — Phase 2 emotion filter excluded):")
    print(f"  {'Query':<44} {'Original (s)':>12} {'CLIP (s)':>10} {'Speedup':>8}")
    for q, o, c in zip(queries, orig_infer_list, clip_infer_list):
        short   = (q[:42] + "…") if len(q) > 42 else q
        speedup = o["avg_time"] / max(c["avg_time"], 1e-9)
        print(f"  {short:<44} {o['avg_time']:>12.4f} {c['avg_time']:>10.4f} {speedup:>7.1f}×")

    print(f"\n  ⚠  Architectural Note:")
    print(f"     Original: semantic graph + dense retrieval + score fusion.")
    print(f"     CLIP: direct dot-product on full vector index.")
    print(f"     Speedup reflects architectural difference as much as raw speed.")
    print(f"     Phase 2 (CamemBERT) is excluded from timing for both methods.")
    print(f"\n{sep}\n")


def save_csv(orig_train, orig_infer_list, clip_train, clip_infer_list,
             queries, out_dir: str):
    path = os.path.join(out_dir, "benchmark_results.csv")
    rows = []
    for method, data in [("Original", orig_train), ("CLIP", clip_train)]:
        rows.append({"phase": "training", "method": method, "query": "N/A",
                     "avg_time_s": data["avg_time"], "std_time_s": data["std_time"],
                     "avg_delta_rss_mb": data["avg_mem"], "std_delta_rss_mb": data["std_mem"]})
    for q, o, c in zip(queries, orig_infer_list, clip_infer_list):
        for method, data in [("Original", o), ("CLIP", c)]:
            rows.append({"phase": "inference (Phase 2 exclue)", "method": method, "query": q,
                         "avg_time_s": data["avg_time"], "std_time_s": data["std_time"],
                         "avg_delta_rss_mb": data["avg_mem"], "std_delta_rss_mb": data["std_mem"]})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV sauvegardé : {path}")


# ── Helpers graphiques ────────────────────────────────────────────────────────

def _bar_label(ax, bars, fmt="{:.1f}", unit="", fontsize=10, rotation=0, padding=5):
    """Ajoute des étiquettes numériques au-dessus de chaque barre."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + padding * ax.get_ylim()[1] / 100,
                fmt.format(h) + unit,
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold",
                rotation=rotation, color="#333333",
            )


# ── Figure 1 — Temps d'entraînement ──────────────────────────────────────────

def plot_training_time(orig_train, clip_train, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 5))

    methods = ["Original", "CLIP"]
    t_vals  = [orig_train["avg_time"], clip_train["avg_time"]]
    t_stds  = [orig_train["std_time"], clip_train["std_time"]]
    colors  = [COLORS[m] for m in methods]

    bars = ax.bar(methods, t_vals, yerr=t_stds, color=colors, width=0.45,
                  alpha=ALPHA_BAR, capsize=CAPSIZE, error_kw=ERROR_KW,
                  edgecolor="white", linewidth=0.8, zorder=3)

    _bar_label(ax, bars, fmt="{:.1f}", unit=" s", fontsize=11, padding=2)

    ax.set_title("Training Phase — Indexing Time", pad=16)
    ax.set_ylabel("Average Duration (seconds)")
    ax.set_ylim(0, max(t_vals) * 1.38)
    ax.tick_params(axis="x", bottom=False)

    # Annotation ratio
    if t_vals[1] > 0:
        ratio = t_vals[0] / t_vals[1]
        ax.annotate(
            f"Ratio : {ratio:.1f}×",
            xy=(0.97, 0.93), xycoords="axes fraction",
            ha="right", fontsize=9, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EEEFF3", ec="#CCCED2", lw=0.8),
        )

    fig.tight_layout()
    _save_fig(fig, "fig1_training_time.png", out_dir)


# ── Figure 2 — Temps d'inférence par requête ─────────────────────────────────

def plot_inference_time(orig_infer_list, clip_infer_list, queries, out_dir: str):
    short_q = [(q[:20] + "…") if len(q) > 20 else q for q in queries]
    x       = np.arange(len(queries))
    width   = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))

    o_times = [r["avg_time"] for r in orig_infer_list]
    o_stds  = [r["std_time"] for r in orig_infer_list]
    c_times = [r["avg_time"] for r in clip_infer_list]
    c_stds  = [r["std_time"] for r in clip_infer_list]

    b1 = ax.bar(x - width / 2, o_times, width, yerr=o_stds,
                label="Original", color=COLORS["Original"],
                alpha=ALPHA_BAR, capsize=CAPSIZE, error_kw=ERROR_KW,
                edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + width / 2, c_times, width, yerr=c_stds,
                label="CLIP",     color=COLORS["CLIP"],
                alpha=ALPHA_BAR, capsize=CAPSIZE, error_kw=ERROR_KW,
                edgecolor="white", linewidth=0.8, zorder=3)

    y_max = max(max(o_times), max(c_times))
    for bars, vals in [(b1, o_times), (b2, c_times)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + y_max * 0.02,
                    f"{v:.3f}s", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", rotation=45, color="#333333")

    ax.set_title("Inference Phase — Time per Query\n(Phase 2 / Emotion Filter Excluded)", pad=14)
    ax.set_ylabel("Average Duration per Query (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_q, rotation=20, ha="right")
    ax.set_ylim(0, y_max * 1.45)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save_fig(fig, "fig2_inference_time.png", out_dir)


# ── Figure 3 — Mémoire d'entraînement ────────────────────────────────────────

def plot_training_memory(orig_train, clip_train, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 5))

    methods = ["Original", "CLIP"]
    m_vals  = [orig_train["avg_mem"], clip_train["avg_mem"]]
    m_stds  = [orig_train["std_mem"], clip_train["std_mem"]]
    colors  = [COLORS[m] for m in methods]

    bars = ax.bar(methods, m_vals, yerr=m_stds, color=colors, width=0.45,
                  alpha=ALPHA_BAR, capsize=CAPSIZE, error_kw=ERROR_KW,
                  edgecolor="white", linewidth=0.8, zorder=3)

    _bar_label(ax, bars, fmt="{:.0f}", unit=" MB", fontsize=11, padding=2)

    # Annotation poids du modèle sur la barre CLIP
    model_mem = clip_train["model_mem_mb"]
    if model_mem > 0 and len(bars) > 1:
        clip_bar  = bars[1]
        clip_x    = clip_bar.get_x() + clip_bar.get_width() / 2
        ax.annotate(
            f"incl. ~{model_mem:.0f} MB\n(model weights)",
            xy=(clip_x, model_mem),
            xytext=(clip_x + 0.3, model_mem + m_vals[1] * 0.12),
            fontsize=8.5, color="#666666",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8),
        )

    ax.set_title("Training Phase — Δ RSS Memory Footprint\n(FIX: No Double-Counting)", pad=14)
    ax.set_ylabel("RSS Memory Increase (MB)")
    ax.set_ylim(0, max(m_vals) * 1.48)
    ax.tick_params(axis="x", bottom=False)
    fig.tight_layout()
    _save_fig(fig, "fig3_training_memory.png", out_dir)


# ── Figure 4 — Accélération CLIP vs Original ─────────────────────────────────

def plot_speedup(orig_infer_list, clip_infer_list, queries, out_dir: str):
    short_q  = [(q[:20] + "…") if len(q) > 20 else q for q in queries]
    speedups = [o["avg_time"] / max(c["avg_time"], 1e-9)
                for o, c in zip(orig_infer_list, clip_infer_list)]
    colors   = ["#2BA84A" if s >= 1 else "#C44E52" for s in speedups]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(short_q, speedups, color=colors, alpha=ALPHA_BAR,
                  edgecolor="white", linewidth=0.8, zorder=3)

    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2,
                s + max(speedups) * 0.02,
                f"{s:.1f}×",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333333")

    ax.axhline(1, color="#CC2222", linestyle="--", linewidth=1.4,
               label="Baseline (1×)", zorder=4)

    # Manual legend for speedup / slowdown
    patch_fast = mpatches.Patch(color="#2BA84A", alpha=ALPHA_BAR, label="CLIP faster")
    patch_slow = mpatches.Patch(color="#C44E52", alpha=ALPHA_BAR, label="CLIP slower")
    ax.legend(handles=[patch_fast, patch_slow,
                       plt.Line2D([0], [0], color="#CC2222", lw=1.4, ls="--",
                                  label="Baseline (1×)")],
              loc="upper right", framealpha=0.92)

    ax.set_title(
        "Inference Speedup — CLIP vs Original\n"
        "(retrieval + fusion only, Phase 2 excluded)",
        pad=14,
    )
    ax.set_ylabel("Speedup Factor (×)")
    ax.set_ylim(0, max(speedups) * 1.30)
    ax.set_xticklabels(short_q, rotation=20, ha="right")
    fig.tight_layout()
    _save_fig(fig, "fig4_inference_speedup.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ─── FIGURE 5 — COMPARAISON QUALITATIVE (top-5 images côte-à-côte) ───────────
# ══════════════════════════════════════════════════════════════════════════════

def _load_img_safe(path: str):
    """Charge une image depuis le disque, retourne None en cas d'échec."""
    try:
        import matplotlib.image as mpimg
        return mpimg.imread(path)
    except Exception:
        return None


def plot_qualitative_comparison(
    queries: list,
    orig_infer_list: list,
    clip_infer_list: list,
    orig_ctx: dict,
    out_dir: str,
    top_k: int = 5,
):
    """
    Figure 5 — Qualitative Comparison: for each query, displays in a grid
    the top-{top_k} results from the Original method (top row, blue)
    and CLIP (bottom row, orange).

    Structure: 1 figure per query → fig5a_qualitative_q1.png, …
    Each figure = 2 rows × top_k columns of images.
    """
    dataset    = orig_ctx["dataset"]
    id_to_item = {item["workID"]: item for item in dataset}

    for q_idx, (query, orig_res, clip_res) in enumerate(
        zip(queries, orig_infer_list, clip_infer_list), 1
    ):
        # ── Retrieval of top_k IDs for each method ─────────────────
        orig_top  = orig_res["result"]["top_ids"][:top_k]
        clip_top  = clip_res["result"]["top_ids"][:top_k]

        # Pad to top_k if fewer results available
        while len(orig_top) < top_k:
            orig_top.append(None)
        while len(clip_top) < top_k:
            clip_top.append(None)

        # ── Layout ────────────────────────────────────────────────────
        fig = plt.figure(figsize=(top_k * 3.2, 8.0), facecolor="white")
        fig.suptitle(
            f'Qualitative Comparison — Query {q_idx}: "{query}"',
            fontsize=13, fontweight="bold", y=1.01,
        )

        # Row subtitles (two phantom axes full width)
        row_labels = [
            ("Original Method (semantic graph + dense retrieval)", COLORS["Original"]),
            ("CLIP (dot-product ViT-B/32)",                        COLORS["CLIP"]),
        ]
        label_ax_positions = [0.74, 0.30]  # in fig coordinates (top → bottom)
        for label_text, label_color, y_pos in zip(
            [r[0] for r in row_labels],
            [r[1] for r in row_labels],
            label_ax_positions,
        ):
            fig.text(
                0.5, y_pos, label_text,
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.5", fc=label_color, ec="none", alpha=0.90),
            )

        # Grid: 2 rows × top_k columns
        for row, (top_ids, method_name, border_color) in enumerate([
            (orig_top, "Original", COLORS["Original"]),
            (clip_top, "CLIP",     COLORS["CLIP"]),
        ]):
            for col, wid in enumerate(top_ids):
                ax = fig.add_subplot(2, top_k, row * top_k + col + 1)
                ax.axis("off")

                # Cadre coloré selon la méthode
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2.5)

                if wid is None:
                    ax.set_facecolor("#EEEEEE")
                    ax.text(0.5, 0.5, "No\nresults",
                            ha="center", va="center", fontsize=8,
                            color="#AAAAAA", transform=ax.transAxes)
                    ax.set_title(f"#{col+1}", fontsize=8, pad=3, color="#AAAAAA")
                    continue

                img_path = find_image(wid)
                img_arr  = _load_img_safe(img_path) if img_path else None

                if img_arr is not None:
                    ax.imshow(img_arr, aspect="auto")
                else:
                    ax.set_facecolor("#E8E8E8")
                    ax.text(0.5, 0.5, f"Image\nnot found\n(ID {wid})",
                            ha="center", va="center", fontsize=7,
                            color="#888888", transform=ax.transAxes)

                # Title: rank + ID + artwork title (truncated)
                item  = id_to_item.get(wid, {})
                title = (item.get("objectWork") or {}).get("titleText", "") or "—"
                short = (title[:22] + "…") if len(title) > 22 else title

                # Score
                if method_name == "Original" and orig_res["result"].get("fused_scores"):
                    score = orig_res["result"]["fused_scores"].get(wid, 0.0)
                    score_str = f"{score:.3f}"
                elif method_name == "CLIP" and clip_res["result"].get("all_scores"):
                    score = clip_res["result"]["all_scores"].get(wid, 0.0)
                    score_str = f"{score:.3f}"
                else:
                    score_str = "—"

                ax.set_title(
                    f"#{col+1}  ·  {short}\nScore: {score_str}",
                    fontsize=7.5, pad=4, linespacing=1.5,
                    color="#222222",
                )

        plt.subplots_adjust(hspace=0.50, wspace=0.08, top=0.88, bottom=0.05)
        _save_fig(fig, f"fig5{chr(96+q_idx)}_qualitative_q{q_idx}.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ─── MAIN ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Configuration ─────────────────────────────────────────────────────────
    JSON_PATH    = "fabritius_data_base/fabritius_export.json.xz"
    N_SAMPLES    = 1000
    N_RUNS_TRAIN = 3
    N_RUNS_INFER = 5
    WARMUP       = 1
    OUT_DIR      = "benchmark"
    os.makedirs(OUT_DIR, exist_ok=True)

    TEST_QUERIES = [
        "un portrait d'une femme avec un chien",
        "une scène de bataille historique",
        "un paysage avec des montagnes et un lac",
        "une nature morte avec des fruits",
    ]

    # ── Chargement du dataset ─────────────────────────────────────────────────
    print("\n[1/6] Chargement du dataset…")
    raw      = load_dataset(JSON_PATH)
    eligible = [item for item in raw if item["iconographicInterpretation"]]
    random.seed(42)
    dataset  = random.sample(eligible, min(N_SAMPLES, len(eligible)))
    print(f"  Dataset: {len(dataset)} items with description")

    # ── Training Original ─────────────────────────────────────────────────
    print(f"\n[2/6] TRAINING — Original Method ({N_RUNS_TRAIN} run(s))…")
    orig_train = benchmark_original_training(dataset, raw, n_runs=N_RUNS_TRAIN)
    print(f"  → avg. {orig_train['avg_time']:.2f}s ± {orig_train['std_time']:.2f}s  |  "
          f"ΔRSS {orig_train['avg_mem']:.0f} MB ± {orig_train['std_mem']:.0f} MB")

    # ── Training CLIP ─────────────────────────────────────────────────────
    clip_train = None
    clip_ctx   = None
    if CLIP_AVAILABLE:
        print(f"\n[3/6] TRAINING — CLIP ({N_RUNS_TRAIN} run(s))…")
        clip_train = benchmark_clip_training(dataset, n_runs=N_RUNS_TRAIN)
        clip_ctx   = clip_train["ctx"]
        print(f"  → avg. {clip_train['avg_time']:.2f}s ± {clip_train['std_time']:.2f}s  |  "
              f"ΔRSS {clip_train['avg_mem']:.0f} MB ± {clip_train['std_mem']:.0f} MB  "
              f"(model only: {clip_train['model_mem_mb']:.0f} MB)")
    else:
        print("\n[3/6] CLIP non disponible — ignoré.")

    # ── Inférence Original ────────────────────────────────────────────────────
    orig_infer_list = []
    print(f"\n[4/6] INFERENCE — Original Method "
          f"({N_RUNS_INFER} runs + {WARMUP} warm-up, Phase 2 excluded)…")
    for query in TEST_QUERIES:
        print(f"  Query: {query}")
        res = benchmark_original_inference(query, orig_train["ctx"],
                                           n_runs=N_RUNS_INFER, warmup=WARMUP)
        orig_infer_list.append(res)
        print(f"    → {res['avg_time']:.4f}s ± {res['std_time']:.4f}s")

    # ── Inference CLIP ────────────────────────────────────────────────────────
    clip_infer_list = []
    if CLIP_AVAILABLE and clip_ctx:
        print(f"\n[5/6] INFERENCE — CLIP "
              f"({N_RUNS_INFER} runs + {WARMUP} warm-up)…")
        for query in TEST_QUERIES:
            print(f"  Requête : {query}")
            res = benchmark_clip_inference(query, clip_ctx,
                                           n_runs=N_RUNS_INFER, warmup=WARMUP)
            clip_infer_list.append(res)
            print(f"    → {res['avg_time']:.4f}s ± {res['std_time']:.4f}s")
    else:
        clip_infer_list = [
            {"avg_time": 0.0, "std_time": 0.0, "avg_mem": 0.0, "std_mem": 0.0,
             "result": {"wid": None, "emotion": "autre", "score": 0.0,
                        "top_ids": [], "all_scores": {}}}
        ] * len(TEST_QUERIES)

    # ── Report & figures ─────────────────────────────────────────────────────
    print("\n[6/6] Generating report…")
    if clip_train:
        print_summary(orig_train, orig_infer_list,
                      clip_train, clip_infer_list, TEST_QUERIES)
        save_csv(orig_train, orig_infer_list,
                 clip_train, clip_infer_list, TEST_QUERIES, OUT_DIR)

        print("\n  Generating performance figures (fig1–4)…")
        plot_training_time  (orig_train, clip_train, OUT_DIR)
        plot_inference_time (orig_infer_list, clip_infer_list, TEST_QUERIES, OUT_DIR)
        plot_training_memory(orig_train, clip_train, OUT_DIR)
        plot_speedup        (orig_infer_list, clip_infer_list, TEST_QUERIES, OUT_DIR)

        print("\n  Generating qualitative figures (fig5a–d)…")
        plot_qualitative_comparison(
            TEST_QUERIES,
            orig_infer_list,
            clip_infer_list,
            orig_train["ctx"],
            OUT_DIR,
            top_k=5,
        )

        print(f"\n  All figures are in the '{OUT_DIR}/' folder.")
    else:
        print("  CLIP not available — partial report (Original only).")
