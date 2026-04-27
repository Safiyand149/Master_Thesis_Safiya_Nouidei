"""
compare_systems.py
──────────────────
Comparaison visuelle côte à côte :
  • Ligne du haut  → Baseline (keyword / Jaccard)
  • Ligne du bas   → Step1    (graphe + embeddings CamemBERT)

Les deux systèmes partagent exactement le même dataset (seed=42).
"""

import os, sys, time, random, re, json, lzma, glob, math, warnings
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
for _backend in ["Qt5Agg", "Qt6Agg", "GTK3Agg", "GTK4Agg", "wxAgg", "Agg"]:
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as _plt_test
        _plt_test.figure(); _plt_test.close()
        print(f"   Backend matplotlib : {_backend}")
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


JSON_PATH = "fabritius_data_base/fabritius_export.json.xz"
N_SAMPLES = 5000
TOP_K     = 5
ALPHA     = 0.5

sys.path.insert(0, str(Path(__file__).parent))

print("  Chargement de Step1 (graphe + embeddings)...")

import main as S1

# Baseline
STOPWORDS = S1.STOPWORDS

def _tokenize(text):
    return S1.caption_preprocessing(text)

def _filter_tokens(tokens, common_words=None):
    common = common_words or set()
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS and t not in common}

def _jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def _overlap(a, b):
    if not a: return 0.0
    return len(a & b) / len(a)

def _combined(a, b, alpha=ALPHA):
    return alpha * _jaccard(a, b) + (1 - alpha) * _overlap(a, b)

def _retrieve_baseline(query, token_sets, common_words, top_k=TOP_K):
    q_tok = _filter_tokens(_tokenize(query), common_words)
    if not q_tok:
        return []
    scores = {wid: _combined(q_tok, tok) for wid, tok in token_sets.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# Dataset loading 

print("  Chargement du dataset...")

raw = S1.load_dataset(JSON_PATH)

print("Pre-indexation des images (une seule passe disque)...")
t0 = time.time()
img_index = S1.build_image_index()
print(f"   {len(img_index)} fichiers indexes en {time.time()-t0:.1f}s")

def _wid_has_image(work_id):
    pat = re.compile(
        rf"-{re.escape(str(work_id))}[a-z]*[-.]",
    re.IGNORECASE
    )
    return any(pat.search(name) for name in img_index)

eligible_with    = [item for item in raw if item["iconographicInterpretation"]]
print("Filtrage des items sans description ayant une image...")
eligible_without = [
    item for item in raw
    if not item["iconographicInterpretation"] and _wid_has_image(item["workID"])
]
random.seed(42)
sample_with    = random.sample(eligible_with, min(N_SAMPLES, len(eligible_with)))
n_without      = min(len(eligible_without), max(1, N_SAMPLES // 5))
sample_without = random.sample(eligible_without, n_without) if eligible_without else []
dataset        = sample_with + sample_without

print(f"Dataset partage : {len(sample_with)} avec desc + {len(sample_without)} sans desc "
      f"= {len(dataset)} total")

id_to_item = {item["workID"]: item for item in dataset}

global_freq = Counter()
doc_freq    = Counter()
for item in raw:
    tokens = S1.caption_preprocessing(item["iconographicInterpretation"])
    global_freq.update(tokens)
    doc_freq.update(set(tokens))
n_docs = len(raw)
common = {w for w, c in global_freq.items() if c > 50}

# Index BASELINE

print("\n" + "-" * 65)
print("  [BASELINE] Construction de l'index token...")
t0 = time.time()
token_sets = {
    item["workID"]: _filter_tokens(_tokenize(item["iconographicInterpretation"]), common)
    for item in dataset
    if item["iconographicInterpretation"].strip()
}
print(f"  Index pret en {time.time()-t0:.2f}s | {len(token_sets)} oeuvres indexees")

#  Graphe STEP1
print("\n" + "-" * 65)
print("  [STEP1] Construction du graphe...")

cap_labels = {
    item["workID"]: [
        t for t in S1.caption_preprocessing(item["iconographicInterpretation"])
        if len(t) > 2 and t not in STOPWORDS and t not in common
    ]
    for item in dataset
    if item["iconographicInterpretation"].strip()
}
hier_labels = {
    item["workID"]: S1.extract_hierarchical_labels(
        item["subjectTerms"], item["iconographicTerms"], item["conceptualTerms"]
    )
    for item in dataset
}

t0    = time.time()
moods = S1.extract_captions_moods_batch(dataset)
print(f"  Moods extraits en {time.time()-t0:.1f}s")

image_feat_dict = {}
if S1.USE_IMAGE_FEATURES:
    print("  Extraction features visuelles (ResNet50)...")
    t0 = time.time()
    image_feat_dict = S1.extract_image_features_batch([it["workID"] for it in dataset])
    print(f"  Features : {len(image_feat_dict)}/{len(dataset)} images - {time.time()-t0:.1f}s")

trained = S1.train_model(cap_labels, hier_labels)
G = S1.construct_graph(
    cap_dict=cap_labels,
    hier_dict=hier_labels,
    mood_dict=moods,
    trained=trained,
    image_feat_dict=image_feat_dict if image_feat_dict else None,
)
print(f"  Graphe : {G.number_of_nodes()} noeuds, {G.number_of_edges()} aretes")

print("  Prechauffage index vectoriel...")
t0 = time.time()
S1._get_graph_index(G)
print(f"  Index pret en {time.time()-t0:.1f}s | cache {len(S1._EMBED_CACHE)} embeddings")


# Affichage
def _load_img(ax, work_id):
    path = S1.find_image(str(work_id))
    if path:
        try:
            ax.imshow(mpimg.imread(path))
            return
        except Exception:
            pass
    ax.set_facecolor("#e8e8e8")
    ax.text(0.5, 0.5, f"Image\nintrouvable\n(ID {work_id})",
            ha="center", va="center", transform=ax.transAxes,
            color="#888888", fontsize=8)


def _short(text, n):
    return (text[:n] + "...") if len(text) > n else text


def _rank_badge(rank):
    return {1: "1.", 2: "2.", 3: "3.", 4: "4.", 5: "5."}.get(rank, f"{rank}.")


def display_comparison(query, baseline_ranked, step1_ids, step1_scores,
                        t_baseline, t_step1):
    n = TOP_K
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 9))

    fig.suptitle(f'Requete : "{query}"', fontsize=13, fontweight="bold", y=1.005)

    baseline_ids_set = {wid for wid, _ in baseline_ranked[:n]}
    step1_top_set    = set(step1_ids[:n])

    for row, ranked in enumerate([
        [(wid, s) for wid, s in baseline_ranked[:n]],
        [(wid, step1_scores.get(wid, 0.0)) for wid in step1_ids[:n]],
    ]):
        for col, (wid, score) in enumerate(ranked):
            ax = axes[row, col]
            _load_img(ax, wid)
            ax.axis("off")

            item  = id_to_item.get(wid, {})
            title = item.get("objectWork", {}).get("titleText", "") or "-"
            mood  = moods.get(wid, ["-"])[0]
            badge = _rank_badge(col + 1)

            label = (
                f"{badge}  ID {wid}\n"
                f"{_short(title, 30)}\n"
                f"Score : {score:.4f}\n"
                f"Emotion : {mood}"
            )

            if wid in baseline_ids_set and wid in step1_top_set:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("#f0a500")
                    spine.set_linewidth(3)
                label += "\n[commun]"

            ax.set_title(label, fontsize=7.2, loc="center", pad=4,
                         linespacing=1.45, color="#111111")

    common_patch = mpatches.Patch(facecolor="#f0a500", label="ID present dans les deux resultats")
    fig.legend(handles=[common_patch], loc="lower center", ncol=1,
               fontsize=8, framealpha=0.85, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0.0, 0.02, 1.0, 1.0])
    plt.subplots_adjust(hspace=0.55, wspace=0.12, left=0.02, right=0.98)
    plt.show(block=True)


def print_comparison_summary(query, baseline_ranked, step1_ids, step1_scores,
                              t_baseline, t_step1):
    b_ids      = [wid for wid, _ in baseline_ranked[:TOP_K]]
    s_ids      = step1_ids[:TOP_K]
    common_ids = set(b_ids) & set(s_ids)

    t_b_ms = round(t_baseline * 1000)
    t_s_ms = round(t_step1 * 1000)

    print(f"  Requete : \"{query}\"")
    print(f"{'-'*65}")
    print(f"  {'Haut (' + str(t_b_ms) + ' ms)':40s}  Bas ({t_s_ms} ms)")

    for i in range(TOP_K):
        b_wid      = b_ids[i] if i < len(b_ids) else "-"
        s_wid      = s_ids[i] if i < len(s_ids) else "-"
        b_score    = dict(baseline_ranked).get(b_wid, 0)
        s_score    = step1_scores.get(s_wid, 0)
        match_flag = "[=]" if b_wid != "-" and b_wid == s_wid else "   "
        print(f"  {i+1}. {b_wid:>10s} ({b_score:.4f})  {match_flag}  {s_wid:>10s} ({s_score:.4f})")

    print(f"{'-'*65}")
    print(f"  IDs en commun (top {TOP_K}) : {len(common_ids)} / {TOP_K}")
    overlap_pct = len(common_ids) / TOP_K * 100
    bar = "#" * int(overlap_pct / 5) + "." * (20 - int(overlap_pct / 5))
    print(f"  Overlap : [{bar}] {overlap_pct:.0f} %")


# Input loop
print("  Les deux systemes sont prets.")
print("  Tapez une requete pour lancer la comparaison,")
print("  ou 'q' pour quitter.")


while True:
    query = input("\nRequete : ").strip()
    if query.lower() in ("q", "quit", "exit"):
        break
    if not query:
        continue

    # BASELINE
    t0              = time.time()
    baseline_ranked = _retrieve_baseline(query, token_sets, common, top_k=TOP_K)
    t_baseline      = time.time() - t0

    # STEP1
    t0           = time.time()
    plain_labels = S1.extract_caption_labels(query, global_freq=global_freq, common=common)
    mood_labels  = [f"mood_{m}" for m in S1.extract_captions_moods(query)]
    q_labels     = plain_labels + mood_labels

    # Expansion sémantique des labels
    graph_nodes = list(G.nodes())
    expanded_labels = S1.expand_query_semantically(q_labels, graph_nodes, top_k=3)
    q_labels = expanded_labels

    # Recherche dense
    dense_results = S1.dense_retrieval(query, dataset, top_k=50)

    # Récupération d'un large pool depuis le graphe
    top_ids_raw, graph_scores = S1._compute_similarity(
        G, q_labels,
        top_k=50,
        doc_freq=doc_freq,
        n_docs=n_docs,
    )

    # Fusion des scores graphe et dense
    all_scores = S1.fusion_scores(graph_scores, dense_results, alpha=0.5)

    # Trier par scores fusionnés
    sorted_fused = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    step1_ids = [wid for wid, _ in sorted_fused[:TOP_K]]
    t_step1   = time.time() - t0

    print_comparison_summary(query, baseline_ranked, step1_ids, all_scores,
                             t_baseline, t_step1)
    display_comparison(query, baseline_ranked, step1_ids, all_scores,
                       t_baseline, t_step1)