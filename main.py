import os

# ── CONFIGURATION GPU — AVANT tout import de torch ────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import re
import json
import lzma
import math
import time
import random
import warnings
import numpy as np
import networkx as nx
import torch
from pathlib import Path
from collections import Counter

# ── Détection GPU ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    print(f"🚀 GPU détecté : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM disponible : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
    torch.set_num_threads(os.cpu_count())
    try:
        torch.set_num_interop_threads(os.cpu_count())
    except RuntimeError:
        pass
    print(f"⚠️  Pas de GPU — fallback CPU ({os.cpu_count()} threads)")

warnings.filterwarnings("ignore")

import glob
import subprocess

# ── IMAGE FEATURE EXTRACTION (CNN) ────────────────────────────────────────────
try:
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms
    from PIL import Image as PILImage
    USE_IMAGE_FEATURES = True
    print("🖼️  torchvision disponible — extraction de features visuelles activée")
except ImportError:
    USE_IMAGE_FEATURES = False
    print("   torchvision non installé — features visuelles désactivées")

# ── BACKEND MATPLOTLIB INTERACTIF ─────────────────────────────────────────────
import matplotlib
for _backend in ["Qt5Agg", "Qt6Agg", "GTK3Agg", "GTK4Agg", "wxAgg", "Agg"]:
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as _plt_test
        _plt_test.figure()
        _plt_test.close()
        print(f"   Backend matplotlib : {_backend}")
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ── FAISS ─────────────────────────────────────────────────────────────────────
try:
    import faiss
    USE_FAISS = True
    if DEVICE.type == "cuda":
        try:
            _faiss_res = faiss.StandardGpuResources()
            USE_FAISS_GPU = True
            print("⚡ FAISS-GPU disponible — recherche vectorielle sur GPU")
        except Exception:
            USE_FAISS_GPU = False
            print("⚡ FAISS disponible (CPU)")
    else:
        USE_FAISS_GPU = False
        print("⚡ FAISS disponible — recherche vectorielle accélérée")
except ImportError:
    USE_FAISS = False
    USE_FAISS_GPU = False
    print("   FAISS non installé — fallback numpy")

# ── STOPWORDS ─────────────────────────────────────────────────────────────────
STOPWORDS = {
    "le","la","les","de","des","un","une","et","en","du","dans","sur",
    "avec","pour","par","au","aux","ce","ces","cet","cette","qui","que",
    "quoi","dont","où","mais","ou","donc","or","ni","car","est","son",
    "ses","mon","mes","ton","tes","leur","leurs","plus","bien","tout",
    "très","aussi","même","comme","alors","après","avant","entre","vers",
    "sous","chez","sans","lors","peu","trop","assez","encore","déjà",
    "non","oui","pas","plus","rien","tout","ils","elles","nous","vous",
    "lui","eux","elle","il","je","tu","on","se","si","ne","y","en"
}

try:
    from fabritius_extract.fab_sel_workid_v1 import process as fab_process
    from hierarchy import creation_hierarchy, compute_levels
except ImportError:
    def creation_hierarchy(d): return {}
    def compute_levels(h, r): return {}

# ── CONSTANTES ────────────────────────────────────────────────────────────────
BASE_WEIGHT  = 0.5
DEPTH_BONUS  = 0.5

# Pool interne élargi pour garantir 5 résultats avec image en Phase 1
PHASE1_POOL_SIZE = 5   # candidats récupérés depuis le graphe
PHASE1_DISPLAY   = 5    # toujours affiché 5 en Phase 1

EMOTION_LABELS = [
    "amusement", "admiration", "satisfaction", "excitation",
    "colère", "dégout", "peur", "tristesse", "autre"
]

EMOTION_VALENCE = {
    "amusement"   :  1,
    "admiration"  :  1,
    "satisfaction":  1,
    "excitation"  :  1,
    "colère"      : -1,
    "dégout"      : -1,
    "peur"        : -1,
    "tristesse"   : -1,
    "autre"       :  0,
}

# ── SINGLETONS ────────────────────────────────────────────────────────────────
_MODEL            = None
_GRAPH_INDEX      = None
_GRAPH_INDEX_HASH = None
_EMBED_CACHE      = {}

# ── IMAGE FEATURE SINGLETONS ──────────────────────────────────────────────────
_IMAGE_FEAT_MODEL = None
_IMAGE_FEAT_CACHE = {}        # work_id -> np.array (feature vector L2-normalisé)
_IMAGE_FEAT_DIM   = 2048      # ResNet50 avgpool output dim
_IMAGE_TRANSFORM  = None


# ── INDEX D'IMAGES ────────────────────────────────────────────────────────────

IMAGE_ROOT  = "/DATA/public/siamese/dataset_mrbab/art-foto"
IMAGE_INDEX = None


def build_image_index():
    global IMAGE_INDEX
    if IMAGE_INDEX is None:
        print("Indexation des images...")
        pattern = os.path.join(IMAGE_ROOT, "**", "*.jpg")
        files = glob.glob(pattern, recursive=True)
        IMAGE_INDEX = {os.path.basename(f): f for f in files}
    return IMAGE_INDEX


def find_image(work_id: str):
    index = build_image_index()
    id_pattern = re.compile(
    rf"-{re.escape(str(work_id))}[a-z]*[-.]",
    re.IGNORECASE
    )
    for name, path in index.items():
        if id_pattern.search(name):
            return path
    return None

def _wid_has_image(work_id: str) -> bool:
    """Lookup rapide sur l'index déjà en mémoire."""
    img_index = build_image_index()
    pat = re.compile(
        rf"-{re.escape(str(work_id))}[a-z]*[-.]", re.IGNORECASE
    )
    return any(pat.search(name) for name in img_index)


def filter_ids_with_images(sorted_ids, all_scores, target_k):
    """Filtre et retourne exactement target_k IDs ayant une image."""
    valid_ids = []
    for wid in sorted_ids:
        if find_image(wid):
            valid_ids.append(wid)
        if len(valid_ids) >= target_k:
            break
    return valid_ids


# ── AFFICHAGE ─────────────────────────────────────────────────────────────────

def _load_img_or_placeholder(ax, work_id: str):
    """Charge l'image dans ax, ou affiche un placeholder gris si introuvable."""
    path = find_image(str(work_id))
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


def display_phase1_grid(top_ids: list, all_scores: dict, moods: dict,
                        dataset: list, query: str, n_display: int = 5):
    """Affiche les n_display meilleures œuvres de la Phase 1 en grille 1×N."""
    id_to_item = {item["workID"]: item for item in dataset}
    entries    = top_ids[:n_display]
    n          = len(entries)

    if n == 0:
        print("   ⚠️  Aucune image à afficher pour la Phase 1.")
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(f'Phase 1 — Requête : "{query}"',
                 fontsize=11, fontweight="bold", y=1.01)

    for ax, wid in zip(axes, entries):
        item  = id_to_item.get(wid, {})
        title = item.get("objectWork", {}).get("titleText", "") or "—"
        score = all_scores.get(wid, 0.0)
        mood  = moods.get(wid, ["?"])[0]

        _load_img_or_placeholder(ax, wid)
        ax.axis("off")

        short_title = (title[:28] + "…") if len(title) > 28 else title
        ax.set_title(
            f"ID {wid}\n{short_title}\nScore : {score:.3f}\nÉmotion : {mood}",
            fontsize=7.5, loc="center", pad=4, linespacing=1.5
        )

    plt.tight_layout()
    plt.show(block=True)


def display_phase2_single(wid: str, emotion: str, score: float,
                          dataset: list, query: str):
    """Affiche l'unique résultat de la Phase 2 dans une fenêtre dédiée."""
    id_to_item = {item["workID"]: item for item in dataset}
    item       = id_to_item.get(wid, {})
    title      = item.get("objectWork", {}).get("titleText", "") or "—"

    fig, ax = plt.subplots(1, 1, figsize=(6, 7))
    fig.suptitle(f'Phase 2 — Meilleur match émotionnel\nRequête : "{query}"',
                 fontsize=10, fontweight="bold", y=1.02)

    _load_img_or_placeholder(ax, wid)
    ax.axis("off")

    short_title = (title[:40] + "…") if len(title) > 40 else title
    ax.set_title(
        f"ID {wid}\n{short_title}\nÉmotion : {emotion}   |   Similarité : {score:.3f}",
        fontsize=9, loc="center", pad=6, linespacing=1.6
    )

    plt.tight_layout()
    plt.show(block=True)


# ── CNN FEATURES (ResNet50) ───────────────────────────────────────────────────

def _get_image_feat_transform():
    global _IMAGE_TRANSFORM
    if _IMAGE_TRANSFORM is None and USE_IMAGE_FEATURES:
        _IMAGE_TRANSFORM = tv_transforms.Compose([
            tv_transforms.Resize(256),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])
    return _IMAGE_TRANSFORM


def get_image_feat_model():
    """Charge ResNet50 pré-entraîné (tronqué avant FC) en singleton."""
    global _IMAGE_FEAT_MODEL
    if _IMAGE_FEAT_MODEL is None and USE_IMAGE_FEATURES:
        base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        _IMAGE_FEAT_MODEL = torch.nn.Sequential(*list(base.children())[:-1])
        _IMAGE_FEAT_MODEL.eval()
        _IMAGE_FEAT_MODEL = _IMAGE_FEAT_MODEL.to(DEVICE)
        print(f"   ResNet50 chargé pour features visuelles (dim={_IMAGE_FEAT_DIM})")
    return _IMAGE_FEAT_MODEL


def _flush_image_batch(model, wids, tensors, out_dict):
    """Helper : exécute un batch CNN et stocke les features L2-normalisées."""
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        feats = model(batch).squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float32)
    for wid, feat in zip(wids, feats):
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        _IMAGE_FEAT_CACHE[wid] = feat
        out_dict[wid] = feat


def extract_image_features_batch(work_ids: list, batch_size: int = 32) -> dict:
    """
    Extrait les features visuelles pour une liste de work_ids en batches.
    Retourne un dict {work_id: np.array} pour les IDs ayant une image.
    """
    if not USE_IMAGE_FEATURES:
        return {}

    transform  = _get_image_feat_transform()
    model      = get_image_feat_model()
    results    = {}
    to_process = [wid for wid in work_ids if wid not in _IMAGE_FEAT_CACHE]

    # IDs déjà en cache
    for wid in work_ids:
        if wid in _IMAGE_FEAT_CACHE:
            results[wid] = _IMAGE_FEAT_CACHE[wid]

    # Traitement par batch
    batch_wids, batch_tensors = [], []
    for wid in to_process:
        path = find_image(str(wid))
        if path is None:
            continue
        try:
            img    = PILImage.open(path).convert("RGB")
            tensor = transform(img)
            batch_wids.append(wid)
            batch_tensors.append(tensor)
        except Exception:
            continue

        if len(batch_tensors) >= batch_size:
            _flush_image_batch(model, batch_wids, batch_tensors, results)
            batch_wids, batch_tensors = [], []

    if batch_tensors:
        _flush_image_batch(model, batch_wids, batch_tensors, results)

    return results


# ── MODÈLE DE LANGUE ──────────────────────────────────────────────────────────

def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer, models
        word_embedding = models.Transformer(
            "dangvantuan/sentence-camembert-large",
            max_seq_length=512
        )
        pooling = models.Pooling(
            word_embedding.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        _MODEL = SentenceTransformer(modules=[word_embedding, pooling])
        _MODEL.max_seq_length = 512

        if DEVICE.type == "cuda":
            _MODEL = _MODEL.to(DEVICE)
            print(f"   Modèle chargé sur GPU : {torch.cuda.get_device_name(0)} (max_seq_length=512)")
        else:
            print("   Modèle chargé sur CPU (max_seq_length=512)")
    return _MODEL


def _embed(texts):
    if not texts:
        return np.empty((0, 1024), dtype=np.float32)

    model     = get_model()
    new_texts = [t for t in texts if t not in _EMBED_CACHE]

    if new_texts:
        batch_size = 128 if DEVICE.type == "cuda" else 16
        embs = model.encode(
            new_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=DEVICE,
        )
        for t, e in zip(new_texts, embs):
            _EMBED_CACHE[t] = e

    return np.vstack([_EMBED_CACHE[t] for t in texts])


# ── VALENCE ÉMOTIONNELLE ──────────────────────────────────────────────────────

def _get_valence(emotion_text: str) -> int:
    low = emotion_text.lower().strip()
    for key, val in EMOTION_VALENCE.items():
        if low == key:
            return val
    known = list(EMOTION_VALENCE.keys())
    embs  = _embed([emotion_text] + known)
    sims  = embs[0] @ embs[1:].T
    best  = known[int(np.argmax(sims))]
    return EMOTION_VALENCE[best]


# ── CHARGEMENT DATASET ────────────────────────────────────────────────────────

def load_dataset(json_path):
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {json_path}")

    opener = lzma.open(path, "rt", encoding="utf-8") if str(path).endswith(".xz") \
             else open(path, "r", encoding="utf-8")
    with opener as f:
        raw = json.load(f)

    dataset = []
    for rec in raw:
        work_obj = rec.get("objectWork", {})
        subject  = rec.get("subjectMatter", {})
        wid = str(work_obj.get("workID", ""))
        if "," in wid or "/" in wid or wid.count("-") > 1 or not wid:
            continue

        interp = rec.get("iconographicInterpretation") or subject.get("iconographicInterpretation", "")
        if isinstance(interp, list):
            interp = " ".join(str(x) for x in interp if x is not None)

        dataset.append({
            "workID"                    : wid,
            "iconographicInterpretation": str(interp or "").strip(),
            "subjectTerms"              : subject.get("subjectTerms", ""),
            "iconographicTerms"         : subject.get("iconographicTerms", ""),
            "conceptualTerms"           : subject.get("conceptualTerms", ""),
            "subjectMatter"             : subject,
            "objectWork"                : work_obj,
        })

    print(f"Dataset chargé : {len(dataset)} entrées")
    return dataset


# ── PRÉ-TRAITEMENT ────────────────────────────────────────────────────────────

def caption_preprocessing(caption):
    if not caption:
        return []
    text = re.sub(r"[^\w\sàâäéèêëïîôöùûüç]", " ", caption.lower())
    return text.split()


def extract_caption_labels(caption, global_freq=None, doc_freq=None, n_docs=1):
    """
    Extraction des labels de la requête pour gem.py.

    Amélioration vs baseline (filtre dur fréquence > 50) :
    - Les tokens sont pondérés par IDF ; seuls ceux dont l'IDF est trop faible
      (< 0.3) sont rejetés, ce qui conserve les termes de domaine courants mais
      informatifs (ex. "paysage", "portrait") que la baseline supprime.
    - Les tokens très courts (≤ 2 chars) et stopwords sont toujours exclus.
    """
    tokens = caption_preprocessing(caption)
    seen, result = set(), []
    n = max(n_docs, 1)

    for t in tokens:
        if len(t) <= 2 or t in STOPWORDS or t in seen:
            continue
        seen.add(t)

        # Calcul IDF : si doc_freq fourni, on l'utilise ; sinon on se base sur global_freq
        if doc_freq and n > 1:
            idf = math.log(n / (doc_freq.get(t, 1) + 1)) + 1.0
        elif global_freq:
            # Approximation : fréquence globale normalisée comme proxy
            freq = global_freq.get(t, 1)
            idf = math.log(max(n, 50) / (freq + 1)) + 1.0
        else:
            idf = 1.0

        # Seuil IDF très bas : on rejette seulement les tokens vraiment omniprésents
        if idf >= 0.3:
            result.append(t)

    return result


def extract_captions_moods_batch(dataset, chunk_size=200):
    """
    Extrait les moods par batch.
    - Items AVEC description : inférence textuelle via embeddings.
    - Items SANS description  : fallback "autre" (affiné plus tard via features visuelles si dispo).
    """
    if DEVICE.type == "cuda":
        chunk_size = 200
    else:
        chunk_size = 50

    emo_embs = _embed(EMOTION_LABELS)
    result   = {}

    items_with_cap    = [item for item in dataset if item["iconographicInterpretation"].strip()]
    items_without_cap = [item for item in dataset if not item["iconographicInterpretation"].strip()]

    captions = [item["iconographicInterpretation"] for item in items_with_cap]
    for i in range(0, len(captions), chunk_size):
        chunk_items = items_with_cap[i:i + chunk_size]
        chunk_caps  = captions[i:i + chunk_size]
        cap_embs    = _embed(chunk_caps)
        cos_sims    = cap_embs @ emo_embs.T
        for item, sims in zip(chunk_items, cos_sims):
            result[item["workID"]] = [EMOTION_LABELS[int(np.argmax(sims))]]

    for item in items_without_cap:
        result[item["workID"]] = ["autre"]

    return result


def extract_captions_moods(caption):
    if not caption:
        return ["autre"]
    embeddings = _embed([caption] + EMOTION_LABELS)
    sims = embeddings[0] @ embeddings[1:].T
    return [EMOTION_LABELS[int(np.argmax(sims))]]


def _normalize_term_field(value):
    return "; ".join(map(str, value)) if isinstance(value, list) else str(value or "")


def extract_hierarchical_labels(st, it, ct):
    combined = {}
    for key, val in [("subjectTerms", st), ("iconographicTerms", it), ("conceptualTerms", ct)]:
        s = _normalize_term_field(val).strip()
        if s:
            hier = creation_hierarchy({key: s})
            if hier:
                root = next(iter(hier))
                for node, d in compute_levels(hier, root).items():
                    if node:
                        combined[node] = max(combined.get(node, 0), d)
    return combined


# ── ENTRAÎNEMENT ──────────────────────────────────────────────────────────────

def train_model(cap_dict, hier_dict):
    counts = Counter()
    _p = lambda a, b: tuple(sorted((str(a), str(b))))

    for labels in cap_dict.values():
        for i, l1 in enumerate(labels):
            for l2 in labels[i + 1:]:
                counts[_p(l1, l2)] += 1

    for d_map in hier_dict.values():
        nodes = list(d_map.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                w = (1 + math.log1p(d_map[nodes[i]])) * (1 + math.log1p(d_map[nodes[j]]))
                counts[_p(nodes[i], nodes[j])] += w

    for wid in set(cap_dict) & set(hier_dict):
        for cl in cap_dict[wid]:
            for hl, d in hier_dict[wid].items():
                counts[_p(cl, hl)] += (1 + math.log1p(d))

    max_c = max(counts.values()) if counts else 1
    return {pair: c / max_c for pair, c in counts.items()}


# ── CONSTRUCTION DU GRAPHE ────────────────────────────────────────────────────

def construct_graph(cap_dict, hier_dict, mood_dict, trained, image_feat_dict=None):
    global _GRAPH_INDEX, _GRAPH_INDEX_HASH
    _GRAPH_INDEX      = None
    _GRAPH_INDEX_HASH = None

    G = nx.Graph()

    # Tous les wids connus (avec ou sans description)
    all_wids = set(cap_dict) | set(hier_dict) | set(mood_dict)
    for wid in all_wids:
        G.add_node(wid, type="image", size=600, depth=0)

    for wid, labels in cap_dict.items():
        c_node = f"caption_{wid}"
        G.add_node(c_node, type="caption", size=400, depth=0)
        G.add_edge(wid, c_node, weight=1.0)
        for lbl in labels:
            if not G.has_node(lbl):
                G.add_node(lbl, type="caption_label", size=220, depth=0)
            G.add_edge(c_node, lbl, weight=0.8)

    for wid, d_map in hier_dict.items():
        for lbl, d in d_map.items():
            w = BASE_WEIGHT + DEPTH_BONUS * math.log1p(d)
            if not G.has_node(lbl):
                G.add_node(lbl, type="hier_label", size=200 + 60 * d, depth=d)
            else:
                if d > G.nodes[lbl].get("depth", 0):
                    G.nodes[lbl].update({"depth": d, "size": 200 + 60 * d})
                G.nodes[lbl]["type"] = "hier_label"
            G.add_edge(wid, lbl, weight=w)

    for wid, moods in mood_dict.items():
        for m in moods:
            m_n = f"mood_{m}"
            if not G.has_node(m_n):
                G.add_node(m_n, type="mood", size=200, depth=0)
            G.add_edge(wid, m_n, weight=0.8)   # 0.6 → 0.8 : l'émotion est un signal fort

    for (l1, l2), weight in trained.items():
        if G.has_node(l1) and G.has_node(l2):
            # Racine carrée : rendements décroissants mais minimum garanti à 0.1
            boosted  = max(math.sqrt(weight), 0.1)
            existing = G.get_edge_data(l1, l2, {}).get("weight", 0)
            G.add_edge(l1, l2, weight=max(existing, boosted))

    # ── Liens visuels CNN pour les images SANS description ────────────────────
    if image_feat_dict:
        no_desc_wids = [
            wid for wid in all_wids
            if not G.has_node(f"caption_{wid}") and wid in image_feat_dict
        ]
        anchored_wids = [
            wid for wid in all_wids
            if G.has_node(f"caption_{wid}") and wid in image_feat_dict
        ]

        if no_desc_wids and anchored_wids:
            nd_feats  = np.vstack([image_feat_dict[w] for w in no_desc_wids])
            anc_feats = np.vstack([image_feat_dict[w] for w in anchored_wids])
            sim_mat   = nd_feats @ anc_feats.T

            for i, wid in enumerate(no_desc_wids):
                row  = sim_mat[i]
                best = np.where(row >= 0.5)[0]
                best_sorted = sorted(best, key=lambda j: row[j], reverse=True)[:3]
                for j in best_sorted:
                    sim_val = float(row[j])
                    G.add_edge(wid, anchored_wids[j],
                               weight=sim_val * 0.7,
                               edge_type="visual_similarity")
                    G.nodes[wid]["has_visual_link"] = True

        # Stocke le vecteur de feature visuelle sur le nœud image
        for wid, feat in image_feat_dict.items():
            if G.has_node(wid):
                G.nodes[wid]["image_feat"] = feat

    for u, v, d in G.edges(data=True):
        d["distance"] = 1.0 / max(d.get("weight", 0.5), 1e-6)

    return G


# ── INDEX VECTORIEL ───────────────────────────────────────────────────────────

def _get_graph_index(G):
    global _GRAPH_INDEX, _GRAPH_INDEX_HASH

    current_hash = id(G)
    if _GRAPH_INDEX is not None and _GRAPH_INDEX_HASH == current_hash:
        return _GRAPH_INDEX

    nodes   = [
        n for n, d in G.nodes(data=True)
        if d.get("type") in {"caption_label", "hier_label", "mood"}
    ]
    vectors = _embed(nodes).astype("float32")

    if USE_FAISS and USE_FAISS_GPU:
        cpu_index = faiss.IndexFlatIP(vectors.shape[1])
        gpu_index = faiss.index_cpu_to_gpu(_faiss_res, 0, cpu_index)
        gpu_index.add(vectors)
        _GRAPH_INDEX = (nodes, gpu_index)
        print(f"   Index FAISS-GPU : {len(nodes)} vecteurs")
    elif USE_FAISS:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        _GRAPH_INDEX = (nodes, index)
    else:
        _GRAPH_INDEX = (nodes, vectors)

    _GRAPH_INDEX_HASH = current_hash
    return _GRAPH_INDEX


def _map_query_to_graph(G, query_labels, threshold=0.60):
    """
    Mappe les labels de requête sur les nœuds du graphe par similarité cosinus
    (CamemBERT). Améliorations vs baseline :
      - Seuil abaissé à 0.60 (vs 0.75) pour couvrir les requêtes paraphrasées
      - Fallback top-1 garanti : si aucun voisin ne dépasse le seuil, on prend
        le nœud le plus proche avec un poids réduit (évite les requêtes silencieuses)
      - Top-5 mappings par label (vs 2) pour une couverture sémantique plus large
    """
    nodes, index = _get_graph_index(G)
    mapped = {}

    text_labels = [ql for ql in query_labels if not ql.startswith("mood_")]
    mood_labels  = [ql for ql in query_labels if ql.startswith("mood_")]

    for ql in mood_labels:
        if G.has_node(ql):
            mapped[ql] = [(ql, 1.0)]

    if not text_labels:
        return mapped

    q_embeddings = _embed(text_labels).astype("float32")

    if USE_FAISS:
        sims, idx = index.search(q_embeddings, 5)
        for i, ql in enumerate(text_labels):
            if G.has_node(ql):
                mapped[ql] = [(ql, 1.0)]
                continue
            pairs = [
                (nodes[j], float(s))
                for j, s in zip(idx[i], sims[i])
                if j >= 0 and s >= threshold
            ]
            # Fallback garanti : si rien ne dépasse le seuil, on prend le meilleur
            if not pairs and idx[i][0] >= 0:
                pairs = [(nodes[idx[i][0]], float(sims[i][0]) * 0.6)]
            if pairs:
                mapped[ql] = pairs[:5]
    else:
        sim_matrix = q_embeddings @ index.T
        for i, ql in enumerate(text_labels):
            if G.has_node(ql):
                mapped[ql] = [(ql, 1.0)]
                continue
            best = np.where(sim_matrix[i] >= threshold)[0]
            sims_sorted = sorted(
                [(nodes[j], float(sim_matrix[i][j])) for j in best],
                key=lambda x: x[1], reverse=True
            )
            # Fallback garanti
            if not sims_sorted:
                best_j = int(np.argmax(sim_matrix[i]))
                sims_sorted = [(nodes[best_j], float(sim_matrix[i][best_j]) * 0.6)]
            if sims_sorted:
                mapped[ql] = sims_sorted[:5]

    return mapped


# ── PHASE 1 : RÉCUPÉRATION SÉMANTIQUE ────────────────────────────────────────

def _compute_similarity(G, query_labels, top_k=PHASE1_POOL_SIZE, doc_freq=None, n_docs=1):
    """
    Récupération sémantique Phase 1.  Améliorations vs baseline :

    1. IDF réel pour les mood labels (basé sur doc_freq) plutôt qu'une
       constante 0.5 — les émotions rares sont plus discriminantes.
    2. Bonus de type de nœud : les nœuds caption_label (texte libre) reçoivent
       un boost ×1.2, les hier_label un boost proportionnel à leur profondeur
       (terme plus spécifique = plus précis), les mood nodes ×1.0.
    3. Cutoff de propagation étendu à 4.0 (vs 3.0) pour atteindre les images
       plus périphériques qui partagent des concepts proches mais indirects.
    4. Normalisation par sqrt(degree) au lieu de log(degree) : récompense les
       images bien connectées (riches en labels) sans les écraser.
    5. Score de couverture multi-labels : bonus additionnel si plusieurs labels
       de la requête convergent vers la même image (signal de pertinence fort).
    """
    image_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "image"]
    mapped      = _map_query_to_graph(G, query_labels)

    # ── IDF unifié : même formule pour tous les labels (plus de constante 0.5) ─
    if doc_freq and n_docs > 1:
        idf = {
            ql: math.log(n_docs / (doc_freq.get(ql, 1) + 1)) + 1.0
            for ql in query_labels
        }
    else:
        idf = {ql: 1.0 for ql in query_labels}

    # ── Bonus par type de nœud graphe ─────────────────────────────────────────
    NODE_TYPE_BOOST = {
        "caption_label": 1.2,
        "hier_label"   : 1.1,   # modulé aussi par depth ci-dessous
        "mood"         : 1.0,
    }

    sources = [
        (ql, gn, cosim)
        for ql, matches in mapped.items()
        for gn, cosim in matches
    ]

    scores      = {img: 0.0 for img in image_nodes}
    label_hits  = {img: set() for img in image_nodes}   # pour bonus couverture

    for ql, gn, cosim in sources:
        gn_data    = G.nodes[gn]
        gn_type    = gn_data.get("type", "caption_label")
        depth      = gn_data.get("depth", 0)
        type_boost = NODE_TYPE_BOOST.get(gn_type, 1.0)
        if gn_type == "hier_label":
            type_boost *= (1.0 + 0.1 * depth)   # termes plus spécifiques = bonus

        dists = nx.single_source_dijkstra_path_length(G, gn, cutoff=4.0, weight="distance")
        wf    = idf.get(ql, 1.0) * cosim * type_boost

        for target, d in dists.items():
            if G.nodes[target].get("type") == "image":
                scores[target]     += wf / (1.0 + d)
                label_hits[target].add(ql)

    # ── Bonus de couverture : plusieurs labels qui pointent vers la même image ─
    for img in scores:
        n_hits = len(label_hits[img])
        if n_hits > 1:
            scores[img] *= (1.0 + 0.15 * math.log1p(n_hits - 1))

    # ── Normalisation douce : sqrt(degree) favorise les hubs sans les écraser ─
    for img in scores:
        deg = G.degree(img, weight="weight")
        scores[img] /= (1.0 + math.sqrt(max(deg, 1e-6)))

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [img for img, _ in top], scores


# ── PHASE 2 : FILTRE PAR ÉMOTION ─────────────────────────────────────────────

def phase2_emotion_filter(top_ids, desired_emotion, mood_dict):
    OPPOSITE_VALENCE_PENALTY = 0.05

    desired_valence = _get_valence(desired_emotion)
    candidates      = [(wid, mood_dict.get(wid, ["autre"])[0]) for wid in top_ids]

    embeddings = _embed([desired_emotion] + [e for _, e in candidates])
    raw_scores = embeddings[1:] @ embeddings[0]

    adjusted_scores = []
    for i, (wid, emotion) in enumerate(candidates):
        candidate_valence = _get_valence(emotion)
        score = float(raw_scores[i])

        if desired_valence != 0 and candidate_valence != 0:
            if desired_valence != candidate_valence:
                score *= OPPOSITE_VALENCE_PENALTY

        adjusted_scores.append(score)

    best_idx = int(np.argmax(adjusted_scores))
    return candidates[best_idx][0], candidates[best_idx][1], float(adjusted_scores[best_idx])


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    JSON_PATH = "fabritius_data_base/fabritius_export.json.xz"
    N_SAMPLES = 1000

    raw = load_dataset(JSON_PATH)

    # ── Pré-construction de l'index images UNE SEULE FOIS ─────────────────────
    print("Pré-indexation des images (une seule passe disque)...")
    t0 = time.time()
    img_index = build_image_index()
    print(f"   {len(img_index)} fichiers indexés en {time.time()-t0:.1f}s")

    # ── Sélection du dataset : items AVEC description (éligibles texte)
    eligible_with = [item for item in raw if item["iconographicInterpretation"]]
    # ── Items SANS description mais ayant une image (enrichit le graphe via CNN)
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

    print(f"Dataset : {len(sample_with)} avec description + {len(sample_without)} sans description "
          f"(image uniquement) = {len(dataset)} total")

    global_freq = Counter()
    doc_freq    = Counter()

    for item in raw:
        tokens = caption_preprocessing(item["iconographicInterpretation"])
        global_freq.update(tokens)
        doc_freq.update(set(tokens))

    n_docs = len(raw)
    common = {w for w, c in global_freq.items() if c > 50}

    # cap_labels : seulement les items avec description
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

    print("Extraction des moods (chunks + cache)...")
    t0    = time.time()
    moods = extract_captions_moods_batch(dataset)
    print(f"   Moods extraits en {time.time() - t0:.1f}s")

    # ── Extraction des features visuelles (ResNet50) ───────────────────────────
    image_feat_dict = {}
    if USE_IMAGE_FEATURES:
        print("Extraction des features visuelles (ResNet50)...")
        t0 = time.time()
        all_wids = [item["workID"] for item in dataset]
        image_feat_dict = extract_image_features_batch(all_wids)
        print(f"   Features extraites pour {len(image_feat_dict)}/{len(all_wids)} images "
              f"en {time.time() - t0:.1f}s")

    trained = train_model(cap_labels, hier_labels)
    G       = construct_graph(
        cap_dict=cap_labels,
        hier_dict=hier_labels,
        mood_dict=moods,
        trained=trained,
        image_feat_dict=image_feat_dict if USE_IMAGE_FEATURES else None,
    )
    print(f"Graphe : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

    # Statistiques liens visuels
    if sample_without:
        visual_links = sum(
            1 for wid in sample_without
            if G.has_node(wid) and G.nodes[wid].get("has_visual_link")
        )
        print(f"   Images sans description liées via similarité visuelle : "
              f"{visual_links}/{len(sample_without)}")

    print("Préchauffage de l'index vectoriel...")
    t0 = time.time()
    _get_graph_index(G)
    print(f"   Index prêt en {time.time() - t0:.1f}s")
    print(f"   Cache embeddings : {len(_EMBED_CACHE)} entrées")

    print(f"\nÉmotions disponibles : {', '.join(EMOTION_LABELS)}")

    while True:
        query = input("\nEntrez une histoire (ou 'q') : ").strip()
        if query.lower() in ("q", "quit"):
            break

        t0 = time.time()

        plain_labels = extract_caption_labels(query, global_freq=global_freq)
        mood_labels  = [f"mood_{m}" for m in extract_captions_moods(query)]
        q_labels     = plain_labels + mood_labels

        # Récupération d'un large pool depuis le graphe
        top_ids_raw, all_scores = _compute_similarity(
            G, q_labels,
            top_k=PHASE1_POOL_SIZE,   # pool large (30)
            doc_freq=doc_freq,
            n_docs=n_docs,
        )

        # Filtre pour garder exactement PHASE1_DISPLAY IDs avec image
        top_ids = filter_ids_with_images(
            top_ids_raw,
            all_scores,
            target_k=PHASE1_DISPLAY,  # 5
        )

        elapsed = time.time() - t0
        print(f"\n── Phase 1 ({elapsed:.2f}s) — top {PHASE1_DISPLAY} résultats avec image ──")
        for i, wid in enumerate(top_ids[:PHASE1_DISPLAY], 1):
            mood  = moods.get(wid, ["?"])[0]
            score = all_scores[wid]
            print(f"  {i}. ID: {wid} | Score: {score:.3f} | Émotion: {mood}")
            item = next((it for it in dataset if it["workID"] == wid), None)
            if item:
                title = item.get("objectWork", {}).get("titleText")
                if title:
                    print(f"     Titre : {title}")

        # ── Grille Phase 1 : toujours 5 images, bloquante ─────────────────────
        display_phase1_grid(top_ids, all_scores, moods, dataset, query,
                            n_display=PHASE1_DISPLAY)

        # ── Phase 2 : filtre émotionnel ────────────────────────────────────────
        desired = input("\nÉmotion souhaitée (Entrée pour ignorer) : ").strip()
        if not desired:
            continue

        best_wid, best_emotion, best_score = phase2_emotion_filter(top_ids, desired, moods)
        print(f"\n── Phase 2 ──")
        print(f"  Meilleur match : ID {best_wid} | Émotion : {best_emotion} | Similarité : {best_score:.3f}")
        item = next((it for it in dataset if it["workID"] == best_wid), None)
        if item:
            best_title = item.get("objectWork", {}).get("titleText", "")
            if best_title:
                print(f"  Titre : {best_title}")

        # ── Image unique Phase 2 : 1 image, bloquante ─────────────────────────
        display_phase2_single(best_wid, best_emotion, best_score, dataset, query)