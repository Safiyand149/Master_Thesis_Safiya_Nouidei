import numpy as np
from sentence_transformers import SentenceTransformer, models

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        word_embedding = models.Transformer(
            "dangvantuan/sentence-camembert-large",
            max_seq_length=512,
        )
        pooling = models.Pooling(
            word_embedding.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        _MODEL = SentenceTransformer(modules=[word_embedding, pooling])
        _MODEL.max_seq_length = 512

        # Move the model to the GPU when one is available.
        try:
            import torch
            if torch.cuda.is_available():
                _MODEL = _MODEL.to("cuda")
        except Exception:
            pass

    return _MODEL


def semantic_similarity(a: str, b: str) -> float:
    """Return the normalized cosine similarity between two labels."""
    if not a or not b:
        return 0.0

    model = get_model()
    emb = model.encode(
        [a, b],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return float(np.dot(emb[0], emb[1]))


def _all_nodes(hierarchy):
    # Every node, whether it appears as a parent or only as a child.
    nodes = set(hierarchy.keys())
    for children in hierarchy.values():
        nodes.update(children)
    return nodes


def _normalize_hierarchy(hierarchy):
    # Deduplicate each child list and make sure every node has an entry.
    corrected = {}
    for parent, children in hierarchy.items():
        corrected[parent] = list(dict.fromkeys(children))
    for node in _all_nodes(hierarchy):
        corrected.setdefault(node, [])
    return corrected


def _build_parent_map(hierarchy):
    # Invert the hierarchy: for each child, list its parents.
    parent_map = {}
    for parent, children in hierarchy.items():
        for child in children:
            parent_map.setdefault(child, []).append(parent)
    return parent_map


def _collect_descendants(hierarchy, node, visited=None):
    # Gather every node reachable below `node` (used to avoid creating cycles).
    if visited is None:
        visited = set()
    for child in hierarchy.get(node, []):
        if child not in visited:
            visited.add(child)
            _collect_descendants(hierarchy, child, visited)
    return visited


def _node_embeddings(nodes):
    model = get_model()
    embeddings = model.encode(
        list(nodes),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return {node: emb for node, emb in zip(nodes, embeddings)}


def correct_hierarchy(hierarchy, similarity_threshold=0.45, min_improvement=0.10, max_iterations=3):
    """Correct a label hierarchy using semantic similarity.

    The algorithm reassigns a node only when an alternative parent is clearly
    better than the current one. Plain detachment is avoided so as not to break
    the explicit relationships already present in the source hierarchy.
    """
    hierarchy = _normalize_hierarchy(hierarchy)
    if not hierarchy:
        return {}

    # Precompute a full node-to-node similarity matrix.
    nodes = list(hierarchy.keys())
    embeddings = _node_embeddings(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    matrix = np.stack([embeddings[node] for node in nodes])
    similarity = matrix @ matrix.T

    def node_similarity(parent, child):
        return float(similarity[index[parent], index[child]])

    for _ in range(max_iterations):
        parent_map = _build_parent_map(hierarchy)
        descendants = {node: _collect_descendants(hierarchy, node) for node in nodes}

        best_move = None
        best_improvement = 0.0

        for child in nodes:
            if child not in parent_map:
                continue

            current_parents = parent_map[child]
            if not current_parents:
                continue

            current_scores = [node_similarity(parent, child) for parent in current_parents]
            best_current_score = max(current_scores)

            # Find the best possible parent, skipping the node itself and its
            # descendants to keep the hierarchy acyclic.
            best_parent = None
            best_score = -1.0
            for candidate in nodes:
                if candidate == child or candidate in descendants[child]:
                    continue
                score = node_similarity(candidate, child)
                if score > best_score:
                    best_score = score
                    best_parent = candidate

            if best_parent is None or best_parent in current_parents:
                continue

            # Only accept the move if it clears the threshold and improves enough.
            improvement = best_score - best_current_score
            if best_score >= similarity_threshold and improvement >= min_improvement:
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (child, current_parents, best_parent)

        # No worthwhile move this pass: the hierarchy has converged.
        if not best_move:
            break

        # Apply the single best move: detach from old parents, attach to the new one.
        child, current_parents, new_parent = best_move
        for parent in current_parents:
            if child in hierarchy[parent]:
                hierarchy[parent].remove(child)
        hierarchy.setdefault(new_parent, [])
        if child not in hierarchy[new_parent]:
            hierarchy[new_parent].append(child)
        hierarchy = _normalize_hierarchy(hierarchy)

    return hierarchy


def correct_work_hierarchy(json_input, work_id, similarity_threshold=0.45, parse_similarity_threshold=0.3):
    """Load a JSON artwork and correct its subjectMatter hierarchies."""
    try:
        from fabritius_extract.fab_sel_workid_v2 import process
        from hierarchy import creation_hierarchy
    except ImportError as exc:
        raise ImportError(
            "Impossible d'importer les modules nécessaires pour charger l'oeuvre JSON. "
            "Vérifiez que fabritius_extract et hierarchy sont accessibles."
        ) from exc

    result = process(json_input, work_id)
    subject_matter = result.get("subjectMatter", {})
    if not subject_matter:
        return {}, {}

    raw_hierarchies = creation_hierarchy(subject_matter, similarity_threshold=parse_similarity_threshold)
    corrected_hierarchies = {}
    for field, hierarchy in raw_hierarchies.items():
        corrected = correct_hierarchy(hierarchy, similarity_threshold=similarity_threshold)
        corrected_hierarchies[field] = corrected

    return raw_hierarchies, corrected_hierarchies


def _print_hierarchy(title, hierarchy):
    print(f"\n--- {title} ---")
    if not hierarchy:
        print("Aucune hiérarchie trouvée.")
        return
    for parent, children in hierarchy.items():
        print(f"{parent} -> {children}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python correction.py <json_input> <work_id>")
        sys.exit(1)

    json_input = sys.argv[1]
    work_id = sys.argv[2]

    print("Chargement et correction de la hiérarchie...")
    raw_hierarchies, corrected_hierarchies = correct_work_hierarchy(json_input, work_id)

    if not raw_hierarchies:
        print("Aucune hiérarchie valide trouvée pour cette œuvre.")
        sys.exit(0)

    for field in raw_hierarchies:
        _print_hierarchy(f"Original {field}", raw_hierarchies[field])
        _print_hierarchy(f"Corrigée {field}", corrected_hierarchies.get(field, {}))