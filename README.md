# Graph-Based Retrieval of Artworks from Narrative Inputs for Memory-Triggering Applications

A retrieval system that returns artworks from a query written in French, based on a multimodal graph.


The system is compared to a lexical baseline and to CLIP.


The dataset of artworks and descriptions comes from **Fabritius / RMFAB** (Musées royaux des Beaux-Arts de Belgique).

## Structure

| File | Role |
|---|---|
| `main.py` | Core of the system: dataset loading, label/emotion extraction, visual features (ResNet50), graph construction, Phase 1 retrieval, Phase 2 emotion filter, interactive loop. |
| `hierarchy.py` | Construction and visualization of term hierarchies (`subjectTerms`, `iconographicTerms`, `conceptualTerms`) from the RMFAB syntax, with level computation and graph plotting. |
| `correction.py` | Semantic correction of a label hierarchy: reassigning a node to a better parent based on CamemBERT similarity, while avoiding cycles. |
| `compare.py` | Side-by-side visual comparison: lexical baseline (Jaccard/overlap) vs. the graph + embeddings system, on the same sample. |
| `benchmark.py` | Quantitative evaluation (time, memory, quality) of the system against **CLIP**. |
| `benchmark_plots.py` | Generation of the benchmark figures. |
| `visualization.py` | Generates 6 images illustrating the progressive construction of the graph and shortest-path reasoning (for master's thesis purposes). |
| `fabritius_extract/` | Extraction and selection of artworks from the JSON export (`fab_sel_workid_v1/v2`, `fab_dump_v2`). |
| `fabritius_data_base/` | Source data: `fabritius_export.json.xz`. |

---

## Prerequisites

* Python 3.9+
* A CUDA GPU is not mandatory but a plus (CPU fallback available)

### Dependencies

```bash
pip install torch torchvision numpy networkx matplotlib pillow sentence-transformers
pip install faiss-gpu   # or faiss-cpu
pip install psutil
pip install git+https://github.com/openai/CLIP.git
```

The embedding model is **`dangvantuan/sentence-camembert-large`** (downloaded automatically).

---

## Data

The expected dataset is:
```
fabritius_data_base/fabritius_export.json.xz
```
The images are searched under this path:

```python
IMAGE_ROOT = "/DATA/public/siamese/dataset_mrbab/art-foto"
```

The artworks are private and were accessed via a VPN and MobaXterm.

---

## Usage

### Interactive search

```bash
python main.py
```

```
Entrez une histoire (ou 'q') : un rassemblement familial joyeux, des visages, un souvenir d'enfance
```
It shows the 5 best artworks, then asks for a target emotion.

### Comparison

```bash
python compare.py
```

### Hierarchy

```bash
python hierarchy.py <json_input> <work_id>
python correction.py <json_input> <work_id>
```

### Benchmark and figures

```bash
python benchmark.py
python benchmark_plots.py
```

### Pedagogical figures

```bash
python visualization.py     # generates frame_0.png … frame_5.png
```

---

## License

The code in this repository is released under the MIT License (see the
[LICENSE](LICENSE) file for details).

## Third-party components and data

This project builds on pre-trained models and external libraries, each
governed by its own license: CLIP, ResNet50 (torchvision), 
sentence-camembert-large, and FAISS. Their terms apply to their respective
components and are not covered by the MIT License above.

The artwork data comes from the Royal Museums of Fine Arts of Belgium
(RMFAB) and is subject to the museum's own usage conditions. The MIT
License covers only the source code in this repository, not the underlying
dataset or images.

## Author

**Safiya Nouidei**
