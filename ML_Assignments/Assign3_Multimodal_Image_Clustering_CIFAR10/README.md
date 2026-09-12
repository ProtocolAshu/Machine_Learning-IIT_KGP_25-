# Assignment 3 — Multimodal Image Clustering on CIFAR-10

**Course:** Machine Learning (IIT Kharagpur, 2025) · **Roll No:** 22CS30009

Unsupervised clustering of CIFAR-10 images combining **visual features** (CNN) and **textual features** (transformer language embeddings), with a formal written report.

---

## Pipeline

1. **Data** — Full CIFAR-10 (60,000 images, 10 classes) loaded via torchvision; class-name captions generated per image
2. **Visual features** — Pre-trained **ResNet18** (classification head removed), batched inference → 512-d embeddings
3. **Textual features** — **SentenceBERT** (`all-MiniLM-L6-v2`) caption encodings (CLIP `ViT-B/32` text encoder also implemented)
4. **Fusion** — L2-normalize both, then concatenate (weighted-average fusion with PCA projection also implemented)
5. **Clustering** — **K-Means** and **GMM** (10 components) on each feature set
6. **Evaluation** — Cluster→majority-label mapping, **Cohen Kappa Score** vs ground truth
7. **Visualization** — t-SNE (perplexity 30) plots of ground truth vs clusters

---

## Results (Cohen Kappa Score)

| Feature Set | K-Means | GMM |
|---|---|---|
| Visual (ResNet18) | **0.212** | 0.186 |
| Textual (SentenceBERT) | 1.000 | 1.000 |
| Fused (Visual + Textual) | 1.000 | 0.889 |

**Interpretation:**
- The **visual-only** result (~0.21 kappa) is the genuine unsupervised clustering difficulty of CIFAR-10 with ResNet18 features — unsupervised methods can't fully separate all 10 classes without label supervision.
- The textual/fused scores of 1.0 are an upper-bound artifact: captions were constructed *from the ground-truth labels* (`"This is a {class}."`), so text embeddings trivially encode the answer. The fused-GMM dip to 0.89 shows visual noise perturbing otherwise perfect text clusters.
- Takeaway: textual constraints dominate fusion here; for honest multimodal clustering, captions would need to come from an independent captioner, not the labels.

---

## Files

| File | Description |
|---|---|
| `22CS30009.ipynb` | Full solved notebook (pipeline + all results) |
| `22CS30009_Clustering_Report.pdf` | Written analysis report |
| `Visual_Features.png` | t-SNE — ground truth vs K-Means clusters (visual features) |
| `Textual_Features.png` | t-SNE — ground truth vs clusters (textual features) |
| `Fused_Features.png` | t-SNE — ground truth vs clusters (fused features) |
| `subset_indices.npy` | Indices of the 2,000-image subset used for t-SNE |
| `Assignment-3.pdf` | Original assignment statement |
