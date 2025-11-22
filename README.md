# Cancer Subtype Clustering – Machine Learning Pipeline

This repository contains all scripts and modules used to preprocess TCGA-style RNA-seq data, perform k-selection, run K-means clustering, execute recursive sub-clustering, evaluate subtype distances, and compute gene-importance rankings using multiple statistical models.

---

## Folder Structure
data/               # raw, processed, cleaned data (empty placeholders)  
src/                # pipeline modules  
scripts/            # executable scripts  
results/            # saved outputs (empty placeholders)  
notebooks/          # optional notebooks  

---

## How to Run

### 1. Create `conda` environment
conda create -n cancer_env python=3.9 -y  
conda activate cancer_env  
pip install -r requirements.txt  

### 2. Run preprocessing
python scripts/0_prepare_data.py \
  --input data/raw/tcga_subset.csv \
  --outdir data/clean

### 3. Run k-selection (Elbow + Hierarchical Dendrogram)
python scripts/1_choose_k.py \
  --input data/clean/expr_scaled.csv \
  --outdir results/k_selection \
  --kmax 30

### 4. Run main clustering
python scripts/2_run_clustering.py \
  --input data/clean/expr_scaled.csv \
  --k 20 \
  --outdir results/base_kmeans

### 5. Run recursive refinement
python scripts/3_recursive_refine.py \
  --input data/clean/expr_scaled.csv \
  --labels results/base_kmeans/labels_k20.csv \
  --outdir results/recursive \
  --purity_threshold 0.95

### 6. Run subtype-distance analysis
python scripts/4_subtype_distance_check.py \
  --input data/clean/expr_scaled.csv \
  --labels results/recursive/final_labels.csv \
  --outdir results/subtype_distance

### 7. Run gene importance ranking
python scripts/5_rank_genes.py \
  --input data/clean/expr_scaled.csv \
  --labels results/recursive/final_labels.csv \
  --outdir results/gene_rankings

---

## Data Availability

This repository does not include TCGA RNA-seq data due to GDC restrictions.  
Users must manually download TCGA data.

**Expected file:**
data/raw/tcga_subset.csv

Synthetic demo generator:
python scripts/make_fake_data.py \
  --out data/raw/fake_tcga_subset.csv

---

## Hardware Tested
- CPU: AMD Ryzen Threadripper 1900X  
- OS: Ubuntu 18.04  
- Python: 3.9  
- RAM: 32 GB  

---

## Notes
- Random seeds fixed across all scripts  
- Intermediate outputs automatically saved  
- Compatible with any similar high-dimensional gene-expression dataset  
- TCGA data cannot be redistributed  

