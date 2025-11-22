#!/usr/bin/env bash
set -e

echo "Step 0: Prepare data"
python scripts/0_prepare_data.py

echo "Step 1: Choose k (elbow + hierarchical)"
python scripts/1_choose_k.py

echo "Step 2: Run initial KMeans clustering"
python scripts/2_run_clustering.py

echo "Step 3: Recursive subclustering refinement"
python scripts/3_recursive_refine.py

echo "Step 4: Subtype distance re-check"
python scripts/4_subtype_distance_check.py

echo "Step 5: Feature importance ranking"
python scripts/5_rank_genes.py

echo "Pipeline complete. Outputs are in results/"
