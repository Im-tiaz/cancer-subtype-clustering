from dataclasses import dataclass

@dataclass
class Config:
    random_state: int = 42
    k_min: int = 10
    k_max: int = 30
    k_start: int = 20  # supported by elbow + silhouette + hierarchical + stability
    purity_threshold: float = 0.95  # “typically ≥95%” criterion
    max_recursions: int = 6
    min_cluster_size: int = 30
    stability_repeats: int = 10
    linkage_method: str = "ward"
    distance_metric: str = "euclidean"
    outdir: str = "results"

