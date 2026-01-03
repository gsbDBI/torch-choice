"""Generate synthetic datasets for the benchmarking pipeline (self-contained)."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import torch_choice
from torch_choice.data import ChoiceDataset

# Use a non-interactive backend so the script is safe in headless runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SECTION = "=" * 80

# Default dimensions used by the notebook.
NUM_USERS = 500
NUM_ITEMS = 500
LATENT_DIM = 30
NUM_USER_CLUSTERS = 5
NUM_ITEM_CLUSTERS = 5

# Experiment roster mirrors the notebook.
ALL_EXPERIMENTS = {
    "num_records_experiment_small",
    "full_dataset",
    "num_params_experiment_small",
    "num_items_experiment_small",
    "num_items_experiment_large",
}


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for benchmark replication.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./synthetic_data"),
        help="Directory to write generated datasets.",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=3_000_000,
        help="Number of synthetic choice records to simulate.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip t-SNE visualizations (faster, headless-safe).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        help="Subset of experiments to generate; use 'all' to run everything.",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        help="Optional path to reference datasets for checksum comparison.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate a minimal set of datasets (single dataset/value per experiment) for quick verification.",
    )
    return parser


def log_versions() -> None:
    print(SECTION)
    print("Environment")
    print(SECTION)
    print(f"Python version : {sys.version}")
    print(f"Torch version  : {torch.__version__}")
    print(f"Torch-Choice   : {torch_choice.__version__}")
    if torch.cuda.is_available():
        print(f"GPU available  : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU available  : None detected")


def generate_clustered_latents(
    num_entities: int,
    latent_dim: int,
    num_clusters: int,
    cluster_std: float = 0.5,
    between_cluster_distance: float = 3.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate latent vectors organized in clusters."""
    rng = np.random.default_rng(random_state)
    cluster_centers = rng.standard_normal(size=(num_clusters, latent_dim)) * between_cluster_distance

    entities_per_cluster = np.full(num_clusters, num_entities // num_clusters)
    remainder = num_entities % num_clusters
    if remainder > 0:
        extra_indices = rng.choice(num_clusters, remainder, replace=False)
        entities_per_cluster[extra_indices] += 1

    latents: list[np.ndarray] = []
    cluster_assignments: list[int] = []
    for cluster_idx in range(num_clusters):
        cluster_latents = rng.normal(
            loc=cluster_centers[cluster_idx],
            scale=cluster_std,
            size=(entities_per_cluster[cluster_idx], latent_dim),
        )
        latents.append(cluster_latents)
        cluster_assignments.extend([cluster_idx] * entities_per_cluster[cluster_idx])

    stacked = np.vstack(latents)
    assignments = np.asarray(cluster_assignments)
    print(
        f"[Latents] Created {stacked.shape[0]} entities x {stacked.shape[1]} dims "
        f"across {num_clusters} clusters."
    )
    return stacked, assignments


def generate_price_coefficients(
    num_users: int,
    mean: float = -1.5,
    std: float = 0.5,
    random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=mean, scale=std, size=num_users)


def generate_prices_and_availability(
    num_items: int,
    num_sessions: int,
    price_mean: float = 10.0,
    price_std: float = 3.0,
    availability_prob: float = 0.9,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    base_prices = rng.normal(loc=price_mean, scale=price_std, size=num_items)
    base_prices = np.maximum(base_prices, 1.0)

    prices = np.zeros((num_sessions, num_items))
    for i in range(num_items):
        variations = rng.normal(0, price_std * 0.2, num_sessions)
        prices[:, i] = base_prices[i] + variations
    prices = np.maximum(prices, 1.0)

    availability = rng.binomial(1, availability_prob, size=(num_sessions, num_items)).astype(bool)
    for session_id in range(num_sessions):
        if not np.any(availability[session_id]):
            availability[session_id, rng.integers(0, num_items)] = True

    return prices, availability


def simulate_choice_behavior(
    user_latents: np.ndarray,
    item_latents: np.ndarray,
    prices: np.ndarray,
    availability: np.ndarray,
    price_coefficients: np.ndarray,
    num_records: int,
    choice_noise: float = 0.7,
    exploration_prob: float = 0.15,
    temperature: float = 2.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate user choices with utility, noise, and occasional exploration."""
    rng = np.random.default_rng(random_state)
    num_users, _ = user_latents.shape
    num_items = item_latents.shape[0]
    num_sessions = prices.shape[0]

    user_indices = rng.integers(0, num_users, size=num_records)
    session_indices = rng.integers(0, num_sessions, size=num_records)
    item_indices = np.zeros(num_records, dtype=int)

    print("[Simulation] Pre-computing user-item compatibility matrix...")
    batch_size = 100
    ui_compatibility = np.zeros((num_users, num_items))
    for start in tqdm(range(0, num_users, batch_size), desc="Computing compatibility"):
        end = min(start + batch_size, num_users)
        ui_compatibility[start:end] = user_latents[start:end] @ item_latents.T

    print("[Simulation] Running stochastic choice process...")
    for i in tqdm(range(num_records), desc="Simulating choices"):
        user_idx = user_indices[i]
        session_idx = session_indices[i]
        available_mask = availability[session_idx]
        if not np.any(available_mask):
            available_mask[rng.integers(0, num_items)] = True

        available_items = np.where(available_mask)[0]
        if rng.random() < exploration_prob:
            item_indices[i] = rng.choice(available_items)
            continue

        base_utility = ui_compatibility[user_idx]
        price_effect = price_coefficients[user_idx] * prices[session_idx]
        utility = base_utility + price_effect
        utility += rng.normal(0, choice_noise, size=utility.shape)

        masked_utility = utility.copy()
        masked_utility[~available_mask] = float("-inf")

        if rng.random() < 0.35:
            max_utility = np.max(masked_utility)
            exp_utility = np.exp((masked_utility - max_utility) / temperature)
            exp_utility[~available_mask] = 0
            probs = exp_utility / np.sum(exp_utility)
            item_indices[i] = rng.choice(num_items, p=probs)
        else:
            item_indices[i] = np.argmax(masked_utility)

    return user_indices, session_indices, item_indices


def build_latent_frames(user_latents: np.ndarray, item_latents: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    user_df = pd.DataFrame(user_latents, columns=[f"user_latent_{i}" for i in range(LATENT_DIM)])
    user_df["user_id"] = np.arange(user_latents.shape[0])

    item_df = pd.DataFrame(item_latents, columns=[f"item_latent_{i}" for i in range(LATENT_DIM)])
    item_df["item_id"] = np.arange(item_latents.shape[0])
    return user_df, item_df


def summarize_records(train_all: pd.DataFrame) -> None:
    print("[Records] Summary")
    print(f"  Total records     : {len(train_all):,}")
    print(f"  Unique users      : {train_all['user_id'].nunique():,}")
    print(f"  Avg sessions/user : {train_all.value_counts('user_id').mean():.2f}")
    print(f"  Unique items      : {train_all['item_id'].nunique():,}")
    print(f"  Avg choices/item  : {train_all.value_counts('item_id').mean():.2f}")


def simulate_choice_data(
    train_all: pd.DataFrame,
    user_latents: pd.DataFrame,
    item_latents: pd.DataFrame,
    num_subsampled_items: int,
    num_subsampled_users: int,
    num_subsampled_latents: Optional[int],
    num_subsampled_records: Optional[int],
    seed: int,
    experiment_tag: str,
    output_path: Path,
    num_items_total: int,
    export_csv: bool,
    item_pool: Optional[np.ndarray] = None,
    user_pool: Optional[np.ndarray] = None,
) -> None:
    """Subsample users/items/records and write ChoiceDataset and optional CSV."""
    rng = np.random.default_rng(seed)

    missing_user_latents = [f"user_latent_{i}" for i in range(LATENT_DIM) if f"user_latent_{i}" not in user_latents.columns]
    missing_item_latents = [f"item_latent_{i}" for i in range(LATENT_DIM) if f"item_latent_{i}" not in item_latents.columns]
    assert not missing_user_latents, f"user_latents missing dimensions: {missing_user_latents}"
    assert not missing_item_latents, f"item_latents missing dimensions: {missing_item_latents}"

    if num_subsampled_latents is not None:
        user_latents = user_latents.copy().drop(columns=[f"user_latent_{i}" for i in range(num_subsampled_latents, LATENT_DIM)])
        item_latents = item_latents.copy().drop(columns=[f"item_latent_{i}" for i in range(num_subsampled_latents, LATENT_DIM)])

    if item_pool is None or len(item_pool) < num_subsampled_items:
        item_pool = np.arange(num_items_total)
    if user_pool is None or len(user_pool) < num_subsampled_users:
        user_pool = np.arange(NUM_USERS)

    selected_items = rng.choice(item_pool, size=num_subsampled_items, replace=False)
    selected_users = rng.choice(user_pool, size=num_subsampled_users, replace=False)

    suitable_records = train_all[
        (train_all["item_id"].isin(selected_items)) & (train_all["user_id"].isin(selected_users))
    ]
    if num_subsampled_records is None:
        records = suitable_records.copy()
        num_subsampled_records = len(records)
    else:
        if len(suitable_records) == 0:
            raise ValueError(
                f"No records available after subsampling for experiment_tag={experiment_tag}. "
                "Try increasing --num-records or adjusting experiment settings."
            )
        if len(suitable_records) < num_subsampled_records:
            print(
                f"[Warn] Requested {num_subsampled_records} records but only {len(suitable_records)} available "
                f"for {experiment_tag}; using all available records."
            )
            num_subsampled_records = len(suitable_records)
        records = suitable_records.sample(num_subsampled_records, replace=False, random_state=seed)

    relevant_user_latents = user_latents[user_latents["user_id"].isin(selected_users)].copy()
    relevant_item_latents = item_latents[item_latents["item_id"].isin(selected_items)].copy()

    item_encoder = LabelEncoder().fit(selected_items)
    user_encoder = LabelEncoder().fit(selected_users)

    relevant_item_latents.loc[:, "item_id"] = item_encoder.transform(relevant_item_latents["item_id"].values)
    relevant_user_latents.loc[:, "user_id"] = user_encoder.transform(relevant_user_latents["user_id"].values)
    records.loc[:, "item_id"] = item_encoder.transform(records["item_id"].values)
    records.loc[:, "user_id"] = user_encoder.transform(records["user_id"].values)

    relevant_item_latents = relevant_item_latents.set_index("item_id").loc[np.arange(num_subsampled_items)]
    relevant_user_latents = relevant_user_latents.set_index("user_id").loc[np.arange(num_subsampled_users)]

    dataset = ChoiceDataset(
        item_index=torch.LongTensor(records["item_id"].values),
        user_index=torch.LongTensor(records["user_id"].values),
        session_index=torch.arange(len(records)).long(),
        user_latents=torch.FloatTensor(relevant_user_latents.values),
        item_latents=torch.FloatTensor(relevant_item_latents.values),
        num_items=num_subsampled_items,
    )
    dataset._num_items = num_subsampled_items
    dataset._num_users = num_subsampled_users

    pt_path = output_path / f"simulated_choice_data_{experiment_tag}_seed_{seed}.pt"
    torch.save(dataset, pt_path)
    print(f"[Saved] {pt_path} ({pt_path.stat().st_size / 1e6:.1f} MB)")

    if not export_csv:
        return

    temp = []
    for record_id in range(num_subsampled_records):
        item_chosen = records.iloc[record_id]["item_id"]
        one_hot = np.zeros(num_subsampled_items)
        one_hot[item_chosen] = 1
        temp.append(
            pd.DataFrame(
                {
                    "session_id": record_id,
                    "item_id": np.arange(num_subsampled_items),
                    "choice": one_hot,
                    "user_id": records.iloc[record_id]["user_id"],
                }
            )
        )
    df_long = pd.concat(temp, axis=0)
    df_long = df_long.merge(relevant_user_latents.reset_index(), on="user_id", how="left")
    df_long = df_long.merge(relevant_item_latents.reset_index(), on="item_id", how="left")

    csv_path = output_path / f"simulated_choice_data_{experiment_tag}_seed_{seed}.csv"
    df_long.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")


def save_tsne_plots(
    user_latents: np.ndarray,
    item_latents: np.ndarray,
    user_clusters: np.ndarray,
    item_clusters: np.ndarray,
    output_path: Path,
) -> None:
    print("[Plot] Running t-SNE for visualization (saved to disk).")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    user_latents_2d = tsne.fit_transform(user_latents)

    tsne = TSNE(n_components=2, random_state=43, perplexity=30, n_iter=1000)
    item_latents_2d = tsne.fit_transform(item_latents)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    for cluster_id in range(NUM_USER_CLUSTERS):
        mask = user_clusters == cluster_id
        plt.scatter(user_latents_2d[mask, 0], user_latents_2d[mask, 1], label=f"Cluster {cluster_id}", alpha=0.6)
    plt.title("User Latent Clusters (t-SNE projection)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()

    plt.subplot(1, 2, 2)
    for cluster_id in range(NUM_ITEM_CLUSTERS):
        mask = item_clusters == cluster_id
        plt.scatter(
            item_latents_2d[mask, 0],
            item_latents_2d[mask, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
            marker="s",
        )
    plt.title("Item Latent Clusters (t-SNE projection)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.tight_layout()
    first_plot = output_path / "tsne_latent_clusters.png"
    plt.savefig(first_plot, dpi=200)
    plt.close()
    print(f"[Saved] {first_plot}")


def get_md5sum(path: Path) -> str:
    md5_hash = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def compare_against_reference(output_path: Path, reference_path: Path) -> None:
    if not reference_path.exists():
        print(f"[Compare] Reference path {reference_path} not found; skipping.")
        return

    print("[Compare] Checking generated files against reference.")
    comparison: list[dict[str, str]] = []
    for ref_file in sorted(reference_path.iterdir()):
        if not ref_file.is_file():
            continue
        if ref_file.name.startswith("._"):
            continue
        if ref_file.suffix not in {".pt", ".csv"}:
            continue

        generated_file = output_path / ref_file.name
        if not generated_file.exists():
            comparison.append(
                {
                    "file": ref_file.name,
                    "consistent": "missing",
                    "md5_generated": "N/A",
                    "md5_reference": get_md5sum(ref_file),
                    "detail": "generated file missing",
                }
            )
            continue

        md5_ref = get_md5sum(ref_file)
        md5_gen = get_md5sum(generated_file)

        if ref_file.suffix == ".csv" and md5_ref != md5_gen:
            detail = "csv differs"
        elif ref_file.suffix == ".pt":
            try:
                ds1 = torch.load(ref_file, weights_only=False)
                ds2 = torch.load(generated_file, weights_only=False)
                detail = "match" if ds1 == ds2 else "tensor differs"
            except Exception as exc:  # noqa: BLE001
                detail = f"load error: {exc}"
        else:
            detail = "match"

        comparison.append(
            {
                "file": ref_file.name,
                "consistent": "yes" if md5_ref == md5_gen else "no",
                "md5_generated": md5_gen,
                "md5_reference": md5_ref,
                "detail": detail,
            }
        )

    for row in comparison:
        print(
            f"[Compare] {row['file']}: {row['consistent']} | "
            f"md5 gen={row['md5_generated']} ref={row['md5_reference']} | {row['detail']}"
        )


def run(
    output_path: Path,
    experiments: Optional[Iterable[str]] = None,
    num_records: int = 3_000_000,
    skip_plots: bool = False,
    reference_path: Optional[Path] = None,
    smoke_test: bool = False,
) -> None:
    experiments = list(experiments or ["all"])
    selected_experiments: Iterable[str] = ALL_EXPERIMENTS if "all" in experiments else set(experiments)

    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    log_versions()

    print(SECTION)
    print("Generating latent factors")
    print(SECTION)
    user_latents, user_clusters = generate_clustered_latents(
        num_entities=NUM_USERS,
        latent_dim=LATENT_DIM,
        num_clusters=NUM_USER_CLUSTERS,
        cluster_std=0.5,
        between_cluster_distance=3.0,
        random_state=42,
    )
    item_latents, item_clusters = generate_clustered_latents(
        num_entities=NUM_ITEMS,
        latent_dim=LATENT_DIM,
        num_clusters=NUM_ITEM_CLUSTERS,
        cluster_std=0.3,
        between_cluster_distance=2.5,
        random_state=43,
    )
    print(f"[Latents] Users shape {user_latents.shape}, Items shape {item_latents.shape}")
    print(f"[Latents] User cluster distribution: {np.bincount(user_clusters)}")
    print(f"[Latents] Item cluster distribution: {np.bincount(item_clusters)}")

    if not skip_plots:
        save_tsne_plots(user_latents, item_latents, user_clusters, item_clusters, output_path)

    print(SECTION)
    print("Pricing and availability")
    print(SECTION)
    price_coefficients = generate_price_coefficients(num_users=NUM_USERS, mean=-1.5, std=0.5, random_state=44)
    prices, availability = generate_prices_and_availability(
        num_items=NUM_ITEMS,
        num_sessions=1000,
        price_mean=10.0,
        price_std=3.0,
        availability_prob=0.95,
        random_state=45,
    )
    print(
        f"[Prices] range=({prices.min():.2f}, {prices.max():.2f}) "
        f"availability={availability.mean() * 100:.1f}% "
        f"coeff range=({price_coefficients.min():.2f}, {price_coefficients.max():.2f})"
    )

    print(SECTION)
    print("Simulating choices")
    print(SECTION)
    user_indices, session_indices, item_indices = simulate_choice_behavior(
        user_latents=user_latents,
        item_latents=item_latents,
        prices=prices,
        availability=availability,
        price_coefficients=price_coefficients,
        num_records=num_records,
        choice_noise=0.5,
        exploration_prob=0.5,
        random_state=42,
    )
    print(f"[Simulation] Generated {len(user_indices):,} records.")

    print(SECTION)
    print("Preparing data frames")
    print(SECTION)
    user_latents_df, item_latents_df = build_latent_frames(user_latents, item_latents)
    train_all = pd.DataFrame({"user_id": user_indices, "item_id": item_indices})
    summarize_records(train_all)

    # For smoke-test runs, prefer sampling from frequent users/items to avoid empty subsamples.
    item_pool: Optional[np.ndarray] = None
    user_pool: Optional[np.ndarray] = None
    if smoke_test:
        item_pool = train_all["item_id"].value_counts().head(200).index.to_numpy()
        user_pool = train_all["user_id"].value_counts().index.to_numpy()

    smoke_num_records_per_dataset = 1_000

    print(SECTION)
    print("Writing benchmark datasets")
    print(SECTION)
    if "num_records_experiment_small" in selected_experiments:
        simulate_choice_data(
            train_all=train_all,
            user_latents=user_latents_df,
            item_latents=item_latents_df,
            num_subsampled_items=30,
            num_subsampled_users=NUM_USERS,
            num_subsampled_latents=10,
            num_subsampled_records=(smoke_num_records_per_dataset if smoke_test else None),
            seed=42,
            experiment_tag="num_records_experiment_small",
            output_path=output_path,
            num_items_total=NUM_ITEMS,
            export_csv=True,
            item_pool=item_pool,
            user_pool=user_pool,
        )

    if "full_dataset" in selected_experiments and not smoke_test:
        simulate_choice_data(
            train_all=train_all,
            user_latents=user_latents_df,
            item_latents=item_latents_df,
            num_subsampled_items=NUM_ITEMS,
            num_subsampled_users=NUM_USERS,
            num_subsampled_latents=LATENT_DIM,
            num_subsampled_records=None,
            seed=42,
            experiment_tag="full_dataset",
            output_path=output_path,
            num_items_total=NUM_ITEMS,
            export_csv=False,
        )

    if "num_params_experiment_small" in selected_experiments:
        simulate_choice_data(
            train_all=train_all,
            user_latents=user_latents_df,
            item_latents=item_latents_df,
            num_subsampled_items=50,
            num_subsampled_users=NUM_USERS,
            num_subsampled_latents=LATENT_DIM,
            num_subsampled_records=(smoke_num_records_per_dataset if smoke_test else 10_000),
            seed=42,
            experiment_tag="num_params_experiment_small",
            output_path=output_path,
            num_items_total=NUM_ITEMS,
            export_csv=True,
            item_pool=item_pool,
            user_pool=user_pool,
        )

    if "num_items_experiment_small" in selected_experiments:
        items_grid = [10, 20, 30] if smoke_test else [10, 20, 30, 50, 100, 150, 200]
        for num_items_sampled in tqdm(items_grid, desc="num_items_experiment_small"):
            simulate_choice_data(
                train_all=train_all,
                user_latents=user_latents_df,
                item_latents=item_latents_df,
                num_subsampled_items=num_items_sampled,
                num_subsampled_users=NUM_USERS,
                num_subsampled_latents=5,
                num_subsampled_records=(smoke_num_records_per_dataset if smoke_test else 10_000),
                seed=42,
                experiment_tag=f"num_items_experiment_small_{num_items_sampled}_items",
                output_path=output_path,
                num_items_total=NUM_ITEMS,
                export_csv=True,
                item_pool=item_pool,
                user_pool=user_pool,
            )

    if "num_items_experiment_large" in selected_experiments and not smoke_test:
        for num_items_sampled in tqdm(
            [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            desc="num_items_experiment_large",
        ):
            simulate_choice_data(
                train_all=train_all,
                user_latents=user_latents_df,
                item_latents=item_latents_df,
                num_subsampled_items=num_items_sampled,
                num_subsampled_users=NUM_USERS,
                num_subsampled_latents=LATENT_DIM,
                num_subsampled_records=30_000,
                seed=42,
                experiment_tag=f"num_items_experiment_large_{num_items_sampled}_items",
                output_path=output_path,
                num_items_total=NUM_ITEMS,
                export_csv=False,
            )

    if reference_path:
        compare_against_reference(output_path, reference_path.resolve())

    print("[Done] Finished generating synthetic datasets.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(
        output_path=args.output_path,
        experiments=args.experiments,
        num_records=args.num_records,
        skip_plots=args.skip_plots,
        reference_path=args.reference_path,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user.")
        sys.exit(1)

