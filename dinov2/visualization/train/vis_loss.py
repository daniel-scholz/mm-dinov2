import json
import logging
import os
from pathlib import Path

import pandas as pd
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger("dinov2")


def _load_training_metrics(result_directory: Path):
    training_metrics_fp = result_directory / "training_metrics.json"
    if not training_metrics_fp.exists():
        return None
    # assert training_metrics_fp.exists(), f"File {training_metrics_fp} does not exist"

    lines = training_metrics_fp.read_text().splitlines()
    lines = [json.loads(line) for line in lines]
    return pd.DataFrame(lines)


def gather_training_metrics(out_dir: Path):
    df_training_metrics_dict = {}
    for cur_seq_dir in out_dir.iterdir():
        if not cur_seq_dir.is_dir():
            continue

        df_training_metrics = _load_training_metrics(cur_seq_dir)
        if df_training_metrics is None:
            continue
        df_training_metrics_dict[cur_seq_dir.name] = df_training_metrics
    return df_training_metrics_dict


def _transform_result_to_dataframe(result, iter):
    df_iter = pd.DataFrame(result)
    df_iter.reset_index(inplace=True, names=["metric"])

    df_iter_flat = df_iter.melt(id_vars=["metric"], var_name="split", value_name=iter)

    # set metric and split as index
    df_iter_flat.set_index(["metric", "split"], inplace=True, drop=True)

    return df_iter_flat


def _load_eval_metrics(experiment_dir: Path, prefix="training"):
    results = []
    for results_json_fp in experiment_dir.glob(f"{prefix}*/results.json"):
        iter = results_json_fp.parent.name.split("_")[-1]
        if iter.isdigit():
            iter = int(iter)
        else:
            print("iter is not digit", iter)

        results.append((iter, pd.read_json(results_json_fp)))
    if not len(results):
        return None
    iter_0, dict_seq = results[0]
    df_eval_metrics = _transform_result_to_dataframe(dict_seq, iter_0)

    for iter, result in results[1:]:
        df_eval_metrics_iter = _transform_result_to_dataframe(result, iter)
        df_eval_metrics = pd.merge(
            df_eval_metrics, df_eval_metrics_iter, left_index=True, right_index=True
        )

    return df_eval_metrics


def gather_eval_metrics(train_out_dir: Path):
    df_results_dict = {}
    for experiment_dir in train_out_dir.iterdir():
        if not experiment_dir.is_dir():
            continue

        cur_eval_dir = experiment_dir / "eval"
        if not cur_eval_dir.exists():
            continue
        # assert cur_eval_dir.exists(), f"Directory {cur_eval_dir} does not exist"

        df_eval_metrics = _load_eval_metrics(cur_eval_dir, prefix="manual")
        if df_eval_metrics is None:
            continue

        df_results_dict[experiment_dir.name] = df_eval_metrics
    return df_results_dict


def _plot_loss_and_metrics(
    experiment_dir, df_eval_metrics, df_training_metrics: pd.DataFrame
):
    df_metrics_t = df_eval_metrics.T
    # flatten columns
    df_metrics_t.columns = [
        f"{metric}_{split}" for metric, split in df_metrics_t.columns
    ]

    # replace final iteration with maximum iteration from training metrics
    df_metrics_t.rename(
        index={"final": df_training_metrics["iteration"].max()}, inplace=True
    )
    # merge on iteration
    df_eval_metrics = pd.merge(
        df_metrics_t,
        df_training_metrics,
        left_index=True,
        right_on="iteration",
        how="outer",
    ).reset_index(drop=True)

    # # keep only columns with either "loss", "val" or "test", and "iteration"
    # df_metrics = df_metrics[
    #     [col for col in df_metrics.columns if "loss" in col or "val" in col or "test" in col or col == "iteration"]
    # ]

    # plot metrics
    df_metrics_melted = df_eval_metrics.melt(
        id_vars="iteration", var_name="metric", value_name="value"
    )

    g = sns.relplot(
        data=df_metrics_melted,
        x="iteration",
        y="value",
        row="metric",
        kind="line",
        height=4,
        aspect=1.5,
        facet_kws={"sharey": False, "sharex": True},
    )

    # save
    plot_fp = experiment_dir / "training_metrics.png"
    g.savefig(plot_fp)


def plot_gathered_loss_and_metrics(
    out_dir: Path,
    df_results_dict: dict[str, pd.DataFrame],
    df_training_metrics_dict: dict[str, pd.DataFrame],
):
    pbar = tqdm(
        df_results_dict.items(),
        desc="Plotting gathered loss and metrics",
        total=len(df_results_dict),
    )
    for experiment_name, df_metrics in pbar:
        pbar.set_postfix_str(experiment_name)
        df_training_metrics = df_training_metrics_dict[experiment_name]

        _plot_loss_and_metrics(
            out_dir / experiment_name, df_metrics, df_training_metrics
        )


def vis_loss_and_metrics(cfg):
    experiment_dir = os.path.join(cfg.train.output_dir, "eval")
    experiment_dir = Path(experiment_dir)
    df_training_metrics = _load_training_metrics(experiment_dir.parent)

    df_eval_metrics = _load_eval_metrics(experiment_dir)

    if (df_training_metrics is None) or (df_eval_metrics is None):
        return

    _plot_loss_and_metrics(experiment_dir.parent, df_eval_metrics, df_training_metrics)


def gather_missing_sequence_results(train_out_dir: str):
    df_results_dict = {}
    for experiment_dir in train_out_dir.iterdir():
        if not experiment_dir.is_dir():
            continue

        cur_eval_dir = experiment_dir / "eval"
        if not cur_eval_dir.exists():
            continue

        # check if missing sequence analysis ran
        if not (cur_eval_dir / "best_mri_sequences-r-a-n-d-o-m").exists():
            continue

        # assert cur_eval_dir.exists(), f"Directory {cur_eval_dir} does not exist"

        df_eval_metrics = _load_eval_metrics_missing_sequences(cur_eval_dir)
        if df_eval_metrics is None:
            continue

        df_results_dict[experiment_dir.name] = df_eval_metrics
    return df_results_dict


def _load_eval_metrics_missing_sequences(experiment_dir: Path):
    # load all results and store them in a table
    experiment_dir = Path(experiment_dir)
    df_eval_metrics = _load_eval_metrics(experiment_dir, prefix="best_mri_sequences")

    df_eval_metrics = df_eval_metrics.T
    # flatten columns
    df_eval_metrics.columns = [
        f"{metric}.{split}" for metric, split in df_eval_metrics.columns
    ]
    # extract mri_sequences from index column (final_mri_sequences-)
    mri_sequences_col = df_eval_metrics.index.str.extract(r"sequences-(.*)")
    # split mri_sequences into list
    mri_sequences_col = mri_sequences_col[0].str.split("-", expand=True)

    mri_sequences = ["t1", "t1c", "t2", "flair"]

    # create new emoty column for each sequence
    for mri_sequence in mri_sequences:
        df_eval_metrics[mri_sequence] = 0

    # set row to 1 if sequence is in mri_sequences_col
    for mri_sequence in mri_sequences:
        seq_is_present_mask = (mri_sequences_col == mri_sequence).any(axis=1)
        df_eval_metrics.loc[seq_is_present_mask.values, mri_sequence] = 1
    # sort by mri_sequences
    df_eval_metrics = df_eval_metrics.sort_values(by=mri_sequences)
    # make the mri_sequence columns the first
    df_eval_metrics = df_eval_metrics[
        [*mri_sequences, *df_eval_metrics.columns[: -len(mri_sequences)]]
    ]
    # save
    df_eval_metrics.to_csv(experiment_dir / "best_mri_sequences_eval_metrics.csv")
    logger.info(
        f"Saved best_mri_sequences_eval_metrics.csv to {experiment_dir}/best_mri_sequences_eval_metrics.csv"
    )
    logger.info(df_eval_metrics.loc[:, ["mcc.val", "t1", "t1c", "t2", "flair"]])
    return df_eval_metrics
