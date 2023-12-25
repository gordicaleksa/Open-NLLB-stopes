#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
import shutil
import time
import copy
import gzip
import itertools
import logging
import os
import random
from contextlib import ExitStack
import multiprocessing
from pathlib import Path
import pickle
from typing import Dict, Optional

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from submitit import AutoExecutor
import wandb

from stopes.pipelines.filtering.first_stage_filtering import FirstStage, FuzzyFilterStage
from stopes.core import utils
from stopes.pipelines.filtering.configs import (
    FilterConfig,
    GroupFilterConfig,
    register_configs,
)
from stopes.pipelines.filtering.dataset import Dataset, DatasetLine, DatasetReader
from stopes.pipelines.filtering.filters import FilteringCounts
from stopes.pipelines.filtering.utils import cache_step_sync, normalize_unicode
from stopes.pipelines.filtering.filters.file_chunker_utils_bitext import BitextChunker, find_offsets, find_offsets_given_line_numbers, build_line_number_to_byte_offset_map, convert_offsets_to_line_numbers
from stopes.pipelines.monolingual.utils.text_normalizer import replace_unicode_punct, remove_non_printing_char, normalize_whitespace

logger = logging.getLogger(__name__)


register_configs()


class FilteringStage(Enum):
    FirstStage = 1
    SecondStage = 2
    ThirdStage = 3


def get_preprocess_stage_infix(stage: FilteringStage):
    if stage == FilteringStage.FirstStage:
        return ""
    elif stage == FilteringStage.SecondStage:
        return "_global_exact_dedup"
    elif stage == FilteringStage.ThirdStage:
        return "_fuzzy_dedup"
    else:
        raise ValueError(f"Unknown stage {stage}")


def get_postprocess_stage_infix(stage: FilteringStage):
    if stage == FilteringStage.FirstStage:
        return "_before_fuzzy_*", -1
    elif stage == FilteringStage.SecondStage:
        return "_before_fuzzy_*_global_exact_dedup", -4
    elif stage == FilteringStage.ThirdStage:
        return "_fuzzy_dedup_*", -1
    else:
        raise ValueError(f"Unknown stage {stage}")


def get_step(steps_so_far, dataset_counts):
    return steps_so_far + dataset_counts.total_before


def stage_preprocess(dataset, corpus_name, dataset_output_dir, num_workers=12, stage: FilteringStage = FilteringStage.FirstStage):
    num_lines = utils.count_lines(dataset.src)
    if num_lines == 0:
        return -1, -1, -1, -1

    num_workers_dynamic = min(((num_lines - 1) // 5000) + 1, num_workers)
    filename_infix = get_preprocess_stage_infix(stage)

    src_offsets_path = dataset_output_dir / f"{corpus_name}_src_offsets{filename_infix}.pickle"
    if os.path.exists(src_offsets_path):
        with open(src_offsets_path, "rb") as f:
            src_offsets = pickle.load(f)
    else:
        src_offsets = find_offsets(dataset.src, num_workers_dynamic)
        with open(src_offsets_path, "wb") as f:
            pickle.dump(src_offsets, f)

    src_file_chunks = list(zip(src_offsets, src_offsets[1:]))

    src_chunks_line_numbers_path = dataset_output_dir / f"{corpus_name}_src_chunks_line_numbers{filename_infix}.pickle"
    if os.path.exists(src_chunks_line_numbers_path):
        with open(src_chunks_line_numbers_path, "rb") as f:
            src_chunks_line_numbers = pickle.load(f)
    else:
        src_chunks_line_numbers = convert_offsets_to_line_numbers(src_offsets, dataset.src)
        with open(src_chunks_line_numbers_path, "wb") as f:
            pickle.dump(src_chunks_line_numbers, f)

    tgt_offsets_path = dataset_output_dir / f"{corpus_name}_tgt_offsets{filename_infix}.pickle"
    if os.path.exists(tgt_offsets_path):
        with open(tgt_offsets_path, "rb") as f:
            tgt_offsets = pickle.load(f)
    else:
        tgt_offsets = find_offsets_given_line_numbers(dataset.tgt, src_chunks_line_numbers)
        with open(tgt_offsets_path, "wb") as f:
            pickle.dump(tgt_offsets, f)

    tgt_chunks_line_numbers_path = dataset_output_dir / f"{corpus_name}_tgt_chunks_line_numbers{filename_infix}.pickle"
    if os.path.exists(tgt_chunks_line_numbers_path):
        with open(tgt_chunks_line_numbers_path, "rb") as f:
            tgt_chunks_line_numbers = pickle.load(f)
    else:
        tgt_chunks_line_numbers = convert_offsets_to_line_numbers(tgt_offsets, dataset.tgt)
        with open(tgt_chunks_line_numbers_path, "wb") as f:
            pickle.dump(tgt_chunks_line_numbers, f)

    assert tgt_chunks_line_numbers == src_chunks_line_numbers, f"src and tgt have different number of lines for {dataset.src} and {dataset.tgt}"

    tgt_file_chunks = list(zip(tgt_offsets, tgt_offsets[1:]))

    src_chunks_line_numbers = list(zip(src_chunks_line_numbers, src_chunks_line_numbers[1:]))
    tgt_chunks_line_numbers = list(zip(tgt_chunks_line_numbers, tgt_chunks_line_numbers[1:]))

    return src_file_chunks, tgt_file_chunks, src_chunks_line_numbers, num_workers_dynamic


def stage_postprocess(
        dataset_output_dir,
        corpus_name,
        src_lang,
        tgt_lang,
        path_out_src_before_fuzzy,
        path_out_tgt_before_fuzzy,
        path_counts,
        stage: FilteringStage = FilteringStage.FirstStage,
    ):
    is_third = stage == FilteringStage.ThirdStage
    filename_infix, index = get_postprocess_stage_infix(stage)

    src_files_list = sorted([str(path) for path in dataset_output_dir.glob(f"{corpus_name}.{src_lang}{filename_infix}")], key=lambda x: int(x.split('_')[index]))
    tgt_files_list = sorted([str(path) for path in dataset_output_dir.glob(f"{corpus_name}.{tgt_lang}{filename_infix}")], key=lambda x: int(x.split('_')[index]))
    assert len(src_files_list) == len(tgt_files_list), f"Number of src files {len(src_files_list)} is not equal to number of tgt files {len(tgt_files_list)}"

    # Merge output files from the workers
    src_cnt = 0
    tgt_cnt = 0
    with gzip.open(path_out_src_before_fuzzy, "wt") if is_third else open(path_out_src_before_fuzzy, "w") as outfile_src, gzip.open(path_out_tgt_before_fuzzy, "wt") if is_third else open(path_out_tgt_before_fuzzy, "w") as outfile_tgt:
        for src_file, tgt_file in zip(src_files_list, tgt_files_list):
            with open(src_file) as src_infile, open(tgt_file) as tgt_infile:
                for line in src_infile:
                    src_cnt += 1
                    outfile_src.write(line)
                for line in tgt_infile:
                    tgt_cnt += 1
                    outfile_tgt.write(line)

    assert src_cnt == tgt_cnt, f"Number of src lines {src_cnt} is not equal to number of tgt lines {tgt_cnt}"
    # delete the temporary files
    for src_file, tgt_file in zip(src_files_list, tgt_files_list):
        os.remove(src_file)
        os.remove(tgt_file)

    counts_files_list = [str(path) for path in dataset_output_dir.glob(f"{corpus_name}{filename_infix}.yaml")]
    counts_partial = []
    with open(path_counts, "wt") as fout:
        for counts_file in counts_files_list:
            with open(counts_file) as fin:
                counts_partial.append(FilteringCounts(**yaml.safe_load(fin)))

        if len(counts_partial) == 0:
            counts_partial = [FilteringCounts()]

        yaml.safe_dump(sum(counts_partial).__dict__, fout)

    # delete the temporary files
    for counts_file in counts_files_list:
        os.remove(counts_file)

    # remove offsets & chunks that were computed prior to main stage
    offset_files_list = [str(path) for path in dataset_output_dir.glob(f"*offsets*")]
    chunk_files_list = [str(path) for path in dataset_output_dir.glob(f"*chunks*")]
    assert len(offset_files_list) == len(chunk_files_list), f"Number of src files {len(offset_files_list)} is not equal to number of tgt files {len(chunk_files_list)}"
    for offset_file, chunk_file in zip(offset_files_list, chunk_files_list):
        os.remove(offset_file)
        os.remove(chunk_file)

    return sum(counts_partial)


# TODO have this use the MultiprocLineProcessor module
@cache_step_sync("filter_direction")
def filter_direction(
    group_name: str,
    src_lang: str,
    tgt_lang: Optional[str],  # if None, treat as monolingual datasets
    datasets: Dict[str, Dataset],
    length_factors: Dict[str, float],
    config: GroupFilterConfig,
    dataset_output_dir: Path,
    custom_step_name: str,
    output_dir: Path,
    wandb_run_name: str,
    num_workers: int = 32
):
    direction = f"{src_lang}-{tgt_lang}" if tgt_lang is not None else src_lang
    wandb.init(project="slavic-NLLB-filtering", reinit=False, name=wandb_run_name)
    print(f"Filtering {group_name}.{direction}")

    counts: Dict[str, FilteringCounts] = {}
    next_stage_datasets = {}
    # 1st stage of filtering: everything except for fuzzy deduplication.
    timings = []

    #
    # 1ST STAGE - CORPUS-LEVEL FILTERING
    #
    # global_exact_dedup = False
    # for corpus_name, dataset in datasets.items():

    #     path_out_src_before_fuzzy = dataset_output_dir / f"{corpus_name}.{src_lang}_before_fuzzy"
    #     path_out_tgt_before_fuzzy = dataset_output_dir / f"{corpus_name}.{tgt_lang}_before_fuzzy"

    #     next_stage_datasets[corpus_name] = Dataset(src=path_out_src_before_fuzzy, tgt=path_out_tgt_before_fuzzy, tsv=None, metadata=None, lang_dir=None, fold=None)

    #     path_counts = dataset_output_dir / f"{corpus_name}_before_fuzzy.yaml"

    #     if os.path.isfile(path_counts):
    #         with open(path_counts, "rt") as fin:
    #             counts[corpus_name] = FilteringCounts(**yaml.safe_load(fin))
    #         print(f"Skipping {corpus_name} as a corresponding YAML file already exists")
    #         continue

    #     ts = time.time()

    #     # Prepare for parallel processing
    #     src_file_chunks, tgt_file_chunks, _, num_workers_dynamic = stage_preprocess(
    #         dataset,
    #         corpus_name,
    #         dataset_output_dir,
    #         num_workers,
    #         stage=FilteringStage.FirstStage
    #     )

    #     if src_file_chunks != -1:  # if not empty
    #         FirstStage(
    #             dataset.src,
    #             dataset.tgt,
    #             src_file_chunks,
    #             tgt_file_chunks,
    #             dataset_output_dir,
    #             group_name,
    #             corpus_name,
    #             src_lang,
    #             tgt_lang,
    #             config,
    #             length_factors,
    #             num_workers_dynamic,
    #             dedup_dict=None,  # externally we only pass dedup_dict for global exact deduplication
    #             global_exact_dedup=global_exact_dedup,
    #         ).run()
    #         timings.append(time.time() - ts)

    #     counts[corpus_name] = stage_postprocess(
    #         dataset_output_dir,
    #         corpus_name,
    #         src_lang,
    #         tgt_lang,
    #         path_out_src_before_fuzzy,
    #         path_out_tgt_before_fuzzy,
    #         path_counts,
    #         stage=FilteringStage.FirstStage
    #     )

    #
    # 2ND STAGE - GLOBAL EXACT DEDUP
    #
    global_exact_dedup = True
    dedup_dict = multiprocessing.Manager().dict()
    for corpus_name, dataset in datasets.items():

        path_out_src_before_fuzzy = dataset_output_dir / f"{corpus_name}.{src_lang}_before_fuzzy_global_exact_dedup"
        path_out_tgt_before_fuzzy = dataset_output_dir / f"{corpus_name}.{tgt_lang}_before_fuzzy_global_exact_dedup"

        next_stage_datasets[corpus_name] = Dataset(src=path_out_src_before_fuzzy, tgt=path_out_tgt_before_fuzzy, tsv=None, metadata=None, lang_dir=None, fold=None)

        path_counts = dataset_output_dir / f"{corpus_name}_before_fuzzy_global_exact_dedup.yaml"

        if os.path.isfile(path_counts):
            with open(path_counts, "rt") as fin:
                counts[corpus_name] = FilteringCounts(**yaml.safe_load(fin))
            print(f"Skipping {corpus_name} as a corresponding YAML file already exists")
            continue

        ts = time.time()

        src_file_chunks, tgt_file_chunks, _, num_workers_dynamic = stage_preprocess(
            dataset,
            corpus_name,
            dataset_output_dir,
            num_workers,
            stage=FilteringStage.SecondStage
        )

        if src_file_chunks != -1:  # if not empty

            FirstStage(
                dataset.src,
                dataset.tgt,
                src_file_chunks,
                tgt_file_chunks,
                dataset_output_dir,
                group_name,
                corpus_name,
                src_lang,
                tgt_lang,
                config,
                length_factors,
                num_workers_dynamic,
                dedup_dict,
                global_exact_dedup=global_exact_dedup,
            ).run()
            timings.append(time.time() - ts)

        counts[corpus_name] = stage_postprocess(
            dataset_output_dir,
            corpus_name,
            src_lang,
            tgt_lang,
            path_out_src_before_fuzzy,
            path_out_tgt_before_fuzzy,
            path_counts,
            stage=FilteringStage.SecondStage
        )

    #
    # 3RD STAGE - FUZZY DEDUP
    #

    # Optimization - if all the datasets are already filtered, skip the fuzzy deduplication
    skip_fuzzy = True
    for corpus_name, dataset in next_stage_datasets.items():
        path_counts = dataset_output_dir / f"{corpus_name}.yaml"
        if not os.path.isfile(path_counts):
            skip_fuzzy = False
            break

    if skip_fuzzy:
        print(f"Skipping fuzzy deduplication for {group_name}.{direction} as all the datasets are already fuzzy filtered")
        return direction, counts

    lines_left_to_process = 0
    for corpus_name, dataset in next_stage_datasets.items():
        lines_left_to_process += counts[corpus_name].total_after

    # Update threshold in config.fuzzy_dedup_filter depending on number of lines - simple heuristic.
    if lines_left_to_process < 1_000_000:
        config.fuzzy_dedup_filter.threshold = 0.8  # only if "very" similar do we remove for smaller datasets.
    elif lines_left_to_process < 5_000_000:
        config.fuzzy_dedup_filter.threshold = 0.75
    elif lines_left_to_process < 10_000_000:
        config.fuzzy_dedup_filter.threshold = 0.65
    else:
        config.fuzzy_dedup_filter.threshold = 0.5

    # It's stateful so we have to do it before the loop - we're doing fuzzy across all datasets for this lang direction.
    # If the dataset is too big we repeat once more on the filtered smaller dataset from the 1st iteration.
    fuzzy_filter = hydra.utils.instantiate(
        config.fuzzy_dedup_filter,
        datasets=next_stage_datasets,
        num_workers=num_workers,
        output_dir=Path(output_dir) / f"{group_name.split('_')[-1]}_minhashes_{direction}")

    cnt = 0

    # TODO: TMP UNTIL I FIGURE OUT A GOOD WAY TO PARALLELIZE WORK
    num_workers = 1
    for corpus_name, dataset in next_stage_datasets.items():

        path_out_src = dataset_output_dir / f"{corpus_name}.{src_lang}.gz"
        path_out_tgt = dataset_output_dir / f"{corpus_name}.{tgt_lang}.gz"

        path_counts = dataset_output_dir / f"{corpus_name}.yaml"

        if os.path.isfile(path_counts):
            with open(path_counts, "rt") as fin:
                counts[corpus_name] = FilteringCounts(**yaml.safe_load(fin))
                cnt += counts[corpus_name].total_before
            print(f"Skipping {corpus_name} as a corresponding YAML file already exists")
            continue

        src_file_chunks, tgt_file_chunks, src_chunks_line_numbers, num_workers_dynamic = stage_preprocess(
            dataset,
            corpus_name,
            dataset_output_dir,
            num_workers,
            stage=FilteringStage.ThirdStage
        )

        if src_file_chunks != -1:  # if not empty

            FuzzyFilterStage(
                dataset.src,
                dataset.tgt,
                src_file_chunks,
                tgt_file_chunks,
                dataset_output_dir,
                corpus_name,
                src_lang,
                tgt_lang,
                num_workers_dynamic,
                fuzzy_filter,
                src_chunks_line_numbers,
                cnt,
            ).run()

            cnt += src_chunks_line_numbers[-1][1] - src_chunks_line_numbers[0][0]

        counts[corpus_name] = stage_postprocess(
            dataset_output_dir,
            corpus_name,
            src_lang,
            tgt_lang,
            path_out_src,
            path_out_tgt,
            path_counts,
            stage=FilteringStage.ThirdStage
        )

    if counts:
        print(f"Total counts: {sum(counts.values()).__dict__}")
        with open(dataset_output_dir / "total.yaml", "wt") as fout:
            yaml.safe_dump(sum(counts.values()).__dict__, fout)

    return direction, counts


def filter_group(group_name: str, config: DictConfig):
    assert group_name in config, f"unknown data group {group_name}"
    executor = AutoExecutor(
        folder=Path(config.output_dir) / config.executor.log_folder / group_name,
        cluster=config.executor.cluster,
    )
    executor.update_parameters(
        slurm_partition=config.executor.slurm_partition,
        timeout_min=2880,
        nodes=1,
        cpus_per_task=16,
        mem_gb=48,
        name=f"filter.{group_name}",
    )
    logger.info(f"Filtering {group_name}")

    group_config = config.get(group_name)
    assert (
        group_config.included_corpora is None or group_config.excluded_corpora is None
    )
    data_conf_dir = Path(config.data_conf_dir)
    datasets = OmegaConf.load(
        data_conf_dir / "unfiltered_corpora" / f"{group_name}.yaml"
    )
    with open(data_conf_dir / "length_factors.yaml", "rt") as fin:
        length_factors = yaml.safe_load(fin)
    # submit all directions as part of the same array
    jobs = []
    with executor.batch():
        for direction, corpora in datasets.items():
            if direction not in config.directions:
                continue

            if direction not in ["eng_Latn-pol_Latn", "eng_Latn-ces_Latn", "eng_Latn-rus_Cyrl"]:  # If we change the config.directions we change the output directory...
                continue

            try:
                src, tgt = direction.split("-")
            except ValueError:  # this is monolingual data
                src = direction
                tgt = None

            assert group_config.included_corpora or group_config.excluded_corpora
            # select the datasets we want to include
            if group_config.included_corpora is not None:
                group_datasets = {
                    corpus_name: dataset
                    for corpus_name, dataset in corpora.items()
                    if corpus_name in group_config.included_corpora
                }
            else:
                group_datasets = {
                    corpus_name: dataset
                    for corpus_name, dataset in corpora.items()
                    if corpus_name not in group_config.excluded_corpora
                }
            if not group_datasets:
                logger.warning(f"Skipping empty {group_name}.{direction}")
                continue
            assert "total" not in group_datasets.keys(), "'total' is a reserved name"

            dataset_output_dir = Path(config.output_dir) / group_name / direction
            os.makedirs(dataset_output_dir, exist_ok=True)
            logger.info(f"Preparing {group_name}.{direction} job")

            # TODO use the stopes launcher + async
            job = executor.submit(
                filter_direction,
                group_name=group_name,
                src_lang=src,
                tgt_lang=tgt,
                datasets=group_datasets,
                length_factors=length_factors,
                config=config.get(group_name),
                dataset_output_dir=dataset_output_dir,
                custom_step_name=f"{group_name}.{direction}",
                output_dir=Path(config.output_dir),
                wandb_run_name=config.wandb_run_name
            )
            jobs.append(job)
    logger.info(f"All jobs for {group_name} have been scheduled")
    _ = [job.result() for job in jobs]
    logger.info(f"All jobs for {group_name} are done")


@hydra.main(config_path="conf", config_name="example")
def main(config: DictConfig) -> None:
    config.wandb_run_name = f"slavic_filtering_{random.randint(0, 1000000)}"
    config.data_conf_dir = (Path(__file__).parent / "filter_configs").resolve()  # Default config path we use everywhere.
    directions_path = config.directions[0]
    with open(directions_path, "rt") as fin:
        all_directions = yaml.safe_load(fin)
        # Values contain the number of sentences for directions which we don't need anymore they were just used for sorting (optimization)
        all_directions = list(all_directions.keys())

    assert config.train_primary is not None, "train_primary config is required"
    included_corpora_path = config.train_primary.included_corpora[0]
    with open(included_corpora_path, "rt") as fin:
        corpora = yaml.safe_load(fin)
    config.train_primary.included_corpora = corpora

    if config.train_mined is not None:
        included_corpora_path = config.train_mined.included_corpora[0]
        with open(included_corpora_path, "rt") as fin:
            corpora = yaml.safe_load(fin)
        config.train_mined.included_corpora = corpora

    if config.train_bt is not None:
        included_corpora_path = config.train_bt.included_corpora[0]
        with open(included_corpora_path, "rt") as fin:
            corpora = yaml.safe_load(fin)
        config.train_bt.included_corpora = corpora

    os.makedirs(config.output_dir, exist_ok=True)
    logger.info(f"Running with config:\n{OmegaConf.to_yaml(config)}")
    with open(Path(config.output_dir) / "config.yaml", "wt") as fout:
        fout.write(OmegaConf.to_yaml(config, sort_keys=True))

    # If you want to run this locally we have to batch jobs otherwise the number of processes overwhelms the machine and leads to crashes
    if config.executor.slurm_partition is None:
        batch_size = 16  # TODO: set this to a reasonable value (number of cores on your CPU)
        root_output_dir = config.output_dir
        for i in range(0, len(all_directions), batch_size):
            config.directions = all_directions[i : i + batch_size]
            config.output_dir = os.path.join(root_output_dir, f"batch_{i}_to_{i+len(config.directions)}")
            os.makedirs(config.output_dir, exist_ok=True)
            for group_name in ("train_mined", "train_bt"):  # TODO: TMP REMOVED "train_primary", 
                if config.get(group_name, None):
                    filter_group(group_name=group_name, config=config)

        logger.info(f"All jobs done – data written to {root_output_dir}")
    else:
        for group_name in ("train_primary", "train_mined", "train_bt"):
            if config.get(group_name, None):
                filter_group(group_name=group_name, config=config)

        logger.info(f"All jobs done – data written to {config.output_dir}")

    bad_corpora = []
    for root_dir, _, files in os.walk(config.output_dir):
        for file in files:
            if file != "total.yaml" and file != "config.yaml" and file.endswith(".yaml"):
                with open(os.path.join(root_dir, file), "r") as fin:
                    yaml_info = yaml.safe_load(fin)
                    total_after = yaml_info["total_after"]
                    if total_after == 0:
                        bad_corpora.append(os.path.join(root_dir, file))
    if len(bad_corpora) > 0:
        print('*' * 80)
        print(f"Found {len(bad_corpora)} empty corpora: {bad_corpora}")
        print('*' * 80)

if __name__ == "__main__":
    main()
