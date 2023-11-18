#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

from stopes.pipelines.filtering.first_stage_filtering import FirstStage
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


def get_step(steps_so_far, dataset_counts):
    return steps_so_far + dataset_counts.total_before


def first_stage_preprocess(dataset, corpus_name, dataset_output_dir, num_workers=12, global_exact_dedup=False):
    num_lines = utils.count_lines(dataset.src)
    if num_lines == 0:
        return -1, -1, -1
    num_workers_dynamic = min(((num_lines - 1) // 5000) + 1, num_workers)

    src_offsets_path = dataset_output_dir / f"{corpus_name}_src_offsets{'_global_exact_dedup' if global_exact_dedup else ''}.pickle"
    if os.path.exists(src_offsets_path):
        with open(src_offsets_path, "rb") as f:
            src_offsets = pickle.load(f)
    else:
        src_offsets = find_offsets(dataset.src, num_workers_dynamic)
        with open(src_offsets_path, "wb") as f:
            pickle.dump(src_offsets, f)

    src_file_chunks = list(zip(src_offsets, src_offsets[1:]))

    src_chunks_line_numbers_path = dataset_output_dir / f"{corpus_name}_src_chunks_line_numbers{'_global_exact_dedup' if global_exact_dedup else ''}.pickle"
    if os.path.exists(src_chunks_line_numbers_path):
        with open(src_chunks_line_numbers_path, "rb") as f:
            src_chunks_line_numbers = pickle.load(f)
    else:
        src_chunks_line_numbers = convert_offsets_to_line_numbers(src_offsets, dataset.src)
        with open(src_chunks_line_numbers_path, "wb") as f:
            pickle.dump(src_chunks_line_numbers, f)

    tgt_offsets_path = dataset_output_dir / f"{corpus_name}_tgt_offsets{'_global_exact_dedup' if global_exact_dedup else ''}.pickle"
    if os.path.exists(tgt_offsets_path):
        with open(tgt_offsets_path, "rb") as f:
            tgt_offsets = pickle.load(f)
    else:
        tgt_offsets = find_offsets_given_line_numbers(dataset.tgt, src_chunks_line_numbers)
        with open(tgt_offsets_path, "wb") as f:
            pickle.dump(tgt_offsets, f)

    tgt_chunks_line_numbers_path = dataset_output_dir / f"{corpus_name}_tgt_chunks_line_numbers{'_global_exact_dedup' if global_exact_dedup else ''}.pickle"
    if os.path.exists(tgt_chunks_line_numbers_path):
        with open(tgt_chunks_line_numbers_path, "rb") as f:
            tgt_chunks_line_numbers = pickle.load(f)
    else:
        tgt_chunks_line_numbers = convert_offsets_to_line_numbers(tgt_offsets, dataset.tgt)
        with open(tgt_chunks_line_numbers_path, "wb") as f:
            pickle.dump(tgt_chunks_line_numbers, f)

    assert tgt_chunks_line_numbers == src_chunks_line_numbers, f"src and tgt have different number of lines for {dataset.src} and {dataset.tgt}"

    tgt_file_chunks = list(zip(tgt_offsets, tgt_offsets[1:]))

    return src_file_chunks, tgt_file_chunks, num_workers_dynamic


def first_stage_postprocess(
        dataset_output_dir,
        corpus_name,
        src_lang,
        tgt_lang,
        path_out_src_before_fuzzy,
        path_out_tgt_before_fuzzy,
        path_counts,
        global_exact_dedup
    ):
    src_files_list = sorted([str(path) for path in dataset_output_dir.glob(f"{corpus_name}.{src_lang}_before_fuzzy_*{'_global_exact_dedup' if global_exact_dedup else ''}")], key=lambda x: int(x.split('_')[-4]))
    tgt_files_list = sorted([str(path) for path in dataset_output_dir.glob(f"{corpus_name}.{tgt_lang}_before_fuzzy_*{'_global_exact_dedup' if global_exact_dedup else ''}")], key=lambda x: int(x.split('_')[-4]))
    assert len(src_files_list) == len(tgt_files_list), f"Number of src files {len(src_files_list)} is not equal to number of tgt files {len(tgt_files_list)}"

    # Merge output files from the workers
    src_cnt = 0
    tgt_cnt = 0
    with open(path_out_src_before_fuzzy, "w") as outfile_src, open(path_out_tgt_before_fuzzy, "w") as outfile_tgt:
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

    counts_files_list = [str(path) for path in dataset_output_dir.glob(f"{corpus_name}_before_fuzzy_*{'_global_exact_dedup' if global_exact_dedup else ''}.yaml")]
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
    num_workers: int = 12
):
    direction = f"{src_lang}-{tgt_lang}" if tgt_lang is not None else src_lang
    wandb.init(project="slavic-NLLB-filtering", reinit=False, name=wandb_run_name)
    print(f"Filtering {group_name}.{direction}")

    counts: Dict[str, FilteringCounts] = {}
    fuzzy_datasets = {}
    # 1st stage of filtering: everything except for fuzzy deduplication.
    timings = []
    global_exact_dedup = True
    # if os.path.exists(os.path.join(dataset_output_dir, 'global_exact_dedup_dictionary_keys.pkl')):
    #     with open(os.path.join(dataset_output_dir, 'global_exact_dedup_dictionary_keys.pkl'), 'rb') as f:
    #         dedup_dict_keys = pickle.load(f)
    #         # add 1 as values for all the keys
    #         dedup_dict = {key: 1 for key in dedup_dict_keys}
    #         dedup_dict = multiprocessing.Manager().dict(dedup_dict)
    # else:
    dedup_dict = multiprocessing.Manager().dict()

    for corpus_name, dataset in datasets.items():

        if global_exact_dedup:
            dataset.src = dataset_output_dir / f"{corpus_name}.{src_lang}_before_fuzzy"
            dataset.tgt = dataset_output_dir / f"{corpus_name}.{tgt_lang}_before_fuzzy"

        path_out_src_before_fuzzy = dataset_output_dir / f"{corpus_name}.{src_lang}_before_fuzzy{'_global_exact_dedup' if global_exact_dedup else ''}"
        path_out_tgt_before_fuzzy = dataset_output_dir / f"{corpus_name}.{tgt_lang}_before_fuzzy{'_global_exact_dedup' if global_exact_dedup else ''}"

        fuzzy_datasets[corpus_name] = Dataset(src=path_out_src_before_fuzzy, tgt=path_out_tgt_before_fuzzy, tsv=None, metadata=None, lang_dir=None, fold=None)

        path_counts = dataset_output_dir / f"{corpus_name}_before_fuzzy{'_global_exact_dedup' if global_exact_dedup else ''}.yaml"

        if os.path.isfile(path_counts):
            with open(path_counts, "rt") as fin:
                counts[corpus_name] = FilteringCounts(**yaml.safe_load(fin))
            print(f"Skipping {corpus_name} as a corresponding YAML file already exists")
            continue

        # steps_so_far = sum([el.total_before for el in counts.values()])
        ts = time.time()

        src_file_chunks, tgt_file_chunks, num_workers_dynamic = first_stage_preprocess(
            dataset,
            corpus_name,
            dataset_output_dir,
            num_workers,
            global_exact_dedup=True
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
                global_exact_dedup=True,
            ).run()
            timings.append(time.time() - ts)

        counts[corpus_name] = first_stage_postprocess(
            dataset_output_dir,
            corpus_name,
            src_lang,
            tgt_lang,
            path_out_src_before_fuzzy,
            path_out_tgt_before_fuzzy,
            path_counts,
            global_exact_dedup=global_exact_dedup
        )

    # Optimization - if all the datasets are already filtered, skip the fuzzy deduplication
    skip_fuzzy = True
    for corpus_name, dataset in fuzzy_datasets.items():
        path_counts = dataset_output_dir / f"{corpus_name}.yaml"
        if not os.path.isfile(path_counts):
            skip_fuzzy = False
            break

    if skip_fuzzy:
        print(f"Skipping fuzzy deduplication for {group_name}.{direction} as all the datasets are already fuzzy filtered")
        return direction, counts

    lines_left_to_process = 0
    for corpus_name, dataset in fuzzy_datasets.items():
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

    # 2nd stage: fuzzy deduplication
    # It's stateful so we have to do it before the loop - we're doing fuzzy across all datasets for this lang direction.
    # If the dataset is too big we repeat once more on the filtered smaller dataset from the 1st iteration.
    fuzzy_filter = hydra.utils.instantiate(
        config.fuzzy_dedup_filter,
        datasets=fuzzy_datasets,
        num_workers=num_workers,
        output_dir=Path(output_dir) / f"{group_name.split('_')[-1]}_minhashes_{direction}",
        attempt_num=0)
    # too little RAM for complete dedup, repeat the fuzzy deduplication to approximate the full deduplication
    repeat_fuzzy = fuzzy_filter.should_repeat_fuzzy
    for i in range(2 if repeat_fuzzy else 1):
        cnt = 0
        control_flag = repeat_fuzzy and i == 0

        if i == 1 and repeat_fuzzy:  # Update the state
            shutil.rmtree(Path(output_dir) / f"{group_name.split('_')[-1]}_minhashes_{direction}")
            fuzzy_filter = hydra.utils.instantiate(
                config.fuzzy_dedup_filter,
                datasets=fuzzy_datasets,  # Updated in the first iteration
                num_workers=num_workers,
                output_dir=Path(output_dir) / f"{group_name.split('_')[-1]}_minhashes_{direction}",
                attempt_num=1)

        for corpus_name, dataset in fuzzy_datasets.items():
            dataset_counts = counts[corpus_name]
            if i == 1 and repeat_fuzzy:
                dataset_counts.total_after_fuzzy = 0  # reset the counter

            path_out_src = dataset_output_dir / f"{corpus_name}.{src_lang}{'_1_fuzzy' if control_flag else '.gz'}"
            path_out_tgt = dataset_output_dir / f"{corpus_name}.{tgt_lang}{'_1_fuzzy' if control_flag else '.gz'}"

            path_counts = dataset_output_dir / f"{corpus_name}{'_1' if control_flag else ''}.yaml"

            if os.path.isfile(path_counts):
                with open(path_counts, "rt") as fin:
                    counts[corpus_name] = FilteringCounts(**yaml.safe_load(fin))
                    cnt += counts[corpus_name].total_after
                print(f"Skipping {corpus_name} as a corresponding YAML file already exists")
                continue

            print(f"Processing {corpus_name} - fuzzy deduplication")
            with ExitStack() as outputs, DatasetReader(dataset, corpus_name) as inputs:
                fout_src = outputs.enter_context(open(path_out_src, "wt") if control_flag else gzip.open(path_out_src, "wt"))
                fout_tgt = None

                # if tgt_lang is not provided, we have a monolingual dataset;
                # otherwise, the parallel file needs to be opened
                if tgt_lang is not None:
                    fout_tgt = outputs.enter_context(open(path_out_src, "wt") if control_flag else gzip.open(path_out_tgt, "wt"))

                for i, line in enumerate(inputs):
                    if cnt % 10000 == 0:
                        wandb.log(
                            {f"{group_name.split('_')[-1]}_fuzzy/{direction}": cnt / dataset_counts.total_after},
                            step=cnt
                        )

                    # hacky - in order not to break the existing func signature I'm encoding the line number information in the line.src
                    line.src = f'{str(cnt).zfill(9)}{line.src}'
                    line = fuzzy_filter.filter_line(line, dataset_counts)
                    cnt += 1  # this needs to be precisely here, don't move it unless you know why you are doing that.

                    if line is None:
                        continue

                    fout_src.write(line.src + "\n")
                    if fout_tgt is not None:
                        fout_tgt.write(line.tgt + "\n")
                    dataset_counts.total_after_fuzzy += 1

                if dataset_counts:
                    with open(path_counts, "wt") as fout:
                        yaml.safe_dump(dataset_counts.__dict__, fout)

            # update the dataset path so that the output from this iteration is the input to the next
            # modify the dataset.src and dataset.tgt paths
            if control_flag:
                dataset.src = path_out_src
                dataset.tgt = path_out_tgt

            if repeat_fuzzy and i == 1:  # delete iteration 1 intermediate files
                os.remove(dataset.src)
                os.remove(dataset.tgt)
                os.remove(dataset_output_dir / f"{corpus_name}_1.yaml")

            # TODO(gordicaleksa): maybe if fuzzy finished successfully delete the before_fuzzy files

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
    for direction, corpora in datasets.items():
        with executor.batch():
            if direction not in config.directions:
                continue

            if direction not in ["eng_Latn-slk_Latn"]:
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
        jobs[0].result()
        jobs = []
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
            for group_name in ("train_mined", "train_bt"):  # "train_primary", 
                if config.get(group_name, None):
                    filter_group(group_name=group_name, config=config)

        logger.info(f"All jobs done – data written to {root_output_dir}")
    else:
        for group_name in ("train_mined", "train_bt"):  # TODO: TMP HACK: "train_primary", 
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
