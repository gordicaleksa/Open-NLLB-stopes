#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import gzip
import itertools
import logging
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Optional

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from submitit import AutoExecutor

from stopes.pipelines.filtering.configs import (
    FilterConfig,
    GroupFilterConfig,
    register_configs,
)
from stopes.pipelines.filtering.dataset import Dataset, DatasetLine, DatasetReader
from stopes.pipelines.filtering.filters import FilteringCounts
from stopes.pipelines.filtering.utils import cache_step_sync, normalize_unicode
from stopes.pipelines.monolingual.utils.text_normalizer import replace_unicode_punct, remove_non_printing_char

logger = logging.getLogger(__name__)


register_configs()


def balance_quotation_marks(line):
    if line.startswith('"') and not line.endswith('"'):
        line = line[1:]

    if not line.startswith('"') and line.endswith('"'):
        line = line[:-1]

    return line


def normalize_line(line):
    line = line.strip()
    line = replace_unicode_punct(line)
    line_tmp = remove_non_printing_char(line)
    return balance_quotation_marks(line_tmp)


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
):
    direction = f"{src_lang}-{tgt_lang}" if tgt_lang is not None else src_lang
    print(f"Filtering {group_name}.{direction}")

    # build the list of filters to be applied to this direction
    filters = [
        hydra.utils.instantiate(config.laser_filter),
        hydra.utils.instantiate(
            config.length_filter,
            length_factors=length_factors,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        ),
        hydra.utils.instantiate(
            config.symbols_filter, keep_dates_and_numbers=group_name != "train_mined"
        ),
        hydra.utils.instantiate(
            config.lid_filter, src_lang=src_lang, tgt_lang=tgt_lang
        ),
        hydra.utils.instantiate(
            config.toxicity_filter, src_lang=src_lang, tgt_lang=tgt_lang
        ),
        hydra.utils.instantiate(config.dedup_filter),
        hydra.utils.instantiate(config.fuzzy_dedup_filter, datasets)
    ]

    # filter datasets sequentially
    counts: Dict[str, FilteringCounts] = {}
    for corpus_name, dataset in datasets.items():
        dataset_counts = FilteringCounts()  # filtering counts for the current dataset

        path_out_src = dataset_output_dir / f"{corpus_name}.{src_lang}.gz"
        path_out_tgt = dataset_output_dir / f"{corpus_name}.{tgt_lang}.gz"
        path_counts = dataset_output_dir / f"{corpus_name}.yaml"

        if os.path.isfile(path_counts):
            with open(path_counts, "rt") as fin:
                counts[corpus_name] = yaml.safe_load(fin)
            print(f"Skipping {corpus_name} as a corresponding YAML file already exists")
            continue

        print(f"Processing {corpus_name}")
        with ExitStack() as outputs, DatasetReader(dataset, corpus_name) as inputs:
            fout_src = outputs.enter_context(gzip.open(path_out_src, "wt"))
            fout_tgt = None

            # if tgt_lang is not provided, we have a monolingual dataset;
            # otherwise, the parallel file needs to be opened
            if tgt_lang is not None:
                fout_tgt = outputs.enter_context(gzip.open(path_out_tgt, "wt"))

            for i, line in enumerate(inputs):
                dataset_counts.total_before += 1

                if config.normalize_unicode:
                    line.src = normalize_unicode(line.src)
                    if line.tgt is not None:
                        line.tgt = normalize_unicode(line.tgt)

                if config.normalize_line:
                    line.src = normalize_line(line.src)
                    if line.tgt is not None:
                        line.tgt = normalize_line(line.tgt)

                # apply filters sequentially
                for fltr in filters:
                    if fltr is None:
                        continue

                    # hacky - in order not to break the existing func signature I'm encoding the line number information in the line.src
                    if fltr.__class__.__name__ == "FuzzyDedupFilter":
                        line.src = f'{str(i).zfill(9)}{line.src}'

                    line = fltr.filter_line(line, dataset_counts)
                    # no need to keep filtering if the line was already discarded
                    if line is None:
                        break
                if line is None:
                    continue

                fout_src.write(line.src + "\n")
                if fout_tgt is not None:
                    fout_tgt.write(line.tgt + "\n")
                dataset_counts.total_after += 1

            if dataset_counts:
                counts[corpus_name] = dataset_counts
                with open(path_counts, "wt") as fout:
                    yaml.safe_dump(dataset_counts.__dict__, fout)
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
            )
            jobs.append(job)
    logger.info(f"All jobs for {group_name} have been scheduled")
    _ = [job.result() for job in jobs]
    logger.info(f"All jobs for {group_name} are done")


@hydra.main(config_path="conf", config_name="example")
def main(config: DictConfig) -> None:
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
            for group_name in ("train_primary", "train_mined", "train_bt"):
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
