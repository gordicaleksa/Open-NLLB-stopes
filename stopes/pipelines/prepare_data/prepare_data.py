# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from collections import defaultdict
import logging
import os
import shutil
import tarfile
import typing as tp
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import requests

from stopes.core import utils
from stopes.pipelines.filtering.dataset import Dataset
from stopes.pipelines.prepare_data.binarize import binarize
from stopes.pipelines.prepare_data.build_vocab import build_vocab
from stopes.pipelines.prepare_data.configs import CorporaConfig, PrepareDataConfig
from stopes.pipelines.prepare_data.dedup_sharding import dedup_sharding
from stopes.pipelines.prepare_data.retrieve_data import retrieve_data
from stopes.pipelines.prepare_data.validate import validate

logger = logging.getLogger("prepare_data")


def sort_datasets_based_on_num_sentences(datasets, is_local_run):
    # if local run sort to make things faster otherwise we have too much data
    # and this would slow us down too much
    if is_local_run:
        dataset_num_sentences_tuples = []
        for dataset in datasets:
            src_path = dataset.src
            lang_direction = Path(src_path).parent.name
            src_cnt = utils.count_lines(src_path)
            tgt_path = dataset.tgt
            tgt_cnt = utils.count_lines(tgt_path)
            assert src_cnt == tgt_cnt, f"src and tgt have different number of lines for {src_path}"
            dataset_num_sentences_tuples.append((dataset, src_cnt, lang_direction))
        dataset_num_sentences_tuples.sort(key=lambda x: x[1])
        datasets = [dataset for dataset, _, _ in dataset_num_sentences_tuples]
        return datasets, dataset_num_sentences_tuples
    else:
        return datasets, None


def maybe_sort(datasets_with_metadata, is_local_run):
    if is_local_run:
        lang_direction_num_sentences = defaultdict(int)
        for _, num_sentences, lang_direction in datasets_with_metadata:
            lang_direction_num_sentences[lang_direction] += num_sentences
        datasets_with_metadata.sort(key=lambda x: lang_direction_num_sentences[x[2]])
        return [dataset for dataset, _, _ in datasets_with_metadata]
    else:
        return datasets_with_metadata


class PrepareData:
    def __init__(self, config: PrepareDataConfig):
        self.config = config
        self.ensure_all_dirs()
        # Cache won't be re-used if you change the output_dir.
        self.config.launcher.cache.caching_dir = Path(self.output_dir) / "cache"
        self.launcher = hydra.utils.instantiate(self.config.launcher)
        unsorted_datasets = self._get_datasets(self.config.corpora)
        # Sort the datasets according to the number of sentences in them.
        # For the local runs where we batch the datasets this will speed up thing by quite a lot.
        datasets, datasets_with_metadata = sort_datasets_based_on_num_sentences(unsorted_datasets, is_local_run=self.launcher.partition is None)
        self.datasets = datasets
        self.datasets_with_metadata = datasets if datasets_with_metadata is None else datasets_with_metadata
        self._check_files_exist(self.datasets)
        OmegaConf.save(
            config=config,
            f=str(self.output_dir / "prepare_data.yaml"),
        )
        OmegaConf.set_readonly(self.config, True)

    async def run(self):
        train_src_counts_map, train_tgt_counts_map, train_counts_map = await validate(
            self.datasets, self.launcher
        )
        self.datasets = maybe_sort(self.datasets_with_metadata, is_local_run=self.launcher.partition is None)
        retrieved_datasets: tp.List[Dataset] = await retrieve_data(
            self.datasets,
            self.config.preprocessing,
            self.launcher,
            self.retrieved_data_dir,
        )
        (src_vocab, tgt_vocab), (
            sharded_train_datasets,
            retrieved_eval_datasets,
            max_num_shards,
        ) = await asyncio.gather(
            build_vocab(
                retrieved_datasets,
                self.config.vocab,
                train_src_counts_map,
                train_tgt_counts_map,
                self.launcher,
                self.vocab_dir,
            ),
            dedup_sharding(
                retrieved_datasets,
                train_counts_map,
                self.config.dedup,
                self.config.sharding,
                self.launcher,
                self.tmp_dir / "sharded",
            ),
        )
        await binarize(
            sharded_train_datasets,
            retrieved_eval_datasets,
            src_vocab,
            tgt_vocab,
            max_num_shards,
            self.config.sharding.binarize_num_workers,
            train_src_counts_map,
            train_tgt_counts_map,
            self.launcher,
            self.tmp_dir / "binarized",
            self.data_bin,
        )

        # Delete tmp_dir.
        shutil.rmtree(self.tmp_dir)

    def ensure_all_dirs(self):
        self.output_dir = Path(self.config.output_dir).resolve()
        self.retrieved_data_dir = self.output_dir / "retrieved_data"
        self.vocab_dir = self.output_dir / "vocab_bin"
        self.data_bin = self.output_dir / "data_bin"
        self.tmp_dir = self.output_dir / "tmp"
        utils.ensure_dir(self.output_dir)
        utils.ensure_dir(self.retrieved_data_dir)
        utils.ensure_dir(self.vocab_dir)
        utils.ensure_dir(self.data_bin)
        utils.ensure_dir(self.tmp_dir)
        utils.ensure_dir(self.tmp_dir / "sharded")
        utils.ensure_dir(self.tmp_dir / "binarized")

    @staticmethod
    def _get_datasets(
        corpora_conf: CorporaConfig,
    ) -> tp.List[Dataset]:
        datasets = []
        for fold in corpora_conf:
            if corpora_conf[fold]:
                for lang_dir in corpora_conf[fold]:
                    for corpus in corpora_conf[fold][lang_dir]:
                        src_file = corpora_conf[fold][lang_dir][corpus].src
                        tgt_file = corpora_conf[fold][lang_dir][corpus].tgt
                        metadata = getattr(
                            corpora_conf[fold][lang_dir][corpus], "metadata", None
                        )
                        dataset = Dataset(
                            src=src_file,
                            tgt=tgt_file,
                            metadata=metadata,
                            lang_dir=lang_dir,
                            fold=fold,
                        )
                        datasets.append(dataset)
        return datasets

    @staticmethod
    def _check_files_exist(datasets: tp.List[Dataset]):
        for dataset in datasets:
            assert Path(dataset.src).exists(), f"Nonexistent source path: {dataset.src}"
            assert Path(dataset.tgt).exists(), f"Nonexistent target path: {dataset.tgt}"


def download_spm200():
    # Get path to the current file
    target_directory = Path(__file__).resolve().parent / "spm_models"
    urls_dict = [
        ("https://tinyurl.com/flores200sacrebleuspm", target_directory / "flores200_sacrebleu_tokenizer_spm.model"),
        ("https://tinyurl.com/nllb200dictionary", target_directory / "flores200_sacrebleu_tokenizer_spm.dict.txt"),
    ]
    for url, path in urls_dict:
        if not os.path.exists(path):
            os.makedirs(target_directory, exist_ok=True)
            response = requests.get(url)
            if not response.ok:
                raise Exception(f"Could not download from {url}!")
            open(path, "wb").write(response.content)
            print(f"Wrote: {path}")


@hydra.main(config_path="conf", config_name="prepare_data")
def main(config: PrepareDataConfig) -> None:
    if config.vocab.src_vocab.pretrained or config.vocab.tgt_vocab.pretrained:
        # Download SPM-200 model and dictionary automatically. We assume that's the tokenizer you want to use.
        download_spm200()

    # For readability purposes we pass in a path instead of dumping the data directly into prepare_data.yaml.
    train_corpora_path = config.corpora.train[0]
    assert isinstance(train_corpora_path, str)
    with open(train_corpora_path, "rt") as fin:
        train_corpora = OmegaConf.load(fin)
    config.corpora.train = train_corpora

    pipeline = PrepareData(config)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
