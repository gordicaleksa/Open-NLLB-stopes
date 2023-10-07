#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import glob
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import time
from typing import Dict, List, Optional, Set
from threading import Lock

from datasketch import MinHash, MinHashLSH
import xxhash

from stopes.core import utils
from stopes.pipelines.filtering.dataset import Dataset, DatasetLine, DatasetReader
from stopes.pipelines.filtering.filters.base import Filter, FilteringCounts
from stopes.pipelines.monolingual.utils.text_normalizer import normalize_for_dedup

from stopes.pipelines.filtering.filters.file_chunker_utils_bitext import BitextChunker, find_offsets, find_offsets_given_line_numbers, build_line_number_to_byte_offset_map, convert_offsets_to_line_numbers

class DedupFilter(Filter):
    def __init__(
        self,
        dedup_pairs: bool,
        max_source_dedup: Optional[int],
        max_target_dedup: Optional[int],
        shared_memory: bool = False,
        mp_dict: any = None,
        lock: Lock = None,
    ):
        self.shared_memory = shared_memory
        if shared_memory:
            assert mp_dict is not None, "shared_memory requires mp_dict"
            assert lock is not None, "shared_memory requires lock"
            self.lock = lock

        # pair deduplication
        self.dedup_pairs = dedup_pairs
        self.seen_pair_hashes = mp_dict if shared_memory else set()

        # source-side deduplication
        self.source_dedup = max_source_dedup
        self.source_dup_counts: Dict[int, int] = Counter()

        # target-side deduplication
        self.target_dedup = max_target_dedup
        self.target_dup_counts: Dict[int, int] = Counter()

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:

        if line.tgt is not None and (self.dedup_pairs or self.target_dedup):
            normalized_tgt = normalize_for_dedup(line.tgt)

        if self.dedup_pairs or self.source_dedup:
            normalized_src = normalize_for_dedup(line.src)

        if self.dedup_pairs:
            normalized = str(normalized_src)
            if line.tgt is not None:
                normalized += f"\t{normalized_tgt}"
            line_hash = xxhash.xxh3_64_intdigest(normalized)

            if self.shared_memory:
                with self.lock:
                    if line_hash in self.seen_pair_hashes:
                        counts.pair_dedup += 1
                        return None
                    self.seen_pair_hashes[line_hash] = 1
            else:
                if line_hash in self.seen_pair_hashes:
                    counts.pair_dedup += 1
                    return None
                self.seen_pair_hashes.add(line_hash)

        if self.target_dedup and line.tgt is not None:
            line_hash = xxhash.xxh3_64_intdigest(normalized_tgt)
            if self.target_dup_counts[line_hash] >= self.target_dedup:
                counts.target_dedup += 1
                return None
            self.target_dup_counts[line_hash] += 1

        if self.source_dedup:
            line_hash = xxhash.xxh3_64_intdigest(normalized_src)
            if self.source_dup_counts[line_hash] >= self.source_dedup:
                counts.source_dedup += 1
                return None
            self.source_dup_counts[line_hash] += 1

        return line


def yield_minhashes(minhashes_directory):
    query = os.path.join(minhashes_directory, "*minhashes_*.pkl")
    minhash_files = list(glob.glob(query))
    minhash_files.sort(key=lambda x: int(re.search(r'minhashes_(\d+)\.pkl', x).group(1)))

    for file in minhash_files:
        data = pickle.load(open(file, "rb"))
        for offset, minhash in data:
            yield (offset, minhash)


def get_shingle_set(text: str, k: int = 5):
    shingle_set = []
    for i in range(len(text) - k+1):
        shingle_set.append(text[i:i+k])
    return set(shingle_set)


def compute_minhash_worker(output_dir, src_path, tgt_path, src_offset, tgt_offset, global_offset, line_numbers_covered_range, num_perms):
    mhs = []

    # Caching.
    if os.path.exists(os.path.join(output_dir, f"minhashes_{global_offset + line_numbers_covered_range[0]}.pkl")):
        return

    with BitextChunker(src_path, tgt_path, src_offset, tgt_offset) as line_iterator:
        for i, line in enumerate(line_iterator):
            mh = MinHash(num_perm=num_perms)

            for shingle in get_shingle_set(normalize_for_dedup(line.src) + " " + normalize_for_dedup(line.tgt)):
                mh.update(shingle.encode('utf8'))

            mhs.append((global_offset + line_numbers_covered_range[0] + i, mh))

    if len(mhs) == line_numbers_covered_range[1] - line_numbers_covered_range[0]:
        minhashes_file = os.path.join(output_dir, f"minhashes_{global_offset + line_numbers_covered_range[0]}.pkl")
        pickle.dump(mhs, open(minhashes_file, "wb"))
    else:
        minhashes_file = os.path.join(output_dir, f"error_minhashes_{global_offset + line_numbers_covered_range[0]}.pkl")
        pickle.dump(mhs, open(minhashes_file, "wb"))


class FuzzyDedupFilter(Filter):
    MAX_NUM_LINES = '999999999'  # arbitrary setting, but should be enough for most datasets

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        num_bands: int = 10,
        subvector_size: int = 10,
        num_perms: int = 100,
        threshold: float = None,
        num_workers: int = 10,
        output_dir: str = None,
        debug: bool = False,
    ):
        self.debug = debug
        if self.debug:
            current_path = os.path.dirname(os.path.realpath(__file__))
            self.debug_file = open(os.path.join(current_path, "debug_fuzzy_dedup_filter.txt"), "a")
            self.dataset_root_path = os.path.dirname(next(iter(datasets.items()))[1].src)

        self.num_perms = num_perms
        if threshold:
            self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perms)
        else:
            self.lsh = MinHashLSH(params=[num_bands, subvector_size], num_perm=num_perms)

        self.build_lsh_parallel(self.lsh, output_dir, datasets, num_perms, num_workers)

    def build_lsh_parallel(self, lsh, output_dir, datasets, num_perms, num_workers):
        os.makedirs(output_dir, exist_ok=True)
        global_offset = 0

        if self.debug:
            self.index_to_dataset = {}
            self.dataset_to_line_num_byte_map_src = {}
            self.dataset_to_line_num_byte_map_tgt = {}

        for i, (_, dataset) in enumerate(datasets.items()):
            num_lines = utils.count_lines(dataset.src)
            if num_lines == 0:
                print(f'Skipping {i+1}-th dataset: {dataset.src} because it has 0 lines.')
                continue

            if self.debug:
                self.index_to_dataset[global_offset] = dataset
                self.dataset_to_line_num_byte_map_src[os.path.split(dataset.src)[-1]] = build_line_number_to_byte_offset_map(dataset.src)
                self.dataset_to_line_num_byte_map_tgt[os.path.split(dataset.tgt)[-1]] = build_line_number_to_byte_offset_map(dataset.tgt)

            print(f'Computing minhashes for {i+1}-th dataset: {dataset.src}.')
            # Modify the number of workers dynamically based on the number of lines in the dataset using the passed value as the upper bound.
            num_workers_dynamic = min(((num_lines - 1) // 5000) + 1, num_workers)

            src_offsets = find_offsets(dataset.src, num_workers_dynamic)
            src_file_chunks = list(zip(src_offsets, src_offsets[1:]))
            src_chunks_line_numbers = convert_offsets_to_line_numbers(src_offsets, dataset.src)
            tgt_offsets = find_offsets_given_line_numbers(dataset.tgt, src_chunks_line_numbers)
            # Below 2 lines can be removed they are just there to catch potential edge cases / bugs.
            tgt_chunks_line_numbers = convert_offsets_to_line_numbers(tgt_offsets, dataset.tgt)
            assert tgt_chunks_line_numbers == src_chunks_line_numbers, f"src and tgt have different number of lines for {dataset.src} and {dataset.tgt}"
            tgt_file_chunks = list(zip(tgt_offsets, tgt_offsets[1:]))
            src_chunks_line_numbers = list(zip(src_chunks_line_numbers, src_chunks_line_numbers[1:]))

            with ProcessPoolExecutor(max_workers=num_workers_dynamic) as executor:
                futures = [
                    executor.submit(
                        compute_minhash_worker, output_dir, dataset.src, dataset.tgt, src_offset, tgt_offset, global_offset, line_numbers_covered_range, num_perms)
                    for (src_offset, tgt_offset, line_numbers_covered_range) in zip(src_file_chunks, tgt_file_chunks, src_chunks_line_numbers)
                ]
                # with tqdm(total=num_chunks) as pbar:
                #     for _ in concurrent.futures.as_completed(futures):
                #         pbar.update(1)
                for future in futures:
                    future.result()

            global_offset += num_lines

        cnts = []

        for cnt, mh in yield_minhashes(output_dir):
            cnts.append(cnt)
            lsh.insert(f'{str(cnt).zfill(len(self.MAX_NUM_LINES))}', mh)

        test = list(range(len(cnts)))
        assert test == cnts, f'minhash corrupt'

    def build_lsh_sequential(self, lsh, datasets, num_perms):
        cnt = 0
        for corpus_name, dataset in datasets.items():
            with DatasetReader(dataset, corpus_name) as inputs:
                for line in inputs:
                    mh = MinHash(num_perm=num_perms)

                    for shingle in get_shingle_set(normalize_for_dedup(line.src) + " " + normalize_for_dedup(line.tgt)):
                        mh.update(shingle.encode('utf8'))

                    lsh.insert(f'{str(cnt).zfill(len(self.MAX_NUM_LINES))}', mh)
                    cnt += 1

    def fuzzy_debug_subroutine(self, normalized_line, result, i):
        self.debug_file.write(f"QUERY-{normalized_line}" + "\n")
        for ii in result:
            if int(ii) != i:
                candidate_line = None
                for key in sorted(self.index_to_dataset.keys(), reverse=True):
                    if int(ii) < key:
                        pass
                    else:
                        dataset = self.index_to_dataset[key]
                        line_num_byte_map_src = self.dataset_to_line_num_byte_map_src[os.path.split(dataset.src)[-1]]
                        line_num_byte_map_tgt = self.dataset_to_line_num_byte_map_tgt[os.path.split(dataset.tgt)[-1]]
                        line_offset_index = int(ii) - key
                        src_byte_offset = line_num_byte_map_src[line_offset_index]
                        tgt_byte_offset = line_num_byte_map_tgt[line_offset_index]
                        with open(dataset.src, 'r') as f, open(dataset.tgt, 'r') as f_tgt:
                            f.seek(src_byte_offset)
                            f_tgt.seek(tgt_byte_offset)
                            candidate_line = normalize_for_dedup(f.readline())
                            candidate_line += " "
                            candidate_line += normalize_for_dedup(f_tgt.readline())
                        break
                assert candidate_line is not None
                self.debug_file.write(f"CANDIDATE-{ii}-{candidate_line}" + "\n")
        self.debug_file.write("\n\n")
        self.debug_file.flush()

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        # a hacky way to pass in the line number in order not to break the function signature
        i = int(line.src[:len(self.MAX_NUM_LINES)])

        line.src = line.src[len(self.MAX_NUM_LINES):]  # remove the line number from the beginning of the line

        mh = MinHash(num_perm=self.num_perms)

        normalized_line = normalize_for_dedup(line.src) + " " + normalize_for_dedup(line.tgt)
        for shingle in get_shingle_set(normalized_line):
            mh.update(shingle.encode('utf8'))

        result = self.lsh.query(mh)

        for index in result:
            if int(index) != i:
                if self.debug:
                    self.fuzzy_debug_subroutine(normalized_line, result, i)

                self.lsh.remove(f'{str(i).zfill(len(self.MAX_NUM_LINES))}')  # Remove the current line from the LSH index
                counts.pair_fuzzy_dedup += 1
                return None

        return line