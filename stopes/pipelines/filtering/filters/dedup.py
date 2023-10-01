#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import time
from typing import Dict, List, Optional, Set

from datasketch import MinHash, MinHashLSH
import xxhash

from stopes.core import utils
from stopes.pipelines.filtering.dataset import Dataset, DatasetLine, DatasetReader
from stopes.pipelines.filtering.filters.base import Filter, FilteringCounts
from stopes.pipelines.monolingual.utils.text_normalizer import normalize_for_dedup

from stopes.pipelines.filtering.filters.file_chunker_utils_bitext import BitextChunker, find_offsets, find_offsets_given_line_numbers

class DedupFilter(Filter):
    def __init__(
        self,
        dedup_pairs: bool,
        max_source_dedup: Optional[int],
        max_target_dedup: Optional[int],
    ):
        # pair deduplication
        self.dedup_pairs = dedup_pairs
        self.seen_pair_hashes: Set[int] = set()

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


def convert_offsets_to_line_numbers(offsets, filename):
    with open(filename, 'r') as f:
        line_numbers_offsets = []
        cnt = 0
        for offset in offsets:
            # count the number of lines up to the offset
            while f.tell() < offset:
                f.readline()
                cnt += 1
            assert f.tell() == offset
            line_numbers_offsets.append(cnt)

        assert len(line_numbers_offsets) == len(offsets)
        assert line_numbers_offsets[0] == 0
        assert line_numbers_offsets[-1] == utils.count_lines(filename)

        return line_numbers_offsets


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
    ):
        self.num_perms = num_perms
        if threshold:
            self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perms)
        else:
            self.lsh = MinHashLSH(params=[num_bands, subvector_size], num_perm=num_perms)

        self.build_lsh_parallel(self.lsh, datasets, num_perms, num_workers)

    def build_lsh_parallel(self, lsh, datasets, num_perms, num_workers):
        global_offset = 0
        minhashes = []
        for _, dataset in datasets.items():
            num_lines = utils.count_lines(dataset.src)
            src_offsets = find_offsets(dataset.src, num_workers)
            src_file_chunks = list(zip(src_offsets, src_offsets[1:]))
            src_chunks_line_numbers = convert_offsets_to_line_numbers(src_offsets, dataset.src)
            tgt_offsets = find_offsets_given_line_numbers(dataset.tgt, src_chunks_line_numbers)
            # Below 2 lines can be removed they are just there to catch potential edge cases / bugs.
            tgt_chunks_line_numbers = convert_offsets_to_line_numbers(tgt_offsets, dataset.tgt)
            assert tgt_chunks_line_numbers == src_chunks_line_numbers, f"src and tgt have different number of lines for {dataset.src} and {dataset.tgt}"
            tgt_file_chunks = list(zip(tgt_offsets, tgt_offsets[1:]))

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        self.compute_minhash_worker, dataset.src, dataset.tgt, src_offset, tgt_offset, global_offset, file_offset, num_perms)
                    for (src_offset, tgt_offset, file_offset) in zip(src_file_chunks, tgt_file_chunks, src_chunks_line_numbers)
                ]
                # with tqdm(total=num_chunks) as pbar:
                #     for _ in concurrent.futures.as_completed(futures):
                #         pbar.update(1)
                for future in futures:
                    minhashes_partial = future.result()
                    minhashes.extend(minhashes_partial)

            global_offset += num_lines

        for cnt, mh in minhashes:
            lsh.insert(f'{str(cnt).zfill(len(self.MAX_NUM_LINES))}', mh)

    def build_lsh_sequential(self, lsh, datasets, num_perms):
        cnt = 0
        for corpus_name, dataset in datasets.items():
            with DatasetReader(dataset, corpus_name) as inputs:
                for line in inputs:
                    mh = MinHash(num_perm=num_perms)

                    for shingle in self.get_shingle_set(normalize_for_dedup(line.src) + " " + normalize_for_dedup(line.tgt)):
                        mh.update(shingle.encode('utf8'))

                    lsh.insert(f'{str(cnt).zfill(len(self.MAX_NUM_LINES))}', mh)
                    cnt += 1

    def compute_minhash_worker(self, src_path, tgt_path, src_offset, tgt_offset, global_offset, document_offset, num_perms):
        mhs = []
        with BitextChunker(src_path, tgt_path, src_offset, tgt_offset) as line_iterator:
            for i, line in enumerate(line_iterator):
                mh = MinHash(num_perm=num_perms)

                for shingle in self.get_shingle_set(normalize_for_dedup(line.src) + " " + normalize_for_dedup(line.tgt)):
                    mh.update(shingle.encode('utf8'))

                mhs.append((global_offset + document_offset + i, mh))
        return mhs

    def get_shingle_set(self, text: str, k: int = 5):
        shingle_set = []
        for i in range(len(text) - k+1):
            shingle_set.append(text[i:i+k])
        return set(shingle_set)

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        # a hacky way to pass in the line number in order not to break the function signature
        i = int(line.src[:len(self.MAX_NUM_LINES)])

        line.src = line.src[len(self.MAX_NUM_LINES):]  # remove the line number from the beginning of the line

        mh = MinHash(num_perm=self.num_perms)

        for shingle in self.get_shingle_set(normalize_for_dedup(line.src) + " " + normalize_for_dedup(line.tgt)):
            mh.update(shingle.encode('utf8'))

        result = self.lsh.query(mh)

        for index in result:
            if int(index) != i:
                self.lsh.remove(f'{str(i).zfill(len(self.MAX_NUM_LINES))}')  # Remove the current line from the LSH index
                counts.pair_fuzzy_dedup += 1
                return None

        return line