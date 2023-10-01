#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from typing import Dict, List, Optional, Set

from datasketch import MinHash, MinHashLSH
import xxhash

from stopes.pipelines.filtering.dataset import Dataset, DatasetLine, DatasetReader
from stopes.pipelines.filtering.filters.base import Filter, FilteringCounts
from stopes.pipelines.monolingual.utils.text_normalizer import normalize_for_dedup


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


class FuzzyDedupFilter(Filter):
    MAX_NUM_LINES = '999999999'  # arbitrary setting, but should be enough for most datasets

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        num_bands: int = 10,
        subvector_size: int = 10,
        num_perms: int = 100,
        threshold: float = None,
    ):
        self.num_perms = num_perms
        if threshold:
            self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perms)
        else:
            self.lsh = MinHashLSH(params=[num_bands, subvector_size], num_perm=num_perms)

        cnt = 0
        for corpus_name, dataset in datasets.items():
            with DatasetReader(dataset, corpus_name) as inputs:
                for line in inputs:
                    # TODO(gordicaleksa): computation of minhashes can be parallelized
                    mh = MinHash(num_perm=num_perms)

                    for shingle in self.get_shingle_set(normalize_for_dedup(line.src) + " " + normalize_for_dedup(line.tgt)):
                        mh.update(shingle.encode('utf8'))

                    self.lsh.insert(f'{str(cnt).zfill(len(self.MAX_NUM_LINES))}', mh)
                    cnt += 1

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
            if index != i:
                self.lsh.remove(i)  # Remove the current line from the LSH index
                counts.pair_fuzzy_dedup += 1
                return None

        return line