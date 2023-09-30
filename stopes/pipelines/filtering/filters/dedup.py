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

from stopes.pipelines.filtering.dataset import DatasetLine
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
    MAX_NUM_LINES = '999999999'

    def __init__(
        self,
        src_lines,
        tgt_lines,
        num_perms: int = 100,
    ):
        assert len(src_lines) <= int(self.MAX_NUM_LINES), f'FuzzyDedupFilter: too many lines ({len(src_lines)})'

        self.num_perms = num_perms
        self.lsh = MinHashLSH(params=[10, 10], num_perm=num_perms)  # TODO: experiment with b (number of bands) & r (size of subvector) params
        for i, (src_line, trg_line) in enumerate(zip(src_lines, tgt_lines)):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            mh = MinHash(num_perm=num_perms)

            for shingle in self.get_shingle_set(src_line + " " + trg_line):
                mh.update(shingle.encode('utf8'))

            self.lsh.insert(f'{str(i).zfill(len(self.MAX_NUM_LINES))}', mh)

    def get_shingle_set(self, text: str, k: int = 5):
        shingle_set = []
        for i in range(len(text) - k+1):
            shingle_set.append(text[i:i+k])
        return set(shingle_set)

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        i = int(line.src[:len(self.MAX_NUM_LINES)])  # hacky way to pass in the line number in order not to break the function signature
        line.src = line.src[len(self.MAX_NUM_LINES):]  # remove the line number from the beginning of the line
        # TODO(gordicaleksa): punctuation is removed, text is lowercased, NFD Unicode normalization is applied, accents are removed, and all whitespace is normalized.
        mh = MinHash(num_perm=self.num_perms)

        for shingle in self.get_shingle_set(line.src + " " + line.tgt):
            mh.update(shingle.encode('utf8'))

        result = self.lsh.query(mh)  # [4073, 1877, 1054, 1738, 4163, 1407, 1695, 4445, 1065, 4557, 863, 1293, 999, 754, 4754]

        if len(result) > 1:  # we have found fuzzy duplicates
            if result[-1] == i:  # we keep only the final duplicate
                return line
            else:
                counts.pair_fuzzy_dedup += 1
                return None

        return line
