#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import re
import string
from typing import Dict, List, Optional, Tuple

import fasttext

from stopes.pipelines.filtering.dataset import DatasetLine
from stopes.pipelines.filtering.filters.base import Filter, FilteringCounts
from stopes.pipelines.monolingual.utils.text_normalizer import replace_unicode_punct, normalize_whitespace, remove_non_printing_char, DIGIT_RE

class LidFilter(Filter):
    def __init__(
        self,
        model_path: Path,
        default_threshold: float,
        thresholds: Dict[str, float],
        excluded_corpora: Optional[List[str]],
        excluded_languages: Optional[List[str]],
        src_lang: str,
        tgt_lang: Optional[str],
        debug: bool = False,
    ):
        self.src_threshold = thresholds.get(src_lang, default_threshold)
        self.tgt_threshold = thresholds.get(tgt_lang, default_threshold)
        self.excluded_corpora = excluded_corpora
        self.excluded_languages = excluded_languages or []
        self.lid = fasttext.load_model(model_path)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.debug = debug

        if self.debug:
            current_path = os.path.dirname(os.path.realpath(__file__))
            self.debug_file = open(os.path.join(current_path, f"debug_lid_filter_{src_lang}-{tgt_lang}.txt"), "a")
            self.debug_file_scores = open(os.path.join(current_path, f"debug_lid_filter_scores_{src_lang}-{tgt_lang}.txt"), "a")

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        # certain corpora may be excluded from LID
        if self.excluded_corpora and line.corpus in self.excluded_corpora:
            return line

        # store LID probs in DatasetLine
        lid_l, lid_p = self.lid.predict(line.src, k=-1)
        lid_probs = {lang[9:]: prob for lang, prob in zip(lid_l, lid_p)}
        line.src_lid_prob = lid_probs.get(self.src_lang, 0.0)
        if self.tgt_lang is not None:
            lid_l, lid_p = self.lid.predict(line.tgt, k=-1)
            lid_probs = {lang[9:]: prob for lang, prob in zip(lid_l, lid_p)}
            line.tgt_lid_prob = lid_probs.get(self.tgt_lang, 0.0)

        if self.debug:
            self.debug_file_scores.write(f"{line.src_lid_prob:.2f}" + " " + f"{line.tgt_lid_prob:.2f}" + "\n")
            self.debug_file_scores.flush()

        if self.src_threshold >= 0 and self.src_lang not in self.excluded_languages:
            if line.src_lid_prob <= self.src_threshold:
                if self.debug:
                    prefix = f"SRC.{line.src_lid_prob:.2f}" if line.tgt_lid_prob > self.tgt_threshold else f"SRC.{line.src_lid_prob:.2f}-TGT.{line.tgt_lid_prob:.2f}"
                    self.debug_file.write(f"{prefix}-{line.src}" + " || " + f"{line.tgt}" + "\n")
                    self.debug_file.flush()
                counts.lid_threshold += 1
                return None
        if (
            self.tgt_lang is not None
            and self.tgt_threshold >= 0
            and self.tgt_lang not in self.excluded_languages
        ):
            if line.tgt_lid_prob <= self.tgt_threshold:
                if self.debug:
                    prefix = f"TGT.{line.tgt_lid_prob:.2f}" if line.src_lid_prob > self.src_threshold else f"SRC.{line.src_lid_prob:.2f}-TGT.{line.tgt_lid_prob:.2f}"
                    self.debug_file.write(f"{prefix}-{line.src}" + " || " + f"{line.tgt}" + "\n")
                    self.debug_file.flush()
                counts.lid_threshold += 1
                return None
        return line


class HBSLidFilter(Filter):
    # HBS - Croatian, Bosnian, Serbian specific LID filter.
    def __init__(
        self,
        model_path: Path,
        default_threshold: float,
        thresholds: Dict[str, float],
        excluded_corpora: Optional[List[str]],
        excluded_languages: Optional[List[str]],
        src_lang: str,
        tgt_lang: Optional[str],
        min_length: int = 40,
        debug: bool = False,
    ):
        self.src_threshold = thresholds.get(src_lang, default_threshold)
        self.tgt_threshold = thresholds.get(tgt_lang, default_threshold)
        self.excluded_corpora = excluded_corpora
        self.excluded_languages = excluded_languages or []
        self.lid = fasttext.load_model(model_path)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.min_length = min_length
        self.debug = debug

        if self.debug:
            current_path = os.path.dirname(os.path.realpath(__file__))
            self.debug_file = open(os.path.join(current_path, f"debug_lid_filter_{src_lang}-{tgt_lang}.txt"), "a")

    def normalize(self, line: str) -> str:
        # We already do this start of 1st stage
        # line = normalize_whitespace(line)
        line = line.lower()
        line = DIGIT_RE.sub("", line)  # remove digits
        # We already do this start of 1st stage
        # line = remove_non_printing_char(line)
        # line = replace_unicode_punct(line)
        line = line.translate(str.maketrans('', '', string.punctuation))  # Remove all punctuation.
        # line = strip_accents(line)
        line = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', line)  # remove ip addresses
        line = re.sub(r'http\S+', '', line)  # remove urls
        line = normalize_whitespace(line)
        return line

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        # certain corpora may be excluded from LID
        if self.excluded_corpora and line.corpus in self.excluded_corpora:
            return line

        # we only consider lines with at least min_length characters
        if len(line.src) < self.min_length or len(line.tgt) < self.min_length:
            return line

        # store LID probs in DatasetLine
        lid_l, lid_p = self.lid.predict(self.normalize(line.src), k=-1)
        lid_probs = {lang[9:]: prob for lang, prob in zip(lid_l, lid_p)}
        line.src_lid_prob = lid_probs.get("non_hbs", 0.0)
        if self.tgt_lang is not None:
            lid_l, lid_p = self.lid.predict(self.normalize(line.tgt), k=-1)
            lid_probs = {lang[9:]: prob for lang, prob in zip(lid_l, lid_p)}
            line.tgt_lid_prob = lid_probs.get("hbs", 0.0)

        if self.src_threshold >= 0 and self.src_lang not in self.excluded_languages:
            if line.src_lid_prob <= self.src_threshold:
                if self.debug:
                    prefix = f"SRC.{line.src_lid_prob:.2f}" if line.tgt_lid_prob > self.tgt_threshold else f"SRC.{line.src_lid_prob:.2f}-TGT.{line.tgt_lid_prob:.2f}"
                    self.debug_file.write(f"{prefix}-{line.src}" + " || " + f"{line.tgt}" + "\n")
                    self.debug_file.flush()
                counts.lid_threshold += 1
                return None
        if (
            self.tgt_lang is not None
            and self.tgt_threshold >= 0
            and self.tgt_lang not in self.excluded_languages
        ):
            if line.tgt_lid_prob <= self.tgt_threshold:
                if self.debug:
                    prefix = f"TGT.{line.tgt_lid_prob:.2f}" if line.src_lid_prob > self.src_threshold else f"SRC.{line.src_lid_prob:.2f}-TGT.{line.tgt_lid_prob:.2f}"
                    self.debug_file.write(f"{prefix}-{line.src}" + " || " + f"{line.tgt}" + "\n")
                    self.debug_file.flush()
                counts.lid_threshold += 1
                return None
        return line
