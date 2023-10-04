import os
import re
from typing import Optional

from stopes.pipelines.filtering.dataset import DatasetLine
from stopes.pipelines.filtering.filters.base import Filter, FilteringCounts
from stopes.pipelines.monolingual.utils.text_normalizer import replace_unicode_punct


def get_digit2char_ratio(line):
    digit_count = sum([1 for char in line if char.isdigit()])
    char_count = sum([1 for char in line if char.isalpha()])

    return digit_count / (char_count + 1)


def word_has_alpha_char(word):
    for char in word:
        if char.isalpha():
            return True
    return False


def get_ratio_of_words_with_alpha(line):
    line = re.sub(r'\s', ' ', line)
    words = line.split(' ')

    cnt = 0
    for word in words:
        if word_has_alpha_char(word):
            cnt += 1

    return cnt / len(words)


def detect_ellipses(line):
    line = replace_unicode_punct(line)

    pattern = re.compile(r'\.{3,}')

    match = pattern.search(line)

    return match


class SymbolsFilter(Filter):
    def __init__(
        self,
        hashtag_num: int = 3,
        digit2char_ratio: float = 0.5,
        words_with_alpha_ratio: float = 0.5,
        keep_dates_and_numbers: bool = True,
        debug: bool = False,
    ):
        self.hashtag_num = hashtag_num
        self.digit2char_ratio = digit2char_ratio
        self.words_with_alpha_ratio = words_with_alpha_ratio
        self.keep_dates_and_numbers = keep_dates_and_numbers
        self.debug = debug

        self.dates_and_numbers_patterns = [
            re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
            re.compile(r'\b\d{2}/\d{2}/\d{4}\b'),
            re.compile(r'\b\d{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\b'),
            re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{2}, \d{4}\b'),
            re.compile(r'\d')
        ]

        if self.debug:
            current_path = os.path.dirname(os.path.realpath(__file__))
            self.debug_file = open(os.path.join(current_path, "debug_symbols_filter.txt"), "a")

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:

        # For mined data we want to remove all dates and numbers as they are usually incorrect.
        if not self.keep_dates_and_numbers:
            for pattern in self.dates_and_numbers_patterns:
                if pattern.search(line.src) or pattern.search(line.tgt):
                    counts.symbols += 1
                    if self.debug:
                        self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
                        self.debug_file.flush()
                    return None

        for current_line in [line.src, line.tgt]:
            hashtag_count = current_line.count("#")
            if self.hashtag_num and hashtag_count > self.hashtag_num:
                if self.debug:
                    self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
                    self.debug_file.flush()
                counts.symbols += 1
                return None

            if get_digit2char_ratio(current_line) > self.digit2char_ratio:
                if self.debug:
                    self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
                    self.debug_file.flush()
                counts.symbols += 1
                return None

            if get_ratio_of_words_with_alpha(current_line) < self.words_with_alpha_ratio:
                if self.debug:
                    self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
                    self.debug_file.flush()
                counts.symbols += 1
                return None

            # if detect_ellipses(line.src):
            #     if self.debug:
            #         self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
            #         self.debug_file.flush()
            #     counts.symbols += 1
            #     return None

        return line
