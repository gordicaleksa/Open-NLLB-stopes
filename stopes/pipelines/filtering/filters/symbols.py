import os
import re
import string
from typing import Optional

from stopes.pipelines.filtering.dataset import DatasetLine
from stopes.pipelines.filtering.filters.base import Filter, FilteringCounts


# Copied all below from Open-NLLB-stopes: https://github.com/gordicaleksa/Open-NLLB-stopes
UNICODE_PUNCT = {
    "，": ",",
    "。": ".",
    "、": ",",
    "„": '"',
    "”": '"',
    "“": '"',
    "«": '"',
    "»": '"',
    "１": '"',
    "」": '"',
    "「": '"',
    "《": '"',
    "》": '"',
    "´": "'",
    "∶": ":",
    "：": ":",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "；": ";",
    "–": "-",
    "—": " - ",
    "．": ". ",
    "～": "~",
    "’": "'",
    "…": "...",
    "━": "-",
    "〈": "<",
    "〉": ">",
    "【": "[",
    "】": "]",
    "％": "%",
    "►": "-",
}


NON_PRINTING_CHARS_RE = re.compile(
    f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]"
)


UNICODE_PUNCT_RE = re.compile(f"[{''.join(UNICODE_PUNCT.keys())}]")


PUNCT_OR_NON_PRINTING_CHARS_RE = re.compile(
    (UNICODE_PUNCT_RE.pattern + NON_PRINTING_CHARS_RE.pattern).replace("][", "")
)

def replace_unicode_punct(text: str) -> str:
    return "".join((UNICODE_PUNCT.get(c, c) for c in text))


def digit2char_ratio(line):
    line = PUNCT_OR_NON_PRINTING_CHARS_RE.sub("", line)
    # line = re.sub(r'\s', '', line)  # Remove all white spaces.
    # line = line.replace('-', '')  # Remove all dashes.
    # line = line.replace('_', '')  # Remove all underscores.
    # line = line.translate(str.maketrans('', '', string.punctuation))  # Remove all punctuation.

    digit_count = sum([1 for char in line if char.isdigit()])
    char_count = sum([1 for char in line if char.isalpha()])

    return digit_count / char_count


def has_alpha(word):
    for char in word:
        if char.isalpha():
            return True
    return False


def ratio_of_words_with_alpha(line):
    # Split the line on whitespace
    line = re.sub(r'\s', ' ', line)  # Remove all white spaces.
    words = line.split(' ')

    cnt = 0
    for word in words:
        if has_alpha(word):
            cnt += 1

    return cnt / len(words)


def detect_ellipses(line):
    line = replace_unicode_punct(line)

    pattern = re.compile(r'\.{6,}')

    match = pattern.search(line)

    return match


class SymbolsFilter(Filter):
    def __init__(
        self,
        hashtag_num: int = 3,
        digit2char_ratio: float = 0.5,
        words_with_alpha_ratio: float = 0.5,
        debug: bool = True,
    ):
        self.hashtag_num = hashtag_num
        self.digit2char_ratio = digit2char_ratio
        self.words_with_alpha_ratio = words_with_alpha_ratio
        self.debug = debug
        if self.debug:
            current_path = os.path.dirname(os.path.realpath(__file__))
            self.debug_file = open(os.path.join(current_path, "debug_symbols.txt"), "a")

    def balance_quotation_marks(self, line):
        line = replace_unicode_punct(line)

        flag = False
        if line.startswith('"') and not line.endswith('"'):
            line = line[1:]
            flag = True

        if not line.startswith('"') and line.endswith('"'):
            line = line[:-1]
            flag = True

        return flag

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:

        # if self.balance_quotation_marks(line.src):
        #     if self.debug:
        #         self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
        #         self.debug_file.flush()

        # if self.balance_quotation_marks(line.tgt):
        #     if self.debug:
        #         self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
        #         self.debug_file.flush()

        # Source
        # hashtag_count = line.src.count("#")
        # if hashtag_count > self.hashtag_num:
        #     if self.debug:
        #         self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
        #         self.debug_file.flush()
        #     counts.symbols += 1
        #     return None

        # if digit2char_ratio(line.src) > self.digit2char_ratio:
        #     if self.debug:
        #         self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
        #         self.debug_file.flush()
        #     counts.symbols += 1
        #     return None

        # if ratio_of_words_with_alpha(line.src) < self.words_with_alpha_ratio:
        #     if self.debug:
        #         self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
        #         self.debug_file.flush()
        #     counts.symbols += 1
        #     return None

        # if detect_ellipses(line.src):
        #     if self.debug:
        #         self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
        #         self.debug_file.flush()
        #     counts.symbols += 1
        #     return None

        # Target
        # if line.tgt is not None:
        #     hashtag_count = line.tgt.count("#")
        #     if hashtag_count > self.hashtag_num:
        #         if self.debug:
        #             self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
        #             self.debug_file.flush()
        #         counts.symbols += 1
        #         return None

        # if digit2char_ratio(line.tgt) > self.digit2char_ratio:
        #     if self.debug:
        #         self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
        #         self.debug_file.flush()
        #     counts.symbols += 1
        #     return None

        # if ratio_of_words_with_alpha(line.tgt) < self.words_with_alpha_ratio:
        #     if self.debug:
        #         self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
        #         self.debug_file.flush()
        #     counts.symbols += 1
        #     return None

        # if detect_ellipses(line.src):
        #     if self.debug:
        #         self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
        #         self.debug_file.flush()
        #     counts.symbols += 1
        #     return None

        return line
