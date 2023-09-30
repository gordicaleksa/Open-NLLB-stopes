import os
import re
import string
from typing import Optional

from stopes.pipelines.filtering.dataset import DatasetLine
from stopes.pipelines.filtering.filters.base import Filter, FilteringCounts


class MinedFilter(Filter):
    def __init__(
        self,
    ):
        self.debug = True  # TODO: make this configurable
        if self.debug:
            current_path = os.path.dirname(os.path.realpath(__file__))
            self.debug_file = open(os.path.join(current_path, "dates.txt"), "a")

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:

        patterns = [
            re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
            re.compile(r'\b\d{2}/\d{2}/\d{4}\b'),
            re.compile(r'\b\d{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\b'),
            re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{2}, \d{4}\b'),
            re.compile(r'\d')
        ]

        for pattern in patterns:
            if pattern.search(line.src) or pattern.search(line.tgt):
                counts.symbols += 1
                if self.debug:
                    self.debug_file.write(f"{line.src}" + " || " + f"{line.tgt}" + "\n")
                    self.debug_file.flush()
                return None

        return line
