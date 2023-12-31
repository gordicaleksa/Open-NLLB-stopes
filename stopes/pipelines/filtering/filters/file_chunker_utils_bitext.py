import os
import typing as tp

from stopes.pipelines.filtering.dataset import DatasetLine
from stopes.core import utils


def _safe_readline(fd) -> str:
    pos = fd.tell()
    while True:
        try:
            return fd.readline()
        except UnicodeDecodeError:
            pos -= 1
            fd.seek(pos)  # search where this character begins


def find_offsets(filename: str, num_chunks: int) -> tp.List[int]:
    """
    given a file and a number of chunk, find the offsets in the file
    to be able to chunk around full lines.
    """
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            _safe_readline(f)
            offsets[i] = f.tell()
        offsets[-1] = size
        return offsets


def find_offsets_given_line_numbers(filename: str, line_numbers: tp.List[int]) -> tp.List[int]:
    with open(filename, "r", encoding="utf-8") as f:
        offsets = []
        for i in range(1, len(line_numbers)):
            chunk_line_count = line_numbers[i] - line_numbers[i - 1]
            offsets.append(f.tell())
            for _ in range(chunk_line_count):
                _safe_readline(f)

        assert f.tell() == os.fstat(f.fileno()).st_size
        offsets.append(os.fstat(f.fileno()).st_size)
        return offsets


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


def build_line_number_to_byte_offset_map(filename: str) -> tp.Dict[int, int]:
    line_number_to_byte_offset = {}
    line_num = 0
    with open(filename, "r", encoding="utf-8") as f:
        file_size = os.fstat(f.fileno()).st_size
        while f.tell() < file_size:
            line_number_to_byte_offset[line_num] = f.tell()
            _safe_readline(f)
            line_num += 1

        assert utils.count_lines(filename) == line_num
        assert f.tell() == os.fstat(f.fileno()).st_size
        return line_number_to_byte_offset


class BitextChunkLineIterator:
    """
    Iterator to properly iterate over lines of a file chunck.
    """

    def __init__(self, src_fd, tgt_fd, src_start_offset: int, src_end_offset: int, tgt_start_offset: int, tgt_end_offset: int):
        self._src_fd = src_fd
        self._tgt_fd = tgt_fd
        self._src_start_offset = src_start_offset
        self._src_end_offset = src_end_offset
        self._tgt_start_offset = tgt_start_offset
        self._tgt_end_offset = tgt_end_offset

    def __iter__(self) -> tp.Iterable[str]:
        self._src_fd.seek(self._src_start_offset)
        self._tgt_fd.seek(self._tgt_start_offset)
        src_line = _safe_readline(self._src_fd)
        tgt_line = _safe_readline(self._tgt_fd)
        while src_line and tgt_line:
            src_pos = self._src_fd.tell()
            tgt_pos = self._tgt_fd.tell()
            # f.tell() does not always give the byte position in the file
            # sometimes it skips to a very large number
            # it is unlikely that through a normal read we go from
            # end bytes to end + 2**32 bytes (4 GB) and this makes it unlikely
            # that the procedure breaks by the undeterministic behavior of
            # f.tell()
            if (
                self._src_end_offset > 0
                and (src_pos > self._src_end_offset
                and src_pos < self._src_end_offset + 2**32)
                or (tgt_pos > self._tgt_end_offset
                and tgt_pos < self._tgt_end_offset + 2**32)
            ):
                break
            yield DatasetLine(corpus="", src=src_line.strip(), tgt=tgt_line.strip())
            src_line = _safe_readline(self._src_fd)
            tgt_line = _safe_readline(self._tgt_fd)


class BitextChunker:
    """
    contextmanager to read a chunk of a file line by line.
    """

    def __init__(self, src_path: str, tgt_path: str, src_offset: tp.Tuple[int], tgt_offset: tp.Tuple[int]):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.src_start_offset = src_offset[0]
        self.src_end_offset = src_offset[1]
        self.tgt_start_offset = tgt_offset[0]
        self.tgt_end_offset = tgt_offset[1]

    def __enter__(self) -> BitextChunkLineIterator:
        self.src_fd = open(self.src_path, "r", encoding="utf-8")
        self.tgt_fd = open(self.tgt_path, "r", encoding="utf-8")
        return BitextChunkLineIterator(
            self.src_fd,
            self.tgt_fd,
            self.src_start_offset,
            self.src_end_offset,
            self.tgt_start_offset,
            self.tgt_end_offset,
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.src_fd.close()
        self.tgt_fd.close()