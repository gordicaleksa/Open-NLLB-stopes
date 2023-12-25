from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import gzip
import multiprocessing
import os

import hydra
import yaml

from stopes.pipelines.filtering.filters.file_chunker_utils_bitext import BitextChunker
from stopes.pipelines.filtering.filters import FilteringCounts
from stopes.pipelines.monolingual.utils.text_normalizer import replace_unicode_punct, remove_non_printing_char, normalize_whitespace
from stopes.pipelines.filtering.utils import normalize_unicode


def balance_quotation_marks(line):
    if line.startswith('"') and not line.endswith('"'):
        line = line[1:]

    if not line.startswith('"') and line.endswith('"'):
        line = line[:-1]

    return line


def normalize_line(line):
    line = normalize_whitespace(line)
    line = replace_unicode_punct(line)
    line_tmp = remove_non_printing_char(line)
    return balance_quotation_marks(line_tmp)


def first_stage_filtering_worker(
        worker_id,
        config,
        src_path,
        tgt_path,
        src_offset,
        tgt_offset,
        dataset_output_dir,
        group_name,
        corpus_name,
        src_lang,
        tgt_lang,
        length_factors,
        mp_dict,
        global_exact_dedup,
    ):
    if global_exact_dedup:
        filters = [
            hydra.utils.instantiate(config.dedup_filter, shared_memory=True, mp_dict=mp_dict, lock=lock),
        ]
    else:
        filters = [
            hydra.utils.instantiate(config.laser_filter),
            hydra.utils.instantiate(
                config.length_filter,
                length_factors=length_factors,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            ),
            hydra.utils.instantiate(
                config.symbols_filter, keep_dates_and_numbers=group_name != "train_mined"
            ),
            hydra.utils.instantiate(
                config.lid_filter, src_lang=src_lang, tgt_lang=tgt_lang
            ),
            hydra.utils.instantiate(
                config.toxicity_filter, src_lang=src_lang, tgt_lang=tgt_lang
            ),
            hydra.utils.instantiate(config.dedup_filter, shared_memory=True, mp_dict=mp_dict, lock=lock),
        ]

    path_out_src_before_fuzzy = dataset_output_dir / f"{corpus_name}.{src_lang}_before_fuzzy_{worker_id}{'_global_exact_dedup' if global_exact_dedup else ''}"
    path_out_tgt_before_fuzzy = dataset_output_dir / f"{corpus_name}.{tgt_lang}_before_fuzzy_{worker_id}{'_global_exact_dedup' if global_exact_dedup else ''}"

    path_counts = dataset_output_dir / f"{corpus_name}_before_fuzzy_{worker_id}{'_global_exact_dedup' if global_exact_dedup else ''}.yaml"

    # if os.path.isfile(path_counts):
    #     print(f"Skipping {path_counts}: already exists")
    #     return

    dataset_counts = FilteringCounts()  # filtering counts for the current dataset

    with ExitStack() as outputs, BitextChunker(src_path, tgt_path, src_offset, tgt_offset) as inputs:
        fout_src = outputs.enter_context(open(path_out_src_before_fuzzy, "wt"))
        fout_tgt = None

        # if tgt_lang is not provided, we have a monolingual dataset;
        # otherwise, the parallel file needs to be opened
        if tgt_lang is not None:
            fout_tgt = outputs.enter_context(open(path_out_tgt_before_fuzzy, "wt"))

        for line in inputs:
            dataset_counts.total_before += 1

            # Temporarily disabled
            # if get_step(steps_so_far, dataset_counts) % 10000 == 0:
            #     wandb.log(
            #         {f"{group_name.split('_')[-1]}_before_fuzzy/{direction}": get_step(steps_so_far, dataset_counts) / total_num_lines},
            #         step=get_step(steps_so_far, dataset_counts)
            #     )

            if config.normalize_unicode:
                line.src = normalize_unicode(line.src)
                if line.tgt is not None:
                    line.tgt = normalize_unicode(line.tgt)

            if config.normalize_line:
                line.src = normalize_line(line.src)
                if line.tgt is not None:
                    line.tgt = normalize_line(line.tgt)

            # apply filters sequentially
            for fltr in filters:
                if fltr is None:
                    continue

                line = fltr.filter_line(line, dataset_counts)
                # no need to keep filtering if the line was already discarded
                if line is None:
                    break
            if line is None:
                continue

            fout_src.write(line.src + "\n")
            if fout_tgt is not None:
                fout_tgt.write(line.tgt + "\n")
            dataset_counts.total_after += 1

        if dataset_counts:
            with open(path_counts, "wt") as fout:
                yaml.safe_dump(dataset_counts.__dict__, fout)


def init_pool_processes(lock_):
    global lock
    lock = lock_


class FirstStage:
    def __init__(
            self,
            src_path,
            tgt_path,
            src_file_chunks,
            tgt_file_chunks,
            dataset_output_dir,
            group_name,
            corpus_name,
            src_lang,
            tgt_lang,
            config,
            length_factors,
            num_workers_dynamic,
            dedup_dict,
            global_exact_dedup,
        ):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.src_file_chunks = src_file_chunks
        self.tgt_file_chunks = tgt_file_chunks
        self.dataset_output_dir = dataset_output_dir
        self.corpus_name = corpus_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.config = config
        self.group_name = group_name
        self.length_factors= length_factors
        self.num_workers = num_workers_dynamic
        self.dedup_dict = dedup_dict
        self.global_exact_dedup = global_exact_dedup

    def run(self):
        dedup_lock = multiprocessing.Lock()
        if self.dedup_dict is None:
            self.dedup_dict = multiprocessing.Manager().dict()  # corpus-level dedup.

        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=init_pool_processes,
            initargs=(dedup_lock,)) as executor:

            # Process file chunks in parallel.
            futures = [
                executor.submit(
                    first_stage_filtering_worker,
                    worker_id,
                    self.config,
                    self.src_path,
                    self.tgt_path,
                    src_offset,
                    tgt_offset,
                    self.dataset_output_dir,
                    self.group_name,
                    self.corpus_name,
                    self.src_lang,
                    self.tgt_lang,
                    self.length_factors,
                    self.dedup_dict,
                    self.global_exact_dedup)
                for worker_id, (src_offset, tgt_offset) in enumerate(zip(self.src_file_chunks, self.tgt_file_chunks))
            ]

            for future in futures:
                future.result()


def fuzzy_filtering_worker(
        worker_id,
        src_path,
        tgt_path,
        src_offset,
        tgt_offset,
        dataset_output_dir,
        corpus_name,
        src_lang,
        tgt_lang,
        fuzzy_filter,
        line_numbers_range_chunk,
        offset_cnt
):
    # TODO: fix the cnt logic -> crucial for correct fuzzy deduplication
    path_out_src = dataset_output_dir / f"{corpus_name}.{src_lang}_fuzzy_dedup_{worker_id}"
    path_out_tgt = dataset_output_dir / f"{corpus_name}.{tgt_lang}_fuzzy_dedup_{worker_id}"

    path_counts = dataset_output_dir / f"{corpus_name}_fuzzy_dedup_{worker_id}.yaml"

    dataset_counts = FilteringCounts()  # filtering counts for the current dataset

    cnt = offset_cnt + line_numbers_range_chunk[0]  # TODO: test this works correctly
    with ExitStack() as outputs, BitextChunker(src_path, tgt_path, src_offset, tgt_offset) as inputs:
        fout_src = outputs.enter_context(open(path_out_src, "wt"))
        fout_tgt = None

        # if tgt_lang is not provided, we have a monolingual dataset;
        # otherwise, the parallel file needs to be opened
        if tgt_lang is not None:
            fout_tgt = outputs.enter_context(open(path_out_tgt, "wt"))

        for i, line in enumerate(inputs):
            dataset_counts.total_before += 1

            # hacky - in order not to break the existing func signature I'm encoding the line number information in the line.src
            line.src = f'{str(cnt).zfill(9)}{line.src}'
            line = fuzzy_filter.filter_line(line, dataset_counts)
            cnt += 1  # this needs to be precisely here, don't move it unless you know why you are doing that.

            if line is None:
                continue

            fout_src.write(line.src + "\n")
            if fout_tgt is not None:
                fout_tgt.write(line.tgt + "\n")
            dataset_counts.total_after_fuzzy += 1

        if dataset_counts:
            with open(path_counts, "wt") as fout:
                yaml.safe_dump(dataset_counts.__dict__, fout)


class FuzzyFilterStage:
    def __init__(
            self,
            src_path,
            tgt_path,
            src_file_chunks,
            tgt_file_chunks,
            dataset_output_dir,
            corpus_name,
            src_lang,
            tgt_lang,
            num_workers_dynamic,
            fuzzy_filter,
            chunks_line_numbers,
            cnt,
        ):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.src_file_chunks = src_file_chunks
        self.tgt_file_chunks = tgt_file_chunks
        self.dataset_output_dir = dataset_output_dir
        self.corpus_name = corpus_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.num_workers = num_workers_dynamic
        self.fuzzy_filter = fuzzy_filter
        self.line_numbers_range_all = chunks_line_numbers
        self.cnt = cnt

    def run(self):

        if self.num_workers > 1:
            with ProcessPoolExecutor(
                max_workers=self.num_workers) as executor:

                # Process file chunks in parallel.
                futures = [
                    executor.submit(
                        fuzzy_filtering_worker,
                        worker_id,
                        self.src_path,
                        self.tgt_path,
                        src_offset,
                        tgt_offset,
                        self.dataset_output_dir,
                        self.corpus_name,
                        self.src_lang,
                        self.tgt_lang,
                        self.fuzzy_filter,
                        line_numbers_range_chunk,
                        self.cnt)
                    for worker_id, (src_offset, tgt_offset, line_numbers_range_chunk) in enumerate(zip(self.src_file_chunks, self.tgt_file_chunks, self.line_numbers_range_all))
                ]

                for future in futures:
                    future.result()
        else:
            fuzzy_filtering_worker(
                0,
                self.src_path,
                self.tgt_path,
                self.src_file_chunks[0],
                self.tgt_file_chunks[0],
                self.dataset_output_dir,
                self.corpus_name,
                self.src_lang,
                self.tgt_lang,
                self.fuzzy_filter,
                self.line_numbers_range_all[0],
                self.cnt
            )