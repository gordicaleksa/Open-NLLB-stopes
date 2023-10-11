from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import multiprocessing

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
        mp_dict
    ):
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
    path_out_src_before_fuzzy = dataset_output_dir / f"{corpus_name}.{src_lang}_before_fuzzy_{worker_id}"
    path_out_tgt_before_fuzzy = dataset_output_dir / f"{corpus_name}.{tgt_lang}_before_fuzzy_{worker_id}"

    path_counts = dataset_output_dir / f"{corpus_name}_before_fuzzy_{worker_id}.yaml"

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
            num_workers_dynamic
        ):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.src_file_chunks = src_file_chunks
        self.tgt_file_chunks = tgt_file_chunks
        self.dataset_output_dir = dataset_output_dir
        self.group_name = group_name
        self.corpus_name = corpus_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.config = config
        self.length_factors= length_factors
        self.num_workers = num_workers_dynamic

    def run(self):
        dedup_dict = multiprocessing.Manager().dict()
        dedup_lock = multiprocessing.Lock()

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
                    dedup_dict)
                for worker_id, (src_offset, tgt_offset) in enumerate(zip(self.src_file_chunks, self.tgt_file_chunks))
            ]

            for future in futures:
                future.result()
