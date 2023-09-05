# NLLB prepare_data pipeline

## Quick start
1. Run `prepare_extra_configs.py` to create additional yaml file (`train_corpora.yaml`) that you'll need to pass to `prepare_data.yaml`. It will automatically save it into `prepare_data_configs/` directory (similar to how the filtering stage saves to `filter_configs`). For validation data specify it directly inside `prepare_data.yaml` (you'll just need to modify the existing settings). Note: this script expects you already ran the filtering pipeline and have the `train_primary.yaml` config in the right location.

2. Modify config files under `conf/`. The main one is `prepare_data.yaml`. You'll also need to modify the paths in the `conf/vocab/`. The `prepare_data.py` script will automatically download SPM-200 model and dictionary into `prepare_data/spm_models/` directory.

3. Run the `prepare_data.py` script, no arguments are needed it's controlled solely by the `prepare_data.yaml` config.

## Intro

This pipeline takes in the filtered corpora text files (can be compressed), trains an SPM model, deduplicates, shards, encodes & binarizes them in the format required by fairseq. An array of jobs is scheduled wherever possible, in particular for validate, retrieve_data, dedup_sharding and binarizing. The pipeline uses the caching feature of Stopes.

## Input Config:

* fold: train, train_mining, train_mmt_bt, train_smt_bt, valid, test are possible options
* lang_dir: language direction

corpora: `CorporaConfig`
    <FOLD>:
        <LANG_DIR>:
            <CORPUS_NAME>:
                src: <SRC_FILE_PATH>
                tgt: <TGT_FILE_PATH>
                metadata: <METADATA_FILE_PATH> (optional)
            ...
        ...
    ...
Specify paths to src, tgt, and optionally metadata files per (fold, lang_dir) for each corpus.

preprocessing: `PreprocessingConfig`
Specify boolean values for MOSES preprocessing.

vocab: `VocabConfig`
Specify vocab params for src, tgt vocab. By default vocab is trained jointly, in which case we only use the src_vocab config for the joint vocab.

dedup: `DedupConfig`
How to deduplicate? Whether across folds and how to use individual sentences for deduplication.

sharding: `ShardingConfig`
How to shard? Total sentences per shard and the minimum number of sentences for each lang_dir per shard. Also the number of workers to binarize sharded files in a Multiproc way.

launcher:
How to launch your jobs? locally or submitit

## Run Command:

Please override the default config options as required.
```
python stopes/pipelines/prepare_data/prepare_data.py output_dir=<OUTPUT_DIR>
```

## Pipeline Breakdown

* validate: Counts the number of lines for all parallel corpora and makes sure they're the same for src & tgt and stores train line counts statistics.
* retrieve_data: Concatenates all corpora for each (fold, lang_dir), runs Moses preprocessing over each of them as per preprocessing config and saves them to the `retrieved_data` directory.
* build_vocab: Samples a corpus as per sampling_config and trains an SPM on the sampled corpus. We need to sample a corpus since training an SPM on all of the corpora is time consuming. This is done jointly for src, tgt directinos by default but can be done separately as well. The trained SPM, the model file and vocab file are saved in the `vocab_bin` directory
* dedup_sharding: Deduplicates training corpora across eval corpora (valid, test) & optionally across folds as per dedup_config and shards training corpora.
* binarize: Binarizes all the sharded files (train, eval) using `MultiProcFairSeqBinarizerEncoder` and writes them to the sharded directories in the `data_bin` directory.

## Caveat

This pipeline doesn't work if metadata is not specified for all corpora for a (fold, lang_dir) because we concatenate all corpora files for each (fold, lang_dir) into one file and shard them. So we need metadata information for every one of these lines, if specified at all for the (fold, lang_dir).

## Note
There is hard expectation on how SPM-200 dictionary should be named otherwise you'll hit some bugs. Example if the spm model is named `flores200_sacrebleu_tokenizer_spm.model` then the dictionary file should be named like `flores200_sacrebleu_tokenizer_spm.dict.txt` (same prefix but suffix is `.dict.txt`). This is now automatically handled by the `prepare_data.py` script and you don't need to worry about it.
