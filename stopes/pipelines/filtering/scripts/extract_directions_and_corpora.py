"""
This scripts prepares additional yaml configs "directions.yaml" and "{primary/mined}_included_corpora.yaml"
using train_primary.yaml and train_mined.yaml as a source of information.

The purpose is to make example.yaml config more readable and easier to maintain. Instead of dumping
directions and corpora information directly into example config we instead link paths to yaml files.
"""
import argparse
from pathlib import Path


import yaml


from stopes.core.utils import count_lines


def main(args):
    data_conf_dir = args.data_conf_dir
    in_config_files = ["train_primary.yaml", "train_mined.yaml"]
    lang_direction_num_sentences_list = []
    for in_config_file in in_config_files:
        suffix = in_config_file.split(".")[0].split('_')[-1]
        config_path = data_conf_dir / in_config_file

        if not config_path.exists():
            raise ValueError(f"File {config_path} does not exist. Please run the `populate_data_conf.py` script first.")

        with open(config_path, "r") as fout:
            config = yaml.safe_load(fout)

            lang_direction_num_sentences = {}
            for lang_direction, datasets in config.items():
                num_sentences_for_lang_direction = 0
                for corpus_name, dataset in datasets.items():
                    src_path = dataset["src"]
                    src_cnt = count_lines(src_path)
                    tgt_path = dataset["tgt"]
                    tgt_cnt = count_lines(tgt_path)
                    assert src_cnt == tgt_cnt, f"src and tgt have different number of lines for {src_path}"
                    num_sentences_for_lang_direction += src_cnt
                lang_direction_num_sentences[lang_direction] = num_sentences_for_lang_direction

            # Sort the directions by number of sentences (optimization that helps us run filtering stage faster!)
            lang_direction_num_sentences = dict(
                sorted(lang_direction_num_sentences.items(), key=lambda item: item[1], reverse=False)
            )
            lang_direction_num_sentences_list.append(lang_direction_num_sentences)

            # Save all the different corpora names
            corpora = []
            for el in list(config.values()):
                corpora.extend(list(el.keys()))
            corpora = list(set(corpora))

            with open(data_conf_dir / f"{suffix}_included_corpora.yaml", "w") as fout:
                yaml.dump(corpora, fout)

    keys = []
    for lang_direction_num_sentences in lang_direction_num_sentences_list:
        keys.extend(list(lang_direction_num_sentences.keys()))
    keys = list(set(keys))
    lang_direction_num_sentences = {}
    for key in keys:
        for lang_direction_num_sentences_ in lang_direction_num_sentences_list:
            if key in lang_direction_num_sentences_:
                if key not in lang_direction_num_sentences:
                    lang_direction_num_sentences[key] = lang_direction_num_sentences_[key]
                else:
                    lang_direction_num_sentences[key] += lang_direction_num_sentences_[key]

    with open(data_conf_dir / "directions.yaml", "w") as fout:
        yaml.dump(lang_direction_num_sentences, fout, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-conf-dir",
        type=Path,
        default=Path("stopes/pipelines/filtering/filter_configs/unfiltered_corpora"),
        help="Directory where the configuration files are stored.",
    )
    args = parser.parse_args()
    main(args)
