"""
This scripts prepares 2 additional yaml configs "directions.yaml" and "included_corpora.yaml"
using train_primary.yaml as a source of information.

The purpose is to make example.yaml config more readable and easier to maintain. Instead of dumping
directions and corpora information directly into example config we instead link paths to yaml files.
"""
import pathlib
import yaml

from stopes.core.utils import count_lines

# We're making an assumption here for now that you'll be keeping your configs in a particular location.
yaml_config_path = pathlib.Path(__file__).parent.parent / "filter_configs" / "unfiltered_corpora"

with open(yaml_config_path / "train_primary.yaml", "r") as fout:
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

    lang_direction_num_sentences = dict(
        sorted(lang_direction_num_sentences.items(), key=lambda item: item[1], reverse=False)
    )
    with open(yaml_config_path / "directions.yaml", "w") as fout:
        yaml.dump(lang_direction_num_sentences, fout, sort_keys=False)

    corpora = []
    for el in list(config.values()):
        corpora.extend(list(el.keys()))
    corpora = list(set(corpora))
    with open(yaml_config_path / "included_corpora.yaml", "w") as fout:
        yaml.dump(corpora, fout)
