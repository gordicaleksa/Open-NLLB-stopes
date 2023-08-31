"""
This scripts prepares 2 additional yaml configs "directions.yaml" and "included_corpora.yaml"
using train_primary.yaml as a source of information.

The purpose is to make example.yaml config more readable and easier to maintain. Instead of dumping
directions and corpora information directly into example config we instead link paths to yaml files.
"""
import pathlib
import yaml

# We're making an assumption here for now that you'll be keeping your configs in a particular location.
yaml_config_path = pathlib.Path(__file__).parent.parent / "filter_configs" / "unfiltered_corpora"

with open(yaml_config_path / "train_primary.yaml", "r") as fout:
    config = yaml.safe_load(fout)
    directions = list(config.keys())
    print(config.keys())
    with open(yaml_config_path / "directions.yaml", "w") as fout:
        yaml.dump(directions, fout)

    corpora = []
    for el in list(config.values()):
        corpora.extend(list(el.keys()))
    corpora = list(set(corpora))
    with open(yaml_config_path / "included_corpora.yaml", "w") as fout:
        yaml.dump(corpora, fout)
