# Open-NLLB Bitext Filtering

The `filter.py` pipeline applies various filters to bitext, with optional support for
monolingual text. It is configured with [Hydra](https://hydra.cc/).

Before being able to run the filtering pipeline, you will need to create configuration
files containing the list of corpora and the set of length factors + 2 more helper configs.

This can be done by running:
1. `scripts/populate_data_conf.py`
2. `scripts/compute_length_factors.py`
3. `scripts/extract_directions_and_corpora.py`
respectively.

Please consult the help of `populate_data_conf.py` (by running them with `-h`) to learn more about how to configure it.
Default settings will work for the two other scripts.

Finally modify the `conf/examples.yaml`, we've submitted an example of how it is supposed to look like.

A basic filtering stage run using default parameters might look like this:
```
python filter.py \
  output_dir=/home/$USER/filter_test
```

or if you're running in vscode, specify the following args in your launch.json's config:
```
"args": [
    "+output_dir='$STOPES_ROOT/stopes/pipelines/filtering/filter_out_test'",
]
```
`$STOPES_ROOT` is the path to the root directory of your stopes project!

This command will run using the output directory and will additionally load the
default example config from `conf/example.yaml`. Anything not specified on the command line
or in `conf/example.yaml` will be set to the default values specified in
`data_types.FilterConfig`.

When needing to run a new filtering job with many parameter overrides, instead of
manually overriding parameters on the command line it is better to create an entirely
new config file, e.g. `conf/my_config.yaml`, containing all overrides. The script can
then be instructed to load it as follows:
```
python filter.py \
  --config-name=my_config \
  output_dir=/home/$USER/filter_test
```

## Toxicity filtering (advanced)
If you want to use toxicity filtering you'll have to download the toxicity-200 word lists from [here](https://github.com/facebookresearch/flores/blob/main/toxicity/README.md).

To make things a bit easier please download the above file to your **Open-NLLB root** because we are downloading and doing initial datasets preparations there. You can just run `wget --trust-server-names https://tinyurl.com/NLLB200TWL` from Open-NLLB project root directory.

After that run the following bash script `examples/nllb/data/unzip_toxicity_lists.sh` (again it's in the Open-NLLB repo) to unzip all the word lists. It assumes that you've downloaded the toxicity lists to Open-NLLB root.

Finally you'll have to modify the path (`twl_path_template`) to your word lists inside  `ToxicityFilterConfig`. Atm it's hardcoded, we should eventually make that a part of yaml config.


