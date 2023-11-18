# from matplotlib import pyplot as plt

# scores_path_mkd = "/home/aleksa/Projects/nllb/stopes/stopes/pipelines/filtering/filters/debug_lid_filter_scores_eng_Latn-mkd_Cyrl.txt"
# scores_path_szl = "/home/aleksa/Projects/nllb/stopes/stopes/pipelines/filtering/filters/debug_lid_filter_scores_eng_Latn-szl_Latn.txt"

# filepaths = [scores_path_szl]

# for filepath in filepaths:
#     with open(filepath, "r") as f:
#         lines = f.readlines()
#         lines = [line.strip() for line in lines]

#         src_scores = [float(line.split(" ")[0]) for line in lines]
#         tgt_scores = [float(line.split(" ")[1]) for line in lines]

#         assert len(src_scores) == len(tgt_scores), f'Expected same number of scores'

#         # plot the histogram for scores (they are in range [0, 1])
#         plt.hist(src_scores, bins=100, alpha=0.5, label='src')
#         plt.hist(tgt_scores, bins=100, alpha=0.5, label='tgt')
#         plt.legend(loc='upper right')
#         plt.show()

# print('ok')

# tmp = "/home/aleksa/Projects/nllb/stopes/stopes/pipelines/filtering/filter_configs/unfiltered_corpora/directions.yaml"
# import yaml
# # read the yaml
# with open(tmp, "r") as f:
#     data = yaml.safe_load(f)

#     # sort the data by number of sentences
#     data = dict(sorted(data.items(), key=lambda item: item[1], reverse=False))

#     # print the data
#     for key, value in data.items():
#         print(key, value)

# # save back
# with open(tmp, "w") as f:
#     yaml.dump(data, f, sort_keys=False)
from stopes.pipelines.filtering.filters import FilteringCounts

import os
from collections import defaultdict
import yaml
root_path = "/hdd/open-nllb-data/slavic_data_final_primary/train_primary/"
root_path2 = "/hdd/open-nllb-data/slavic_data_final_mined/train_mined/"

num_lines_per_lang = defaultdict(list)
num_lines_per_lang_original = defaultdict(list)
num_lines_per_lang_final = defaultdict(list)

def get_gt_num(dirname):
    root_dir = "/hdd/open-nllb-data/slavic_data/primary/"
    cnt = 0
    for _, subdirs, _ in os.walk(root_dir):
        for subdir in subdirs:
            if subdir == dirname:
                cnt += 1

    return cnt

for path in [root_path, root_path2]:
    for dirname in os.listdir(path):
        # if not dirname in ["eng_Latn-bul_Cyrl"]:
        #     continue
        dirpath = os.path.join(path, dirname)
        # get all yaml file paths
        yaml_paths = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath) if filename.endswith("total.yaml")]
        # yaml_final_paths = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath) if "fuzzy" not in filename and filename.endswith('yaml') and not filename.startswith('total')]
        # assert len(yaml_final_paths) == 0, f'Expected no files'
        # gt_num = get_gt_num(dirname)

        # assert len(yaml_paths) == len(yaml_final_paths) == gt_num, f'Expected same number of files'
        for yaml_path in yaml_paths:
            with open(yaml_path, "rt") as fin:
                data = yaml.safe_load(fin)
                num_lines_per_lang[dirname].append(data['total_after_fuzzy'])

        # for yaml_path in yaml_final_paths:
        #     with open(yaml_path, "rt") as fin:
        #         data = yaml.safe_load(fin)
        #         num_lines_per_lang_final[dirname].append(data['total_after_fuzzy'])


for key, value in num_lines_per_lang.items():
    num_lines_per_lang[key] = sum(value)

for key, value in num_lines_per_lang_final.items():
    num_lines_per_lang_final[key] = sum(value)

print('ok')
# for key, value in num_lines_per_lang_original.items():
#     num_lines_per_lang_original[key] = sum(value)

# for key, value in num_lines_per_lang_original.items():
#     print(key)
#     print(value.__dict__)

# # sort the data by number of sentences
# num_lines_per_lang = dict(sorted(num_lines_per_lang.items(), key=lambda item: item[1], reverse=False))
# num_lines_per_lang_original = dict(sorted(num_lines_per_lang_original.items(), key=lambda item: item[1], reverse=False))

# print('=' * 100)
# for key, value in num_lines_per_lang.items():
#     print(key, value)
# print('=' * 100)
# for key, value in num_lines_per_lang_original.items():
#     print(key, value)

# import os
# root_path = "/home/aleksa/Projects/nllb/stopes/stopes/pipelines/filtering/filter_out_slavic_non_hbs/batch_0_to_10/train_mined/eng_Latn-ces_Latn"
# # delete all files that have global_exact_dedup infix in the filename
# for filename in os.listdir(root_path):
#     if "global_exact_dedup" in filename:
#         os.remove(os.path.join(root_path, filename))