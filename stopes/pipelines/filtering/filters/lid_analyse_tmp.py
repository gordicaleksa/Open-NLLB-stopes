import os

fpath = "/home/aleksa/Projects/nllb/stopes/stopes/pipelines/filtering/filters/debug_lid_filter_eng_Latn-szl_Latn.txt"
out_dir = "/home/aleksa/Projects/nllb/stopes/stopes/pipelines/filtering/filters/"
fname = os.path.basename(fpath)
with open(fpath, "r") as f:
    lines = f.readlines()

src_lines = [line for line in lines if line.startswith("SRC")]
tgt_lines = [line[:9] + " " + line.split("||")[-1] for line in lines if line.startswith("TGT")]

src_lines_low_score = [line for line in src_lines if float(line[4:8]) == 0]
src_lines_high_score = [line for line in src_lines if 0.0 < float(line[4:8]) <= 0.2]
tgt_lines_low_score = [line for line in tgt_lines if float(line[4:8]) == 0]
tgt_lines_high_score = [line for line in tgt_lines if 0.8 < float(line[4:8]) <= 0.9]

# print the histogram of scores, count on y axis, score on x axis
# scores = [float(line[4:8]) for line in lines if line.startswith("SRC")]

# import matplotlib.pyplot as plt
# plt.hist(scores, bins=100)
# plt.show()


with open(os.path.join(out_dir, f"src_low_{fname}"), "w") as f:
    f.writelines(src_lines_low_score)

with open(os.path.join(out_dir, f"src_high_{fname}"), "w") as f:
    f.writelines(src_lines_high_score)

with open(os.path.join(out_dir, f"tgt_low_{fname}"), "w") as f:
    f.writelines(tgt_lines_low_score)

with open(os.path.join(out_dir, f"tgt_high_{fname}"), "w") as f:
    f.writelines(tgt_lines_high_score)
