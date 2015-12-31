import glob
import json
import os
import sys

outfile = "all_boxscores.json"
data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

files_names = glob.glob(os.path.join(data_dir, "*.json"))
huge_dict = {}
for f in files_names:
	with open(f) as infile:
		huge_dict[os.path.basename(f)] = json.load(infile)

with open("../{}".format(outfile), 'w') as outfile:
	json.dump(huge_dict, outfile)
