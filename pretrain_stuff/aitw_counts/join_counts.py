import json
import glob

from collections import Counter, defaultdict

count_files = glob.glob("./*.json")
print(count_files)

all_counts = Counter()
for p in count_files:
    with open(p) as f:
        pc = json.load(f)
        all_counts.update(pc)

len(all_counts.keys())
final_dict = defaultdict(int)
for key in all_counts:
    final_dict[key.lower()] += all_counts[key]
len(final_dict.keys())

with open("all_counts.json", "w") as f:
    json.dump(final_dict, f)