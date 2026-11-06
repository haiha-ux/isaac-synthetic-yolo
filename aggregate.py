"""Mean over seeds of results/*.json -> markdown table on stdout."""
import glob
import json
from collections import defaultdict

rows = defaultdict(list)
for f in sorted(glob.glob("results/*_s[0-9].json")):
    r = json.load(open(f))
    rows[(r["train"].split("/")[1].replace("_train", ""), r["aug"])].append(r["eval"])
sets = ["fixed_val", "rand_val", "shift_test"]
print("| Train set | Ultralytics aug | seeds | " + " | ".join("%s mAP50 / mAP50-95" % s for s in sets) + " |")
print("|---|---|---|" + "---|" * len(sets))
for (tr, aug), evs in sorted(rows.items()):
    cells = []
    for s in sets:
        m50 = [e[s]["mAP50"] for e in evs]
        m = [e[s]["mAP50-95"] for e in evs]
        cells.append("%.3f / %.3f (%.3f-%.3f)" % (sum(m50) / len(m50), sum(m) / len(m), min(m), max(m)))
    print("| %s | %s | %d | %s |" % (tr, aug, len(evs), " | ".join(cells)))
