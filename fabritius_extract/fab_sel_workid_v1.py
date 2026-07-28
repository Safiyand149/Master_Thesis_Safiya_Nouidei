#!/usr/bin/env python
import json
import lzma
import sys

def process(json_in, work_id):
    if json_in.endswith(".xz"):
        with lzma.open(json_in, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(json_in, "r") as f:
            data = json.load(f)
    for rec in data:
        if rec["objectWork"]["workID"] == work_id:
            print(json.dumps(rec, indent=4, ensure_ascii=False))
            break

    return rec

if __name__ == '__main__':
    process(*sys.argv[1:])