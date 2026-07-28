#!/usr/bin/env python
import json, lzma, sys

def process(json_in):
    if json_in.endswith(".xz"):
        with lzma.open(json_in, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(json_in, "r") as f:
            data = json.load(f)
    for rec in data:
        print(f'''\
recordID: {rec["recordID"]}
  objectWork:
    workID: {rec["objectWork"]["workID"]}
    creatorDescription: {rec["objectWork"]["creatorDescription"]}
    titleText: {rec["objectWork"]["titleText"]}''')
        if "subjectMatter" in rec:
            print("  subjectMatter:")
            for k in rec["subjectMatter"]:
                print(f"    {k}: {rec["subjectMatter"][k]}")
        print()

if __name__ == '__main__':
    process(*sys.argv[1:])