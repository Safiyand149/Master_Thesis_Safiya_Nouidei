#!/usr/bin/env python
import json
import lzma
import sys


def _get_general_subject_description(subject_matter):
    return (
        subject_matter.get("generalSubjectDescription")
        or subject_matter.get("generalSubjectSecription")
        or subject_matter.get("generalsubjectsecription")
    )


def process(json_in, work_id):
    if json_in.endswith(".xz"):
        with lzma.open(json_in, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(json_in, "r", encoding="utf-8") as f:
            data = json.load(f)

    record = None
    for rec in data:
        if str(rec.get("objectWork", {}).get("workID")) == str(work_id):
            record = rec
            break

    if not record:
        print(f"No record found for workID: {work_id}")
        return {}

    object_work = record.get("objectWork", {}) or {}
    print(
        f"""recordID: {record.get('recordID', '')}
  objectWork:
    workID: {object_work.get('workID', '')}
    creatorDescription: {object_work.get('creatorDescription', '')}
    titleText: {object_work.get('titleText', '')}"""
    )

    subject_matter = record.get("subjectMatter", {}) or {}
    if subject_matter:
        print("  subjectMatter:")

        general_subject_description = _get_general_subject_description(subject_matter)
        if general_subject_description:
            print(f"    generalSubjectDescription: {general_subject_description}")

        for key, value in subject_matter.items():
            if key in {
                "generalSubjectDescription",
                "generalSubjectSecription",
                "generalsubjectsecription",
            }:
                continue
            print(f"    {key}: {value}")

    return record


if __name__ == '__main__':
    process(*sys.argv[1:])
