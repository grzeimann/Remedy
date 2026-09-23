#!/usr/bin/env python
"""Create the reviewed post-QA rejection file from an amplifier QA table."""

import argparse
import csv
from pathlib import Path


REJECT_FIELDS = ("H5", "exposure", "SPECID", "IFUSLOT", "IFUID", "AMP")


def _key(row):
    return (
        Path(row["H5"]).name,
        int(row["exposure"]),
        int(row["SPECID"]),
        int(row["IFUSLOT"]),
        int(row["IFUID"]),
        row["AMP"].strip().upper(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Write an explicit reject file for WARN amplifier QA rows."
    )
    parser.add_argument(
        "--qa-table",
        type=Path,
        default=None,
        help="QA CSV; defaults to production QA, then current Diagnose QA.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("m101_cube_current_model_qa_reject.csv"),
        help="Output rejection CSV.",
    )
    parser.add_argument(
        "--expected-warn-count",
        type=int,
        default=56,
        help="Expected current WARN count; default: 56.",
    )
    args = parser.parse_args()

    if args.qa_table is not None:
        qa_path = args.qa_table
    else:
        candidates = (
            Path("m101_cube_current_model_amplifier_qa.csv"),
            Path("m101_exposure_scale_diagnostic/post_iter2_amplifier_qa.csv"),
        )
        qa_path = next((path for path in candidates if path.exists()), None)

    if qa_path is None or not qa_path.exists():
        raise FileNotFoundError(
            "No QA table found. Supply one with --qa-table."
        )

    with qa_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(REJECT_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "QA table lacks required columns: %s" % sorted(missing)
            )
        warn_rows = [
            row for row in reader
            if row.get("QA_STATUS", "").strip().upper() == "WARN"
        ]

    if len(warn_rows) != args.expected_warn_count:
        raise ValueError(
            "Expected %d WARN rows, found %d; no reject file written."
            % (args.expected_warn_count, len(warn_rows))
        )

    keys = [_key(row) for row in warn_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate WARN amplifier/exposure keys found.")

    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REJECT_FIELDS)
        writer.writeheader()
        for row in warn_rows:
            writer.writerow({field: row[field] for field in REJECT_FIELDS})

    print("Source QA table: %s" % qa_path)
    print("WARN rows selected: %d" % len(warn_rows))
    print("Reject file written: %s" % args.output)


if __name__ == "__main__":
    main()
