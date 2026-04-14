from __future__ import annotations

from pathlib import Path
import csv
import sys

import numpy as np


ROOT = Path("/workspace")
GRAPH_SOL = ROOT / "GraphSol"
PREDICT_DIR = GRAPH_SOL / "Predict"
COMMON_DIR = PREDICT_DIR / "Data" / "common"
GENERATE_DIR = PREDICT_DIR / "Data" / "generate"
UPLOAD_DIR = PREDICT_DIR / "Data" / "upload"
TEST_DIR = GRAPH_SOL / "test_415"

INPUT_FASTA = ROOT / "PLMsol_test.fasta"
OUTPUT_FASTA = TEST_DIR / "graphsol_test.fasta"
MAPPING_TSV = TEST_DIR / "graphsol_test_mapping.tsv"

ALLOWED = set("ACDEFGHIKLMNPQRSTVWYX")
MIN_LEN = 30
MAX_LEN = 700
MAX_SEQS = 5


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name = None
    seq_parts: list[str] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(seq_parts).upper()))
            name = line[1:].split()[0]
            seq_parts = []
        else:
            seq_parts.append(line)

    if name is not None:
        records.append((name, "".join(seq_parts).upper()))

    return records


def wrap80(sequence: str) -> str:
    return "\n".join(sequence[index:index + 80] for index in range(0, len(sequence), 80))


def select_records(records: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    selected: list[tuple[str, str, str]] = []
    seen_sequences: set[str] = set()

    for original_name, sequence in records:
        if len(sequence) < MIN_LEN or len(sequence) > MAX_LEN:
            continue
        if any(amino not in ALLOWED for amino in sequence):
            continue
        if sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)
        short_name = f"gs{len(selected) + 1:03d}"
        selected.append((short_name, original_name, sequence))
        if len(selected) >= MAX_SEQS:
            break

    return selected


def load_blosum() -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    with (COMMON_DIR / "BLOSUM62_dim23.txt").open() as handle:
        next(handle)
        for raw_line in handle:
            fields = raw_line.strip().split()
            result[fields[0]] = [int(value) for value in fields[1:]]
    return result


def band_graph(length: int) -> np.ndarray:
    matrix = np.ones((length, length), dtype=float)
    mask1 = np.tril(np.ones((length, length), dtype=float), -3)
    mask2 = np.triu(np.ones((length, length), dtype=float), 3)
    return matrix - (mask1 + mask2)


def main() -> int:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    GENERATE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    records = parse_fasta(INPUT_FASTA)
    selected = select_records(records)
    if not selected:
        raise RuntimeError("No compatible sequences were found in PLMsol_test.fasta")

    one_d_mean = np.load(COMMON_DIR / "eSol_oneD_mean.npy")
    blosum = load_blosum()

    with OUTPUT_FASTA.open("w") as fasta_writer, MAPPING_TSV.open("w", newline="") as mapping_writer:
        writer = csv.writer(mapping_writer, delimiter="\t")
        writer.writerow(["graphsol_id", "original_id", "length"])

        for short_name, original_name, sequence in selected:
            fasta_writer.write(f">{short_name}\n{wrap80(sequence)}\n")
            writer.writerow([short_name, original_name, len(sequence)])

            blosum_matrix = np.array([blosum[amino] for amino in sequence], dtype=float)
            one_d_matrix = np.concatenate(
                [blosum_matrix, np.tile(one_d_mean, (len(sequence), 1))],
                axis=1,
            )
            np.save(GENERATE_DIR / f"{short_name}_oneD.npy", one_d_matrix)
            np.save(GENERATE_DIR / f"{short_name}_twoD.npy", band_graph(len(sequence)))

    target_input = UPLOAD_DIR / "input.fasta"
    target_input.write_text(OUTPUT_FASTA.read_text())

    print(f"Prepared {len(selected)} sequences")
    for short_name, original_name, sequence in selected:
        print(f"{short_name}\t{original_name}\t{len(sequence)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
