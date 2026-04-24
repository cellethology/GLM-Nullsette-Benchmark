from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHAGENOME_ROOT = REPO_ROOT / "alphagenome_research"
DEFAULT_INPUT_DIRS = {
    # "Abf1TATA": REPO_ROOT / "inference_data" / "deboer" / "Abf1TATA", # CTAAAGGTGAAGAATTATTCACTGGTGTTGTCCCAATTTTGGTTGAATTAGATGGTGATGTTAATGGTCACAAATTTTCTGTCTCCGGTGAAGGTGAAGGTGATGCTACTTACGGTAAATTGACCTTAAAATTGATTTGTACTACTGGTAAATTGCCAGTTCCATGGCCAACCTTAGTCACTACTTTAGGTTATGGTTTGCAATGTTTTGCTAGATACCCAGATCATATGAAACAACATGACTTTTTCAAGTCTGCCATGCCAGAAGGTTATGTTCAAGAAAGAACTATTTTTTTCAAAGATGACGGTAACTACAAGACCAGAGCTGAAGTCAAGTTTGAAGGTGATACCTTAGTTAATAGAATCGAATTAAAAGGTATTGATTTTAAAGAAGATGGTAACATTTTAGGTCACAAATTGGAATACAACTATAACTCTCACAATGTTTACATCACTGCTGACAAACAAAAGAATGGTATCAAAGCTAACTTCAAAATTAGACACAACATTGAAGATGGTGGTGTTCAATTAGCTGACCATTATCAACAAAATACTCCAATTGGTGATGGTCCAGTCTTGTTACCAGACAACCATTACTTATCCTATCAATCTGCCTTATCCAAAGATCCAAACGAAAAGAGAGACCACATGGTCTTGTTAGAATTTGTTACTGCTGCTGGTATTACCCATGGTATGGATGAATTGTACAAA
    # "pTpA": REPO_ROOT / "inference_data" / "deboer" / "pTpA", # CTAAAGGTGAAGAATTATTCACTGGTGTTGTCCCAATTTTGGTTGAATTAGATGGTGATGTTAATGGTCACAAATTTTCTGTCTCCGGTGAAGGTGAAGGTGATGCTACTTACGGTAAATTGACCTTAAAATTGATTTGTACTACTGGTAAATTGCCAGTTCCATGGCCAACCTTAGTCACTACTTTAGGTTATGGTTTGCAATGTTTTGCTAGATACCCAGATCATATGAAACAACATGACTTTTTCAAGTCTGCCATGCCAGAAGGTTATGTTCAAGAAAGAACTATTTTTTTCAAAGATGACGGTAACTACAAGACCAGAGCTGAAGTCAAGTTTGAAGGTGATACCTTAGTTAATAGAATCGAATTAAAAGGTATTGATTTTAAAGAAGATGGTAACATTTTAGGTCACAAATTGGAATACAACTATAACTCTCACAATGTTTACATCACTGCTGACAAACAAAAGAATGGTATCAAAGCTAACTTCAAAATTAGACACAACATTGAAGATGGTGGTGTTCAATTAGCTGACCATTATCAACAAAATACTCCAATTGGTGATGGTCCAGTCTTGTTACCAGACAACCATTACTTATCCTATCAATCTGCCTTATCCAAAGATCCAAACGAAAAGAGAGACCACATGGTCTTGTTAGAATTTGTTACTGCTGCTGGTATTACCCATGGTATGGATGAATTGTACAAA
    # "kosuri": REPO_ROOT / "inference_data" / "kosuri", # CGTAAAGGCGAAGAGCTGTTCACTGGTTTCGTCACTATTCTGGTGGAACTGGATGGTGATGTCAACGGTCATAAGTTTTCCGTGCGTGGCGAGGGTGAAGGTGACGCAACTAATGGTAAACTGACGCTGAAGTTCATCTGTACTACTGGTAAACTGCCGGTACCTTGGCCGACTCTGGTAACGACGCTGACTTATGGTGTTCAGTGCTTTGCTCGTTATCCGGACCACATGAAGCAGCATGACTTCTTCAAGTCCGCCATGCCGGAAGGCTATGTGCAGGAACGCACGATTTCCTTTAAGGATGACGGCACGTACAAAACGCGTGCGGAAGTGAAATTTGAAGGCGATACCCTGGTAAACCGCATTGAGCTGAAAGGCATTGACTTTAAAGAAGACGGCAATATCCTGGGCCATAAGCTGGAATACAATTTTAACAGCCACAATGTTTACATCACCGCCGATAAACAAAAAAATGGCATTAAAGCGAATTTTAAAATTCGCCACAACGTGGAGGATGGCAGCGTGCAGCTGGCTGATCACTACCAGCAAAACACTCCAATCGGTGATGGTCCTGTTCTGCTGCCAGACAATCACTATCTGAGCACGCAAAGCGTTCTGTCTAAAGATCCGAACGAGAAACGCGATCACATGGTTCTGCTGGAGTTCGTAACCGCAGCGGGCATCACGCATGGTATGGATGAACTGTACAAA
    # "lagator": REPO_ROOT / "inference_data" / "lagator", # AGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTGATGTTAATGGGCACAAATTTTCTGTCAGTGGAGAGGGTGAAGGTGATGCAACATACGGAAAACTTACCCTTAAATTGATTTGCACTACTGGAAAACTACCTGTTCCATGGCCAACACTTGTCACTACTTTGGGTTATGGTCTAATGTGCTTTGCTAGATACCCAGATCATATGAAACGGCATGACTTTTTCAAGAGTGCCATGCCCGAAGGTTATGTACAGGAAAGAACTATATTTTTCAAAGATGACGGGAACTACAAGACACGTGCTGAAGTCAAGTTTGAAGGTGATACCCTTGTTAATAGAATCGAGTTAAAAGGTATTGATTTTAAAGAAGATGGAAACATTCTTGGACACAAATTGGAATACAACTATAACTCACACAATGTATACATCACTGCAGACAAACAAAAGAATGGAATCAAAGCTAACTTCAAAATTAGACACAACATTGAAGATGGAGGTGTTCAACTAGCAGCCATTATCAACAAAATACTCCAATTGGCGATGGCCCTGTCCTTTTACCAGACAACCATTACCTGTCCTATCAATCTGCCCTTTCGAAAGATCCCAACGAAAAGAGAGACCACATGGTCCTTCTTGAGTTTGTAACAGCTGCTGGGATTACACATGGCATGGATGAACTATACAAA
    "zahm": REPO_ROOT / "inference_data" / "zahm", # GAAGATGCCAAAAACATTAAGAAGGGCCCAGCGCCATTCTACCCACTCGAAGACGGGACCGCCGGCGAGCAGCTGCACAAAGCCATGAAGCGCTACGCCCTGGTGCCCGGCACCATCGCCTTTACCGACGCACATATCGAGGTGGACATTACCTACGCCGAGTACTTCGAGATGAGCGTTCGGCTGGCAGAAGCTATGAAGCGCTATGGGCTGAATACAAACCATCGGATCGTGGTGTGCAGCGAGAATAGCTTGCAGTTCTTCATGCCCGTGTTGGGTGCCCTGTTCATCGGTGTGGCTGTGGCCCCAGCTAACGACATCTACAACGAGCGCGAGCTGCTGAACAGCATGGGCATCAGCCAGCCCACCGTCGTATTCGTGAGCAAGAAAGGGCTGCAAAAGATCCTCAACGTGCAAAAGAAGCTACCGATCATACAAAAGATCATCATCATGGATAGCAAGACCGACTACCAGGGCTTCCAAAGCATGTACACCTTCGTGACTTCCCATTTGCCACCCGGCTTCAACGAGTACGACTTCGTGCCCGAGAGCTTCGACCGGGACAAAACCATCGCCCTGATCATGAACAGTAGTGGCAGTACCGGATTGCCCAAGGGCGTAGCCCTACCGCACCGCACCGCTTGTGTCCGATTCAGTCATGCCCGCGACCCCATCTTCGGCAACCAGATCATCCCCGACACCGCTATCCTCAGCGTGGTGCCATTTCACCACGGCTTCGGCATGTTCACCACGCTGGGCTACTTGATCTGCGGCTTTCGGGTCGTGCTCATGTACCGCTTCGAGGAGGAGCTATTCTTGCGCAGCTTGCAAGACTATAAGATTCAATCTGCCCTGCTGGTGCCCACACTATTTAGCTTCTTCGCTAAGAGCACTCTCATCGACAAGTACGACCTAAGCAACTTGCACGAGATCGCCAGCGGCGGGGCGCCGCTCAGCAAGGAGGTAGGTGAGGCCGTGGCCAAACGCTTCCACCTACCAGGCATCCGCCAGGGCTACGGCCTGACAGAAACAACCAGCGCCATTCTGATCACCCCCGAAGGGGACGACAAGCCTGGCGCAGTAGGCAAGGTGGTGCCCTTCTTCGAGGCTAAGGTGGTGGACTTGGACACCGGCAAGACACTGGGTGTGAACCAGCGCGGCGAGCTGTGCGTCCGTGGCCCCATGATCATGAGCGGCTACGTTAACAACCCCGAGGCTACAAACGCTCTCATCGACAAGGACGGCTGGCTGCACAGCGGCGACATCGCCTACTGGGACGAGGACGAGCACTTCTTCATCGTGGACCGGCTGAAGAGCCTGATCAAATACAAGGGCTACCAGGTAGCCCCAGCCGAACTGGAGAGCATCCTGCTGCAACACCCCAACATCTTCGACGCCGGGGTCGCCGGCCTGCCCGACGACGATGCCGGCGAGCTGCCCGCCGCAGTCGTCGTGCTGGAACACGGTAAAACCATGACCGAGAAGGAGATCGTGGACTATGTGGCCAGCCAGGTTACAACCGCCAAGAAGCTGCGCGGTGGTGTTGTGTTCGTGGACGAGGTGCCTAAAGGACTGACCGGCAAGTTGGACGCCCGCAAGATCCGCGAGATTCTCATTAAGGCCAAGAAGGGCGGCAAGATCGCCGTG
}
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "inference_out_csv" / "AlphaGenome"
DEFAULT_MODEL_VERSION = "all_folds"
DEFAULT_CONTEXT_LENGTH = 131_072
DEFAULT_HUMAN_ONTOLOGY = "EFO:0001187"
VALID_SUFFIXES = {".txt", ".fa", ".fasta", ".fna", ".fas"}
DNA_ALPHABET = {"A", "C", "G", "T", "N"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run forced AlphaGenome inference on the local Nullsettes benchmark."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional local AlphaGenome checkpoint directory. If omitted, the script downloads from Hugging Face.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=DEFAULT_MODEL_VERSION,
        help="AlphaGenome model version to fetch from Hugging Face when --checkpoint-path is not set.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Length of the padded sequence fed into AlphaGenome.",
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="homo_sapiens",
        choices=("homo_sapiens", "mus_musculus"),
        help="AlphaGenome organism head to use.",
    )
    parser.add_argument(
        "--ontology-term",
        action="append",
        default=None,
        help="Optional ontology CURIE used to subset RNA-seq tracks. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--use-all-rna-tracks",
        action="store_true",
        help="Average across all non-padding RNA-seq tracks instead of subsetting by ontology.",
    )
    parser.add_argument(
        "--score-subsequence",
        type=str,
        default=None,
        help=(
            "Optional DNA subsequence to match inside each input sequence. "
            "If provided, only the matched region is aggregated into the final score."
        ),
    )
    parser.add_argument(
        "--score-subsequence-occurrence",
        type=int,
        default=None,
        help=(
            "Optional 1-based occurrence of --score-subsequence to use when it appears multiple times. "
            "If omitted, the script requires a unique match."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "gpu", "cpu"),
        help="JAX backend to use.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Which JAX device to use when --device is not auto.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional root containing the five benchmark folders. Defaults to aaa_final_data/server_data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where model outputs will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute CSVs even if they already exist.",
    )
    return parser.parse_args()


def resolve_input_dirs(input_root: Optional[Path]) -> Dict[str, Path]:
    if input_root is None:
        return DEFAULT_INPUT_DIRS

    input_root = input_root.resolve()
    return {
        "Abf1TATA": input_root / "deboer" / "Abf1TATA",
        "pTpA": input_root / "deboer" / "pTpA",
        "kosuri": input_root / "kosuri",
        "lagator": input_root / "lagator",
        "zahm": input_root / "zahm",
    }


def iter_sequence_files(input_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES and not path.name.startswith(".")
    )


def sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "default"


def sanitize_sequence(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return "".join(base if base in DNA_ALPHABET else "N" for base in seq)


def build_context_sequence(seq: str, context_length: int) -> Tuple[str, int, int, int]:
    seq = sanitize_sequence(seq)
    if len(seq) >= context_length:
        trim_start = (len(seq) - context_length) // 2
        trimmed = seq[trim_start : trim_start + context_length]
        return trimmed, 0, context_length, trim_start

    total_padding = context_length - len(seq)
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    centered = ("N" * left_padding) + seq + ("N" * right_padding)
    return centered, left_padding, len(seq), 0


def find_all_matches(seq: str, query: str) -> List[int]:
    matches: List[int] = []
    start = seq.find(query)
    while start != -1:
        matches.append(start)
        start = seq.find(query, start + 1)
    return matches


def resolve_score_window(
    centered_seq: str,
    *,
    insert_start: int,
    insert_length: int,
    score_subsequence: Optional[str],
    score_subsequence_occurrence: Optional[int],
) -> Tuple[int, int, int, int]:
    if score_subsequence is None:
        return insert_start, insert_start + insert_length, 0, insert_length

    if score_subsequence_occurrence is not None and score_subsequence_occurrence < 1:
        raise ValueError("--score-subsequence-occurrence must be >= 1.")

    effective_seq = centered_seq[insert_start : insert_start + insert_length]
    query = sanitize_sequence(score_subsequence)
    if not query:
        raise ValueError("--score-subsequence is empty after sanitization.")
    if len(query) > len(effective_seq):
        raise ValueError(
            "The requested score subsequence is longer than the effective input sequence passed to AlphaGenome."
        )

    matches = find_all_matches(effective_seq, query)
    if not matches:
        raise ValueError(
            "Could not find --score-subsequence inside the effective sequence region used for scoring."
        )

    if score_subsequence_occurrence is None:
        if len(matches) != 1:
            raise ValueError(
                f"--score-subsequence matched {len(matches)} regions. "
                "Pass --score-subsequence-occurrence to choose one."
            )
        match_start = matches[0]
    else:
        match_index = score_subsequence_occurrence - 1
        if match_index >= len(matches):
            raise ValueError(
                f"--score-subsequence-occurrence={score_subsequence_occurrence} was requested, "
                f"but only {len(matches)} match(es) were found."
            )
        match_start = matches[match_index]

    match_end = match_start + len(query)
    return insert_start + match_start, insert_start + match_end, match_start, match_end


def default_ontology_terms(args: argparse.Namespace) -> Optional[Sequence[str]]:
    if args.use_all_rna_tracks:
        return None
    if args.ontology_term:
        return args.ontology_term
    if args.organism == "homo_sapiens":
        return [DEFAULT_HUMAN_ONTOLOGY]
    return None


def resolve_jax_device(jax, device_name: str, device_index: int):
    if device_name == "auto":
        return None

    devices = jax.devices(device_name)
    if not devices:
        raise RuntimeError(f"No JAX devices available for backend '{device_name}'.")
    if device_index < 0 or device_index >= len(devices):
        raise IndexError(
            f"Requested device index {device_index}, but only {len(devices)} '{device_name}' device(s) were found."
        )
    return devices[device_index]


def load_model(args: argparse.Namespace, ontology_terms: Optional[Sequence[str]]):
    del ontology_terms

    if not ALPHAGENOME_ROOT.exists():
        raise FileNotFoundError(f"Local AlphaGenome repo not found: {ALPHAGENOME_ROOT}")

    sys.path.insert(0, str(ALPHAGENOME_ROOT / "src"))
    import huggingface_hub
    import jax
    from alphagenome_research.model import dna_model
    from alphagenome_research.model.metadata import metadata as metadata_lib

    organism = (
        dna_model.Organism.HOMO_SAPIENS
        if args.organism == "homo_sapiens"
        else dna_model.Organism.MUS_MUSCULUS
    )
    device = resolve_jax_device(jax, args.device, args.device_index)
    organism_settings = {
        organism: dna_model.OrganismSettings(metadata=metadata_lib.load(organism))
    }

    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    else:
        checkpoint_path = Path(
            huggingface_hub.snapshot_download(
                repo_id=f"google/alphagenome-{args.model_version.replace('_', '-').lower()}",
                token=os.environ.get("HF_TOKEN"),
            )
        )

    model = dna_model.create(
        checkpoint_path,
        organism_settings=organism_settings,
        device=device,
    )
    return model, dna_model, organism, checkpoint_path, device

"""
For AlphaGenome, the scoring logic is:
    The original short sequence is placed at the center of a fixed-length input window, currently 131,072 bp by default.
    The remaining positions are padded with N.
    The padded sequence is passed through AlphaGenome.
    We extract the RNA-seq prediction output.
    We take the prediction values over the positions corresponding to the inserted original sequence.
    We average those values across the selected positions and across the selected RNA-seq tracks.
    This final average is used as the sequence-level score.

So both models use the same overall strategy:
    center the short sequence in a long padded context,
    run the model,
    restrict to the region corresponding to the original sequence,
    average the output into one scalar.

The main difference is:
    Enformer averages over overlapping output bins from the selected Enformer head
    AlphaGenome averages over overlapping RNA-seq 1-bp predictions from the selected RNA-seq tracks
"""
def score_sequence(
    seq: str,
    *,
    model,
    dna_model_module,
    organism,
    context_length: int,
    ontology_terms: Optional[Sequence[str]],
    score_subsequence: Optional[str],
    score_subsequence_occurrence: Optional[int],
) -> Tuple[float, int, int]:
    centered_seq, insert_start, insert_length, source_offset = build_context_sequence(seq, context_length)
    score_start, score_end, relative_start, relative_end = resolve_score_window(
        centered_seq,
        insert_start=insert_start,
        insert_length=insert_length,
        score_subsequence=score_subsequence,
        score_subsequence_occurrence=score_subsequence_occurrence,
    )
    predictions = model.predict_sequence(
        centered_seq,
        organism=organism,
        requested_outputs=[dna_model_module.OutputType.RNA_SEQ],
        ontology_terms=ontology_terms,
    )
    if predictions.rna_seq is None:
        raise RuntimeError("AlphaGenome did not return RNA-seq predictions.")

    values = np.asarray(predictions.rna_seq.values, dtype=np.float32)
    if values.size == 0 or (values.ndim > 1 and values.shape[-1] == 0):
        raise ValueError(
            "AlphaGenome returned zero RNA-seq tracks. Try a different --ontology-term or pass --use-all-rna-tracks."
        )

    selected = values[score_start:score_end]
    if selected.size == 0:
        selected = values
    return (
        float(np.mean(selected)),
        source_offset + relative_start,
        source_offset + relative_end,
    )


def scoring_label(args: argparse.Namespace) -> str:
    if args.score_subsequence is None:
        return "full_sequence"

    subseq = sanitize_sequence(args.score_subsequence)
    label = f"matched_subsequence_{len(subseq)}bp"
    if args.score_subsequence_occurrence is not None:
        label = f"{label}_occ{args.score_subsequence_occurrence}"
    return sanitize_component(label)


def main() -> None:
    args = parse_args()
    input_dirs = resolve_input_dirs(args.input_root)
    ontology_terms = default_ontology_terms(args)
    model, dna_model_module, organism, checkpoint_path, device = load_model(args, ontology_terms)

    run_label = sanitize_component(
        checkpoint_path.name if args.checkpoint_path is not None else args.model_version
    )
    ontology_label = (
        "all_rna_tracks"
        if ontology_terms is None
        else sanitize_component("__".join(ontology_terms))
    )
    output_root = args.output_root.resolve() / run_label / args.organism / ontology_label / scoring_label(args)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Organism: {args.organism}")
    print(f"Ontology terms: {ontology_terms if ontology_terms is not None else 'ALL'}")
    if args.score_subsequence is None:
        print("Scoring region: full inserted sequence")
    else:
        occurrence = (
            f", occurrence {args.score_subsequence_occurrence}"
            if args.score_subsequence_occurrence is not None
            else ", requiring a unique match"
        )
        print(
            f"Scoring region: matched subsequence of length {len(sanitize_sequence(args.score_subsequence))} bp{occurrence}"
        )
    print(f"JAX device: {device if device is not None else 'auto'}")
    print(
        f"Input root: {args.input_root.resolve() if args.input_root else (REPO_ROOT / 'aaa_final_data' / 'server_data')}"
    )
    print(f"Output root: {output_root}")

    for dataset_name, input_dir in input_dirs.items():
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        dataset_output_dir = output_root / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        input_files = list(iter_sequence_files(input_dir))
        print(f"\n[{dataset_name}] {len(input_files)} files")

        for input_file in tqdm(input_files, desc=f"{dataset_name} files"):
            output_file = dataset_output_dir / f"{input_file.stem}.csv"
            if output_file.exists() and not args.overwrite:
                continue

            seqs = [str(record.seq) for record in SeqIO.parse(str(input_file), "fasta")]
            if not seqs:
                print(f"Skipping empty file: {input_file}")
                continue

            score_results = [
                score_sequence(
                    seq,
                    model=model,
                    dna_model_module=dna_model_module,
                    organism=organism,
                    context_length=args.context_length,
                    ontology_terms=ontology_terms,
                    score_subsequence=args.score_subsequence,
                    score_subsequence_occurrence=args.score_subsequence_occurrence,
                )
                for seq in tqdm(seqs, desc=f"{input_file.stem} seqs", leave=False)
            ]
            score_df = pd.DataFrame(
                {
                    "seqs": [sanitize_sequence(seq) for seq in seqs],
                    "scores": [score for score, _, _ in score_results],
                }
            )
            if args.score_subsequence is not None:
                score_df["score_region_start"] = [start for _, start, _ in score_results]
                score_df["score_region_end"] = [end for _, _, end in score_results]
            score_df.to_csv(output_file, sep="\t", index=False)

    print("AlphaGenome inference finished.")


if __name__ == "__main__":
    main()
