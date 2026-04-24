from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIRS = {
    "Abf1TATA": REPO_ROOT / "inference_data" / "deboer" / "Abf1TATA",
    "pTpA": REPO_ROOT / "inference_data" / "deboer" / "pTpA",
    "kosuri": REPO_ROOT / "inference_data" / "kosuri",
    "lagator": REPO_ROOT / "inference_data" / "lagator",
    "zahm": REPO_ROOT / "inference_data" / "zahm",
}
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "inference_out_csv" / "HyenaDNA"
MODEL_MAX_LENGTHS = {
    "LongSafari/hyenadna-tiny-1k-seqlen-hf": 1024,
    "LongSafari/hyenadna-small-32k-seqlen-hf": 32768,
    "LongSafari/hyenadna-medium-160k-seqlen-hf": 160000,
    "LongSafari/hyenadna-medium-450k-seqlen-hf": 450000,
    "LongSafari/hyenadna-large-1m-seqlen-hf": 1_000_000,
}
VALID_SUFFIXES = {".txt", ".fa", ".fasta", ".fna", ".fas"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HyenaDNA zero-shot sequence scoring on the local Nullsettes benchmark."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="LongSafari/hyenadna-large-1m-seqlen-hf",
        help="Hugging Face model id or a local pretrained model directory.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override the model context length. Needed if --model-name is a local path.",
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
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
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


def compute_causal_llr(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    reduction: str = "mean",
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]

    if attention_mask is None:
        shifted_mask = torch.ones_like(shifted_input_ids, dtype=torch.bool)
    else:
        shifted_mask = attention_mask[:, 1:].bool()

    log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~shifted_mask, 0.0)

    if reduction == "sum":
        return token_log_probs.sum(dim=-1)
    if reduction == "mean":
        denom = shifted_mask.sum(dim=-1).clamp_min(1)
        return token_log_probs.sum(dim=-1) / denom
    raise ValueError(f"Invalid reduction method: {reduction}")


def score_sequences(
    seqs: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[float]:
    scores: List[float] = [0.0] * len(seqs)
    ordered_indices = sorted(range(len(seqs)), key=lambda idx: len(seqs[idx]), reverse=True)

    for start in tqdm(range(0, len(ordered_indices), batch_size), desc="Batches", leave=False):
        batch_indices = ordered_indices[start : start + batch_size]
        batch_seqs = [seqs[idx] for idx in batch_indices]
        encoded = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        model_inputs = {"input_ids": encoded["input_ids"].to(device)}
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(**model_inputs)
            batch_scores = compute_causal_llr(
                logits=outputs.logits,
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
            )

        for idx, score in zip(batch_indices, batch_scores.tolist()):
            scores[idx] = float(score)

    return scores


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    input_dirs = resolve_input_dirs(args.input_root)
    device = torch.device(args.device)
    max_length = args.max_length if args.max_length is not None else MODEL_MAX_LENGTHS.get(args.model_name)
    if max_length is None:
        raise ValueError(
            "Unknown HyenaDNA context length. Pass --max-length when using a local model path."
        )
    model_slug = args.model_name.split("/")[-1]
    output_root = args.output_root.resolve() / model_slug
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.cls_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Input root: {args.input_root.resolve() if args.input_root else (REPO_ROOT / 'aaa_final_data' / 'server_data')}")
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

            seqs = [str(record.seq).upper() for record in SeqIO.parse(str(input_file), "fasta")]
            if not seqs:
                print(f"Skipping empty file: {input_file}")
                continue

            scores = score_sequences(
                seqs=seqs,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=args.batch_size,
                max_length=max_length,
            )
            pd.DataFrame({"seqs": seqs, "scores": scores}).to_csv(
                output_file, sep="\t", index=False
            )

    print("HyenaDNA inference finished.")


if __name__ == "__main__":
    main()
