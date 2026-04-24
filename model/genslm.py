from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
GENSLM_ROOT = REPO_ROOT / "genslm" / "genslm"
DEFAULT_INPUT_DIRS = {
    # "Abf1TATA": REPO_ROOT / "inference_data" / "deboer" / "Abf1TATA",
    # "pTpA": REPO_ROOT / "inference_data" / "deboer" / "pTpA",
    # "kosuri": REPO_ROOT / "inference_data" / "kosuri",
    # "lagator": REPO_ROOT / "inference_data" / "lagator",
    "zahm": REPO_ROOT / "inference_data" / "zahm",
}
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "inference_out_csv" / "GenSLM"
VALID_SUFFIXES = {".txt", ".fa", ".fasta", ".fna", ".fas"}
GENSLM_MODELS = {
    # "genslm_25M_patric": {
    #     "config": GENSLM_ROOT / "architectures" / "neox" / "neox_25,290,752.json",
    #     "tokenizer": GENSLM_ROOT / "tokenizer_files" / "codon_wordlevel_69vocab.json",
    #     "weights": "weights/patric_25m_epoch01-val_loss_0.57_bias_removed.pt",
    #     "seq_length": 2048,
    #     "kmer_size": 3,
    # },
    # "genslm_250M_patric": {
    #     "config": GENSLM_ROOT / "architectures" / "neox" / "neox_244,464,576.json",
    #     "tokenizer": GENSLM_ROOT / "tokenizer_files" / "codon_wordlevel_69vocab.json",
    #     "weights": "weights/patric_250m_epoch00_val_loss_0.48_attention_removed.pt",
    #     "seq_length": 2048,
    #     "kmer_size": 3,
    # },
    "genslm_2.5B_patric": {
        "config": GENSLM_ROOT / "architectures" / "neox" / "neox_2,533,931,008.json",
        "tokenizer": GENSLM_ROOT / "tokenizer_files" / "codon_wordlevel_69vocab.json",
        "weights": "weights/patric_2.5b_epoch00_val_los_0.29_bias_removed.pt",
        "seq_length": 2048,
        "kmer_size": 3,
    },
    # "genslm_25B_patric": {
    #     "config": GENSLM_ROOT / "architectures" / "neox" / "neox_25,076,188,032.json",
    #     "tokenizer": GENSLM_ROOT / "tokenizer_files" / "codon_wordlevel_69vocab.json",
    #     "weights": "model-epoch00-val_loss0.70-v2.pt",
    #     "seq_length": 2048,
    #     "kmer_size": 3,
    # },
    # "genslm_25M_covid": {
    #     "config": GENSLM_ROOT / "architectures" / "neox" / "neox_25,290,752_10240_pos_embed.json",
    #     "tokenizer": GENSLM_ROOT / "tokenizer_files" / "dna_wordlevel_100vocab.json",
    #     "weights": "model-epoch91-val_loss0.01.pt",
    #     "seq_length": 10240,
    #     "kmer_size": 1,
    # },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GenSLM zero-shot sequence scoring on the local Nullsettes benchmark."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="genslm_2.5B_patric",
        choices=sorted(GENSLM_MODELS),
        help="GenSLM checkpoint family to load.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(os.environ.get("GENSLM_MODEL_CACHE_DIR", ".")),
        help="Directory containing the downloaded GenSLM .pt checkpoint.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Optional direct path to a GenSLM checkpoint file. Overrides --model-cache-dir.",
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


def group_by_kmer(seq: str, kmer_size: int) -> str:
    seq = seq.upper()
    return " ".join(seq[i : i + kmer_size] for i in range(0, len(seq), kmer_size))


def load_tokenizer(model_id: str) -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    model_info = GENSLM_MODELS[model_id]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(str(model_info["tokenizer"]))
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.model_max_length = int(model_info["seq_length"])
    return tokenizer


def resolve_weight_path(args: argparse.Namespace, model_info: Dict[str, object]) -> Path:
    if args.weights_path is not None:
        weight_path = args.weights_path.resolve()
    else:
        weight_path = (args.model_cache_dir / str(model_info["weights"])).resolve()

    if not weight_path.exists():
        raise FileNotFoundError(
            "GenSLM checkpoint not found. Expected: "
            f"{weight_path}. Pass --weights-path or --model-cache-dir to the downloaded .pt file."
        )
    return weight_path


def load_model(args: argparse.Namespace):
    from transformers import AutoConfig, AutoModelForCausalLM

    model_info = GENSLM_MODELS[args.model_id]
    config = AutoConfig.from_pretrained(str(model_info["config"]))
    model = AutoModelForCausalLM.from_config(config)

    weight_path = resolve_weight_path(args, model_info)
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model, model_info, weight_path


def score_sequences(
    seqs: List[str],
    tokenizer: PreTrainedTokenizerFast,
    model,
    device: torch.device,
    batch_size: int,
    seq_length: int,
    kmer_size: int,
) -> List[float]:
    scores: List[float] = [0.0] * len(seqs)
    ordered_indices = sorted(range(len(seqs)), key=lambda idx: len(seqs[idx]), reverse=True)

    for start in tqdm(range(0, len(ordered_indices), batch_size), desc="Batches", leave=False):
        batch_indices = ordered_indices[start : start + batch_size]
        batch_seqs = [group_by_kmer(seqs[idx], kmer_size) for idx in batch_indices]
        encoded = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=seq_length,
            return_tensors="pt",
        )
        model_inputs = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }

        with torch.no_grad():
            outputs = model(**model_inputs)
            batch_scores = compute_causal_llr(
                logits=outputs.logits,
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
            )

        for idx, score in zip(batch_indices, batch_scores.tolist()):
            scores[idx] = float(score)

    return scores


def main() -> None:
    args = parse_args()
    input_dirs = resolve_input_dirs(args.input_root)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.model_id)
    model, model_info, weight_path = load_model(args)
    model = model.to(device)
    model.eval()

    output_root = args.output_root.resolve() / args.model_id
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_id}")
    print(f"Checkpoint: {weight_path}")
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
                seq_length=int(model_info["seq_length"]),
                kmer_size=int(model_info["kmer_size"]),
            )
            pd.DataFrame({"seqs": seqs, "scores": scores}).to_csv(
                output_file, sep="\t", index=False
            )

    print("GenSLM inference finished.")


if __name__ == "__main__":
    main()
