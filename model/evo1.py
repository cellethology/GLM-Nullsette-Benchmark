import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import pandas as pd

from Bio import SeqIO
from tqdm import tqdm
from evo import Evo
from evo.scoring import prepare_batch, logits_to_logprobs
from stripedhyena.model import StripedHyena
from stripedhyena.tokenizer import CharLevelTokenizer

import numpy as np
import torch
from typing import List


def score_sequences(
        seqs: List[str],
        model: StripedHyena,
        tokenizer: CharLevelTokenizer,
        reduce_method: str = 'mean',
        device: str = 'cuda:0',
) -> List[float]:
    """
    Computes the model log-likelihood scores for sequences in `seqs`.
    Uses `reduce_method` to take the mean or sum across the likelihoods at each
    position (default: `'mean'`).

    Returns a list of scalar scores corresponding to the reduced log-likelihoods for
    each sequence.
    """
    input_ids, seq_lengths = prepare_batch(seqs, tokenizer, device=device, prepend_bos=True)
    assert (len(seq_lengths) == input_ids.shape[0])

    with torch.inference_mode():
        logits, _ = model(input_ids)

    logprobs = logits_to_logprobs(logits, input_ids, trim_bos=True)
    logprobs = logprobs.float().cpu().numpy()

    if reduce_method == 'mean':
        reduce_func = np.mean
    elif reduce_method == 'sum':
        reduce_func = np.sum
    else:
        raise ValueError(f'Invalid reduce_method {reduce_method}')

    conclude_scores = [
        reduce_func(logprobs[idx][:seq_lengths[idx]])
        for idx in range(len(seq_lengths))
    ]
    return conclude_scores


def main():
    parser = argparse.ArgumentParser(description='Generate sequences using the Evo model.')

    parser.add_argument('--input_dir', required=True, help='Input FASTA file path')
    parser.add_argument('--output_dir', required=True, help='Output path to save csv files of tab-separated value')
    parser.add_argument('--model_name', type=str, default='evo-1-131k-base', help='Evo model name')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of sequences to evaluate at a time')
    parser.add_argument('--device', type=str, default='cuda', help='Device for generation')
    args = parser.parse_args()

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # Load model
    evo_model = Evo(args.model_name)
    model, tokenizer = evo_model.model, evo_model.tokenizer

    model.to(args.device)
    model.eval()

    # Load sequences.
    for file in os.listdir(args.input_dir):
        print(f'Processing {file}...')
        file_name = file.split('.')[0]
        seqs = [str(record.seq) for record in SeqIO.parse(os.path.join(args.input_dir, file), 'fasta')]
        scores = []
        for i in tqdm(range(0, len(seqs), args.batch_size)):
            batch_seqs = seqs[i:i + args.batch_size]
            batch_scores = score_sequences(
                batch_seqs,
                model,
                tokenizer,
                device=args.device,
            )
            scores.extend(batch_scores)

        df = pd.DataFrame({'seqs': seqs, 'scores': scores})
        df.to_csv(os.path.join(args.output_dir, f'{file_name}.csv'), sep='\t', index=False)


if __name__ == '__main__':
    main()
