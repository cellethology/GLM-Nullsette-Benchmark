import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio import SeqIO
import pandas as pd
import os
from tqdm import tqdm
import argparse
from utils.ll_calculation import compute_ll_clm


def main():
    parser = argparse.ArgumentParser(description="Compute log-likelihood scores for sequences using METAGENE-1 model.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input FASTA files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output CSV files.')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("metagene-ai/METAGENE-1")
    model = AutoModelForCausalLM.from_pretrained("metagene-ai/METAGENE-1",
                                                 output_hidden_states=True,
                                                 output_attentions=True).cuda()
    model.eval()

    root_dir_path = args.input_dir
    this_dir_name = root_dir_path.split('/')[-1]
    out_path = os.path.join(args.output_dir, this_dir_name)
    os.makedirs(out_path, exist_ok=True)

    for file in os.listdir(root_dir_path):
        input_fasta = os.path.join(root_dir_path, file)
        output_file = os.path.join(out_path, file.split('.')[0] + '.csv')

        seqs = [str(record.seq) for record in SeqIO.parse(input_fasta, 'fasta')]
        scores = []

        for seq in tqdm(seqs, desc="Processing sequences"):
            inputs = tokenizer(seq, return_tensors="pt")
            inputs = inputs.to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits

            ll_score = compute_ll_clm(logits, inputs['input_ids'], reduction='mean').item()
            scores.append(ll_score)

        df = pd.DataFrame({'seqs': seqs, 'scores': scores})
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
