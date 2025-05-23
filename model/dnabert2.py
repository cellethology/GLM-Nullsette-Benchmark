import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio import SeqIO
import pandas as pd
import os
from tqdm import tqdm
from utils.ll_calculation import compute_llr_mlm


def main():
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).cuda()
    model.eval()

    dir_paths = ['processed_data/deboer/Abf1TATA']
    out_root_path = 'output/DNABERT_2'
    os.makedirs(out_root_path, exist_ok=True)

    for root_dir_path in dir_paths:
        this_dir_name = root_dir_path.split('/')[-1]
        out_path = os.path.join(out_root_path, this_dir_name)
        os.makedirs(out_path, exist_ok=True)
        for file in os.listdir(root_dir_path):
            input_fasta = os.path.join(root_dir_path, file)
            output_file = os.path.join(out_path, file.split('.')[0] + '.csv')

            seqs = [str(record.seq) for record in SeqIO.parse(input_fasta, 'fasta')]
            scores = []

            for seq in tqdm(seqs, desc="Processing sequences"):
                inputs = tokenizer(seq, return_tensors="pt")
                input_ids = inputs["input_ids"].cuda()
                attention_mask = (input_ids != tokenizer.pad_token_id)

                with torch.no_grad():
                    logits = model(input_ids,
                                   attention_mask=attention_mask.cuda()
                                   ).logits

                llr_score = compute_llr_mlm(logits, input_ids, reduction='mean').item()
                scores.append(llr_score)

            df = pd.DataFrame({'seqs': seqs, 'scores': scores})
            df.to_csv(output_file, sep='\t', index=False)
            print(f"Results saved to {output_file}.")


if __name__ == '__main__':
    main()
