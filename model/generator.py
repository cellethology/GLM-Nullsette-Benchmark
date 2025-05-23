import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio import SeqIO
import pandas as pd
import os
from tqdm import tqdm
from utils.ll_calculation import compute_ll_clm


def main():
    """
    GenerTeam/GENERator-eukaryote-3b-base
    GenerTeam/GENERator-eukaryote-1.2b-base
    """
    tokenizer = AutoTokenizer.from_pretrained("GenerTeam/GENERator-eukaryote-3b-base",
                                              trust_remote_code=True
                                              )
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained("GenerTeam/GENERator-eukaryote-3b-base",
                                                 trust_remote_code=True
                                                 ).cuda()
    model.eval()

    dir_paths = ['processed_data/deboer/Abf1TATA']
    out_root_path = 'output/GENERator/eukaryote-3b-base'
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
                inputs = tokenizer(seq,
                                   return_tensors="pt",
                                   add_special_tokens=True,
                                   padding=True,
                                   truncation=True
                                   )
                inputs = inputs.to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    logits = outputs.logits

                llr_score = compute_ll_clm(logits, inputs['input_ids'], reduction='mean').item()
                scores.append(llr_score)
            df = pd.DataFrame({'seqs': seqs, 'scores': scores})
            df.to_csv(output_file, sep='\t', index=False)
            print(f"Results saved to {output_file}.")


if __name__ == '__main__':
    main()
