import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio import SeqIO
import pandas as pd
import os
from tqdm import tqdm
from utils.ll_calculation import compute_llr_mlm


def main():
    """
    InstaDeepAI/nucleotide-transformer-v2-500m-multi-species: 500M_multispecies
    InstaDeepAI/nucleotide-transformer-500m-human-ref: 500M_human
    InstaDeepAI/nucleotide-transformer-2.5b-multi-species: 2.5B_multispecies
    """
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
                                              trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
                                                 trust_remote_code=True)
    model = model.to(device)
    model.eval()
    max_length = tokenizer.model_max_length

    dir_paths = ['processed_data/deboer/Abf1TATA']
    out_root_path = 'output/NucleotideTransformer/2.5B_multispecies'
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
                tokens_ids = \
                tokenizer.batch_encode_plus([seq], return_tensors="pt", padding="max_length", max_length=max_length)[
                    "input_ids"]
                tokens_ids = tokens_ids.to(device)
                attention_mask = tokens_ids != tokenizer.pad_token_id
                attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    logits = model(
                        tokens_ids,
                        attention_mask=attention_mask,
                        encoder_attention_mask=attention_mask,
                        output_hidden_states=False
                    ).logits

                llr_score = compute_llr_mlm(logits, tokens_ids, reduction='mean').item()
                scores.append(llr_score)

            df = pd.DataFrame({'seqs': seqs, 'scores': scores})
            df.to_csv(output_file, sep='\t', index=False)
            print(f"Results saved to {output_file}.")


if __name__ == '__main__':
    main()
