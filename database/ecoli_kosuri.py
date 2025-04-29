import pickle

"""
1. read kosuri_dict_filter.pkl file to get combination of promoter and rbs with their corresponding expression values
2. terminator is not defined and found in the .DNA file
3. stop codon is taa from the .DNA file
"""
kosuri_dict = pickle.load(open('database/pkls/kosuri_promoter_rbs.pkl', 'rb'))
ecoli_kosuri = {
    'promoter': 'pair with kosuri_promoter_rbs_dict',
    'rbs': 'pair with kosuri_promoter_rbs_dict',
    'start_codon': 'atg'.upper(),
    'cds': ('cgtaaaggcgaagagctgttcactggtttcgtcactattctggtggaactggatggtgatgtcaacggtcataagttttccgtgcgtggcgagggtgaaggtgacg'
            'caactaatggtaaactgacgctgaagttcatctgtactactggtaaactgccggtaccttggccgactctggtaacgacgctgacttatggtgttcagtgctttgc'
            'tcgttatccggaccacatgaagcagcatgacttcttcaagtccgccatgccggaaggctatgtgcaggaacgcacgatttcctttaaggatgacggcacgtacaaa'
            'acgcgtgcggaagtgaaatttgaaggcgataccctggtaaaccgcattgagctgaaaggcattgactttaaagaagacggcaatatcctgggccataagctggaat'
            'acaattttaacagccacaatgtttacatcaccgccgataaacaaaaaaatggcattaaagcgaattttaaaattcgccacaacgtggaggatggcagcgtgcagct'
            'ggctgatcactaccagcaaaacactccaatcggtgatggtcctgttctgctgccagacaatcactatctgagcacgcaaagcgttctgtctaaagatccgaacgag'
            'aaacgcgatcacatggttctgctggagttcgtaaccgcagcgggcatcacgcatggtatggatgaactgtacaaa').upper(),
    'stop_codon': 'taa'.upper(),
    'linker_between_stopcodon_and_terminator': 'taacgctgatagtgctagtgtagatcgctactagagccaggcat'.upper(),
    'terminator': 'caaataaaacgaaaggctcagtcgaaagactgggcctttcgttttatctgttgtttgtcggtgaacgctctc'.upper()
}
kosuri_database = {
    'kosuri_promoter_rbs_dict': kosuri_dict,
    'kosuri_cassette': ecoli_kosuri
}