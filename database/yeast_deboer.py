import pickle

yeast_deboer_dict = pickle.load(open('database/pkls/deBoer_promoter.pkl', 'rb'))
yeast_deboer_pTpA = {
    'distal_promoter': 'GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATC',
    'promoter': 'pair with deBoer_promoter_dict',
    'proximal_promoter': 'GGTTACGGCTGTTTCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGA',
    'linker1': 'ag'.upper(),
    'kozak_part1': 'gcaaag'.upper(),
    'start_codon': 'ATG',
    'kozak+4': 'T',
    'cds': ('ctaaaggtgaagaattattcactggtgttgtcccaattttggttgaattagatggtgatgttaatggtcacaaattttctgtctccggtgaaggtgaaggtgatgct'
           'acttacggtaaattgaccttaaaattgatttgtactactggtaaattgccagttccatggccaaccttagtcactactttaggttatggtttgcaatgttttgctag'
           'atacccagatcatatgaaacaacatgactttttcaagtctgccatgccagaaggttatgttcaagaaagaactatttttttcaaagatgacggtaactacaagacca'
           'gagctgaagtcaagtttgaaggtgataccttagttaatagaatcgaattaaaaggtattgattttaaagaagatggtaacattttaggtcacaaattggaatacaac'
           'tataactctcacaatgtttacatcactgctgacaaacaaaagaatggtatcaaagctaacttcaaaattagacacaacattgaagatggtggtgttcaattagctga'
           'ccattatcaacaaaatactccaattggtgatggtccagtcttgttaccagacaaccattacttatcctatcaatctgccttatccaaagatccaaacgaaaagagag'
           'accacatggtcttgttagaatttgttactgctgctggtattacccatggtatggatgaattgtacaaa').upper(),
    'stop_codon': 'TAA',
    'linker2': 'ggcgcgccacttctaaataa'.upper(),
    'terminator': ('gcgaatttcttatgatttatgatttttattattaaataagttataaaaaaaataagtgtatacaaattttaaagtgactcttaggttttaaaacgaaaat'
                  'tcttattcttgagtaactctttcctgtaggtcaggttgctttctcaggtatagtatgaggtcgctcttattgaccacacctctaccgg').upper()
}
yeast_deboer_Abf1TATA = {
    'distal_promoter': 'GCTAGCTGATTATGGTAACTCTATCGGACTTGAGGGATCACATTTCACGCAGTATAGTTC',
    'promoter': 'pair with deBoer_promoter_dict',
    'proximal_promoter': 'GGTTTATTGTTTATAAAAATTAGTTTAAACTGTTGTATATTTTTTCATCTAACGGAACAATAGTAGGTTACGCTAGTTTGGATCCTGAGCAGG'
                         'CAAGATAAACGA',
    'linker1': 'ag'.upper(),
    'kozak_part1': 'gcaaag'.upper(),
    'start_codon': 'ATG',
    'kozak+4': 'T',
    'cds': ('ctaaaggtgaagaattattcactggtgttgtcccaattttggttgaattagatggtgatgttaatggtcacaaattttctgtctccggtgaaggtgaaggtgatgct'
           'acttacggtaaattgaccttaaaattgatttgtactactggtaaattgccagttccatggccaaccttagtcactactttaggttatggtttgcaatgttttgctag'
           'atacccagatcatatgaaacaacatgactttttcaagtctgccatgccagaaggttatgttcaagaaagaactatttttttcaaagatgacggtaactacaagacca'
           'gagctgaagtcaagtttgaaggtgataccttagttaatagaatcgaattaaaaggtattgattttaaagaagatggtaacattttaggtcacaaattggaatacaac'
           'tataactctcacaatgtttacatcactgctgacaaacaaaagaatggtatcaaagctaacttcaaaattagacacaacattgaagatggtggtgttcaattagctga'
           'ccattatcaacaaaatactccaattggtgatggtccagtcttgttaccagacaaccattacttatcctatcaatctgccttatccaaagatccaaacgaaaagagag'
           'accacatggtcttgttagaatttgttactgctgctggtattacccatggtatggatgaattgtacaaa').upper(),
    'stop_codon': 'TAA',
    'linker2': 'ggcgcgccacttctaaataa'.upper(),
    'terminator': ('gcgaatttcttatgatttatgatttttattattaaataagttataaaaaaaataagtgtatacaaattttaaagtgactcttaggttttaaaacgaaaat'
                  'tcttattcttgagtaactctttcctgtaggtcaggttgctttctcaggtatagtatgaggtcgctcttattgaccacacctctaccgg').upper()
}
deboer_database = {
    'deBoer_promoter_dict': yeast_deboer_dict,
    'deBoer_cassette': {
        'pTpA': yeast_deboer_pTpA,
        'Abf1TATA': yeast_deboer_Abf1TATA
    }
}