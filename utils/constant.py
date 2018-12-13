"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'Bacteria': 2}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'Habitat': 2, 'Geographical': 3}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'Bacteria': 2, 'Habitat': 3, 'Geographical': 4}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1,
             'DT': 2, 'NN': 3, 'IN': 4, 'JJ': 5, 'NNS': 6, 'VBN': 7, 'NNP': 8, 'HYPH': 9, 'CC': 10,
             '.': 11, 'CD': 12, ',': 13, 'VBD': 14, 'VBP': 15, 'MD': 16, 'VB': 17, 'RBR': 18, 'RB': 19, ':': 20,
             'VBG': 21, 'VBZ': 22, 'TO': 23, 'AFX': 24, 'PRP': 25, 'WDT': 26, '-LRB-': 27, '-RRB-': 28, 'PRP$': 29, 'ADD': 30,
             'JJR': 31, 'WRB': 32, 'LS': 33, 'UH': 34, 'RBS': 35, 'RP': 36, 'XX': 37, 'SYM': 38, 'FW': 39, 'POS': 40,
             'JJS': 41, 'EX': 42, '``': 43, "''": 44, 'WP': 45, 'WP$': 46, '': 47, 'PDT': 48, '_SP': 49}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1,
                'det': 2, 'ROOT': 3, 'prep': 4, 'amod': 5, 'pobj': 6, 'acl': 7, 'compound': 8, 'punct': 9, 'cc': 10,
                'conj': 11, 'nummod': 12, 'nsubjpass': 13, 'auxpass': 14, 'appos': 15, 'dobj': 16, 'nsubj': 17, 'ccomp': 18, 'mark': 19, 'aux': 20,
                'advmod': 21, 'nmod': 22, 'neg': 23, 'acomp': 24, 'npadvmod': 25, 'agent': 26, 'xcomp': 27, 'advcl': 28, 'relcl': 29, 'pcomp': 30,
                'parataxis': 31, 'attr': 32, 'poss': 33, 'csubj': 34, 'csubjpass': 35, 'prt': 36, 'oprd': 37, 'dep': 38, 'case': 39, 'intj': 40,
                'quantmod': 41, 'meta': 42, 'expl': 43, 'preconj': 44, 'dative': 45, '': 46, 'predet': 47}

NEGATIVE_LABEL = 'no_relation'

LABEL_TO_ID = {'0': 0, '1': 1}

INFINITY_NUMBER = 1e12
