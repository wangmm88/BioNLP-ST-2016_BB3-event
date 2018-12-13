
"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
from collections import Counter

from utils import constant
from gensim.models import KeyedVectors

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', help='data directory.')
    parser.add_argument('--emb_dir', help='embedding directory.')
    parser.add_argument('--emb_file', default='wikipedia-pubmed-and-PMC-w2v.bin', help='embedding file.')
    parser.add_argument('--emb_dim', type=int, default=200, help='embedding dimension.')
    parser.add_argument('--out_dir', help='embedding and vocab output directory.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    parser.add_argument('--lower', action='store_true', help='If specified, lowercase all words.')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # input and output files
    emb_file = args.emb_dir + '/' + args.emb_file
    emb_output_file = args.out_dir + '/embedding.npy'
    vocab_output_file = args.out_dir + '/vocab.pkl'

    # load data and build vocab
    data_names = ['train', 'dev', 'test']
    tokens = []
    for dn in data_names:
        data_file = args.data_dir + '/' + dn + '.json'
        tokens += load_tokens(data_file)
    
    if args.lower: tokens = [w.lower() for w in vocab]
        
    # load and build embedding
    print("loading emb...")
    emb = KeyedVectors.load_word2vec_format(emb_file, binary=True)
    emb, word2id = build_embedding(tokens, emb, args.emb_dim, args.min_freq)
    
    print("dumping to files...")
    with open(vocab_output_file, 'wb') as f:
        pickle.dump(word2id, f)
    np.save(emb_output_file, emb)
    print("all done.")

def build_embedding(tokens, emb_origin, emb_dim, min_freq):
    # build vocab
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    counter = Counter(t for t in tokens)
    if min_freq > 0:
        vocab = sorted([t for t in counter if counter.get(t) >= min_freq],
                       key=counter.get, reverse=True)
    else:
        vocab = sorted([t for t in counter if t in emb_origin],
                       key=counter.get, reverse=True)
    
    # add special tokens and entity mask tokens
    vocab = constant.VOCAB_PREFIX + entity_masks() + vocab

    # init embedding
    vocab_size = len(vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, emb_dim))
    emb[constant.PAD_ID] = 0 # <pad> should be all 0
    
    word2id = {w: i for i, w in enumerate(vocab)}
    for w in word2id:
        if w in emb_origin:
            emb[word2id[w]] = emb_origin[w]
    
    return emb, word2id

def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            ts = d['token']
            # ss, se, os, oe = d['subj_start'], d['subj_end'], d['obj_start'], d['obj_end']
            # # do not create vocab for entity words
            # ts[ss:se+1] = ['<PAD>']*(se-ss+1)
            # ts[os:oe+1] = ['<PAD>']*(oe-os+1)
            tokens += list(filter(lambda t: t!='<PAD>', ts))
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

def entity_masks():
    """ Get all entity mask tokens as a list. """
    masks = []
    subj_entities = list(constant.SUBJ_NER_TO_ID.keys())[2:]
    obj_entities = list(constant.OBJ_NER_TO_ID.keys())[2:]
    masks += ["SUBJ-" + e for e in subj_entities]
    masks += ["OBJ-" + e for e in obj_entities]
    return masks

if __name__ == '__main__':
    main()


