"""
Train a model on TACRED.
"""

import os, sys, time
import random
import argparse
from datetime import datetime

import json
import numpy as np
from shutil import copyfile
from sklearn.utils import shuffle
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model.trainer import GCNTrainer
from dataloader import DataLoader
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/bb3')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
parser.add_argument('--emb_dim', type=int, default=200, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)

parser.add_argument('--prune_k', default=-1, type=int, help='Prune the dependency tree to <= K distance off the dependency path; set to -1 for no pruning.')
parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='sum', help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0, help='L2-penalty for all pooling output.')
parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

parser.add_argument('--no-rnn', dest='rnn', action='store_false', help='Do not use RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adadelta', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=50, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu: args.cuda = False
elif args.cuda: torch.cuda.manual_seed(args.seed)

def posCount(data):
    pos = 0
    for d in data:
        if d['relation'] == 1:
            pos +=1 
    return pos

def getData(data_dir):
    # load train and dev datasets
    print(data_dir)
    with open(data_dir + '/train.json') as infile:
        first_data = json.load(infile)
    with open(data_dir + '/dev.json') as infile:
        second_data = json.load(infile)

    # permute dataset
    first_data = shuffle(first_data)
    second_data = shuffle(second_data)
    
    total_data = first_data + second_data
    total_data = shuffle(total_data)

    train_data = total_data[:-300]
    dev_data = total_data[-300:-150]
    test_data = total_data[-150:]

    # print(train_data[:10])
    print('Train pos/total: %s/%s'%(posCount(train_data), len(train_data)))
    print('Dev pos/total: %s/%s'%(posCount(dev_data), len(dev_data)))
    print('Test pos/total: %s/%s'%(posCount(test_data), len(test_data)))

    return train_data, dev_data, test_data

def getData2(data_dir):
    # load train and dev datasets
    with open(data_dir + '/train.json') as infile:
        train_data = json.load(infile)
    with open(data_dir + '/dev.json') as infile:
        dev_data = json.load(infile)

    # print(train_data[:10])
    print('Train pos/total: %s/%s'%(posCount(train_data), len(train_data)))
    print('Dev pos/total: %s/%s'%(posCount(dev_data), len(dev_data)))

    return train_data, dev_data, dev_data

def run(args):
    # make opt
    opt = vars(args)
    label2id = constant.LABEL_TO_ID
    opt['num_class'] = len(label2id)

    # load vocab
    vocab_file = opt['vocab_dir'] + '/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    opt['vocab_size'] = vocab.size
    emb_file = opt['vocab_dir'] + '/embedding.npy'
    emb_matrix = np.load(emb_file)
    assert emb_matrix.shape[0] == vocab.size
    assert emb_matrix.shape[1] == opt['emb_dim']

    # load data
    train_data, dev_data, test_data = getData2(opt['data_dir'])
    train_batch = DataLoader(train_data, opt['batch_size'], opt, vocab, evaluation=False)
    dev_batch = DataLoader(dev_data, opt['batch_size'], opt, vocab, evaluation=True)
    test_batch = DataLoader(test_data, opt['batch_size'], opt, vocab, evaluation=True)

    # eval_dataset = OrderedDict({'dev': dev_batch, 'test': test_batch})
    eval_dataset = OrderedDict({'dev': dev_batch})
    
    # set model_id
    model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
    model_save_dir = opt['save_dir'] + '/' + model_id
    opt['model_save_dir'] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)

    # save config
    helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
    vocab.save(model_save_dir + '/vocab.pkl')
    file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'])
    helper.print_config(opt)

    # model
    if not opt['load']:
        # create new model
        print(emb_matrix.shape)
        trainer = GCNTrainer(opt, emb_matrix=emb_matrix)
    else:
        # load pretrained model
        model_file = opt['model_file'] 
        print("Loading model from {}".format(model_file))
        model_opt = torch_utils.load_config(model_file)
        model_opt['optim'] = opt['optim']
        trainer = GCNTrainer(model_opt)
        trainer.load(model_file)   

    current_lr = opt['lr']
    global_step = 0
    global_start_time = time.time()
    format_str = 'epoch {}/{}, step {}/{} , loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
    max_steps = len(train_batch) * opt['num_epoch']

    best_f1 = 0
    best_epoch =  1
    dev_f1_history = []
    eval_dict = defaultdict()
    for dn in eval_dataset:
        eval_dict[dn] = defaultdict()

    # start training2
    for epoch in range(1, opt['num_epoch']+1):
        counter = -1
        train_loss = 0
        start_time = time.time()
        for i, batch in enumerate(train_batch):
            loss = trainer.update(batch)
            train_loss += loss
            counter += 1
            if counter % 20 == 0:
                logger.info('epoch %s  >> %2.3f : cost = %2.3f completed in %2.3f (sec) <<'%(
                    epoch, (counter+1)*100.0/len(train_batch),
                    loss, (time.time()-start_time)))
                
        logger.info('epoch %s  >> 100.00 : total cost %2.3f completed in %2.3f (sec) <<'%(
            epoch, train_loss, (time.time()-start_time)))

        # train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
        # save model
        model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
        trainer.save(model_file, epoch)
        
        logger.info('>>>>>>>>>>>>>>PERFORMANCE<<<<<<<<<<<<<<<<<<')
        for dn in eval_dataset:
            eval_dict[dn][epoch] = evaluate(trainer, eval_dataset[dn])
            perPrint(eval_dict[dn][epoch], dn)

        dev_f1 = eval_dict['dev'][epoch]['f1']
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            logger.info('>>>>>>>>>>>>>>NEW BEST PERFORMANCE<<<<<<<<<<<<<<<<<<')
            for dn in eval_dataset:
                perPrint(eval_dict[dn][epoch], dn)
            copyfile(model_file, model_save_dir + '/best_model.pt')

        # lr schedule
        if len(dev_f1_history) > opt['decay_epoch'] \
           and dev_f1 <= dev_f1_history[-1] \
           and opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
            current_lr *= opt['lr_decay']
            trainer.update_lr(current_lr)
            
        dev_f1_history += []

    logger.info('>>>>>>>>>>>>>>BEST PERFORMANCE, epoch %d<<<<<<<<<<<<<<<<<<' % best_epoch)
    for dn in eval_dataset:
        perPrint(eval_dict[dn][best_epoch], dn)

    
def evaluate(trainer, data_batch):
    preds = []
    for i, batch in enumerate(data_batch):
        pred, _, _ = trainer.predict(batch)
        preds += pred
    
    return score(data_batch.gold(), preds)

def score(gold, preds):

    gold = np.asarray(gold, dtype='int32')
    preds = np.asarray(preds, dtype='int32')
    
    zeros = np.zeros(preds.shape, dtype='int')
    numPred = np.sum(np.not_equal(preds, zeros))
    numKey = np.sum(np.not_equal(gold, zeros))
    
    predictedIds = np.nonzero(preds)
    preds_eval = preds[predictedIds]
    keys_eval = gold[predictedIds]
    correct = np.sum(np.equal(preds_eval, keys_eval))
    
    logger.info('correct : {}, numPred : {}, numKey : {}'.format(correct, numPred, numKey))
    
    precision = 100.0 * correct / numPred if numPred > 0 else 0.0
    recall = 100.0 * correct / numKey
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0. else 0.0

    return {'p' : precision, 'r' : recall, 'f1' : f1}

def perPrint(perfs, dn):
    logger.info(dn + ': %2.3f %2.3f %2.3f'%(perfs['p'], perfs['r'], perfs['f1']))

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
