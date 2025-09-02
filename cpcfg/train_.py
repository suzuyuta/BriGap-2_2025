# This code is based on
# Kim et al. (2019) "Compound Probabilistic Context-Free Grammars for Grammar Induction"
# The original source code is available on their GitHub page:
# https://github.com/harvardnlp/compound-pcfg

#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import numpy as np
import time
import logging
from data import Dataset
from utils import *
from models import CompPCFG
from torch.nn.init import xavier_uniform_

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-train.pkl')
parser.add_argument('--val_file', default='data/ptb-val.pkl')
parser.add_argument('--save_path', default='compound-pcfg.pt', help='where to save the model')

# Model options
# Generative model parameters
parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
# Inference network parameters
parser.add_argument('--h_dim', default=512, type=int, help='hidden dim for variational LSTM')
parser.add_argument('--w_dim', default=512, type=int, help='embedding dim for variational LSTM')
# Optimization options
parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument('--max_grad_norm', default=3, type=float, help='gradient clipping parameter')
parser.add_argument('--max_length', default=30, type=float, help='max sentence length cutoff start')
parser.add_argument('--len_incr', default=1, type=int, help='increment max length each epoch')
parser.add_argument('--final_max_length', default=40, type=int, help='final max length cutoff')
parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=1000, help='print stats after N batches')

parser.add_argument('--val_every', type=int, default=3000, help='validation every N steps')
parser.add_argument('--incr_step', default=3000, type=int, help='increment max length every N steps')
parser.add_argument('--no_curriculum', action='store_true', help='disable curriculum learning if we write --no_curriculum in command line')
parser.add_argument('--early_stopping_patience', default=5, type=int)
parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--min_steps', default=10000, type=int)

parser.add_argument('--train_data_output', action='store_true')
parser.add_argument('--train_data_output_folder', type=str, default='./batch_data/')
parser.add_argument('--save_each_val', action='store_true')

parser.add_argument('--use_mean', type=bool, default=False)
parser.add_argument('--fix_z', type=bool, default=False)

def main(args):
    total_steps = 0  # <- add
    validation_step = args.val_every  # <- add
    length_incr_step = args.incr_step  # <- add
    next_incr_step = length_incr_step  #

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = Dataset(args.train_file)
    val_data = Dataset(args.val_file)
    train_sents = train_data.batch_size.sum()
    vocab_size = int(train_data.vocab_size)
    max_len = max(val_data.sents.size(1), train_data.sents.size(1))
    max_train_snt_len = train_data.sents.size(1)
    print('Train: %d sents / %d batches, Val: %d sents / %d batches' %
          (train_data.sents.size(0), len(train_data), val_data.sents.size(0), len(val_data)))
    print('Vocab size: %d, Max Sent Len: %d, Max Train Sent Len: %d' % (vocab_size, max_len, max_train_snt_len))
    print('Random seed: ', args.seed)
    print('max_length: ', args.max_length)
    print('final_max_length: ', args.final_max_length)
    print('nb of pre-terminal states: ', args.t_states)
    print('nb of non-terminal states: ', args.nt_states)

    if args.no_curriculum:
        print('\nWe do not use Curriculum Learning\n')
    else:
        print('\nCurriculum Learning')
        print('Learning start with sentence length ', args.max_length)
        print('increment length every ', args.incr_step, ' steps\n')

    if args.early_stopping:
        print('early stopping patience: ', args.early_stopping_patience)
    else:
        print('No early stopping')

    print('\nSave Path', args.save_path)

    output_prefix = str(args.save_path)[:-3]
    snt_output_path = './' + output_prefix + '_snt.json'
    val_output_path = './' + output_prefix + '_val.json'
    snt_output = {}
    val_output = {}

    batch_data_dir = ''
    batch_data_dict = {}
    batch_data_file_name = args.save_path[:-3]
    if args.train_data_output:
        batch_data_dir = args.train_data_output_folder
        if not os.path.exists(batch_data_dir):
            os.makedirs(batch_data_dir)

    cuda.set_device(args.gpu)
    model = CompPCFG(vocab=vocab_size,
                     state_dim=args.state_dim,
                     t_states=args.t_states,
                     nt_states=args.nt_states,
                     h_dim=args.h_dim,
                     w_dim=args.w_dim,
                     z_dim=args.z_dim)
    for name, param in model.named_parameters():
        if param.dim() > 1:
            xavier_uniform_(param)
    print("model architecture")
    print(model)
    model.train()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    best_val_ppl = 1e5
    best_val_f1 = 0
    epoch = 0

    patience_counter = 0  # <- add
    early_stopping_trigger_len = False  # <- add

    while epoch < args.num_epochs:
        early_stopping_trigger = False
        start_time = time.time()
        epoch += 1
        print('Starting epoch %d' % epoch)
        train_nll = 0.
        train_kl = 0.
        num_sents = 0.
        num_words = 0.
        all_stats = [[0., 0., 0.]]
        b = 0
        for i in np.random.permutation(len(train_data)):
            total_steps += 1  # <- add

            if args.train_data_output:
                batch_sents, batch_length, batch_size, batch_gold_actions, batch_gold_spans, \
                    batch_gold_binary_trees, batch_other_data = train_data[i]
                sents_data = batch_sents.tolist()
                # print(sents_data)
                sents_len_data = [len(b) for b in sents_data]  # we do not need to remove BOS and EOS from length of a sentence b/c cpcfg does not use BOS and EOS
                tmp_dict = {total_steps: sents_len_data}
                """
                batch_data = {
                    'sents': batch_sents.tolist(),
                    'length': batch_length,
                    'size': batch_size,
                    'gold_actions': batch_gold_actions,
                    'gold_spans': batch_gold_spans,
                    'gold_binary_trees': batch_gold_binary_trees,
                    'other_data': batch_other_data
                }

                batch_file = os.path.join(batch_data_dir, f'batch_{epoch}_{step}.json')
                with open(batch_file, 'w') as f:
                    json.dump(batch_data, f)
                """
                batch_file = os.path.join(batch_data_dir, f'{batch_data_file_name}.json')
                batch_data_dict.update(tmp_dict)
                with open(batch_file, 'w') as f:
                    json.dump(batch_data_dict, f)

            if total_steps % validation_step == 0:
                print('Val at step %d' % total_steps)  # <- add
                val_ppl, val_f1, output_data_ = eval(val_data, model)  # <- add

                def output_json_val(total_steps_, output_data):
                    recon_ppl_ = output_data[0]
                    kl_ = output_data[1]
                    ppl_elbo_ = output_data[2]
                    corpus_f1_ = output_data[3]
                    snt_f1_ = output_data[4]

                    json_output = {total_steps_: (recon_ppl_, kl_, ppl_elbo_, corpus_f1_, snt_f1_)}
                    return json_output

                tmp_val_output = output_json_val(total_steps, output_data_)
                val_output.update(tmp_val_output)
                with open(val_output_path, 'w', encoding='utf-8') as val_out:
                    json.dump(val_output, val_out)

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_val_f1 = val_f1
                    # patience_counter = 0  # reset patience counter
                    checkpoint = {
                        'args': args.__dict__,
                        'model': model.cpu(),
                        'word2idx': train_data.word2idx,
                        'idx2word': train_data.idx2word
                    }
                    print('Saving checkpoint to %s' % args.save_path)
                    torch.save(checkpoint, args.save_path)

                    if args.save_each_val:
                        save_path_ = args.save_path[:-3] + '_' + str(total_steps) + '.pt'
                        print('Saving checkpoint to %s' % save_path_)
                        torch.save(checkpoint, save_path_)

                    model.cuda()
                else:
                    if total_steps > args.min_steps:
                        patience_counter += 1
                    if args.early_stopping and patience_counter >= args.early_stopping_patience:
                        print('Early stopping triggered. No improvement for ', args.early_stopping_patience, ' consecutive validation steps.')
                        early_stopping_trigger = True

            if early_stopping_trigger:  # and early_stopping_trigger_len:
                break

            # increment max_length based on steps
            if not args.no_curriculum and total_steps >= next_incr_step:
                args.max_length = min(args.final_max_length, args.max_length + args.len_incr)
                print('Incrementing max_length to', args.max_length)
                next_incr_step += length_incr_step
                # if args.max_length == max_train_snt_len:
                # early_stopping_trigger_len = True
                # print('')

            b += 1
            sents, length, batch_size, _, gold_spans, gold_binary_trees, _ = train_data[i]
            if length > args.max_length or length == 1:  # length filter based on curriculum
                continue
            sents = sents.cuda()
            optimizer.zero_grad()
            nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True, use_mean=args.use_mean, fix_z=args.fix_z)
            (nll + kl).mean().backward()
            train_nll += nll.sum().item()
            train_kl += kl.sum().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            num_sents += batch_size
            num_words += batch_size * (length + 1)  # we implicitly generate </s> so we explicitly count it
            for bb in range(batch_size):
                span_b = [(a[0], a[1]) for a in argmax_spans[bb]]  # ignore labels
                span_b_set = set(span_b[:-1])
                update_stats(span_b_set, [set(gold_spans[bb][:-1])], all_stats)
            if b % args.print_every == 0:
                all_f1 = get_f1(all_stats)
                param_norm = sum([p.norm() ** 2 for p in model.parameters()]).item() ** 0.5
                gparam_norm = sum([p.grad.norm() ** 2 for p in model.parameters()
                                   if p.grad is not None]).item() ** 0.5
                log_str = 'Epoch: %d, Batch: %d/%d, |Param|: %.6f, |GParam|: %.2f,  LR: %.4f, ' + \
                          'ReconPPL: %.2f, KL: %.4f, PPLBound: %.2f, ValPPL: %.2f, ValF1: %.2f, ' + \
                          'CorpusF1: %.2f, Throughput: %.2f examples/sec'
                print(log_str %
                      (epoch, b, len(train_data), param_norm, gparam_norm, args.lr,
                       np.exp(train_nll / num_words), train_kl / num_sents,
                       np.exp((train_nll + train_kl) / num_words), best_val_ppl, best_val_f1,
                       all_f1[0], num_sents / (time.time() - start_time)))
                output_tmp = (epoch, total_steps, param_norm, gparam_norm, args.lr,
                              np.exp(train_nll / num_words), train_kl / num_sents,
                              all_f1[0])

                def output_json_snt(output_data):
                    epoch_ = output_data[0]
                    steps_ = output_data[1]
                    para_norm_ = output_data[2]
                    grad_norm_ = output_data[3]
                    lr_ = output_data[4]
                    recon_ppl_ = output_data[5]
                    kl_ = output_data[6]
                    corpus_f1_ = output_data[7]

                    json_output = {steps_: (epoch_, para_norm_, grad_norm_, lr_, recon_ppl_, kl_, corpus_f1_)}

                    return json_output

                tmp_snt_output = output_json_snt(output_tmp)
                snt_output.update(tmp_snt_output)
                with open(snt_output_path, 'w', encoding='utf-8') as snt_out:
                    json.dump(snt_output, snt_out)

                # print an example parse
                tree = get_tree_from_binary_matrix(binary_matrix[0], length)
                action = get_actions(tree)
                sent_str = [train_data.idx2word[word_idx] for word_idx in list(sents[0].cpu().numpy())]
                print("Pred Tree: %s" % get_tree(action, sent_str))
                print("Gold Tree: %s" % get_tree(gold_binary_trees[0], sent_str))

        if early_stopping_trigger:  # and early_stopping_trigger_len:
            break
        """
        # args.max_length = min(args.final_max_length, args.max_length + args.len_incr)
        print('--------------------------------')
        print('Checking validation perf...')
        val_ppl, val_f1, output_data_ = eval(val_data, model)
        print('--------------------------------')

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_val_f1 = val_f1
            checkpoint = {
                'args': args.__dict__,
                'model': model.cpu(),
                'word2idx': train_data.word2idx,
                'idx2word': train_data.idx2word
            }
            print('Saving checkpoint to %s' % args.save_path)
            torch.save(checkpoint, args.save_path)
            model.cuda()
        """

def eval(data, model):
    model.eval()
    num_sents = 0
    num_words = 0
    total_nll = 0.
    total_kl = 0.
    corpus_f1 = [0., 0., 0.]
    sent_f1 = []
    with torch.no_grad():
        for i in range(len(data)):
            sents, length, batch_size, _, gold_spans, gold_binary_trees, other_data = data[i]
            if length == 1:
                continue
            sents = sents.cuda()
            # note that for unsuperised parsing, we should do model(sents, argmax=True, use_mean = True)
            # but we don't for eval since we want a valid upper bound on PPL for early stopping
            # see eval.py for proper MAP inference
            nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True, use_mean=args.use_mean, fix_z=args.fix_z)
            total_nll += nll.sum().item()
            total_kl += kl.sum().item()
            num_sents += batch_size
            num_words += batch_size * (length + 1)  # we implicitly generate </s> so we explicitly count it
            for b in range(batch_size):
                span_b = [(a[0], a[1]) for a in argmax_spans[b]]  # ignore labels
                span_b_set = set(span_b[:-1])
                gold_b_set = set(gold_spans[b][:-1])
                tp, fp, fn = get_stats(span_b_set, gold_b_set)
                corpus_f1[0] += tp
                corpus_f1[1] += fp
                corpus_f1[2] += fn
                # sent-level F1 is based on L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py

                model_out = span_b_set
                std_out = gold_b_set
                overlap = model_out.intersection(std_out)
                prec = float(len(overlap)) / (len(model_out) + 1e-8)
                reca = float(len(overlap)) / (len(std_out) + 1e-8)
                if len(std_out) == 0:
                    reca = 1.
                    if len(model_out) == 0:
                        prec = 1.
                f1 = 2 * prec * reca / (prec + reca + 1e-8)
                sent_f1.append(f1)
    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    recon_ppl = np.exp(total_nll / num_words)
    ppl_elbo = np.exp((total_nll + total_kl) / num_words)
    kl = total_kl / num_sents
    print('ReconPPL: %.2f, KL: %.4f, PPL (Upper Bound): %.2f' %
          (recon_ppl, kl, ppl_elbo))
    print('Corpus F1: %.2f, Sentence F1: %.2f' %
          (corpus_f1 * 100, sent_f1 * 100))
    model.train()
    output_data = (recon_ppl, kl, ppl_elbo, corpus_f1 * 100, sent_f1 * 100)
    return ppl_elbo, sent_f1 * 100, output_data


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
