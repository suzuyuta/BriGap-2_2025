import json
import pprint
import os
import re
import numpy as np
from argparse import ArgumentParser
from misc.tree import parse, SimpleTree, fill_yield

def _get_terminals(tree):
    terminals = []
    fill_yield(tree, terminals)
    return terminals

def remove_unary(tree):
    if not tree.base['children']:
        return tree
    if len(tree) == 1 and len(tree[0]) > 0:
        return remove_unary(tree[0])
    tree.base['children'] = [remove_unary(child) for child in tree.base['children']]
    return SimpleTree.make_tree(tree.label(), tree.base['children'])


def extract_np_spans(tree, non_terminal: str, start_idx=0):
    spans = []
    index = start_idx

    def traverse(node, idx):
        nonlocal spans
        if not node.base['children']:
            return idx + 1
        # if node.label() == non_terminal:
        if non_terminal == 'CP':
            if 'CP_suj_tr' in node.label() or 'CP_suj_obl' in node.label() or 'CP_suj' in node.label() \
                    or 'CP_obl' in node.label() or 'CP_dobj' in node.label():
                start = idx
                for child in node.base['children']:
                    idx = traverse(child, idx)
                spans.append((start, idx))
            else:
                for child in node.base['children']:
                    idx = traverse(child, idx)

        elif non_terminal == 'VP':
            if 'VP' in node.label() or 'VPobl' in node.label():
                start = idx
                for child in node.base['children']:
                    idx = traverse(child, idx)
                spans.append((start, idx))
            else:
                for child in node.base['children']:
                    idx = traverse(child, idx)

        else:
            if non_terminal in node.label():
                start = idx
                for child in node.base['children']:
                    idx = traverse(child, idx)
                spans.append((start, idx))
            else:
                for child in node.base['children']:
                    idx = traverse(child, idx)

        return idx

    traverse(tree, index)
    return spans

def lire_fichier(path):
    with open(path, 'r', encoding='utf8') as f:
        out = f.readlines()
    return out

def find_matching_nt(pred_tree, spans):
    pred_tree = remove_unary(pred_tree)
    found_nts = []
    found_spans = []

    def traverse(node, idx):
        nonlocal found_nts
        if not node.base['children']:
            return idx + 1
        start = idx
        for child in node.base['children']:
            idx = traverse(child, idx)
        end = idx
        if (start, end) in spans:
            found_nts.append(node.label())
            found_spans.append((start, end))
        return idx

    traverse(pred_tree, 0)
    return found_nts, found_spans


def get_stat(stat_dict: dict):
    res_dict = {}

    def _ratio(g_trees: list, p_trees: list):
        return sum(p_trees) / sum(g_trees)

    for l in stat_dict:
        if l == 'overall':
            g_ = stat_dict['overall']['nt_g']
            p_ = stat_dict['overall']['nt_p']
            overall_score = p_ / g_
            res_dict.update({'overall': overall_score})
            print('overall_score', overall_score)
        else:
            g_ = stat_dict[l]['nb_nt_g']
            p_ = stat_dict[l]['nb_nt_p']
            try:
                l_score = _ratio(g_, p_)
            except ZeroDivisionError:
                print(f'ZeroDivisionError at {l}')
                l_score = 0
                # pprint.pprint(stat_dict)
                # print(l)
                # print(g_)
                # exit(0)
            res_dict.update({l: l_score})
            # print('score', l, l_score)
    return res_dict

def save_json(output_file, output_path):
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(output_file, f)

def calcul_nt(g_path, p_path):

    gold_file = lire_fichier(g_path)
    pred_file = lire_fichier(p_path)

    stat_dict = {}
    all_results = {}
    comp_dict = {}
    good_nt = []
    total_nt_g = 0
    total_nt_p = 0

    for i, (g, p) in enumerate(zip(gold_file, pred_file)):
        length = len(_get_terminals(parse(g)))
        if length not in stat_dict:
            stat_dict.update({length: {'nb_nt_p': [], 'nb_nt_g': []}})
        g_tree = remove_unary(parse(g))
        p_tree = remove_unary(parse(p))
        g_spans = extract_np_spans(g_tree, args.nt)
        nb_nt_g = len(g_spans)
        non_terminal_in_pred, spans_in_pred = find_matching_nt(p_tree, g_spans)

        nt = list(set(non_terminal_in_pred))
        nb_nt_p = len(non_terminal_in_pred)
        for nt_ in nt:
            if nt_ not in good_nt:
                good_nt.append(nt_)
        stat_dict[length]['nb_nt_p'].append(nb_nt_p)
        stat_dict[length]['nb_nt_g'].append(nb_nt_g)
        all_results.update({i: {'length': length, 'nt_g': nb_nt_g, 'nt_p': nb_nt_p}})

        if nb_nt_g != nb_nt_p:
            comp_dict.update({i: {'gold': str(g_tree), 'pred': str(p_tree), 'res': (nb_nt_g, nb_nt_p)}})

        total_nt_g += nb_nt_g
        total_nt_p += nb_nt_p

    stat_dict.update({'overall': {'nt_g': total_nt_g, 'nt_p': total_nt_p, 'good_nt': good_nt}})

    return stat_dict, comp_dict

def get_data_type(file_name):
    data_type = re.search(r'(?<=gold-parse_).+(?=_\d*k)', file_name).group(0)
    data_size = re.search(r'(?<=_)\d+k(?=_)', file_name).group(0)
    seed = re.search(r'(?<=_)\d(?=\.txt)', file_name).group(0)

    return data_type, data_size, seed


def main(args):
    result_dict = {}
    overall_stat_dict = {}
    all_comp_dict = {}
    data_type = re.search(r'(?<=\./).+(?=/parsed)', args.folder_path).group(0)
    if 'flat' in args.folder_path:
        data_type = data_type + '_flat-test'

    parsed_files_g = [f for f in os.listdir(args.folder_path) if re.search(r'^gold-parse', f)]

    for f in parsed_files_g:
        print('\nAnalysing', f)
        data_type_tmp, data_size, seed = get_data_type(f)
        if data_type_tmp != data_type:
            data_type = data_type_tmp

        def initialize_dict(d: dict, data_type_, data_size_, seed_):
            if data_type_ not in d:
                d.update({data_type_: {args.data_option: {}}})
            if data_size_ not in d[data_type_][args.data_option]:
                d[data_type_][args.data_option].update({data_size_: {}})
            if seed_ not in d[data_type_][args.data_option][data_size_]:
                d[data_type_][args.data_option][data_size_].update({seed_: {}})

            return d

        result_dict = initialize_dict(result_dict, data_type, data_size, seed)
        overall_stat_dict = initialize_dict(overall_stat_dict, data_type, data_size, seed)
        all_comp_dict = initialize_dict(all_comp_dict, data_type, data_size, seed)
        parsed_file_p = re.sub('gold', 'pred', f)

        g_path = args.folder_path + f
        p_path = args.folder_path + parsed_file_p
        stat_dict, comp_dict = calcul_nt(g_path, p_path)

        result_dict[data_type][args.data_option][data_size][seed].update(stat_dict)
        overall_stat_dict[data_type][args.data_option][data_size][seed].update(get_stat(stat_dict))
        if comp_dict:
            all_comp_dict[data_type][args.data_option][data_size][seed].update(comp_dict)
            # pprint.pprint(all_comp_dict)
    # output_path = './' + data_type + '_' + args.output_option + args.nt + '_good_prediction.json'
    # save_json(result_dict, output_path)
    stat_dict_output_path = './' + data_type + '_' + args.output_option + args.nt + '_good_prediction_stat.json'
    save_json(overall_stat_dict, stat_dict_output_path)
    # comp_dict_output_path = './' + data_type + '_' + args.output_option + args.nt + '_prediction_comp.json'
    # save_json(all_comp_dict, comp_dict_output_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-folder_path', type=str)
    parser.add_argument('-nt', type=str, default='NP')
    parser.add_argument('-data_option', type=str, default='')
    parser.add_argument('-output_option', type=str, default='')
    args = parser.parse_args()
    main(args)

