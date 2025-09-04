import os
import pprint
import re
import json
from collections import Counter, defaultdict
from argparse import ArgumentParser
from misc.tree import parse, SimpleTree, fill_yield, ParseError
from eval_scorer_copy import Scorer

def _get_terminals(tree):
    terminals = []
    fill_yield(tree, terminals)
    return terminals
class EvalDataLoading:

    def __init__(self, data_path_):
        self.data_path = data_path_
        self.sent = []
        self.gold = []
        self.pred = []

    @staticmethod
    def clean(corpus):
        cleaned = re.sub(r'<PAD>|</S>.*|\s*<S>\s*|[sgp]:\s*|\n|\s{2,}', '', corpus)
        return cleaned

    def separate_data(self):
        with open(self.data_path, 'r') as f:
            data = [self.clean(l) for l in f]
        self.sent, self.gold, self.pred = data[::3], data[1::3], data[2::3]
        return zip(self.sent, self.gold, self.pred)

    @staticmethod
    def parse_trees(zipped_data, remove_punc=False):
        results = {'gold': [], 'pred': [], 'rb': []}
        rb_stat = {}

        for s, g, p in zipped_data:
            seq = s.strip().split()
            try:
                pred_t = parse(p)
                if len(seq) != len(_get_terminals(pred_t)):
                    pred_t = EvalDataLoading.right_branching(seq)
                    rb_stat[len(seq)] = rb_stat.get(len(seq), 0) + 1
            except ParseError:
                pred_t = EvalDataLoading.right_branching(seq)
                rb_stat[len(seq)] = rb_stat.get(len(seq), 0) + 1

            gold_t = parse(g)
            pred_t, _ = EvalDataLoading.replace_term(pred_t, seq, 0)
            gold_t, _ = EvalDataLoading.replace_term(gold_t, seq, 0)
            results['pred'].append(str(pred_t))
            results['gold'].append(str(gold_t))

            rb_tree = EvalDataLoading.right_branching(seq)
            rb_tree, _ = EvalDataLoading.replace_term(rb_tree, seq, 0)
            results['rb'].append(str(rb_tree))

        return results

    @staticmethod
    def replace_term(t, seq, i):
        if not t.children():
            return SimpleTree(label=seq[i], children=[]), i + 1
        else:
            children = [EvalDataLoading.replace_term(child, seq, i)[0] for child in t]
            return SimpleTree(label=t.label(), children=children), i

    @staticmethod
    def right_branching(seq):
        if len(seq) == 1:
            return SimpleTree(label='X', children=[SimpleTree(label=seq[0], children=[])])
        left = SimpleTree(label='X', children=[SimpleTree(label=seq[0], children=[])])
        return SimpleTree(label='X', children=[left, EvalDataLoading.right_branching(seq[1:])])

    @staticmethod
    def left_branching(seq):
        if len(seq) == 1:
            return SimpleTree(label='X', children=[SimpleTree(label=seq[0], children=[])])
        right = SimpleTree(label='X', children=[SimpleTree(label=seq[-1], children=[])])
        return SimpleTree(label='X', children=[EvalDataLoading.left_branching(seq[:-1]), right])

    @staticmethod
    def make_tmp_file(input_data_dict, output_prefix='../../eval', output_suffix='_tmp.txt'):
        output_paths = {}
        for key, data in input_data_dict.items():
            path = f'{output_prefix}{key}{output_suffix}'
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(data))
            output_paths[key] = path
        return output_paths

    @staticmethod
    def evaluation_(gold_path, test_path):
        with open(gold_path, encoding='utf8') as gold_f, open(test_path, encoding='utf8') as test_f:
            return Scorer().score_corpus(gold_f, test_f)

class StatisticsCalculate:

    def __init__(self, results, type_='length'):
        self.results = results
        self.type_ = type_

    def calculate_f1_scores(self):
        f1_scores = []
        matched_brackets = 0
        gold_brackets = 0
        test_brackets = 0

        for result in self.results:
            precision = result.matched_brackets / result.test_brackets if result.test_brackets else 0
            recall = result.matched_brackets / result.gold_brackets if result.gold_brackets else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
            f1_scores.append(f1)
            matched_brackets += result.matched_brackets
            gold_brackets += result.gold_brackets
            test_brackets += result.test_brackets

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        micro_precision = matched_brackets / test_brackets if test_brackets else 0
        micro_recall = matched_brackets / gold_brackets if gold_brackets else 0
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0

        return macro_f1, micro_f1

class PreterminalPatternExtract:
    def __init__(self, file_path):
        self.file_path = file_path

    @staticmethod
    def extract_preterminal_pattern(tree_line):
        preterminal_pattern = re.compile(r'\((\w+)\s+\w+\'?\)')
        matches = preterminal_pattern.findall(tree_line)
        return '-'.join(matches)

    def generate_patterns(self):
        patterns_ = {}
        with open(self.file_path, 'r') as f:
            for i, line in enumerate(f):
                line = re.sub(r'\[\[', '', line)
                line = re.sub(r'\]\]', '', line)
                pattern = self.extract_preterminal_pattern(line.strip())
                patterns_[i] = pattern
        return patterns_


def unify_files(gold_path, pred_path, output_file):
    with open(gold_path, 'r') as in_g, open(pred_path, 'r') as in_p:
        gold_trees = [parse(line.strip()) for line in in_g.readlines()]
        pred_trees = [parse(line.strip()) for line in in_p.readlines()]

    results = []
    for g, p in zip(gold_trees, pred_trees):
        words = []
        fill_yield(g, words)
        sentence = ' '.join(words)
        g_str = str(_transform_tree(g))
        p_str = str(_transform_tree(p))
        results.extend([sentence, g_str, p_str])

    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(results))
    return output_file

def _transform_tree(t):
    if t:
        return SimpleTree('X', [_transform_tree(c) for c in t]) if len(t) > 1 or not t[0] else _transform_tree(t[0])
    return t

def evaluate_and_calculate(args):
    data_loader = EvalDataLoading(args.input_path)
    data_zip = data_loader.separate_data()
    parsed_dict = data_loader.parse_trees(data_zip, remove_punc=args.remove_punc)
    output_paths = data_loader.make_tmp_file(parsed_dict)

    # Évaluer les arbres prédits contre les arbres gold
    res = data_loader.evaluation_(output_paths['gold'], output_paths['pred'])
    res_rb = data_loader.evaluation_(output_paths['gold'], output_paths['rb'])

    return res, res_rb

def get_score(result):
    idx = result.ID
    precision = result.matched_brackets / result.test_brackets if result.test_brackets else 0
    recall = result.matched_brackets / result.gold_brackets if result.gold_brackets else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    length = result.length

    return idx, f1, length

def analyze_f1_by_length(results):
    length_f1_stats = {}
    length_macro_f1 = {}
    length_micro_f1 = {}

    for result in results:
        _, f1, length = get_score(result)

        if length not in length_f1_stats:
            length_f1_stats[length] = {'f1_scores': [], 'matched_brackets': 0, 'test_brackets': 0, 'gold_brackets': 0}
        length_f1_stats[length]['f1_scores'].append(f1)
        length_f1_stats[length]['matched_brackets'] += result.matched_brackets
        length_f1_stats[length]['test_brackets'] += result.test_brackets
        length_f1_stats[length]['gold_brackets'] += result.gold_brackets

    for length, stats in length_f1_stats.items():
        f1_scores = stats['f1_scores']
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        matched = stats['matched_brackets']
        test = stats['test_brackets']
        gold = stats['gold_brackets']

        micro_precision = matched / test if test else 0
        micro_recall = matched / gold if gold else 0
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0

        length_macro_f1[length] = macro_f1
        length_micro_f1[length] = micro_f1

    return length_macro_f1, length_micro_f1, {length: dict(Counter(stats['f1_scores'])) for length, stats in length_f1_stats.items()}

def analyse_by_syntax_pattern(results, syntax_patterns: dict):
    f1_syntax = {}
    syntax_f1 = {}
    # {length: {f1: [synt_patterns], f1: [synt_patterns]}, ...}
    for result in results:
        idx, f1, length = get_score(result)
        syntax_pattern = syntax_patterns[idx]
        nb_preterminal = len(syntax_pattern.split('-'))

        if nb_preterminal != length:
            print(idx)
            print('There is mismatch between nb of preterminals and length of sentence.')
            exit(0)

        f1_key = str(round(f1, 3))
        if length not in f1_syntax:
            f1_syntax.update({length: {f1_key: [syntax_pattern]}})
        else:
            if f1_key not in f1_syntax[length]:
                f1_syntax[length].update({f1_key: [syntax_pattern]})
            else:
                if syntax_pattern not in f1_syntax[length][f1_key]:
                    f1_syntax[length][f1_key].append(syntax_pattern)

        if length not in syntax_f1:
            syntax_f1.update({length: {syntax_pattern: [f1]}})
        else:
            if syntax_pattern not in syntax_f1[length]:
                syntax_f1[length].update({syntax_pattern: [f1]})
            else:
                syntax_f1[length][syntax_pattern].append(f1)

    return f1_syntax, syntax_f1

def size_based_folder(files_list, model_option=''):

    pattern = re.compile(r'_(\d+k)_' + re.escape(model_option) + '\d+\.txt$')
    grouped_files = defaultdict(list)

    for filename in files_list:
        match = pattern.search(filename)
        if match:
            size_group = match.group(1)  # match the first regex r'_(\d+k)_'
            grouped_files[size_group].append(filename)

    def sort_key(key):
        return int(re.sub(r'k$', '', key))  # '1k' → 1, '100k' → 100, '50k' → 50, etc.

    sorted_grouped_files = {k: grouped_files[k] for k in sorted(grouped_files.keys(), key=sort_key)}

    return sorted_grouped_files

def get_global_score(result, result_rb):
    # Calcul des F1 macro et micro
    calculator = StatisticsCalculate(result, type_=args.type)
    macro_f1, micro_f1 = calculator.calculate_f1_scores()
    calculator_rb = StatisticsCalculate(result_rb, type_=args.type)
    macro_f1_rb, micro_f1_rb = calculator_rb.calculate_f1_scores()

    print(f'Macro F1: {macro_f1}, Micro F1: {micro_f1}')
    print(f'Macro F1 (RB): {macro_f1_rb}, Micro F1 (RB): {micro_f1_rb}')

    return macro_f1, micro_f1, macro_f1_rb, micro_f1_rb

def simple_evaluation(args):
    output_json = {}
    print(f'\nEvaluation of F1: {args.pred_path}\n')
    unify_files(args.gold_path, args.pred_path, args.input_path)
    res_stat, res_stat_rb = evaluate_and_calculate(args)

    macro_f1, micro_f1, macro_f1_rb, micro_f1_rb = get_global_score(res_stat, res_stat_rb)
    output_json.update({'overall': {'micro': micro_f1, 'macro': macro_f1, 'micro_RB': micro_f1_rb, 'macro_RB': macro_f1_rb}})
    # Calcul détaillé des F1 par longueur
    length_macro_f1, length_micro_f1, f1_summary = analyze_f1_by_length(res_stat)
    length_macro_f1_rb, length_micro_f1_rb, f1_summary_rb = analyze_f1_by_length(res_stat_rb)
    print('\nF1 par longueur:')
    for length, length_rb in zip(sorted(f1_summary.keys()), sorted(f1_summary_rb.keys())):
        print(f'\nLongueur: {length}')
        print(f'Macro F1: {length_macro_f1[length]:.4f}, Macro F1 (RB): {length_macro_f1_rb[length]:.4f}')
        print(f'Micro F1: {length_micro_f1[length]:.4f}, Micro F1 (RB): {length_micro_f1_rb[length]:.4f}')
        output_json.update({length: {'micro': length_micro_f1[length], 'macro': length_macro_f1[length], 'micro_RB': length_micro_f1_rb[length], 'macro_RB': length_macro_f1_rb[length]}})

    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(output_json, f)

    return res_stat


def main(args):

    if args.simple_evaluation:
        simple_evaluation(args)
        exit(0)

    else:
        # {data_type: {model_option: {data_size:
        #   {nb_exp: {all_micro: xx, all_macro: xx, len1: xx, len2: xx, ... }}}}}

        result_dict = {args.data_type: {args.model_option: {}}}

        # parsed_file_dir = './' + args.data_type + args.parsed_dir
        parsed_file_dir = f'.{args.data_folder}/{args.data_type}{args.parsed_dir}'
        parsed_files = [f for f in os.listdir(parsed_file_dir) if re.search(r'^pred-parse_.+\.txt$', f)]

        if args.model_option:
            size_based_files = size_based_folder(parsed_files, args.model_option + '_')
        else:
            size_based_files = size_based_folder(parsed_files, '')

        gold_path = parsed_file_dir + [f for f in os.listdir(parsed_file_dir) if re.search(r'^gold-parse_.+\.txt$', f)][0]

        extractor = PreterminalPatternExtract(gold_path)
        syntax_patterns = extractor.generate_patterns()
        gold_syntax_patterns_path = './' + args.output_folder + args.data_type + '_gold_syntax_patterns.json'
        with open(gold_syntax_patterns_path, 'w', encoding='utf8') as gs_out:
            json.dump(syntax_patterns, gs_out)

        if args.size_list:
            size_list_ = [args.size_list]
        else:
            size_list_ = list(size_based_files.keys())

        for size in size_list_:
            for pred_file in size_based_files[size]:
                data_size = size
                data_type = args.data_type
                model_option = args.model_option
                nb_experiment = re.search(r'(\d+)(?=\.txt$)', pred_file).group(1)
                print(f'\nEvaluating {pred_file}')

                # pred_file_path = args.data_type + args.parsed_dir + pred_file
                pred_file_path = f'.{args.data_folder}/{args.data_type}{args.parsed_dir}{pred_file}'
                unify_files(gold_path, pred_file_path, args.input_path)

                res_stat, res_stat_rb = evaluate_and_calculate(args)
                macro_f1, micro_f1, macro_f1_rb, micro_f1_rb = get_global_score(res_stat, res_stat_rb)

                # result_dict[data_type][model_option].update({data_size: {nb_experiment: {'overall': {'micro': micro_f1, 'macro': macro_f1, 'micro_RB': micro_f1_rb, 'macro_RB': macro_f1_rb}}}})
                if data_size not in result_dict[data_type][model_option]:
                    result_dict[data_type][model_option][data_size] = {}

                result_dict[data_type][model_option][data_size][nb_experiment] = {'overall': {'micro': micro_f1, 'macro': macro_f1, 'micro_RB': micro_f1_rb, 'macro_RB': macro_f1_rb}}

                # make synt dict
                f1_syntax_dict, syntax_f1_dict = analyse_by_syntax_pattern(res_stat, syntax_patterns)

                """
                # syntax_dict_path_prefix = './' + data_type + '/' + data_type + '_' + size + '_' + model_option + '_' + nb_experiment
                if not model_option == '':
                    syntax_dict_path_prefix = f'.{args.data_folder}/{data_type}/{data_type}_{size}_{model_option}_{nb_experiment}'
                else:
                    syntax_dict_path_prefix = f'.{args.data_folder}/{data_type}/{data_type}_{size}_{nb_experiment}'
                with open(syntax_dict_path_prefix + '_f1-syntax-dict.json', 'w', encoding='utf8') as fs_out:
                    json.dump(f1_syntax_dict, fs_out)
                with open(syntax_dict_path_prefix + '_syntax-f1-dict.json', 'w', encoding='utf8') as sf_out:
                    json.dump(syntax_f1_dict, sf_out)
                """

                # Calcul détaillé des F1 par longueur
                length_macro_f1, length_micro_f1, f1_summary = analyze_f1_by_length(res_stat)
                length_macro_f1_rb, length_micro_f1_rb, f1_summary_rb = analyze_f1_by_length(res_stat_rb)

                for length, length_rb in zip(sorted(f1_summary.keys()), sorted(f1_summary_rb.keys())):
                    result_dict[data_type][model_option][data_size][nb_experiment][length] = {'micro': length_micro_f1[length], 'macro': length_macro_f1[length], 'micro_RB': length_micro_f1_rb[length], 'macro_RB': length_macro_f1_rb[length]}

                if args.print_results:
                    print('F1 par longueur:')
                    for length, length_rb in zip(sorted(f1_summary.keys()), sorted(f1_summary_rb.keys())):
                        print(f'\nLongueur: {length}')
                        print(f'Macro F1: {length_macro_f1[length]:.4f}, Macro F1 (RB): {length_macro_f1_rb[length]:.4f}')
                        print(f'Micro F1: {length_micro_f1[length]:.4f}, Micro F1 (RB): {length_micro_f1_rb[length]:.4f}\n')
                        for f1, count in sorted(f1_summary[length].items(), key=lambda x: x[0], reverse=True):
                            print(f'F1: {f1:.4f} - Nombre: {count}')
                        print('\nRight Branching')
                        for f1, count in sorted(f1_summary_rb[length].items(), key=lambda x: x[0], reverse=True):
                            print(f'F1: {f1:.4f} - Nombre: {count}')

        # pprint.pprint(result_dict)

        if args.output_path:
            try:
                with open(args.output_path, 'r', encoding='utf8') as f_in:
                    existing_data = json.load(f_in)
                existing_data.update(result_dict)

                with open(args.output_path, 'w', encoding='utf8') as f_out:
                    json.dump(existing_data, f_out)

            except FileNotFoundError:
                with open(args.output_path, 'w', encoding='utf8') as f_out:
                    json.dump(result_dict, f_out)

        else:
            with open('./result_dict.json', 'w', encoding='utf8') as f:
                json.dump(result_dict, f)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--simple_evaluation', action='store_true')
    parser.add_argument('--data_type', type=str, help='flat, non-flat, non-flat-no-relobj, etc.')
    parser.add_argument('--data_folder', type=str, default='')
    parser.add_argument('--model_option', default='', help='exemple: CL, NC')
    parser.add_argument('--gold_path', default='gold.txt', help='Chemin vers le fichier gold')
    parser.add_argument('--pred_path', default='pred.txt', help='Chemin vers le fichier prédiction')
    parser.add_argument('--input_path', default='./tmp_unified.txt', help='Fichier temporaire unifié')
    parser.add_argument('--type', default='length', help='Type de variation')
    parser.add_argument('--remove_punc', action='store_true', help='Supprimer la ponctuation avant évaluation')
    parser.add_argument('--print_results', action='store_true', help='print results')
    parser.add_argument('--output_path', required=False, type=str, help='add data for graph if json file already exists')
    parser.add_argument('--parsed_dir', type=str, default='/parsed/')
    parser.add_argument('--size_list', type=str, default='')
    parser.add_argument('--output_folder', type=str, default='')
    args = parser.parse_args()
    main(args)
