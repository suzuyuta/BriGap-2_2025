# This is copy and modification of scorer.py
# Original code is available on https://github.com/flyaway1217/PYEVALB

# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.1
#
# Date: 2016-10-13 10:02:18
# Last modified: 2016-10-15 17:19:14

"""
PYEVALB: Evalb in Python version.
"""

from PYEVALB.parser import ParsingError
from PYEVALB import parser

from misc.tree import right_branching
from eval_summary_copy import Result, tree_height


############################################################
# Exceptions
############################################################


class ScoreException(Exception):
    def get_details(self):
        return self.details()


class LengthUnmatch(ScoreException):
    def __init__(self, len_gold_sentence, len_test_sentence):
        self.len_gold_sentence = len_gold_sentence
        self.len_test_sentence = len_test_sentence

    def details(self):
        a = "Length Unmatched !"
        b = "gold sentence:" + str(self.len_gold_sentence)
        c = "test sentence:" + str(self.len_test_sentence)
        s = '\n'.join([a, b, c])
        s += '\n'
        s += '-'*30
        return s


class WordsUnmatch(ScoreException):
    def __init__(self, gold_sentence, test_sentence):
        self.gold_sentence = gold_sentence
        self.test_sentence = test_sentence

    def details(self):
        a = "Words Unmatched !"
        b = "gold sentence:" + str(self.gold_sentence)
        c = "test sentence:" + str(self.test_sentence)
        s = '\n'.join([a, b, c])
        s += '\n'
        s += '-'*30
        return s


class Scorer:
    """The Scorer class.

    This class is a manager of scoring, it can socre tree
    corpus in a specific configuration.
    Every instance corresponding to a configuration.
    """
    def __init__(self):
        pass

    def _cal_spans(self, gold_nodes, test_nodes):
        """Calculate the common span and across span

        Args:
            gold_spans: a list of nodes in gold tree
            test_spans: a list of nodes in test tree

        Returns:
            a tuple span_result:
                span_result[0]: the number of common spans
                span_result[1]: the number of crossing spans
        """
        common = set(gold_nodes) & set(test_nodes)
        unmatched_spans = [node.span for node in test_nodes
                           if node not in common]
        # print('common', common)
        # print('unmatched_spans', unmatched_spans)
        matched_span = [node.span for node in test_nodes if node in common]
        matched_span_list = [(node.span.s, node.span.e) for node in test_nodes if node in common]
        unmatched_spans_list = [(node.span.s, node.span.e) for node in test_nodes if node not in common]
        # print('matched_span', matched_span)
        # print('matched_span', matched_span)
        # print('matched_span_test', matched_span_list)
        # print('unmatched_spans_test', unmatched_spans_list)
        spans_ = matched_span_list + unmatched_spans_list
        # print('spans', spans_)
        # print('gold_nodes', gold_nodes)

        def second_long_span(spans):
            max_len = 0
            second_len = 0
            max_span = None
            second_span = None
            for span in spans:
                length = span[1] - span[0]
                if length > max_len:
                    max_len = length
                    max_span = span
                if max_len > length > second_len:
                    second_len = length
                    second_span = span

            # print('max_len', max_len, max_span)
            # print('second_len', second_len, second_span)

            return second_span

        second_span = second_long_span(spans_)
        # print('second_span', second_span)

        gold_spans = [node.span for node in gold_nodes]

        cross_counter = 0
        # the crossing spans
        for u in unmatched_spans:
            for g in gold_spans:
                if (u.s < g.s and u.e > g.s and u.e < g.e):
                    cross_counter += 1
                    break
                elif (u.s > g.s and u.s < g.e and u.e > g.e):
                    cross_counter += 1
                    break

        return len(common), cross_counter, unmatched_spans_list

    def score_trees(self, gold_tree, test_tree):
        """Score the two trees

        Args:
            gold_tree: the gold tree
            test_tree: the test tree

        Returns:
            An instance of Result
        """
        # Preparing
        gold_label_nodes = gold_tree.non_terminal_labels
        test_label_nodes = test_tree.non_terminal_labels
        # print('len(gold_label_nodes)', len(gold_label_nodes))
        # print('len(test_label_nodes)', len(test_label_nodes))

        gold_poss = gold_tree.poss
        test_poss = test_tree.poss

        gold_sentence = gold_tree.sentence
        test_sentence = test_tree.sentence

        # Check
        if len(gold_sentence) != len(test_sentence):
            print(gold_sentence)
            print(test_sentence)
            raise LengthUnmatch(len(gold_sentence), len(test_sentence))

        if gold_sentence != test_sentence:
            raise WordsUnmatch(gold_sentence, test_sentence)

        # Statistics
        result = Result()
        common_numeber, cross_number, unmatched_spans_list = self._cal_spans(
                gold_label_nodes, test_label_nodes)
        correct_poss_num = sum([gold == test for gold, test
                               in zip(gold_poss, test_poss)])

        result.length = len(gold_sentence)
        result.state = 0
        try:
            # si la phrase se compose d'un seul mot, on ne l'évalue pas,
            # mais sa précision et son rappel sont 1.
            result.recall = common_numeber / len(gold_label_nodes)
            result.prec = common_numeber / len(test_label_nodes)
            result.matched_brackets = common_numeber
            result.gold_brackets = len(gold_label_nodes)
            result.test_brackets = len(test_label_nodes)
            result.cross_brackets = cross_number
            result.words = len(gold_sentence)
            result.correct_tags = correct_poss_num
            result.tag_accracy = correct_poss_num / len(gold_poss)
            result.height = tree_height(str(gold_tree))
            result.snt_f1 = (2 * result.prec * result.recall) / (result.prec + result.recall) if (result.prec + result.recall) else 0
            result.unmatch = unmatched_spans_list

        except ZeroDivisionError:
            # print(result)
            pass

        return result, gold_label_nodes, test_label_nodes

    def score_corpus(self, f_gold, f_test):
        """
        score the treebanks

        Args:
            f_gold: a iterator of gold treebank
            f_test: a iterator of test treebank

        Returns:
            a list of instances of Result
        """
        results = []
        length_unmatch = 0
        for ID, (gold, test) in enumerate(zip(f_gold, f_test)):
            # gold = str(gold)
            # test = str(test)
            try:
                gold_tree = parser.create_from_bracket_string(gold)
                test_tree = parser.create_from_bracket_string(test)
                current_result, gold_label_nodes, test_label_nodes = self.score_trees(gold_tree, test_tree)
                """
                if ID == 2501:
                    print(current_result, '\n')
                    print(gold_label_nodes, '\n')
                    print(test_label_nodes, '\n\n')
                """

            except (WordsUnmatch, LengthUnmatch) as e:
                # current_result = Result()
                # current_result.state = 2
                gold_tree = parser.create_from_bracket_string(gold)
                snt = gold_tree.sentence
                test_tree = right_branching(snt, 0)
                print('RBRBRB')
                test_tree = parser.create_from_bracket_string(str(test_tree))
                current_result, gold_label_nodes, test_label_nodes = self.score_trees(gold_tree, test_tree)
                print(e.details())
                print('ID:', ID)
                print('g: ', gold)
                print('p: ', test)
            except ParsingError as e:
                print(e.errormessage)
                print('g: ', gold)
                print('p: ', test)
            finally:
                current_result.ID = ID
                results.append(current_result)
        return results

