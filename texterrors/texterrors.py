#!/usr/bin/env python
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import List, Tuple, Dict

import numpy as np
import plac
import regex as re
import texterrors_align
from loguru import logger
from termcolor import colored

OOV_SYM = '<unk>'


def convert_to_int(lst_a, lst_b, dct):
    def convert(lst, dct_syms):
        intlst = []
        for w in lst:
            if w not in dct:
                i = max(v for v in dct_syms.values() if isinstance(v, int)) + 1
                dct_syms[w] = i
                dct_syms[i] = w
            intlst.append(dct_syms[w])
        return intlst
    int_a = convert(lst_a, dct)
    int_b = convert(lst_b, dct)
    return int_a, int_b


def lev_distance(a, b):
    """ This function assumes that elements of a and b are fixed width. """
    if isinstance(a, str):
        return texterrors_align.lev_distance_str(a, b)
    else:
        return texterrors_align.lev_distance(a, b)


def seq_distance(a, b):
    """ This function is for when a and b have strings as elements (variable length). """
    len_a = len(a)
    len_b = len(b)
    summed_cost = np.zeros((len_a + 1, len_b + 1), dtype=np.float64, order="C")
    cost = texterrors_align.calc_sum_cost(summed_cost, a, b, False)
    return cost


def _align_texts(words_a, words_b, use_chardiff, debug, insert_tok): 
    summed_cost = np.zeros((len(words_a) + 1, len(words_b) + 1), dtype=np.float64, 
        order="C")
    cost = texterrors_align.calc_sum_cost(summed_cost, words_a, words_b, use_chardiff)

    if debug:
        np.set_printoptions(linewidth=300)
        np.savetxt('summedcost', summed_cost, fmt='%.3f', delimiter='\t')

    best_path_reversed = texterrors_align.get_best_path(summed_cost, 
        words_a, words_b, use_chardiff)

    aligned_a, aligned_b = [], []
    for i, j in reversed(best_path_reversed):
        if i == -1:
            aligned_a.append(insert_tok)
        else:
            aligned_a.append(words_a[i])
        if j == -1:
            aligned_b.append(insert_tok)
        else:
            aligned_b.append(words_b[j])

    return aligned_a, aligned_b, cost


def align_texts_ctm(text_a_str, text_b_str, times_a, times_b, durs_a, durs_b, debug, insert_tok):
    len_a = len(text_a_str)
    len_b = len(text_b_str)
    # doing dynamic time warp
    text_a_str = [insert_tok] + text_a_str
    text_b_str = [insert_tok] + text_b_str
    # +1 because of padded start token
    summed_cost = np.zeros((len_a + 1, len_b + 1), dtype=np.float64, order="C")
    cost = texterrors_align.calc_sum_cost_ctm(summed_cost, text_a_str, text_b_str,
        times_a, times_b, durs_a, durs_b)

    if debug:
        np.set_printoptions(linewidth=300)
        np.savetxt('summedcost', summed_cost, fmt='%.3f', delimiter='\t')
    best_path_lst = []
    texterrors_align.get_best_path_ctm(summed_cost, best_path_lst,
        text_a_str, text_b_str, times_a, times_b, durs_a, durs_b)
    assert len(best_path_lst) % 2 == 0
    path = []
    for n in range(0, len(best_path_lst), 2):
        i = best_path_lst[n]
        j = best_path_lst[n + 1]
        path.append((i, j))

    # convert hook (up left or left up) transitions to diag, not important.
    # -1 because of padding tokens, i = 1 because first is given
    newpath = [path[0]]
    i = 1
    lasttpl = path[0]
    while i < len(path) - 1:
        tpl = path[i]
        nexttpl = path[i + 1]
        if (
            lasttpl[0] - 1 == nexttpl[0] and lasttpl[1] - 1 == nexttpl[1]
        ):  # minus because reversed
            pass
        else:
            newpath.append(tpl)
        i += 1
        lasttpl = tpl
    path = newpath

    aligned_a, aligned_b = [], []
    lasti, lastj = -1, -1
    for i, j in list(reversed(path)):
        # print(text_a[i], text_b[i], file=sys.stderr)
        if i != lasti:
            aligned_a.append(text_a_str[i])
        else:
            aligned_a.append(insert_tok)
        if j != lastj:
            aligned_b.append(text_b_str[j])
        else:
            aligned_b.append(insert_tok)
        lasti, lastj = i, j

    return aligned_a, aligned_b, cost


def align_texts(text_a, text_b, debug, insert_tok='<eps>', use_chardiff=True):

    assert isinstance(text_a, list) and isinstance(text_b, list), 'Input types should be a list!'
    if len(text_a) > 0:
        assert isinstance(text_a[0], str)
    if len(text_b) > 0:
        assert isinstance(text_b[0], str)

    aligned_a, aligned_b, cost = _align_texts(text_a, text_b, use_chardiff,
                                              debug=debug, insert_tok=insert_tok)

    if debug:
        print(aligned_a)
        print(aligned_b)
    return aligned_a, aligned_b, cost


def get_overlap(refw, hypw):
    # 0 if match, -1 if hyp before, 1 if after
    if hypw[1] < refw[1]:
        neg_offset = refw[1] - hypw[1]
        if neg_offset < hypw[2] * 0.5:
            return 0
        else:
            return -1
    else:
        pos_offset = hypw[1] - refw[1]
        if pos_offset < hypw[2] * 0.5:
            return 0
        else:
            return 1


def get_oov_cer(ref_aligned, hyp_aligned, oov_set):
    # https://arxiv.org/abs/2107.08091
    assert len(ref_aligned) == len(hyp_aligned)
    oov_count_denom = 0
    oov_count_error = 0
    for i, ref_w in enumerate(ref_aligned):
        if ref_w in oov_set:
            oov_count_denom += len(ref_w)
            startidx = i - 1 if i - 1 >= 0 else 0
            hyp_w = ''
            for idx in range(startidx, startidx + 2):
                if idx != i:
                    if idx > len(ref_aligned) - 1 or ref_aligned[idx] != '<eps>':
                        continue
                    if idx < i:
                        hyp_w += hyp_aligned[idx] + ' '
                    else:
                        hyp_w += ' ' + hyp_aligned[idx]
                else:
                    hyp_w += hyp_aligned[idx]
            hyp_w = hyp_w.strip()
            hyp_w = hyp_w.replace('<eps>', '')
            d = texterrors_align.lev_distance_str(ref_w, hyp_w)
            oov_count_error += d
    return oov_count_error, oov_count_denom


@dataclass
class Utt:
    uid: str
    words: list
    times: list = None
    durs: list = None

    def __len__(self):
        return len(self.words)


def read_ref_file(ref_f, isark):
    ref_utts = {}
    with open(ref_f) as fh:
        for i, line in enumerate(fh):
            if isark:
                utt, *words = line.split()
                assert utt not in ref_utts, 'There are repeated utterances in reference file! Exiting'
                ref_utts[utt] = Utt(utt, words)
            else:
                words = line.split()
                i = str(i)
                ref_utts[i] = Utt(i, words)
    return ref_utts


def read_hyp_file(hyp_f, isark, oracle_wer):
    hyp_utts = {} if not oracle_wer else defaultdict(list)
    with open(hyp_f) as fh:
        for i, line in enumerate(fh):
            if isark:
                utt, *words = line.split()
                words = [w for w in words if w != OOV_SYM]
                if not oracle_wer:
                    hyp_utts[utt] = Utt(utt, words)
                else:
                    hyp_utts[utt].append(Utt(utt, words))
            else:
                words = line.split()
                i = str(i)
                hyp_utts[i] = Utt(i, [w for w in words if w != OOV_SYM])
    return hyp_utts


def read_ctm_file(f):
    """ Assumes first field is utt and last three fields are word, time, duration """
    utt_to_wordtimes = defaultdict(list)
    with open(f) as fh:
        for line in fh:
            utt, *_, time, dur, word = line.split()
            time = float(time)
            dur = float(dur)
            utt_to_wordtimes[utt].append((word, time, dur,))
    utts = {}
    for utt, wordtimes in utt_to_wordtimes.items():
        words = []
        times = []
        durs = []
        for e in wordtimes:
            words.append(e[0]), times.append(e[1]), durs.append([2])
        utts[utt] = Utt(utt, words, times, durs)
    return utt_to_wordtimes


@dataclass
class LineElement:
    words: Tuple[str]
    lengths: Tuple[int]
    has_color: bool


class MultiLine:
    def __init__(self, terminal_width, num_lines):
        self.line_elements = []
        self.terminal_width = terminal_width
        self.num_lines = num_lines

    def add_lineelement(self, words, lengths, has_color):
        le = LineElement(words, lengths, has_color)
        self.line_elements.append(le)

    def __len__(self):
        return len(self.line_elements)

    def __getitem__(self, item):
        return self.line_elements[item]

    @staticmethod
    def construct(*lines):
        joined_lines = []
        for line in lines:
            joined_lines.append(' '.join(line))
        return joined_lines

    def iter_construct(self):
        index = 0
        lines = [[] for _ in range(self.num_lines)]
        written_len = 0
        while index < len(self.line_elements):
            le = self.line_elements[index]
            lengths = le.lengths
            padded_len = max(*lengths)
            if written_len + padded_len > self.terminal_width:
                joined_lines = self.construct(*lines)
                lines = [[] for _ in range(self.num_lines)]
                yield joined_lines
                written_len = 0
            written_len += padded_len + 1  # +1 because space will be added
            pad_len_plus_color = padded_len + 9 if le.has_color else padded_len
            words = le.words
            for i, line in enumerate(lines):
                wordlen = pad_len_plus_color if lengths[i] != -1 else padded_len
                line.append(f'{words[i]:^{wordlen}}')

            index += 1
        joined_lines = self.construct(*lines)
        yield joined_lines


@dataclass
class ErrorStats:
    total_cost: int = 0
    total_count: int = 0
    utts: List[str] = field(default_factory=list)
    utt_wrong: int = 0
    ins: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dels: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    subs: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    char_error_count: int = 0
    char_count: int = 0
    oov_count_error: int = 0
    oov_count_denom: int = 0
    oov_word_error: int = 0
    oov_word_count: int = 0
    keywords_predicted: int = 0
    keywords_output: int = 0
    keywords_count: int = 0
    word_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


def read_files(ref_f, hyp_f, isark, isctm, keywords_f, utt_group_map_f, oracle_wer):
    if not isctm:
        ref_utts = read_ref_file(ref_f, isark)
        hyp_utts = read_hyp_file(hyp_f, isark, oracle_wer)
    else:
        ref_utts = read_ctm_file(ref_f)
        hyp_utts = read_ctm_file(hyp_f)

    keywords = set()
    if keywords_f:
        for line in open(keywords_f):
            assert len(line.split()) == 1, 'A keyword must be a single word!'
            keywords.add(line.strip())

    utt_group_map = {}
    if utt_group_map_f:
        for line in open(utt_group_map_f):
            uttid, group = line.split(maxsplit=1)
            group = group.strip()
            utt_group_map[uttid] = group

    return ref_utts, hyp_utts, keywords, utt_group_map


def print_detailed_stats(fh, ins, dels, subs, num_top_errors, freq_sort, word_counts):
    fh.write(f'\nInsertions:\n')
    for v, c in sorted(ins.items(), key=lambda x: x[1], reverse=True)[:num_top_errors]:
        fh.write(f'{v}\t{c}\n')
    fh.write('\n')
    fh.write(f'Deletions (second number is word count total):\n')
    for v, c in sorted(dels.items(), key=lambda x: (x[1] if not freq_sort else x[1] / word_counts[x[0]]),
                       reverse=True)[:num_top_errors]:
        fh.write(f'{v}\t{c}\t{word_counts[v]}\n')
    fh.write('\n')
    fh.write(f'Substitutions (reference>hypothesis, second number is reference word count total):\n')
    for v, c in sorted(subs.items(),
                       key=lambda x: (x[1] if not freq_sort else (x[1] / word_counts[x[0].split('>')[0].strip()], x[1],)),
                       reverse=True)[:num_top_errors]:
        ref_w = v.split('>')[0].strip()
        fh.write(f'{v}\t{c}\t{word_counts[ref_w]}\n')


def process_lines(ref_utts, hyp_utts, debug, use_chardiff, isctm, skip_detailed,
                  terminal_width, oracle_wer, keywords, oov_set, cer, utt_group_map,
                  group_stats, nocolor, insert_tok, suppress_warnings=False):
    error_stats = ErrorStats()
    dct_char = {insert_tok: 0, 0: insert_tok}
    multilines = []
    for utt in ref_utts.keys():
        logger.debug('%s' % utt)
        ref = ref_utts[utt]

        is_empty_reference = not len(ref.words)

        if oracle_wer:
            hyps = hyp_utts[utt]
            costs = []
            for hyp in hyps:
                _, _, cost = align_texts(ref.words, hyp.words, debug, use_chardiff=use_chardiff)
                costs.append(cost)
            error_stats.total_cost += min(costs)
            error_stats.total_count += len(ref)
            continue

        hyp = hyp_utts.get(utt)
        if hyp is None:
            logger.warning(f'Missing hypothesis for utterance: {utt}')
            continue
        error_stats.utts.append(utt)
        logger.debug('ref: %s' % ref.words)
        logger.debug('hyp: %s' % hyp.words)

        if not isctm:
            ref_aligned, hyp_aligned, cost = align_texts(ref.words, hyp.words, debug, use_chardiff=use_chardiff)
        else:
            ref_aligned, hyp_aligned, cost = align_texts_ctm(ref.words, hyp.words, ref.times,
                                                             hyp.times, ref.durs, hyp.durs, debug, insert_tok)
        error_stats.total_cost += cost

        # Counting errors
        error_count = 0
        ref_word_count = 0

        double_line = MultiLine(terminal_width, 2)
        for i, (ref_w, hyp_w,) in enumerate(zip(ref_aligned, hyp_aligned)):
            if ref_w in keywords:
                error_stats.keywords_count += 1
            if hyp_w in keywords:
                error_stats.keywords_output += 1
            if ref_w in oov_set:
                error_stats.oov_word_count += 1

            if ref_w == hyp_w:
                if hyp_w in keywords:
                    error_stats.keywords_predicted += 1
                double_line.add_lineelement((ref_w, '',),
                                            (len(ref_w), -1),
                                            False)
                error_stats.word_counts[ref_w] += 1
                ref_word_count += 1
            else:
                error_count += 1
                if ref_w in oov_set:
                    error_stats.oov_word_error += 1
                if ref_w == '<eps>':
                    if not nocolor:
                        double_line.add_lineelement(('', colored(hyp_w, 'red'),),
                                                    (-1, len(hyp_w),),
                                                    True)
                    else:
                        hyp_w_upper = hyp_w.upper()
                        double_line.add_lineelement(('*', hyp_w_upper,),
                                                    (1, len(hyp_w_upper)),
                                                    False)
                    error_stats.ins[hyp_w] += 1
                elif hyp_w == '<eps>':
                    if not nocolor:
                        double_line.add_lineelement((colored(ref_w, 'green'), '',),
                                                    (len(ref_w), -1,),
                                                    True)
                    else:
                        ref_w_upper = ref_w.upper()
                        double_line.add_lineelement((ref_w_upper, '*',),
                                                    (len(ref_w_upper), 1,),
                                                    False)
                    ref_word_count += 1
                    error_stats.dels[ref_w] += 1
                    error_stats.word_counts[ref_w] += 1
                else:
                    ref_word_count += 1
                    key = f'{ref_w}>{hyp_w}'
                    if not nocolor:
                        double_line.add_lineelement((colored(ref_w, 'green'), colored(hyp_w, 'red'),),
                                                    (len(ref_w), len(hyp_w),),
                                                    True)
                    else:
                        ref_w_upper = ref_w.upper()
                        hyp_w_upper = hyp_w.upper()
                        double_line.add_lineelement((ref_w_upper, hyp_w_upper,),
                                                    (len(ref_w_upper), len(hyp_w_upper),),
                                                    False)
                    error_stats.subs[key] += 1
                    error_stats.word_counts[ref_w] += 1
        error_stats.total_count += ref_word_count

        if not skip_detailed:
            multilines.append(double_line)

        if utt_group_map:
            group = utt_group_map[utt]
            group_stats[group]['count'] += ref_word_count
            group_stats[group]['errors'] += error_count

        if error_count:
            error_stats.utt_wrong += 1

        if cer:  # Calculate CER
            def convert_to_char_list(lst):
                new = []
                for i, word in enumerate(lst):
                    new.extend(list(word))
                    if i != len(lst) - 1:
                        new.append(' ')
                return new

            char_ref = convert_to_char_list(ref.words)
            char_hyp = convert_to_char_list(hyp.words)

            ref_int, hyp_int = convert_to_int(char_ref, char_hyp, dct_char)
            error_stats.char_error_count += texterrors_align.lev_distance(ref_int, hyp_int)
            error_stats.char_count += len(ref_int)

        if is_empty_reference:
            continue

        if oov_set:  # Get OOV CER
            err, cnt = get_oov_cer(ref_aligned, hyp_aligned, oov_set)
            error_stats.oov_count_error += err
            error_stats.oov_count_denom += cnt
    # if not skip_detailed:
        # assert len(multilines) == len(error_stats.utts)
    return multilines, error_stats


def _merge_multilines(multilines_a, multilines_b, terminal_width):
    multilines = []
    for multiline_a, multiline_b in zip(multilines_a, multilines_b):
        multiline = MultiLine(terminal_width, 3)
        idx_a, idx_b = 0, 0
        while idx_a < len(multiline_a) and idx_b < len(multiline_b):
            le_a = multiline_a[idx_a]
            le_b = multiline_b[idx_b]
            if le_a.words[0] == le_b.words[0]:
                multiline.add_lineelement((*le_a.words, le_b.words[-1],),
                                          (*le_a.lengths, le_b.lengths[-1],),
                                          le_a.has_color)
                idx_a += 1
                idx_b += 1
            elif le_a.words[0].lower() == le_b.words[0].lower():
                multiline.add_lineelement((le_a.words[0].upper(), le_a.words[1], le_b.words[1],),
                                          (*le_a.lengths, le_b.lengths[-1],),
                                          le_a.has_color)
                idx_a += 1
                idx_b += 1
            elif le_a.words[0] == '*':
                multiline.add_lineelement((*le_a.words, '',),
                                          (*le_a.lengths, -1,),
                                          False)
                idx_a += 1
            elif le_b.words[0] == '*':
                multiline.add_lineelement(('*', '', le_b.words[1],),
                                          (1, -1, le_b.lengths[1],),
                                          False)
                idx_b += 1
            else:
                raise RuntimeError('Should not be possible')
        multilines.append(multiline)
    return multilines


def process_multiple_outputs(ref_utts, hypa_utts, hypb_utts, fh, num_top_errors,
                             use_chardiff, freq_sort, ref_file, file_a, file_b, terminal_width=None):
    if terminal_width is None:
        terminal_width, _ = shutil.get_terminal_size()
        terminal_width = 120 if terminal_width >= 120 else terminal_width

    multilines_ref_hypa, error_stats_ref_hypa = process_lines(ref_utts, hypa_utts, False, use_chardiff, False,
                                            False, terminal_width, False, [], [], False,
                                            None, None, True, '<eps>')
    multilines_ref_hypb, error_stats_ref_hypb = process_lines(ref_utts, hypb_utts, False, use_chardiff, False,
                                                              False, terminal_width, False, [], [], False,
                                                              None, None, True, '<eps>')
    _, error_stats_hypa_hypb = process_lines(hypa_utts, hypb_utts, False, use_chardiff, False,
                                                              True, terminal_width, False, [], [], False,
                                                              None, None, True, '<eps>')

    merged_multiline = _merge_multilines(multilines_ref_hypa, multilines_ref_hypb,
                                         terminal_width)
    fh.write(f'Per utt details, order is \"{ref_file}\", \"{file_a}\", \"{file_b}\":\n')
    for utt, multiline in zip(error_stats_ref_hypa.utts, merged_multiline):
        fh.write(f'{utt}\n')
        for lines in multiline.iter_construct():
            for line in lines:
                fh.write(f'{line}\n')

    # Outputting metrics from gathered statistics.
    ins_count = sum(error_stats_ref_hypa.ins.values())
    del_count = sum(error_stats_ref_hypa.dels.values())
    sub_count = sum(error_stats_ref_hypa.subs.values())
    wer = (ins_count + del_count + sub_count) / float(error_stats_ref_hypa.total_count)
    fh.write(f'\nResults with file {file_a}'
        f'\nWER: {100. * wer:.1f} (ins {ins_count}, del {del_count}, sub {sub_count} / {error_stats_ref_hypa.total_count})'
        f'\nSER: {100. * error_stats_ref_hypa.utt_wrong / len(error_stats_ref_hypa.utts):.1f}\n')

    print_detailed_stats(fh, error_stats_ref_hypa.ins, error_stats_ref_hypa.dels,
                         error_stats_ref_hypa.subs, num_top_errors, freq_sort,
                         error_stats_ref_hypa.word_counts)
    fh.write(f'---\n')

    ins_count = sum(error_stats_ref_hypb.ins.values())
    del_count = sum(error_stats_ref_hypb.dels.values())
    sub_count = sum(error_stats_ref_hypb.subs.values())
    wer = (ins_count + del_count + sub_count) / float(error_stats_ref_hypb.total_count)
    fh.write(f'\nResults with file {file_b}'
             f'\nWER: {100. * wer:.1f} (ins {ins_count}, del {del_count}, sub {sub_count} / {error_stats_ref_hypb.total_count})'
             f'\nSER: {100. * error_stats_ref_hypb.utt_wrong / len(error_stats_ref_hypb.utts):.1f}\n')

    print_detailed_stats(fh, error_stats_ref_hypb.ins, error_stats_ref_hypb.dels,
                         error_stats_ref_hypb.subs, num_top_errors, freq_sort,
                         error_stats_ref_hypb.word_counts)
    fh.write(f'---\n')

    fh.write(f'\nDifference between outputs:\n')
    print_detailed_stats(fh, error_stats_hypa_hypb.ins, error_stats_hypa_hypb.dels,
                         error_stats_hypa_hypb.subs, num_top_errors, freq_sort,
                         error_stats_hypa_hypb.word_counts)


def process_output(ref_utts, hyp_utts, fh, ref_file, hyp_file, cer=False, num_top_errors=10, oov_set=None, debug=False,
                  use_chardiff=True, isctm=False, skip_detailed=False,
                  keywords=None, utt_group_map=None, oracle_wer=False,
                  freq_sort=False, nocolor=False, insert_tok='<eps>'):
    terminal_width, _ = shutil.get_terminal_size()
    terminal_width = 120 if terminal_width >= 120 else terminal_width

    if oov_set is None:
        oov_set = set()
    if keywords is None:
        keywords = set()
    if utt_group_map is None:
        utt_group_map = {}

    group_stats = {}
    groups = set(utt_group_map.values())
    for group in groups:
        group_stats[group] = {}
        group_stats[group]['count'] = 0
        group_stats[group]['errors'] = 0

    multilines, error_stats = process_lines(ref_utts, hyp_utts, debug, use_chardiff, isctm, skip_detailed,
                  terminal_width, oracle_wer, keywords, oov_set, cer,
                  utt_group_map, group_stats, nocolor, insert_tok)

    if not skip_detailed and not oracle_wer:
        if nocolor:
            fh.write(f'\"{ref_file}\" is treated as reference, \"{hyp_file}\" as hypothesis. Errors are capitalized.\n')
        else:
            fh.write(f'\"{ref_file}\" is treated as reference (white and green), \"{hyp_file}\" as hypothesis (white and red).\n')
        fh.write(f'Per utt details:\n')
        for utt, multiline in zip(error_stats.utts, multilines):
            fh.write(f'{utt}\n')
            for upper_line, lower_line in multiline.iter_construct():
                fh.write(f'{upper_line}\n')
                fh.write(f'{lower_line}\n')

    if not use_chardiff and not oracle_wer:
        s = sum(v for v in chain(error_stats.ins.values(), error_stats.dels.values(), error_stats.subs.values()))
        assert s == error_stats.total_cost, f'{s} {error_stats.total_cost}'
    if oracle_wer:
        fh.write(f'Oracle WER: {error_stats.total_cost / error_stats.total_count}\n')
        return

    # Outputting metrics from gathered statistics.
    ins_count = sum(error_stats.ins.values())
    del_count = sum(error_stats.dels.values())
    sub_count = sum(error_stats.subs.values())
    wer = (ins_count + del_count + sub_count) / float(error_stats.total_count)
    if not skip_detailed:
        fh.write('\n')
    fh.write(f'WER: {100.*wer:.1f} (ins {ins_count}, del {del_count}, sub {sub_count} / {error_stats.total_count})'
             f'\nSER: {100.*error_stats.utt_wrong / len(error_stats.utts):.1f}\n')

    if cer:
        cer = error_stats.char_error_count / float(error_stats.char_count)
        fh.write(f'CER: {100.*cer:.1f} ({error_stats.char_error_count} / {error_stats.char_count})\n')
    if oov_set:
        if error_stats.oov_word_count:
            fh.write(f'OOV CER: {100.*error_stats.oov_count_error / error_stats.oov_count_denom:.1f}\n')
            fh.write(f'OOV WER: {100.*error_stats.oov_word_error / error_stats.oov_word_count:.1f}\n')
        else:
            logger.error('None of the words in the OOV list file were found in the reference!')
    if keywords:
        fh.write(f'Keyword results - recall {error_stats.keywords_predicted / error_stats.keywords_count if error_stats.keywords_count else -1:.2f} '
                 f'- precision {error_stats.keywords_predicted / error_stats.keywords_output if error_stats.keywords_output else -1:.2f}\n')
    if utt_group_map:
        fh.write('Group WERs:\n')
        for group, stats in group_stats.items():
            wer = 100. * (stats['errors'] / float(stats['count']))
            fh.write(f'{group}\t{wer:.1f}\n')
        fh.write('\n')

    if not skip_detailed:
        print_detailed_stats(fh, error_stats.ins, error_stats.dels, error_stats.subs, num_top_errors, freq_sort,
                             error_stats.word_counts)


def main(
    ref_file: 'Reference text',
    hyp_file: 'Hypothesis text',
    outf: 'Optional output file' = '',
    oov_list_f: ('List of OOVs', 'option', None) = '',
    isark: ('Text files start with utterance ID.', 'flag')=False,
    isctm: ('Text files start with utterance ID and end with word, time, duration', 'flag')=False,
    use_chardiff: ('Use character lev distance for better alignment in exchange for slightly higher WER.', 'flag') = False,
    cer: ('Calculate CER', 'flag')=False,
    debug: ('Print debug messages, will write cost matrix to summedcost.', 'flag', 'd')=False,
    skip_detailed: ('No per utterance output', 'flag', 's') = False,
    keywords_f: ('Will filter out non keyword reference words.', 'option', None) = '',
    freq_sort: ('Turn on sorting del/sub errors by frequency (default is by count).', 'flag', None) = False,
    oracle_wer: ('Hyp file should have multiple hypothesis per utterance, lowest edit distance will be used for WER.', 'flag', None) = False,
    utt_group_map_f: ('Should be a file which maps uttids to group, WER will be output per group.', 'option', '') = '',
    usecolor: ('Show detailed output with color (use less -R). Red/white is reference, Green/white model output.', 'flag', 'c')=False,
    num_top_errors: ('Number of errors to show per type in detailed output.', 'option')=10,
    second_hyp_f: ('Will compare outputs between two hypothesis files.', 'option')=''
    ):

    logger.remove()
    if debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    if outf:
        fh = open(outf, 'w')
    else:
        fh = sys.stdout
    if not second_hyp_f:
        if oracle_wer:
            assert isark and not isctm
            skip_detailed = True
            if use_chardiff:
                logger.warning(f'You probably would prefer running without `-use_chardiff`, the WER will be slightly better for the cost of a worse alignment')

        oov_set = set()
        if oov_list_f:
            if not use_chardiff:
                logger.warning('Because you are using standard alignment (not `-use_chardiff`) the alignments could be suboptimal\n'
                               ' which will lead to the OOV-CER being slightly wrong. Use `-use_chardiff` for better alignment, ctm based for the best.')
            with open(oov_list_f) as fh_oov:
                for line in fh_oov:
                    oov_set.add(line.split()[0])  # splitting incase line contains another entry (for example count)

        ref_utts, hyp_utts, keywords, utt_group_map = read_files(ref_file,
            hyp_file, isark, isctm, keywords_f, utt_group_map_f, oracle_wer)

        process_output(ref_utts, hyp_utts, fh, cer, debug=debug, oov_set=oov_set,
                     ref_file=ref_file, hyp_file=hyp_file, use_chardiff=use_chardiff, skip_detailed=skip_detailed,
                     keywords=keywords, utt_group_map=utt_group_map, freq_sort=freq_sort,
                     isctm=isctm, oracle_wer=oracle_wer, nocolor=not usecolor, num_top_errors=num_top_errors)
    else:
        ref_utts = read_ref_file(ref_file, isark)
        hyp_uttsa = read_hyp_file(hyp_file, isark, False)
        hyp_uttsb = read_hyp_file(second_hyp_f, isark, False)

        process_multiple_outputs(ref_utts, hyp_uttsa, hyp_uttsb, fh, num_top_errors,
                                 use_chardiff, freq_sort, ref_file, hyp_file, second_hyp_f)

    fh.close()


def cli():  # entrypoint used in setup.py
    plac.call(main)


if __name__ == "__main__":
    plac.call(main)
