#!/usr/bin/env python
import sys
from collections import defaultdict
from itertools import chain

import texterrors_align
import numpy as np
import plac
from loguru import logger
from termcolor import colored
import shutil
from dataclasses import dataclass


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
    if isinstance(a, str):
        return texterrors_align.lev_distance_str(a, b)
    else:
        return texterrors_align.lev_distance(a, b)


def seq_distance(a, b):
    len_a = len(a)
    len_b = len(b)
    # doing dynamic time warp
    a = ['<>'] + a
    b = ['<>'] + b
    # +1 because of padded start token
    summed_cost = np.zeros((len_a + 1, len_b + 1), dtype=np.float64, order="C")
    cost = texterrors_align.calc_sum_cost(summed_cost, a, b, False)
    return cost


def _align_texts(text_a_str, text_b_str, use_chardiff, debug, insert_tok):
    len_a = len(text_a_str)
    len_b = len(text_b_str)
    # doing dynamic time warp
    text_a_str = [insert_tok] + text_a_str
    text_b_str = [insert_tok] + text_b_str
    # +1 because of padded start token
    summed_cost = np.zeros((len_a + 1, len_b + 1), dtype=np.float64, order="C")
    cost = texterrors_align.calc_sum_cost(summed_cost, text_a_str, text_b_str, use_chardiff)

    if debug:
        np.set_printoptions(linewidth=300)
        np.savetxt('summedcost', summed_cost, fmt='%.3f', delimiter='\t')
    best_path_lst = []
    texterrors_align.get_best_path(summed_cost, best_path_lst, text_a_str, text_b_str, use_chardiff)
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
    assert isinstance(text_a[0], str)

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


def read_files(ref_f, hyp_f, isark, oracle_wer):
    utt_to_text_ref = {}
    utts = []
    with open(ref_f) as fh:
        for i, line in enumerate(fh):
            if isark:
                utt, *words = line.split()
                assert utt not in utts, 'There are repeated utterances in reference file! Exiting'
                utts.append(utt)
                utt_to_text_ref[utt] = words
            else:
                words = line.split()
                utt_to_text_ref[i] = words
                utts.append(i)

    utt_to_text_hyp = {} if not oracle_wer else defaultdict(list)
    with open(hyp_f) as fh:
        for i, line in enumerate(fh):
            if isark:
                utt, *words = line.split()
                if not oracle_wer:
                    utt_to_text_hyp[utt] = [w for w in words if w != '<unk>']
                else:
                    utt_to_text_hyp[utt].append([w for w in words if w != '<unk>'])
            else:
                words = line.split()
                utt_to_text_hyp[i] = [w for w in words if w != '<unk>']
    return utt_to_text_ref, utt_to_text_hyp, utts


def read_ctm_files(ref_f, hyp_f):
    """ Assumes first field is utt and last three fields are word, time, duration """
    def read_ctm_file(f):
        utt_to_wordtimes = defaultdict(list)
        current_utt = None
        with open(f) as fh:
            for line in fh:
                utt, *_, time, dur, word = line.split()
                time = float(time)
                dur = float(dur)
                utt_to_wordtimes[utt].append((word, time, dur,))
        return utt_to_wordtimes
    utt_to_ref = read_ctm_file(ref_f)
    utt_to_hyp = read_ctm_file(hyp_f)
    utts = list(utt_to_ref.keys())
    return utt_to_ref, utt_to_hyp, utts

@dataclass
class LineElement:
    upper_word: str
    length: int
    lower_word: str
    length_lower: int
    has_color: bool


class DoubleLine:
    def __init__(self, terminal_width):
        self.line_elements = []
        self.terminal_width = terminal_width

    def add_lineelement(self, upper_word, length, lower_word, lower_length, has_color):
        le = LineElement(upper_word, length, lower_word, lower_length, has_color)
        self.line_elements.append(le)

    @staticmethod
    def construct(upper_line, lower_line):
        upper_line_str = ' '.join(upper_line)
        lower_line_str = ' '.join(lower_line)
        return upper_line_str, lower_line_str

    def iter_construct(self):
        index = 0
        upper_line = []
        lower_line = []
        written_len = 0
        while index < len(self.line_elements):
            le = self.line_elements[index]
            length_lower = le.length_lower
            padded_len = max(le.length, length_lower)
            if written_len + padded_len > self.terminal_width:
                upper_line_str, lower_line_str = self.construct(upper_line, lower_line)
                upper_line, lower_line = [], []
                yield upper_line_str, lower_line_str
                written_len = 0
            upper_word = le.upper_word
            written_len += padded_len + 1
            pad_len_plus_color = padded_len + 9 if le.has_color else 0
            upper_line.append(f'{upper_word:^{pad_len_plus_color}}')
            if length_lower != -1:
                lower_line.append(f'{le.lower_word:^{pad_len_plus_color}}')
            else:
                lower_line.append(f'{le.lower_word:^{padded_len}}')
            index += 1
        upper_line, lower_line = self.construct(upper_line, lower_line)
        yield upper_line, lower_line


def process_files(ref_f, hyp_f, outf, cer=False, count=10, oov_set=None, debug=False,
                  use_chardiff=True, isark=False, skip_detailed=False, insert_tok='<eps>', keywords_list_f='',
                  not_score_end=False, freq_sort=False, phrase_f='', isctm=False, oracle_wer=False, utt_group_map_f='', nocolor=False):
    if oracle_wer:
        assert isark and not isctm
        assert not use_chardiff, 'Run again with `-no-chardiff` !'
    is_above_three_six = sys.version_info[1] >= 7
    terminal_width, _ = shutil.get_terminal_size()
    if not isctm:
        utt_to_text_ref, utt_to_text_hyp, utts = read_files(ref_f, hyp_f, isark, oracle_wer)
    else:
        utt_to_text_ref, utt_to_text_hyp, utts = read_ctm_files(ref_f, hyp_f)

    keywords = set()
    if keywords_list_f:
        for line in open(keywords_list_f):
            assert len(line.split()) == 1, 'A keyword must be a single word!'
            keywords.add(line.strip())

    utt2phrase = {}
    if phrase_f:
        for line in open(phrase_f):
            utt_words = line.split()
            if len(utt_words) > 1:
                utt2phrase[utt_words[0]] = utt_words[1:]
            else:
                utt2phrase[utt_words[0]] = []

    if utt_group_map_f:
        utt_group_map = {}
        group_stats = {}
        for line in open(utt_group_map_f):
            uttid, group = line.split(maxsplit=1)
            group = group.strip()
            utt_group_map[uttid] = group
            group_stats[group] = {}
            group_stats[group]['count'] = 0
            group_stats[group]['errors'] = 0

    if outf:
        fh = open(outf, 'w')
    else:
        fh = sys.stdout

    # Done reading input, processing.
    oov_count_denom = 0
    oov_count_error = 0
    char_count = 0
    char_error_count = 0
    utt_wrong = 0

    ins = defaultdict(int)
    dels = defaultdict(int)
    subs = defaultdict(int)
    total_cost = 0
    total_count = 0
    word_counts = defaultdict(int)
    if not skip_detailed:
        fh.write('Per utt details:\n')
    dct_char = {insert_tok: 0, 0: insert_tok}
    for utt in utts:
        if debug:
            print(utt)
        ref = utt_to_text_ref[utt]
        if utt2phrase:
            phrase = utt2phrase.get(utt)
            if not phrase:
                continue
            is_contained = any([ref[i: i + len(phrase)] == phrase for i in range(len(ref)-len(phrase) + 1)])
            if not is_contained:
                logger.warning(f'A phrase ({phrase}) does not exist in the reference (uttid: {utt})! The phrase'
                               f' must be contained in the reference text! Will not score.')
                continue
        if keywords:
            ref = [w for w in ref if w in keywords]
        if not len(ref):  # skip utterance if empty reference
            continue

        if oracle_wer:
            hyps = utt_to_text_hyp[utt]
            costs = []
            for hyp in hyps:
                _, _, cost = align_texts(ref, hyp, debug, use_chardiff=use_chardiff)
                costs.append(cost)
            total_cost += min(costs)
            total_count += len(ref)
            continue

        hyp = utt_to_text_hyp.get(utt)
        if hyp is None:
            logger.warning(f'Missing hypothesis for utterance: {utt}')
            continue
        if debug:
            print(ref)
            print(hyp)

        if not isctm:
            ref_aligned, hyp_aligned, cost = align_texts(ref, hyp, debug, use_chardiff=use_chardiff)
        else:
            ref_words = [e[0] for e in ref]
            hyp_words = [e[0] for e in hyp]
            ref_times = [e[1] for e in ref]
            hyp_times = [e[1] for e in hyp]
            ref_durs = [e[2] for e in ref]
            hyp_durs = [e[2] for e in hyp]
            ref_aligned, hyp_aligned, cost = align_texts_ctm(ref_words, hyp_words, ref_times,
                hyp_times, ref_durs, hyp_durs, debug, insert_tok)
        total_cost += cost

        if not skip_detailed:
            fh.write(f'{utt}\n')
        if not_score_end:
            last_good_index = -1
            for i, (ref_w, hyp_w,) in enumerate(zip(ref_aligned, hyp_aligned)):
                if ref_w == hyp_w:
                    last_good_index = i
        # Finds phrase in reference. There should be a smarter way lol
        if utt2phrase:
            phrase = utt2phrase[utt]
            if not phrase:
                continue
            start_idx = 0
            word_idx = 0
            ref_offset = 1
            while start_idx < len(ref_aligned):
                if phrase[word_idx] == ref_aligned[start_idx]:
                    found = True
                    for i in range(1, len(phrase)):
                        while ref_aligned[start_idx + ref_offset] == '<eps>':
                            ref_offset += 1
                        if phrase[word_idx + i] != ref_aligned[start_idx + ref_offset]:
                            found = False
                            ref_offset = 1
                            break
                        ref_offset += 1
                    if found:
                       break
                start_idx += 1
                word_idx = 0

            ref_aligned = ref_aligned[start_idx: start_idx + ref_offset]
            hyp_aligned = hyp_aligned[start_idx: start_idx + ref_offset]
        colored_output = []
        error_count = 0
        ref_word_count = 0
        double_line = DoubleLine(terminal_width)
        for i, (ref_w, hyp_w,) in enumerate(zip(ref_aligned, hyp_aligned)):  # Counting errors
            if not_score_end and i > last_good_index:
                break
            if ref_w == hyp_w:
                double_line.add_lineelement(ref_w, len(ref_w), '', -1, False)
                word_counts[ref_w] += 1
                ref_word_count += 1
            else:
                error_count += 1
                if ref_w == '<eps>':
                    if not nocolor:
                        double_line.add_lineelement(colored(hyp_w, 'green'), len(hyp_w), '', -1, True)
                    else:
                        double_line.add_lineelement('*', 1, f'{hyp_w.upper()}', len(hyp_w), False)
                    ins[hyp_w] += 1
                elif hyp_w == '<eps>':
                    if not nocolor:
                        double_line.add_lineelement(colored(ref_w, 'red'), len(ref_w), '', -1, True)
                    else:
                        double_line.add_lineelement(f'{ref_w.upper()}', len(hyp_w), '', -1, False)
                    ref_word_count += 1
                    dels[ref_w] += 1
                    word_counts[ref_w] += 1
                else:
                    ref_word_count += 1
                    key = f'{ref_w}>{hyp_w}'
                    if not nocolor:
                        double_line.add_lineelement(colored(ref_w, 'red'), len(ref_w),
                                                    colored(hyp_w, 'green'), len(hyp_w), True)
                    else:
                        double_line.add_lineelement(ref_w.upper(), len(ref_w),
                                                    hyp_w.upper(), len(hyp_w), False)
                    subs[key] += 1
                    word_counts[ref_w] += 1
        total_count += ref_word_count
        if not skip_detailed:
            for upper_line, lower_line in double_line.iter_construct():
                fh.write(upper_line + '\n')
                fh.write(lower_line + '\n')


        if utt_group_map_f:
            group = utt_group_map[utt]
            group_stats[group]['count'] += ref_word_count
            group_stats[group]['errors'] += error_count

        if error_count: utt_wrong += 1

        if cer:  # Calculate CER
            if phrase_f:
                raise NotImplementedError('Implementation for CER of phrases not done.')
            def convert_to_char_list(lst):
                new = []
                for i, word in enumerate(lst):
                    for c in word:
                        new.append(c)
                    if i != len(lst) - 1:
                        new.append(' ')
                return new
            char_ref = convert_to_char_list(ref)
            char_hyp = convert_to_char_list(hyp)

            ref_int, hyp_int = convert_to_int(char_ref, char_hyp, dct_char)
            char_error_count += texterrors_align.lev_distance(ref_int, hyp_int)
            char_count += len(ref_int)

        if oov_set:  # Get OOV CER
            err, cnt = get_oov_cer(ref_aligned, hyp_aligned, oov_set)
            oov_count_error += err
            oov_count_denom += cnt

    if not use_chardiff and not oracle_wer:
        s = sum(v for v in chain(ins.values(), dels.values(), subs.values()))
        assert s == total_cost, f'{s} {total_cost}'
    if oracle_wer:
        fh.write(f'Oracle WER: {total_cost / total_count}\n')
        return

    # Outputting metrics from gathered statistics.
    ins_count = sum(ins.values())
    del_count = sum(dels.values())
    sub_count = sum(subs.values())
    wer = (ins_count + del_count + sub_count) / float(total_count)
    if not skip_detailed:
        fh.write('\n')
    fh.write(f'WER: {100.*wer:.1f} (ins {ins_count}, del {del_count}, sub {sub_count} / {total_count})'
             f'\nSER: {100.*utt_wrong / len(utts):.1f}\n')

    if cer:
        cer = char_error_count / float(char_count)
        fh.write(f'CER: {100.*cer:.1f} ({char_error_count} / {char_count})\n')
    if oov_set:
        fh.write(f'OOV CER: {100.*oov_count_error / oov_count_denom:.1f}\n')
    if utt_group_map_f:
        fh.write('Group WERS:\n')
        for group, stats in group_stats.items():
            wer = 100. * (stats['errors'] / float(stats['count']))
            fh.write(f'{group}\t{wer:.1f}\n')
        fh.write('\n')

    if not skip_detailed:
        fh.write(f'\nInsertions:\n')
        for v, c in sorted(ins.items(), key=lambda x: x[1], reverse=True)[:count]:
            fh.write(f'{v}\t{c}\n')
        fh.write('\n')
        fh.write(f'Deletions:\n')
        for v, c in sorted(dels.items(), key=lambda x: (x[1] if not freq_sort else x[1] / word_counts[x[0]]),
                           reverse=True)[:count]:
            fh.write(f'{v}\t{c}\t{word_counts[v]}\n')
        fh.write('\n')
        fh.write(f'Substitutions:\n')
        for v, c in sorted(subs.items(),
                           key=lambda x: (x[1] if not freq_sort else (x[1] / word_counts[x[0].split('>')[0].strip()], x[1],)),
                           reverse=True)[:count]:
            ref_w = v.split('>')[0].strip()
            fh.write(f'{v}\t{c}\t{word_counts[ref_w]}\n')
    if outf:
        fh.close()


def main(
    fpath_ref: "Reference text",
    fpath_hyp: "Hypothesis text",
    outf: ('Optional output file') = '',
    oov_list_f: ('List of OOVs', 'option', None) = '',
    isark: ('', 'flag', None)=False,
    isctm: ('', 'flag', None)=False,
    cer: ('', 'flag', None)=False,
    debug: ("Print debug messages", "flag", "d")=False,
    no_chardiff: ("Don't use character lev distance for alignment", 'flag', None) = False,
    skip_detailed: ('No per utterance output', 'flag', 's') = False,
    phrase_f: ('Has per utterance phrase which should be scored against, instead of whole utterance', 'option', None) = '',
    keywords_list_f: ('Will filter out non keyword reference words.', 'option', None) = '',
    freq_sort: ('Turn on sorting del/sub errors by frequency (default is by count)', 'flag', None) = False,
    not_score_end: ('Errors at the end will not be counted', 'flag', None) = False,
    oracle_wer: ('Hyp file should have multiple hypothesis per utterance, lowest edit distance will be used for WER', 'flag', None) = False,
    utt_group_map_f: ('Should be a file which maps uttids to group, WER will be output per group', 'option', '') = '',
    nocolor: ('', 'flag', None)=False):

    oov_set = []
    if oov_list_f:
        with open(oov_list_f) as fh:
            for line in fh:
                oov_set.append(line.split()[0])
        oov_set = set(oov_set)
    process_files(fpath_ref, fpath_hyp, outf, cer, debug=debug, oov_set=oov_set,
                 use_chardiff=not no_chardiff, isark=isark, skip_detailed=skip_detailed,
                 keywords_list_f=keywords_list_f, not_score_end=not_score_end,
                 freq_sort=freq_sort, phrase_f=phrase_f, isctm=isctm, oracle_wer=oracle_wer,
                 utt_group_map_f=utt_group_map_f, nocolor=nocolor)


if __name__ == "__main__":
    plac.call(main)
