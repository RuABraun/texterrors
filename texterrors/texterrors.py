#!/usr/bin/env python
import sys
from collections import defaultdict
import texterrors_align
import numpy as np
import plac
from loguru import logger
from termcolor import colored
import Levenshtein as levd


def convert_to_int(insert_tok, lst_a, lst_b):
    dct = {insert_tok: 0}
    set_words = set()
    for lst in (lst_a, lst_b):
        set_words.update(lst)
    for i, w in enumerate(set_words):
        dct[w] = i + 1
    dct.update({v: k for k, v in dct.items()})
    return [dct[w] for w in lst_a], [dct[w] for w in lst_b], dct


def _align_texts(text_a, text_b, text_a_str, text_b_str, use_chardiff, debug=False):
    len_a = len(text_a)
    len_b = len(text_b)
    # doing dynamic time warp
    text_a = [0] + text_a
    text_b = [0] + text_b
    text_a_str = ['<eps>'] + text_a_str
    text_b_str = ['<eps>'] + text_b_str
    # +1 because of padded start token
    summed_cost = np.zeros((len_a + 1, len_b + 1), dtype=np.int32, order="C")
    cost = texterrors_align.calc_sum_cost(summed_cost, text_a_str, text_b_str, use_chardiff)

    if debug:
        np.set_printoptions(linewidth=300)
        np.savetxt('summedcost', summed_cost, fmt='%d')
    best_path_lst = []
    texterrors_align.get_best_path(summed_cost, best_path_lst, text_a, text_b)
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
            aligned_a.append(text_a[i])
        else:
            aligned_a.append(0)
        if j != lastj:
            aligned_b.append(text_b[j])
        else:
            aligned_b.append(0)
        lasti, lastj = i, j

    return aligned_a, aligned_b, cost


def align_texts(text_a, text_b, debug, insert_tok='<eps>', use_chardiff=True):

    assert isinstance(text_a, list) and isinstance(text_b, list), 'Input types should be a list!'
    assert isinstance(text_a[0], str)

    text_a_int, text_b_int, dct = convert_to_int(insert_tok, text_a, text_b)
    aligned_a, aligned_b, cost = _align_texts(text_a_int, text_b_int, text_a, text_b, use_chardiff,
                                        debug)

    aligned_a = [dct[e] for e in aligned_a]
    aligned_b = [dct[e] for e in aligned_b]
    return aligned_a, aligned_b, cost


def process_arks(ref_f, hyp_f, outf, cer=False, count=10, oov_set=None, debug=False,
                 use_chardiff=True):
    utt_to_text_ref = {}
    utts = []
    with open(ref_f) as fh:
        for line in fh:
            utt, *words = line.split()
            assert utt not in utts, 'There are repeated utterances in reference file! Exiting'
            utts.append(utt)
            utt_to_text_ref[utt] = words

    utt_to_text_hyp = {}
    with open(hyp_f) as fh:
        for line in fh:
            utt, *words = line.split()
            utt_to_text_hyp[utt] = [w for w in words if w != '<unk>']

    oov_count_denom = 0
    oov_count_error = 0
    char_count = 0
    char_error_count = 0

    ins = defaultdict(int)
    dels = defaultdict(int)
    subs = defaultdict(int)        
    total_count = 0
    cost_total = 0
    word_counts = defaultdict(int)
    fh = open(outf, 'w')
    fh.write('Per utt details:\n')
    for utt in utts:
        ref = utt_to_text_ref[utt]
        hyp = utt_to_text_hyp.get(utt)
        total_count += len(ref)
        if hyp is None:
            print(f'!\tMissing hyp for utt {utt}')
            continue
        ref_aligned, hyp_aligned, cost = align_texts(ref, hyp, debug, use_chardiff=use_chardiff)
        cost_total += cost
        fh.write(f'{utt}\n')
        lst = []
        for ref_w, hyp_w in zip(ref_aligned, hyp_aligned):
            if ref_w == hyp_w:
                lst.append(ref_w)
                word_counts[ref_w] += 1
                continue
            elif ref_w == '<eps>':
                lst.append(colored(hyp_w, 'green'))
                ins[hyp_w] += 1
            elif hyp_w == '<eps>':
                lst.append(colored(ref_w, 'red'))
                dels[ref_w] += 1
                word_counts[ref_w] += 1
            else:
                lst.append(colored(f'{ref_w} > {hyp_w}', 'magenta'))
                subs[f'{ref_w} > {hyp_w}'] += 1 
                word_counts[ref_w] += 1
        for w in lst:
            fh.write(f'{w} ')
        fh.write('\n')

        # Calculate CER
        if cer:
            char_ref = [c for word in ref for c in word]
            char_hyp = [c for word in hyp for c in word]
            ref_int, hyp_int, dct = convert_to_int('<eps>', char_ref, char_hyp)
            # print(utt, ref_int, hyp_int)
            char_error_count += texterrors_align.lev_distance(ref_int, hyp_int)
            char_count += len(ref_int)

        # Get OOV CER
        if oov_set:
            for i, ref_w in enumerate(ref_aligned):
                if ref_w in oov_set:
                    oov_count_denom += len(ref_w)
                    # startidx = i - 1 if i - 1 >= 0 else 0
                    hyp_w = hyp_aligned[i]
                    if hyp_w == '<eps>':
                        if len(ref_w) < d:
                            d = len(ref_w)
                    else:
                        d = levd.distance(ref_w, hyp_w)
                    oov_count_error += d

    wer = (sum(ins.values()) + sum(dels.values()) + sum(subs.values())) / float(total_count)
    fh.write(f'\nWER: {100.*wer:.2f}\n\n')

    if cer:
        cer = char_error_count / float(char_count)
        fh.write(f'CER: {100.*cer:.2f}\n\n')
    if oov_set:
        fh.write(f'\nOOV CER: {100.*oov_count_error / oov_count_denom:.2f}\n\n')
    fh.write('Insertions:\n')
    for v, c in sorted(ins.items(), key=lambda x: x[1], reverse=True)[:count]:
        fh.write(f'{v}\t{c}\n')
    fh.write('\n')
    fh.write('Deletions:\n')
    for v, c in sorted(dels.items(), key=lambda x: x[1], reverse=True)[:count]:
        fh.write(f'{v}\t{c}\t{word_counts[v]}\n')
    fh.write('\n')
    fh.write('Substitutions:\n')
    for v, c in sorted(subs.items(), key=lambda x: x[1], reverse=True)[:count]:
        ref_w = v.split('>')[0].strip()
        fh.write(f'{v}\t{c}\t{word_counts[ref_w]}\n')
    fh.close()


def main(
    fpath_ref: "Reference text",
    fpath_hyp: "Hypothesis text",
    outf: 'Output file',
    oov_list_f: ('List of OOVs', 'option', None),
    isark: ('', 'flag', None)=False,
    cer: ('', 'flag', None)=False,
    debug: ("Print debug messages", "flag", "d")=False,
    no_chardiff: ('', 'flag', None) = False
):
    oov_set = []
    if oov_list_f:
        with open(oov_list_f) as fh:
            for line in fh:
                oov_set.append(line.split()[0])
        oov_set = set(oov_set)
    process_arks(fpath_ref, fpath_hyp, outf, cer, debug=debug, oov_set=oov_set,
                 use_chardiff=not no_chardiff)


if __name__ == "__main__":
    plac.call(main)
