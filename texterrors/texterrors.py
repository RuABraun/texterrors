#!/usr/bin/env python
import sys
from collections import defaultdict
import texterrors_align
import numpy as np
import plac
from loguru import logger
from termcolor import colored
import Levenshtein as levd


def _align_texts(text_a, text_b, debug=False):
    len_a = len(text_a)
    len_b = len(text_b)
    # doing dynamic time warp
    text_a = [0] + text_a
    text_b = [0] + text_b
    # +1 because of padded start token
    summed_cost = np.zeros((len_a + 1, len_b + 1), dtype=np.int32, order="C")
    texterrors_align.calc_sum_cost(summed_cost, text_a, text_b)

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

    return aligned_a, aligned_b


def align_texts(text_a, text_b, debug, insert_tok='<eps>'):

    assert isinstance(text_a, list) and isinstance(text_b, list), 'Input types should be a list!'
    isstr = False
    if isinstance(text_a[0], str):
        isstr = True
        dct = {insert_tok: 0}
        all_text = text_a + text_b
        set_words = set(all_text)
        for i, w in enumerate(set_words):
            dct[w] = i + 1
        text_a = [dct[w] for w in text_a]
        text_b = [dct[w] for w in text_b]
        dct.update({v: k for k, v in dct.items()})

    aligned_a, aligned_b = _align_texts(text_a, text_b, debug)
    if isstr:
        aligned_a = [dct[e] for e in aligned_a]
        aligned_b = [dct[e] for e in aligned_b]
    return aligned_a, aligned_b


def process_arks(ref_f, hyp_f, outf, cer=False, count=10, oov_set=None, debug=False):
    utt_to_text_ref = {}
    utts = set()
    with open(ref_f) as fh:
        for line in fh:
            utt, *words = line.split()
            utts.add(utt)
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
    word_counts = defaultdict(int)
    fh = open(outf, 'w')
    fh.write('Per utt details:\n')
    for utt in utts:
        ref = utt_to_text_ref[utt]
        hyp = utt_to_text_hyp.get(utt)
        if hyp is None:
            print(f'!\tMissing hyp for utt {utt}')
            continue
        ref_aligned, hyp_aligned = align_texts(ref, hyp, debug)
        fh.write(f'{utt}\n')
        lst = []
        for ref_w, hyp_w in zip(ref_aligned, hyp_aligned):
            total_count += 1
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
            def flatten_word_list(words):
                lst = []
                for i, word in enumerate(words):
                    for c in word:
                        lst.append(c)
                    if i != len(words) - 1:
                        lst.append(' ')
                return lst
            char_ref = flatten_word_list(ref)
            char_hyp = flatten_word_list(hyp)
            char_ref_aligned, char_hyp_aligned = align_texts(char_ref, char_hyp, debug)
            for ref_c, hyp_c in zip(char_ref_aligned, char_hyp_aligned):
                char_count += 1
                if ref_c != hyp_c:
                    char_error_count += 1

        # Get OOV CER
        if oov_set:
            for i, ref_w in enumerate(ref_aligned):
                if ref_w in oov_set:
                    oov_count_denom += len(ref_w)
                    d = 100000
                    # Alignment doesn't take character distance into account, so actual closest word could be
                    # one alignment step away
                    startidx = i - 1 if i - 1 >= 0 else 0
                    for hyp_w in hyp_aligned[startidx:i+2]:
                        if hyp_w == '<eps>':
                            if len(ref_w) < d:
                                d = len(ref_w)
                        else:
                            d_tentative = levd.distance(ref_w, hyp_w)
                            if d_tentative < d:
                                d = d_tentative
                    assert d != 100000
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
):
    oov_set = []
    if oov_list_f:
        with open(oov_list_f) as fh:
            for line in fh:
                oov_set.append(line.split()[0])
        oov_set = set(oov_set)
    process_arks(fpath_ref, fpath_hyp, outf, cer, debug=debug, oov_set=oov_set)


if __name__ == "__main__":
    plac.call(main)
