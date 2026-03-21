import numpy as np

import texterrors_align
from texterrors_align import StringVector


CPP_WORDS_CONTAINER = True


def _ensure_string_vector(words, arg_name):
    if isinstance(words, StringVector):
        return words
    if isinstance(words, str):
        raise TypeError(f'{arg_name} should be a StringVector or an iterable of tokens, not a string')
    try:
        return StringVector(list(words))
    except TypeError as exc:
        raise TypeError(f'{arg_name} should be a StringVector or an iterable of tokens') from exc


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
    return texterrors_align.lev_distance(a, b)


def calc_edit_distance_fast(a, b):
    return texterrors_align.calc_edit_distance_fast_str(a, b)


def seq_distance(a, b):
    """ This function is for when a and b have strings as elements (variable length). """
    a = _ensure_string_vector(a, 'a')
    b = _ensure_string_vector(b, 'b')
    len_a = len(a)
    len_b = len(b)
    summed_cost = np.zeros((len_a + 1, len_b + 1), dtype=np.float64, order="C")
    return texterrors_align.calc_sum_cost(summed_cost, a, b, False, True)


def _align_texts(words_a, words_b, use_chardiff, debug, insert_tok):
    if CPP_WORDS_CONTAINER and not debug:
        return texterrors_align.align_texts_words(words_a, words_b, use_chardiff, insert_tok, True)

    summed_cost = np.zeros((len(words_a) + 1, len(words_b) + 1), dtype=np.float64, order="C")

    if debug:
        print(words_a)
        print(words_b)
    if CPP_WORDS_CONTAINER:
        cost = texterrors_align.calc_sum_cost(summed_cost, words_a, words_b, use_chardiff, True)
    else:
        cost = texterrors_align.calc_sum_cost_lists(summed_cost, words_a, words_b, use_chardiff, True)

    if debug:
        np.set_printoptions(linewidth=300)
        np.savetxt('summedcost', summed_cost, fmt='%.3f', delimiter='\t')

    if CPP_WORDS_CONTAINER:
        best_path_reversed = texterrors_align.get_best_path(summed_cost, words_a, words_b, use_chardiff, True)
    else:
        best_path_reversed = texterrors_align.get_best_path_lists(summed_cost, words_a, words_b, use_chardiff, True)

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
    text_a_str = [insert_tok] + text_a_str
    text_b_str = [insert_tok] + text_b_str
    summed_cost = np.zeros((len_a + 1, len_b + 1), dtype=np.float64, order="C")
    cost = texterrors_align.calc_sum_cost_ctm(
        summed_cost, text_a_str, text_b_str, times_a, times_b, durs_a, durs_b
    )

    if debug:
        np.set_printoptions(linewidth=300)
        np.savetxt('summedcost', summed_cost, fmt='%.3f', delimiter='\t')
    best_path_lst = []
    texterrors_align.get_best_path_ctm(summed_cost, best_path_lst, text_a_str, text_b_str, times_a, times_b, durs_a, durs_b)
    assert len(best_path_lst) % 2 == 0
    path = []
    for n in range(0, len(best_path_lst), 2):
        path.append((best_path_lst[n], best_path_lst[n + 1]))

    newpath = [path[0]]
    i = 1
    lasttpl = path[0]
    while i < len(path) - 1:
        tpl = path[i]
        nexttpl = path[i + 1]
        if lasttpl[0] - 1 == nexttpl[0] and lasttpl[1] - 1 == nexttpl[1]:
            pass
        else:
            newpath.append(tpl)
        i += 1
        lasttpl = tpl
    path = newpath

    aligned_a, aligned_b = [], []
    lasti, lastj = -1, -1
    for i, j in list(reversed(path)):
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


def align_texts(text_a, text_b, *, debug=False, insert_tok='<eps>', use_chardiff=True):
    text_a = _ensure_string_vector(text_a, 'text_a')
    text_b = _ensure_string_vector(text_b, 'text_b')

    aligned_a, aligned_b, cost = _align_texts(text_a, text_b, use_chardiff, debug=debug, insert_tok=insert_tok)

    if debug:
        print(aligned_a)
        print(aligned_b)
    return aligned_a, aligned_b, cost


def get_overlap(refw, hypw):
    if hypw[1] < refw[1]:
        neg_offset = refw[1] - hypw[1]
        if neg_offset < hypw[2] * 0.5:
            return 0
        return -1
    pos_offset = hypw[1] - refw[1]
    if pos_offset < hypw[2] * 0.5:
        return 0
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
            oov_count_error += texterrors_align.lev_distance_str(ref_w, hyp_w)
    return oov_count_error, oov_count_denom
