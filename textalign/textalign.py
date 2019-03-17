import logging

import fast
import numpy as np
import plac
import sys

logger = logging.getLogger(__name__)


def _align_texts(text_a, text_b):
    len_a = len(text_a)
    len_b = len(text_b)
    # doing dynamic time warp
    text_a = [0] + text_a
    text_b = [0] + text_b
    # +1 because of padded start token
    summed_cost = np.zeros((len_a + 1, len_b + 1), dtype=np.int32, order="C")
    fast.calc_sum_cost(summed_cost, text_a, text_b)

    if logger.level == logging.DEBUG:
        np.set_printoptions(linewidth=300)
        print(summed_cost, file=sys.stderr)
        np.savetxt('summedcost', summed_cost, fmt='%d')
    best_path_lst = []
    fast.get_best_path(summed_cost, best_path_lst, text_a, text_b)
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


def handle_outliers(aligned_short):
    startidx, endidx = -1, -1
    realstartidx, realendidx = -1, -1
    for i, w in enumerate(aligned_short):
        if w != 0:
            if startidx == -1:
                startidx = i
            endidx = i
            cnt = 0
            for nw in aligned_short[i + 1 : i + 10]:
                if nw != 0:
                    cnt += 1
            if cnt > 3:
                if realstartidx == -1:
                    realstartidx = i
            cnt = 0
            for nw in aligned_short[i - 9 : i]:
                if nw != 0:
                    cnt += 1
            if cnt > 3:
                realendidx = i
    if (
        (realstartidx == startidx and realendidx == endidx)
        or realstartidx == -1
        or realendidx == -1
    ):
        return aligned_short

    if realstartidx != startidx:
        words_to_move_indcs = []
        for i in range(realstartidx):
            if aligned_short[i] != 0:
                words_to_move_indcs.append(i)
        offset = 1
        for idx in reversed(words_to_move_indcs):
            aligned_short[realstartidx - offset] = aligned_short[idx]
            aligned_short[idx] = 0
            offset += 1
    if realendidx != endidx:
        words_to_move_indcs = []
        for i in range(realendidx + 1, len(aligned_short)):
            if aligned_short[i] != 0:
                words_to_move_indcs.append(i)
        offset = 1
        for idx in words_to_move_indcs:
            aligned_short[realendidx + offset] = aligned_short[idx]
            aligned_short[idx] = 0
            offset += 1
    return aligned_short


def get_best_align_subpart(text_long, text_short):
    """ We know the first 2 words should match so find them in the long text and start from there """
    first_two_words = text_short[:2]
    indcs = [0]  # Start of text can be different
    for i in range(len(text_long) - 1):
        if (
            first_two_words[0] == text_long[i]
            and first_two_words[1] == text_long[i + 1]
        ):
            indcs.append(i)

    cover_dist = int(len(text_short) * 2.5)
    bestnum = 1
    bestidx = -1
    aligneds = []
    j = 0
    for startidx in indcs:
        textpart = text_long[startidx : startidx + cover_dist]
        aligned_long, aligned_short = _align_texts(textpart, text_short)
        num = len(aligned_short) / (len(textpart) + len(text_short))
        # print(startidx, startidx + cover_dist, num)
        if num < bestnum:
            bestnum = num
            bestidx = j
        aligneds.append((aligned_long, aligned_short))
        j += 1

    assert bestidx != -1
    return aligneds[bestidx]


def align_texts(text_a, text_b, insert_tok='<eps>', max_len_diff_ratio=2):
    """ If one text is a lot longer than the other will do try and align parts of text
        and then postprocess because there could be errors (because of limit on paths found).
    """
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

    ratio = len(text_a) / len(text_b)
    if (ratio > max_len_diff_ratio or ratio < 1 / max_len_diff_ratio) and (
        len(text_a) > 1000 or len(text_b) > 1000
    ):
        if ratio > max_len_diff_ratio:
            aligned_a, aligned_b = get_best_align_subpart(text_a, text_b)
            # aligned_b = handle_outliers(aligned_b)
        if ratio < 1 / max_len_diff_ratio:
            aligned_b, aligned_a = get_best_align_subpart(text_b, text_a)
            # aligned_a = handle_outliers(aligned_a)
    else:
        aligned_a, aligned_b = _align_texts(text_a, text_b)
        # if len(text_a) < len(text_b):
        #     aligned_a = handle_outliers(aligned_a)
        # else:
        #     aligned_b = handle_outliers(aligned_b)
    if isstr:
        aligned_a = [dct[e] for e in aligned_a]
        aligned_b = [dct[e] for e in aligned_b]
    return aligned_a, aligned_b


def main(
    fpath_a: "File A with text",
    fpath_b: "File B with text",
    debug: ("Print debug messages", "flag", "d"),
):

    if debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)

    with open(fpath_a) as fh:
        texta = fh.read().split()

    with open(fpath_b) as fh:
        textb = fh.read().split()

    dct = {'<eps>': 0}
    all_text = texta + textb
    set_words = set(all_text)
    for i, w in enumerate(set_words):
        dct[w] = i + 1
    texta = [dct[w] for w in texta]
    textb = [dct[w] for w in textb]

    dct.update({v: k for k, v in dct.items()})

    aligned_a, aligned_b = align_texts(texta, textb)
    # with open('out', 'w') as fh:
    #     for e1, e2 in zip(aligned_a, aligned_b):
    #         fh.write(f"{i2w[e1]} {i2w[e2]}\n")
    if debug:
        print([dct[w] for w in aligned_a])
        print([dct[w] for w in aligned_b])


if __name__ == "__main__":
    plac.call(main)
