import Levenshtein as levd
import texterrors


def test_levd():
    pairs = ['a', '', '', 'a', 'MOZILLA', 'MUSIAL', 'ARE', 'MOZILLA', 'TURNIPS', 'TENTH', 'POSTERS', 'POSTURE']
    for a, b in zip(pairs[:-1:2], pairs[1::2]):
        d1 = texterrors.lev_distance(a, b)
        # print(a, b, d1)
        d2 = texterrors.lev_distance(a, b)
        if d1 != d2:
            print(a, b, d1, d2)
            raise RuntimeError('Assert failed!')


def calc_wer(ref, b):
    cnt = 0
    err = 0
    for w1, w2 in zip(ref, b):
        if w1 != '<eps>':
            cnt += 1
        if w1 != w2:
            err += 1
    return 100. * (err / cnt)


def test_wer():
    ref = 'IN THE DISCOTHEQUE THE DJ PLAYED PROGRESSIVE HOUSE MUSIC AND TRANCE'.split()
    hyp = 'IN THE DISCO TAK THE D J PLAYED PROGRESSIVE HOUSE MUSIC AND TRANCE'.split()
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 36.36, round(wer, 2)

    ref = 'IT FORMS PART OF THE SOUTH EAST DORSET CONURBATION ALONG THE ENGLISH CHANNEL COAST'.split()
    hyp = "IT FOLLOWS PARDOFELIS LOUSES DORJE THAT COMORE H O LONELY ENGLISH GENOME COTA'S".split()
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 85.71, round(wer, 2)

    ref = 'THE FILM WAS LOADED INTO CASSETTES IN A DARKROOM OR CHANGING BAG'.split()
    hyp = "THE FILM WAS LOADED INTO CASSETTES IN A DARK ROOM OR CHANGING BAG".split()
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 16.67, round(wer, 2)

test_levd()
test_wer()