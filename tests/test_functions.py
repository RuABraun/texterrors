import Levenshtein as levd
import texterrors


def test_levd():
    pairs = ['a', '', '', 'a', 'MOZILLA', 'MUSIAL', 'ARE', 'MOZILLA', 'TURNIPS', 'TENTH', 'POSTERS', 'POSTURE']
    for a, b in zip(pairs[:-1:2], pairs[1::2]):
        d1 = texterrors.lev_distance(a, b)
        d2 = levd.distance(a, b)
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

    ref = 'GEPHYRIN HAS BEEN SHOWN TO BE NECESSARY FOR GLYR CLUSTERING AT INHIBITORY SYNAPSES'.split()
    hyp = "THE VIDEOS RISHIRI TUX BINOY CYSTIDIA PHU LIAM CHOLESTEROL ET INNIT PATRESE SYNAPSES".split()
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False, use_chardiff=True)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 100.0, round(wer, 2)  # kaldi gets 92.31 ! but has worse alignment
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False, use_chardiff=False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 92.31, round(wer, 2)

    ref = 'test sentence okay words ending now'.split()
    hyp = "test a sentenc ok endin now".split()
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False, use_chardiff=True)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 83.33, round(wer, 2)  # kaldi gets 66.67 ! but has worse alignment
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False, use_chardiff=False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 66.67, round(wer, 2)

    ref = 'speedbird eight six two'.split()
    hyp = 'hello speedbird six two'.split()
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False, use_chardiff=True)
    assert ref_aligned[0] == '<eps>'
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 50.0, round(wer, 2)  # kaldi gets 66.67 ! but has worse alignment

def test_oov_cer():
    oov_set = {'airport'}
    ref_aligned = 'the missing word is <eps> airport okay'.split()
    hyp_aligned = 'the missing word is air port okay'.split()
    err, cnt = texterrors.get_oov_cer(ref_aligned, hyp_aligned, oov_set)
    assert round(err / cnt, 2) == 0.14, round(err / cnt, 2)

    ref_aligned = 'the missing word is airport okay'.split()
    hyp_aligned = 'the missing word is airport okay'.split()
    err, cnt = texterrors.get_oov_cer(ref_aligned, hyp_aligned, oov_set)
    assert err / cnt == 0., err / cnt


print('Reminder: texterrors needs to be installed')
test_levd()
test_wer()
test_oov_cer()
print('Passed!')
