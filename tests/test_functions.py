""" ! Reminder: texterrors needs to be installed """
import io

import Levenshtein as levd
import texterrors
from dataclasses import dataclass


def test_levd():
    pairs = ['a', '', '', 'a', 'MOZILLA', 'MUSIAL', 'ARE', 'MOZILLA', 'TURNIPS', 'TENTH', 'POSTERS', 'POSTURE']
    for a, b in zip(pairs[:-1:2], pairs[1::2]):
        d1 = texterrors.lev_distance(a, b)
        d2 = levd.distance(a, b)
        assert d1 == d2, f'{a} {b} {d1} {d2}'


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


def test_seq_distance():
    a, b = 'a b', 'a b'
    d = texterrors.seq_distance(a.split(), b.split())
    assert d == 0

    a, b = 'a b', 'a c'
    d = texterrors.seq_distance(a.split(), b.split())
    assert d == 1

    a, b = 'a b c', 'a b d'
    d = texterrors.seq_distance(a.split(), b.split())
    assert d == 1

    a, b = 'a b c', 'a b d e'
    d = texterrors.seq_distance(a.split(), b.split())
    assert d == 2

    a, b = 'a b c', 'd e f g'
    d = texterrors.seq_distance(a.split(), b.split())
    assert d == 4

    a, b = 'ça va très bien', 'ça ne va pas'
    d = texterrors.seq_distance(a.split(), b.split())
    assert d == 3

    a, b = 'ça ne va pas', 'merci ça va'
    d = texterrors.seq_distance(a.split(), b.split())
    assert d == 3


@dataclass
class Utt:
    uid: str
    words: list
    times: list = None
    durs: list = None


def create_inp(lines):
    utts = {}
    for line in lines:
        i, line = line.split(maxsplit=1)
        utts[i] = Utt(i, line.split())
    return utts


def test_process_output():
    reflines = ['1 zum beispiel work shops wo wir anbieten']
    hyplines = ['1 zum beispiel work shop sommer anbieten']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)

    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, nocolor=True)
    output = buffer.getvalue()

    ref = """Per utt details:
1
zum beispiel work SHOPS   WO   WIR anbieten
                  SHOP  SOMMER  *          

WER: 42.9 (ins 0, del 1, sub 2 / 7)
SER: 100.0

Insertions:

Deletions:
wir\t1\t1

Substitutions:
shops>shop\t1\t1
wo>sommer\t1\t1
"""

    assert output == ref

def test_process_output_multi():
    reflines = ['0 telefonat mit frau spring klee vom siebenundzwanzigsten august einundzwanzig ich erkläre frau spring klee dass die bundes gerichtliche recht sprechung im zusammen hang mit dem unfall begriff beziehungsweise dem ungewöhnlichen äusseren faktor wie auch bezüglich der unfall ähnlichen körper schädigungen insbesondere die analogie zu meniskus rissen klar geregelt ist']
    hypalines = ['0 telefonat mit frau sprinkler vom siebenundzwanzigsten august einundzwanzig ich erkläre frau sprinkle dass die bundes gerichtliche recht sprechung im zusammen hang mit dem unfall begriff beziehungsweise dem ungewöhnlichen äusseren faktoren wie auch bezüglich der unfall ähnlichen körper schädigungen insbesondere die analogie zum meniskus rissen klar geregelt ist\'']
    hypblines = ['0 telefonat mit frau sprinkle vom siebenundzwanzigsten august einundzwanzig ich erkläre frau sprinkle dass die bundes gerichtliche recht sprechung im zusammen hang mit dem unfall begriff beziehungsweise dem ungewöhnlichen äusseren faktors wie auch bezüglich der unfall ähnlichen körper schädigungen insbesondere die analogie zum meniskus riss en klar geregelt ist']
    refs = create_inp(reflines)
    hypa = create_inp(hypalines)
    hypb = create_inp(hypblines)
    buffer = io.StringIO()
    texterrors.process_multiple_outputs(refs, hypa, hypb, buffer, 10, False, False, 'hypa', 'hypb', terminal_width=203)
    output = buffer.getvalue()
    ref = """Per utt details:
Order is reference, hypa, hypb
0
telefonat mit frau  SPRING   KLEE vom siebenundzwanzigsten august einundzwanzig ich erkläre frau  SPRING  KLEE dass die bundes gerichtliche recht sprechung im zusammen hang mit dem unfall begriff
                   SPRINKLER  *                                                                  SPRINKLE  *                                                                                       
                   SPRINKLE   *                                                                  SPRINKLE  *                                                                                       
beziehungsweise dem ungewöhnlichen äusseren  FAKTOR  wie auch bezüglich der unfall ähnlichen körper schädigungen insbesondere die analogie ZU  meniskus RISSEN *  klar geregelt IST 
                                            FAKTOREN                                                                                       ZUM                                  IST'
                                            FAKTORS                                                                                        ZUM           RISS  EN                   

Results with file hypa
WER: 14.3 (ins 0, del 2, sub 5 / 49)
SER: 100.0

Insertions:

Deletions:
klee\t2\t2

Substitutions:
spring>sprinkler\t1\t2
spring>sprinkle\t1\t2
faktor>faktoren\t1\t1
zu>zum\t1\t1
ist>ist'\t1\t1
---

Results with file hypb
WER: 16.3 (ins 1, del 2, sub 5 / 49)
SER: 100.0

Insertions:
en\t1

Deletions:
klee\t2\t2

Substitutions:
spring>sprinkle\t2\t2
faktor>faktors\t1\t1
zu>zum\t1\t1
rissen>riss\t1\t1
---

Difference between outputs:

Insertions:
en\t1

Deletions:

Substitutions:
sprinkler>sprinkle\t1\t1
faktoren>faktors\t1\t1
rissen>riss\t1\t1
ist'>ist\t1\t1
"""

    assert ref == output

print('Reminder: texterrors needs to be installed')
test_levd()
test_wer()
test_oov_cer()
print('Passed!')
