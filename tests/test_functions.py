""" Run command: PYTHONPATH=. pytest .
"""
import os
import io
import time
import sys
from loguru import logger

import Levenshtein as levd
from texterrors import texterrors
from texterrors.texterrors import StringVector
from dataclasses import dataclass
import difflib
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

def show_diff(text1, text2):
    # Split the strings into lines to compare them line by line
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    # Create a Differ object and calculate the differences
    differ = difflib.Differ()
    diff = list(differ.compare(lines1, lines2))

    # Optionally, you can filter out lines that haven't changed
    diff = [line for line in diff if line[0] != ' ']

    # Join the result back into a single string and return it
    return '\n'.join(diff)


def test_levd():
    pairs = ['a', '', '', 'a', 'MOZILLA', 'MUSIAL', 'ARE', 'MOZILLA', 'TURNIPS', 'TENTH', 'POSTERS', 'POSTURE']
    for a, b in zip(pairs[:-1:2], pairs[1::2]):
        d1 = texterrors.lev_distance(a, b)
        d2 = levd.distance(a, b)
        assert d1 == d2, f'{a} {b} {d1} {d2}'


# def test_calc_edit_distance_fast():
#     pairs = ['a', '', '', 'a', 'MOZILLA', 'MUSIAL', 'ARE', 'MOZILLA', 'TURNIPS', 'TENTH', 'POSTERS', 'POSTURE']
#     for a, b in zip(pairs[:-1:2], pairs[1::2]):
#         d1 = texterrors.calc_edit_distance_fast(a, b)
#         d2 = levd.distance(a, b)
#         assert d1 == d2, f'{a} {b} fasteditdist={d1} ref={d2}'


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
    ref = StringVector('IN THE DISCOTHEQUE THE DJ PLAYED PROGRESSIVE HOUSE MUSIC AND TRANCE'.split())
    hyp = StringVector('IN THE DISCO TAK THE D J PLAYED PROGRESSIVE HOUSE MUSIC AND TRANCE'.split())
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 36.36, round(wer, 2)

    ref = StringVector('IT FORMS PART OF THE SOUTH EAST DORSET CONURBATION ALONG THE ENGLISH CHANNEL COAST'.split())
    hyp = StringVector("IT FOLLOWS PARDOFELIS LOUSES DORJE THAT COMORE H O LONELY ENGLISH GENOME COTA'S".split())
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 85.71, round(wer, 2)

    ref = StringVector('THE FILM WAS LOADED INTO CASSETTES IN A DARKROOM OR CHANGING BAG'.split())
    hyp = StringVector("THE FILM WAS LOADED INTO CASSETTES IN A DARK ROOM OR CHANGING BAG".split())
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 16.67, round(wer, 2)

    ref = StringVector('GEPHYRIN HAS BEEN SHOWN TO BE NECESSARY FOR GLYR CLUSTERING AT INHIBITORY SYNAPSES'.split())
    hyp = StringVector("THE VIDEOS RISHIRI TUX BINOY CYSTIDIA PHU LIAM CHOLESTEROL ET INNIT PATRESE SYNAPSES".split())
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False, use_chardiff=True)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 100.0, round(wer, 2)  # kaldi gets 92.31 ! but has worse alignment
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False, use_chardiff=False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 92.31, round(wer, 2)

    ref = StringVector('test sentence okay words ending now'.split())
    hyp = StringVector("test a sentenc ok endin now".split())
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False, use_chardiff=True)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 83.33, round(wer, 2)  # kaldi gets 66.67 ! but has worse alignment
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, False, use_chardiff=False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 66.67, round(wer, 2)

    ref = StringVector('speedbird eight six two'.split())
    hyp = StringVector('hello speedbird six two'.split())
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
    d = texterrors.seq_distance(StringVector(a.split()), StringVector(b.split()))
    assert d == 0

    a, b = 'a b', 'a c'
    d = texterrors.seq_distance(StringVector(a.split()), StringVector(b.split()))
    assert d == 1

    a, b = 'a b c', 'a b d'
    d = texterrors.seq_distance(StringVector(a.split()), StringVector(b.split()))
    assert d == 1

    a, b = 'a b c', 'a b d e'
    d = texterrors.seq_distance(StringVector(a.split()), StringVector(b.split()))
    assert d == 2

    a, b = 'a b c', 'd e f g'
    d = texterrors.seq_distance(StringVector(a.split()), StringVector(b.split()))
    assert d == 4

    a, b = 'ça va très bien', 'ça ne va pas'
    d = texterrors.seq_distance(StringVector(a.split()), StringVector(b.split()))
    assert d == 3

    a, b = 'ça ne va pas', 'merci ça va'
    d = texterrors.seq_distance(StringVector(a.split()), StringVector(b.split()))
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
        utts[i] = Utt(i, StringVector(line.split()))
    return utts


def test_process_output():
    reflines = ['1 zum beispiel work shops wo wir anbieten']
    hyplines = ['1 zum beispiel work shop sommer anbieten']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)

    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, ref_file='A', hyp_file='B', nocolor=True)
    output = buffer.getvalue()

    ref = """\"A\" is treated as reference, \"B\" as hypothesis. Errors are capitalized.
Per utt details:
1
zum beispiel work SHOPS WO  WIR   anbieten
                  SHOP  *  SOMMER         

WER: 42.9 (ins 0, del 1, sub 2 / 7)
SER: 100.0

Insertions:

Deletions (second number is word count total):
wo\t1\t1

Substitutions (reference>hypothesis, second number is reference word count total):
shops>shop\t1\t1
wir>sommer\t1\t1
"""
    assert output == ref

def test_process_output_multi():
    reflines = ['0 telefonat mit frau spring klee vom siebenundzwanzigsten august einundzwanzig ich erkläre frau spring klee dass die bundes gerichtliche recht sprechung im zusammen hang mit dem unfall begriff beziehungsweise dem ungewöhnlichen äusseren faktor wie auch bezüglich der unfall ähnlichen körper schädigungen insbesondere die analogie zu meniskus rissen klar geregelt ist']
    hypalines = ['0 telefonat mit frau sprinkler vom siebenundzwanzigsten august einundzwanzig ich erkläre frau sprinkle dass die bundes gerichtliche recht sprechung im zusammen hang mit dem unfall begriff beziehungsweise dem ungewöhnlichen äusseren faktoren wie auch bezüglich der unfall ähnlichen körper schädigungen insbesondere die analogie zum meniskus rissen klar geregelt ist\'']
    hypblines = ['0 telefonat mit frau sprinkle vom siebenundzwanzigsten august einundzwanzig ich erkläre frau sprinkle dass die bundes gerichtliche recht sprechung im zusammen hang mit dem unfall begriff beziehungsweise dem ungewöhnlichen äusseren faktors wie auch bezüglich der unfall ähnlichen körper schädigungen insbesondere die analogie zum meniskus riss en klar geregelt ist ok']
    refs = create_inp(reflines)
    hypa = create_inp(hypalines)
    hypb = create_inp(hypblines)
    buffer = io.StringIO()
    texterrors.process_multiple_outputs(refs, hypa, hypb, buffer, 10, False, False, 'ref', 'hypa', 'hypb', terminal_width=180, usecolor=True)
    output = buffer.getvalue()
    ref = """Per utt details, order is "ref", "hypa", "hypb":
0
telefonat mit frau \x1b[32mspring\x1b[0m   \x1b[32mklee\x1b[0m    vom siebenundzwanzigsten august einundzwanzig ich erkläre frau \x1b[32mspring\x1b[0m   \x1b[32mklee\x1b[0m   dass die bundes gerichtliche recht sprechung im zusammen hang mit
telefonat mit frau        \x1b[31msprinkler\x1b[0m vom siebenundzwanzigsten august einundzwanzig ich erkläre frau        \x1b[31msprinkle\x1b[0m dass die bundes gerichtliche recht sprechung im zusammen hang mit
telefonat mit frau        \x1b[31msprinkle\x1b[0m  vom siebenundzwanzigsten august einundzwanzig ich erkläre frau        \x1b[31msprinkle\x1b[0m dass die bundes gerichtliche recht sprechung im zusammen hang mit
dem unfall begriff beziehungsweise dem ungewöhnlichen äusseren  \x1b[32mfaktor\x1b[0m  wie auch bezüglich der unfall ähnlichen körper schädigungen insbesondere die analogie \x1b[32mzu\x1b[0m  meniskus     
dem unfall begriff beziehungsweise dem ungewöhnlichen äusseren \x1b[31mfaktoren\x1b[0m wie auch bezüglich der unfall ähnlichen körper schädigungen insbesondere die analogie \x1b[31mzum\x1b[0m meniskus     
dem unfall begriff beziehungsweise dem ungewöhnlichen äusseren \x1b[31mfaktors\x1b[0m  wie auch bezüglich der unfall ähnlichen körper schädigungen insbesondere die analogie \x1b[31mzum\x1b[0m meniskus \x1b[31mriss\x1b[0m
\x1b[32mrissen\x1b[0m klar geregelt \x1b[32mist\x1b[0m    
rissen klar geregelt \x1b[31mist'\x1b[0m   
  \x1b[31men\x1b[0m   klar geregelt ist  \x1b[31mok\x1b[0m

Results with file hypa
WER: 14.3 (ins 0, del 2, sub 5 / 49)
SER: 100.0

Insertions:

Deletions (second number is word count total):
spring\t2\t2

Substitutions (reference>hypothesis, second number is reference word count total):
klee>sprinkler\t1\t2
klee>sprinkle\t1\t2
faktor>faktoren\t1\t1
zu>zum\t1\t1
ist>ist'\t1\t1
---

Results with file hypb
WER: 18.4 (ins 2, del 2, sub 5 / 49)
SER: 100.0

Insertions:
riss\t1
ok\t1

Deletions (second number is word count total):
spring\t2\t2

Substitutions (reference>hypothesis, second number is reference word count total):
klee>sprinkle\t2\t2
faktor>faktors\t1\t1
zu>zum\t1\t1
rissen>en\t1\t1
---

Difference between outputs:

Insertions:
riss\t1
ist\t1

Deletions (second number is word count total):

Substitutions (reference>hypothesis, second number is reference word count total):
sprinkler>sprinkle\t1\t1
faktoren>faktors\t1\t1
rissen>en\t1\t1
ist'>ok\t1\t1
"""
    print(ref, file=open('ref', 'w'))
    print(output, file=open('output', 'w'))
    assert ref == output, show_diff(ref, output)


def test_process_output_colored():
    reflines = ['1 den asu flash würde es sonst auch in allen drei sch- in allen drei sprachen ist der verfügbar ähm jetzt für uns habe ich gedacht reicht es ja auf deutsch he']
    hyplines = ['1 ah der anzug fleisch würde sonst auch in allen drei ist in allen drei sprachen verfügbar ähm jetzt für uns habe ich gedacht reicht sie auch auf deutsch he']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)

    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, ref_file='A', hyp_file='B', nocolor=False, terminal_width=80)
    output = buffer.getvalue()
    ref = """\"A\" is treated as reference (white and green), \"B\" as hypothesis (white and red).
Per utt details:
1
   \x1b[32mden\x1b[0m  \x1b[32masu\x1b[0m   \x1b[32mflash\x1b[0m  würde \x1b[32mes\x1b[0m sonst auch in allen drei \x1b[32msch-\x1b[0m in allen drei
\x1b[31mah\x1b[0m \x1b[31mder\x1b[0m \x1b[31manzug\x1b[0m \x1b[31mfleisch\x1b[0m                                   \x1b[31mist\x1b[0m               
sprachen \x1b[32mist\x1b[0m \x1b[32mder\x1b[0m verfügbar ähm jetzt für uns habe ich gedacht reicht \x1b[32mes\x1b[0m   \x1b[32mja\x1b[0m 
                                                                     \x1b[31msie\x1b[0m \x1b[31mauch\x1b[0m
auf deutsch he
              

WER: 32.3 (ins 1, del 3, sub 6 / 31)
SER: 100.0

Insertions:
ah\t1

Deletions (second number is word count total):
es\t1\t2
ist\t1\t1
der\t1\t1

Substitutions (reference>hypothesis, second number is reference word count total):
den>der\t1\t1
asu>anzug\t1\t1
flash>fleisch\t1\t1
sch->ist\t1\t1
es>sie\t1\t2
ja>auch\t1\t1
"""
    print(ref, file=open('ref', 'w'))
    print(output, file=open('output', 'w'))
    assert ref == output


def test_cli_basic():
    ref_f = 'testref'
    hyp_f = 'testhyp'
    with open(ref_f, 'w') as fh:
        fh.write('1 zum beispiel work shops wo wir anbieten')
    with open(hyp_f, 'w') as fh:
        fh.write('1 zum beispiel work shop sommer anbieten')
    outf = 'testout'

    texterrors.main(ref_f, hyp_f, outf, isark=True, usecolor=False)
    output = open(outf).read()
    os.remove(ref_f)
    os.remove(hyp_f)
    os.remove(outf)
    ref = f"""\"{ref_f}\" is treated as reference, \"{hyp_f}\" as hypothesis. Errors are capitalized.
Per utt details:
1
zum beispiel work SHOPS  WO   WIR   anbieten
                    *   SHOP SOMMER         

WER: 42.9 (ins 0, del 1, sub 2 / 7)
SER: 100.0

Insertions:

Deletions (second number is word count total):
shops\t1\t1

Substitutions (reference>hypothesis, second number is reference word count total):
wo>shop\t1\t1
wir>sommer\t1\t1
"""
    assert output == ref


def test_speed():
    import time
    import sys
    logger.remove()
    logger.add(sys.stdout, level='INFO')
    ref = create_inp(open('tests/reftext').read().splitlines())
    hyp = create_inp(open('tests/hyptext').read().splitlines())
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    
    buffer = io.StringIO()
    start_time = time.perf_counter()
    texterrors.process_output(ref, hyp, fh=buffer, ref_file='ref', hyp_file='hyp', 
                              skip_detailed=True, use_chardiff=True, debug=False)
    process_time = time.perf_counter() - start_time

    # pr.disable()
    # pr.dump_stats('speed.prof')

    logger.info(f'Processing time for speed test is {process_time}')
    assert process_time < 2.

