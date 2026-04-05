""" Run command: PYTHONPATH=. pytest .
"""
import json
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
from typer.testing import CliRunner

logger.remove()
logger.add(sys.stderr, level="INFO")
runner = CliRunner()

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
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 36.36, round(wer, 2)

    ref = StringVector('IT FORMS PART OF THE SOUTH EAST DORSET CONURBATION ALONG THE ENGLISH CHANNEL COAST'.split())
    hyp = StringVector("IT FOLLOWS PARDOFELIS LOUSES DORJE THAT COMORE H O LONELY ENGLISH GENOME COTA'S".split())
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 85.71, round(wer, 2)

    ref = StringVector('THE FILM WAS LOADED INTO CASSETTES IN A DARKROOM OR CHANGING BAG'.split())
    hyp = StringVector("THE FILM WAS LOADED INTO CASSETTES IN A DARK ROOM OR CHANGING BAG".split())
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 16.67, round(wer, 2)

    ref = StringVector('GEPHYRIN HAS BEEN SHOWN TO BE NECESSARY FOR GLYR CLUSTERING AT INHIBITORY SYNAPSES'.split())
    hyp = StringVector("THE VIDEOS RISHIRI TUX BINOY CYSTIDIA PHU LIAM CHOLESTEROL ET INNIT PATRESE SYNAPSES".split())
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, use_chardiff=True)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 100.0, round(wer, 2)  # kaldi gets 92.31 ! but has worse alignment
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, use_chardiff=False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 92.31, round(wer, 2)

    ref = StringVector('test sentence okay words ending now'.split())
    hyp = StringVector("test a sentenc ok endin now".split())
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, use_chardiff=True)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 83.33, round(wer, 2)  # kaldi gets 66.67 ! but has worse alignment
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, use_chardiff=False)
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 66.67, round(wer, 2)

    ref = StringVector('speedbird eight six two'.split())
    hyp = StringVector('hello speedbird six two'.split())
    ref_aligned, hyp_aligned, _ = texterrors.align_texts(ref, hyp, use_chardiff=True)
    assert ref_aligned[0] == '<eps>'
    wer = calc_wer(ref_aligned, hyp_aligned)
    assert round(wer, 2) == 50.0, round(wer, 2)  # kaldi gets 66.67 ! but has worse alignment


def test_align_texts_accepts_lists():
    ref_words = 'speedbird eight six two'.split()
    hyp_words = 'hello speedbird six two'.split()

    ref_aligned, hyp_aligned, cost = texterrors.align_texts(ref_words, hyp_words, use_chardiff=True)
    ref_aligned_sv, hyp_aligned_sv, cost_sv = texterrors.align_texts(
        StringVector(ref_words), StringVector(hyp_words), use_chardiff=True
    )

    assert ref_aligned == ref_aligned_sv
    assert hyp_aligned == hyp_aligned_sv
    assert cost == cost_sv


def test_align_texts_keyword_only_options():
    ref_words = 'a b'.split()
    hyp_words = 'a c'.split()

    try:
        texterrors.align_texts(ref_words, hyp_words, False)
    except TypeError:
        pass
    else:
        raise AssertionError('align_texts should reject positional optional arguments')


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


def test_simple_entity_accuracy_basic():
    reflines = ['1 The ZBX met Alice']
    hyplines = ['1 the zbx met alice']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)
    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, 'A', 'B', simple_entity_accuracy=True, skip_detailed=True)
    output = buffer.getvalue()
    ref ="""WER: 75.0 (ins 0, del 0, sub 3 / 4)
SER: 100.0
Simple Entity Accuracy: 100.0 (1 / 1)
"""
    assert output == ref, show_diff(output, ref)


def test_simple_entity_accuracy_skips_words_seen_elsewhere_lowercase():
    reflines = ['1 Xiomara arrived', '2 xiomara left']
    hyplines = ['1 xiomara arrived', '2 bob left']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)
    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, 'A', 'B', simple_entity_accuracy=True, skip_detailed=True)
    output = buffer.getvalue()
    ref ="""WER: 50.0 (ins 0, del 0, sub 2 / 4)
SER: 100.0
Simple Entity Accuracy: 0.0 (0 / 0)
"""
    assert output == ref, show_diff(output, ref)


def test_simple_entity_accuracy_titlecase_common_word_is_filtered_anywhere():
    reflines = ['1 Xiomara met Time']
    hyplines = ['1 xiomara met time']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)
    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, 'A', 'B', simple_entity_accuracy=True, skip_detailed=True)
    output = buffer.getvalue()
    ref ="""WER: 66.7 (ins 0, del 0, sub 2 / 3)
SER: 100.0
Simple Entity Accuracy: 100.0 (1 / 1)
"""
    assert output == ref, show_diff(output, ref)


def test_simple_entity_accuracy_uses_common_word_list_for_titlecase():
    reflines = ['1 Time met Alice']
    hyplines = ['1 time met alice']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)
    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, 'A', 'B', simple_entity_accuracy=True, skip_detailed=True)
    output = buffer.getvalue()
    ref ="""WER: 66.7 (ins 0, del 0, sub 2 / 3)
SER: 100.0
Simple Entity Accuracy: 0.0 (0 / 0)
"""
    assert output == ref, show_diff(output, ref)


def test_simple_entity_accuracy_ignores_full_stops_for_rare_candidates():
    reflines = ['1 Xiomara left. ZBX arrived']
    hyplines = ['1 xiomara left. zbx arrived']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)
    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, 'A', 'B', simple_entity_accuracy=True, skip_detailed=True)
    output = buffer.getvalue()
    ref ="""WER: 50.0 (ins 0, del 0, sub 2 / 4)
SER: 100.0
Simple Entity Accuracy: 100.0 (2 / 2)
"""
    assert output == ref, show_diff(output, ref)


def test_simple_entity_accuracy_preserves_candidate_when_not_seen_lowercase():
    reflines = ['1 Xiomara arrived', '2 Zbigniew left']
    hyplines = ['1 xiomara arrived', '2 zbigniew left']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)
    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, 'A', 'B', simple_entity_accuracy=True, skip_detailed=True)
    output = buffer.getvalue()
    ref ="""WER: 50.0 (ins 0, del 0, sub 2 / 4)
SER: 100.0
Simple Entity Accuracy: 100.0 (2 / 2)
"""
    assert output == ref, show_diff(output, ref)


def test_simple_entity_accuracy_ignores_spaces_when_matching():
    reflines = ['1 GenAI shipped', '2 CodeWhisperer launched']
    hyplines = ['1 gen ai shipped', '2 code whisperer launched']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)
    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, 'A', 'B', simple_entity_accuracy=True, skip_detailed=True)
    output = buffer.getvalue()
    ref ="""WER: 100.0 (ins 2, del 0, sub 2 / 4)
SER: 100.0
Simple Entity Accuracy: 100.0 (2 / 2)
"""
    assert output == ref, show_diff(output, ref)


def test_simple_entity_accuracy_requires_match_at_aligned_position():
    reflines = ['1 GenAI launched today']
    hyplines = ['1 launched today gen ai']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)
    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, 'A', 'B', simple_entity_accuracy=True, skip_detailed=True)
    output = buffer.getvalue()
    ref ="""WER: 100.0 (ins 2, del 1, sub 0 / 3)
SER: 100.0
Simple Entity Accuracy: 0.0 (0 / 1)
"""
    assert output == ref, show_diff(output, ref)


def test_simple_entity_accuracy_does_not_lowercase_main_scoring():
    reflines = ['1 Xiomara met Zbigniew']
    hyplines = ['1 xiomara met zbigniew']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)
    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, 'A', 'B', simple_entity_accuracy=True, skip_detailed=True)
    output = buffer.getvalue()
    ref ="""WER: 66.7 (ins 0, del 0, sub 2 / 3)
SER: 100.0
Simple Entity Accuracy: 100.0 (2 / 2)
"""
    assert output == ref, show_diff(output, ref)


def test_simple_entity_accuracy_skips_allcaps_common_word():
    reflines = ['1 Xiomara said OK']
    hyplines = ['1 xiomara said ok']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)
    buffer = io.StringIO()
    texterrors.process_output(refs, hyps, buffer, 'A', 'B', simple_entity_accuracy=True, skip_detailed=True)
    output = buffer.getvalue()
    ref ="""WER: 66.7 (ins 0, del 0, sub 2 / 3)
SER: 100.0
Simple Entity Accuracy: 100.0 (1 / 1)
"""
    assert output == ref, show_diff(output, ref)


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


def test_process_output_simple_entity_detailed_stats():
    reflines = ['1 Xiomara met Zbigniew']
    hyplines = ['1 xiomara met bob']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)

    buffer = io.StringIO()
    texterrors.process_output(
        refs,
        hyps,
        buffer,
        ref_file='A',
        hyp_file='B',
        nocolor=True,
        simple_entity_accuracy=True,
        num_top_errors=5,
    )
    output = buffer.getvalue()

    ref = """\"A\" is treated as reference, \"B\" as hypothesis. Errors are capitalized.
Per utt details:
1
XIOMARA met ZBIGNIEW
XIOMARA       BOB   

WER: 66.7 (ins 0, del 0, sub 2 / 3)
SER: 100.0
Simple Entity Accuracy: 50.0 (1 / 2)

Insertions:

Deletions (second number is word count total):

Substitutions (reference>hypothesis, second number is reference word count total):
Xiomara>xiomara\t1\t1
Zbigniew>bob\t1\t1

Unrecognized Simple Entities:
zbigniew\t1

Recognized Simple Entities:
xiomara\t1
"""
    assert output == ref


def test_process_output_simple_entity_details_tsv():
    reflines = ['1 GenAI launched', '2 Xiomara met Zbigniew']
    hyplines = ['1 gen ai launched', '2 xiomara met bob']
    refs = create_inp(reflines)
    hyps = create_inp(hyplines)

    buffer = io.StringIO()
    entity_details = io.StringIO()
    texterrors.process_output(
        refs,
        hyps,
        buffer,
        ref_file='A',
        hyp_file='B',
        simple_entity_accuracy=True,
        skip_detailed=True,
        simple_entity_details_fh=entity_details,
    )

    ref = """reference_file\thypothesis_file\tutt_id\toccurrence_index\treference_entity\tnormalized_reference_entity\thypothesis_output\tcategory
A\tB\t1\t1\tGenAI\tgenai\tgenai\tmatch
A\tB\t2\t1\tXiomara\txiomara\txiomara\tmatch
A\tB\t2\t2\tZbigniew\tzbigniew\tbob\tsubstitution
"""
    assert entity_details.getvalue() == ref


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


def test_cli_multi_hyp_compare_with_out(tmp_path):
    ref_f = tmp_path / 'ref.ark'
    hyp_a = tmp_path / 'hyp_a.ark'
    hyp_b = tmp_path / 'hyp_b.ark'
    out_f = tmp_path / 'comparison.txt'

    ref_f.write_text('1 hello world\n')
    hyp_a.write_text('1 hello world\n')
    hyp_b.write_text('1 hello there\n')

    result = runner.invoke(texterrors.app, ['--isark', '-s', '-o', str(out_f), str(ref_f), str(hyp_a), str(hyp_b)])
    assert result.exit_code == 0, result.output
    output = out_f.read_text()

    ref = f"""Comparison:
file\tWER\tSER
{hyp_a}\t0.0\t0.0
{hyp_b}\t50.0\t100.0
"""
    assert output == ref


def test_cli_json_output_single(tmp_path):
    ref_f = tmp_path / 'ref.ark'
    hyp_f = tmp_path / 'hyp.ark'

    ref_f.write_text('1 hello world\n')
    hyp_f.write_text('1 hello there\n')

    result = runner.invoke(
        texterrors.app,
        ['--isark', '--output-format', 'json', str(ref_f), str(hyp_f)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload['reference_file'] == str(ref_f)
    assert payload['hypothesis_file'] == str(hyp_f)
    assert payload['summary']['wer'] == 50.0
    assert payload['summary']['sub_count'] == 1
    assert payload['summary']['total_ref_words'] == 2
    assert payload['top_errors']['substitutions'] == [
        {
            'reference': 'world',
            'hypothesis': 'there',
            'count': 1,
            'reference_count': 1,
        }
    ]
    assert 'Per utt details' not in result.output


def test_cli_json_output_multi_hyp(tmp_path):
    ref_f = tmp_path / 'ref.ark'
    hyp_a = tmp_path / 'hyp_a.ark'
    hyp_b = tmp_path / 'hyp_b.ark'

    ref_f.write_text('1 hello world\n')
    hyp_a.write_text('1 hello world\n')
    hyp_b.write_text('1 hello there\n')

    result = runner.invoke(
        texterrors.app,
        ['--isark', '--output-format', 'json', str(ref_f), str(hyp_a), str(hyp_b)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload['reference_file'] == str(ref_f)
    assert [entry['hypothesis_file'] for entry in payload['outputs']] == [str(hyp_a), str(hyp_b)]
    assert [entry['summary']['wer'] for entry in payload['outputs']] == [0.0, 50.0]
    assert payload['outputs'][1]['top_errors']['substitutions'] == [
        {
            'reference': 'world',
            'hypothesis': 'there',
            'count': 1,
            'reference_count': 1,
        }
    ]
    assert 'Comparison:' not in result.output


def test_cli_multi_hyp_simple_entity_details_out(tmp_path):
    ref_f = tmp_path / 'ref.ark'
    hyp_a = tmp_path / 'hyp_a.ark'
    hyp_b = tmp_path / 'hyp_b.ark'
    details_f = tmp_path / 'entity_details.tsv'

    ref_f.write_text('1 GenAI launched\n')
    hyp_a.write_text('1 gen ai launched\n')
    hyp_b.write_text('1 launched\n')

    result = runner.invoke(
        texterrors.app,
        ['--isark', '-s', '--entity-details', str(details_f), str(ref_f), str(hyp_a), str(hyp_b)],
    )
    assert result.exit_code == 0, result.output
    output = details_f.read_text()

    ref = f"""reference_file\thypothesis_file\tutt_id\toccurrence_index\treference_entity\tnormalized_reference_entity\thypothesis_output\tcategory
{ref_f}\t{hyp_a}\t1\t1\tGenAI\tgenai\tgenai\tmatch
{ref_f}\t{hyp_b}\t1\t1\tGenAI\tgenai\t\tdeletion
"""
    assert output == ref


def test_cli_warns_on_old_positional_output_interface(tmp_path):
    ref_f = tmp_path / 'ref.ark'
    hyp_f = tmp_path / 'hyp.ark'
    out_f = tmp_path / 'new_output.txt'

    ref_f.write_text('1 hello world\n')
    hyp_f.write_text('1 hello world\n')

    result = runner.invoke(texterrors.app, ['--isark', str(ref_f), str(hyp_f), str(out_f)])
    assert result.exit_code != 0
    assert 'old CLI interface' in result.output
    assert '-o/--out' in result.output


def test_cli_rejects_entity_details_with_oracle_wer(tmp_path):
    ref_f = tmp_path / 'ref.ark'
    hyp_f = tmp_path / 'hyp.ark'
    details_f = tmp_path / 'entity_details.tsv'

    ref_f.write_text('1 hello world\n')
    hyp_f.write_text('1 hello world\n')

    result = runner.invoke(
        texterrors.app,
        ['--isark', '--oracle-wer', '--entity-details', str(details_f), str(ref_f), str(hyp_f)],
    )
    assert result.exit_code != 0
    assert '--entity-details' in result.output
    assert '--oracle-wer' in result.output


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
