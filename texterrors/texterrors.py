#!/usr/bin/env python
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from importlib.resources import as_file, files
from itertools import chain
from typing import List, Tuple, Dict, Optional

import regex as re
import typer
from loguru import logger
from termcolor import colored
from .alignment import (
    CPP_WORDS_CONTAINER,
    StringVector,
    align_texts,
    align_texts_ctm,
    convert_to_int,
    get_oov_cer,
    lev_distance,
    seq_distance,
)


OOV_SYM = '<unk>'
SIMPLE_ENTITY_COMMON_WORD_LIMIT = 10000


@dataclass
class Utt:
    uid: str
    words: StringVector
    times: list = None
    durs: list = None

    def __len__(self):
        return len(self.words)


def read_ref_file(ref_f, isark):
    ref_utts = {}
    with open(ref_f) as fh:
        for i, line in enumerate(fh):
            if isark:
                utt, *words = line.split()
                assert utt not in ref_utts, 'There are repeated utterances in reference file! Exiting'
                if CPP_WORDS_CONTAINER:
                    words = StringVector(words)
                ref_utts[utt] = Utt(utt, words)
            else:
                words = line.split()
                i = str(i)
                if CPP_WORDS_CONTAINER:
                    words = StringVector(words)
                ref_utts[i] = Utt(i, words)
    return ref_utts


def read_hyp_file(hyp_f, isark, oracle_wer):
    hyp_utts = {} if not oracle_wer else defaultdict(list)
    with open(hyp_f) as fh:
        for i, line in enumerate(fh):
            if isark:
                utt, *words = line.split()
                words = [w for w in words if w != OOV_SYM]
                if CPP_WORDS_CONTAINER:
                    words = StringVector(words)
                if not oracle_wer:
                    hyp_utts[utt] = Utt(utt, words)
                else:
                    hyp_utts[utt].append(Utt(utt, words))
            else:
                words = line.split()
                i = str(i)
                words = [w for w in words if w != OOV_SYM]
                if CPP_WORDS_CONTAINER:
                    words = StringVector(words)
                hyp_utts[i] = Utt(i, words)
    return hyp_utts


def read_ctm_file(f):
    """ Assumes first field is utt and last three fields are word, time, duration """
    utt_to_wordtimes = defaultdict(list)
    with open(f) as fh:
        for line in fh:
            utt, *_, time, dur, word = line.split()
            time = float(time)
            dur = float(dur)
            utt_to_wordtimes[utt].append((word, time, dur,))
    utts = {}
    for utt, wordtimes in utt_to_wordtimes.items():
        words = []
        times = []
        durs = []
        for e in wordtimes:
            words.append(e[0]), times.append(e[1]), durs.append([2])
        utts[utt] = Utt(utt, StringVector(words), times, durs)
    return utt_to_wordtimes


@dataclass
class LineElement:
    words: Tuple[str]


class MultiLine:
    def __init__(self, terminal_width, num_lines):
        self.line_elements = []
        self.terminal_width = terminal_width
        self.num_lines = num_lines

    def add_lineelement(self, *words):
        self.line_elements.append(words)

    def __len__(self):
        return len(self.line_elements)

    def __getitem__(self, item):
        return self.line_elements[item]

    @staticmethod
    def construct(*lines):
        joined_lines = []
        for line in lines:
            joined_lines.append(' '.join(line))
        return joined_lines

    def __repr__(self):
        elems = []
        for le in self.line_elements:
            elems.append('|'.join(w for w in le))
        return '\t'.join(elems)

    def iter_construct(self):
        index = 0
        lines = [[] for _ in range(self.num_lines)]
        written_len = 0
        while index < len(self.line_elements):
            le = self.line_elements[index]
            lengths = [len(_remove_color(w)) for w in le]
            padded_len = max(*lengths)
            if written_len + padded_len > self.terminal_width:
                joined_lines = self.construct(*lines)
                lines = [[] for _ in range(self.num_lines)]
                yield joined_lines
                written_len = 0
            written_len += padded_len + 1  # +1 because space will be added
            words = le
            for i, line in enumerate(lines):
                word = words[i]
                wordlen = padded_len
                wordlen += get_color_lengthoffset(word)
                line.append(f'{word:^{wordlen}}')

            index += 1
        joined_lines = self.construct(*lines)
        yield joined_lines


@dataclass
class ErrorStats:
    total_cost: int = 0
    total_count: int = 0
    utts: List[str] = field(default_factory=list)
    utt_wrong: int = 0
    ins: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dels: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    subs: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    char_error_count: int = 0
    char_count: int = 0
    oov_count_error: int = 0
    oov_count_denom: int = 0
    oov_word_error: int = 0
    oov_word_count: int = 0
    keywords_predicted: int = 0
    keywords_output: int = 0
    keywords_count: int = 0
    simple_entity_matches: int = 0
    simple_entity_count: int = 0
    simple_entity_recognized: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    simple_entity_missed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    simple_entity_details: List[Dict[str, str]] = field(default_factory=list)
    word_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class OutputFormat(str, Enum):
    text = 'text'
    json = 'json'


def _pct(numerator, denominator):
    return 100. * numerator / float(denominator) if denominator else 0.


def _has_uppercase_evidence(word):
    return any(ch.isupper() for ch in word)


@lru_cache(maxsize=1)
def _load_simple_entity_common_words():
    common_words = set()
    wordlist_resource = files('texterrors') / 'data' / 'wordlist'
    with as_file(wordlist_resource) as wordlist_path:
        with open(wordlist_path, 'r', encoding='utf-8') as fh_wordlist:
            for idx, line in enumerate(fh_wordlist):
                if idx >= SIMPLE_ENTITY_COMMON_WORD_LIMIT:
                    break
                parts = line.split()
                if not parts:
                    continue
                common_words.add(parts[0].lower())
    return common_words


def _extract_simple_entity_words(ref_utts):
    """Identify entity-like reference words from casing, common-word filtering, and lowercase reuse.

    A token is kept as a simple entity when it shows uppercase evidence, its lowercase
    form is not in the common-word list, and it does not also appear elsewhere in the
    reference as a lowercase token.
    """
    common_words = _load_simple_entity_common_words()
    lowercase_words = set()
    for utt in ref_utts.values():
        for word in utt.words:
            if _has_uppercase_evidence(word):
                continue
            lowercase_words.add(word.lower())

    entity_words = set()
    for utt in ref_utts.values():
        for word in utt.words:
            if not _has_uppercase_evidence(word):
                continue
            lowered = word.lower()
            if lowered in common_words or lowered in lowercase_words:
                continue
            entity_words.add(lowered)
    return entity_words


def _compact_utt_text(utt, lowercase=False):
    if lowercase:
        return ''.join(word.lower() for word in utt.words)
    return ''.join(utt.words)


def _extract_simple_entity_spans(utt, entity_words, lowercase=False):
    spans = []
    start = 0
    for word in utt.words:
        end = start + len(word)
        entity_word = word.lower() if lowercase else word
        if entity_word in entity_words:
            spans.append((word, entity_word, start, end))
        start = end
    return spans


def _get_aligned_ref_index_map(aligned_ref, compact_ref_len):
    aligned_index_by_ref_char = [None] * compact_ref_len
    ref_char_idx = 0
    for aligned_idx, char in enumerate(aligned_ref):
        if char == '<eps>':
            continue
        aligned_index_by_ref_char[ref_char_idx] = aligned_idx
        ref_char_idx += 1
    assert ref_char_idx == compact_ref_len
    return aligned_index_by_ref_char


def _score_simple_entities_without_spaces(ref_utts, hyp_utts, entity_words, error_stats):
    error_stats.simple_entity_matches = 0
    error_stats.simple_entity_count = 0
    error_stats.simple_entity_recognized = defaultdict(int)
    error_stats.simple_entity_missed = defaultdict(int)
    error_stats.simple_entity_details = []

    for uttid, ref in ref_utts.items():
        hyp = hyp_utts.get(uttid)
        if hyp is None:
            continue

        entity_spans = _extract_simple_entity_spans(ref, entity_words, lowercase=True)
        if not entity_spans:
            continue

        compact_ref = _compact_utt_text(ref, lowercase=True)
        compact_hyp = _compact_utt_text(hyp, lowercase=True)
        aligned_ref, aligned_hyp, _ = align_texts(list(compact_ref), list(compact_hyp), use_chardiff=False)
        aligned_ref_index_map = _get_aligned_ref_index_map(aligned_ref, len(compact_ref))

        for occurrence_index, (ref_entity, entity, start, end) in enumerate(entity_spans, start=1):
            error_stats.simple_entity_count += 1
            aligned_start = aligned_ref_index_map[start]
            aligned_end = aligned_ref_index_map[end - 1]
            aligned_hyp_slice = aligned_hyp[aligned_start:aligned_end + 1]
            matched_hyp = ''.join(char for char in aligned_hyp_slice if char != '<eps>')
            if matched_hyp == entity:
                error_stats.simple_entity_matches += 1
                error_stats.simple_entity_recognized[entity] += 1
                category = 'match'
            else:
                error_stats.simple_entity_missed[entity] += 1
                category = 'deletion' if not matched_hyp else 'substitution'
            error_stats.simple_entity_details.append({
                'utt_id': uttid,
                'occurrence_index': str(occurrence_index),
                'reference_entity': ref_entity,
                'normalized_reference_entity': entity,
                'hypothesis_output': matched_hyp,
                'category': category,
            })


def read_files(ref_f, hyp_f, isark, isctm, keywords_f, utt_group_map_f, oracle_wer):
    if not isctm:
        ref_utts = read_ref_file(ref_f, isark)
        hyp_utts = read_hyp_file(hyp_f, isark, oracle_wer)
    else:
        ref_utts = read_ctm_file(ref_f)
        hyp_utts = read_ctm_file(hyp_f)

    keywords = set()
    if keywords_f:
        for line in open(keywords_f):
            assert len(line.split()) == 1, 'A keyword must be a single word!'
            keywords.add(line.strip())

    utt_group_map = {}
    if utt_group_map_f:
        for line in open(utt_group_map_f):
            uttid, group = line.split(maxsplit=1)
            group = group.strip()
            utt_group_map[uttid] = group

    return ref_utts, hyp_utts, keywords, utt_group_map


def print_detailed_stats(fh, ins, dels, subs, num_top_errors, freq_sort, word_counts):
    fh.write(f'\nInsertions:\n')
    for v, c in sorted(ins.items(), key=lambda x: x[1], reverse=True)[:num_top_errors]:
        fh.write(f'{v}\t{c}\n')
    fh.write('\n')
    fh.write(f'Deletions (second number is word count total):\n')
    for v, c in sorted(dels.items(), key=lambda x: (x[1] if not freq_sort else x[1] / word_counts[x[0]]),
                       reverse=True)[:num_top_errors]:
        fh.write(f'{v}\t{c}\t{word_counts[v]}\n')
    fh.write('\n')
    fh.write(f'Substitutions (reference>hypothesis, second number is reference word count total):\n')
    for v, c in sorted(subs.items(),
                       key=lambda x: (x[1] if not freq_sort else (x[1] / word_counts[x[0].split('>')[0].strip()], x[1],)),
                       reverse=True)[:num_top_errors]:
        ref_w = v.split('>')[0].strip()
        fh.write(f'{v}\t{c}\t{word_counts[ref_w]}\n')


def print_simple_entity_stats(fh, simple_entity_missed, simple_entity_recognized, num_top_errors):
    fh.write('\nUnrecognized Simple Entities:\n')
    for word, count in sorted(simple_entity_missed.items(), key=lambda x: (-x[1], x[0]))[:num_top_errors]:
        fh.write(f'{word}\t{count}\n')

    fh.write('\nRecognized Simple Entities:\n')
    for word, count in sorted(simple_entity_recognized.items(), key=lambda x: (-x[1], x[0]))[:num_top_errors]:
        fh.write(f'{word}\t{count}\n')


def _init_group_stats(utt_group_map):
    group_stats = {}
    for group in set(utt_group_map.values()):
        group_stats[group] = {'count': 0, 'errors': 0}
    return group_stats


def _score_outputs(ref_utts, hyp_utts, debug=False, use_chardiff=True, isctm=False, skip_detailed=False,
                   terminal_width=None, oracle_wer=False, keywords=None, oov_set=None, cer=False,
                   utt_group_map=None, freq_sort=False, nocolor=False, insert_tok='<eps>',
                   simple_entity_accuracy=False, simple_entity_words=None):
    if terminal_width is None:
        terminal_width, _ = shutil.get_terminal_size()
        terminal_width = 120 if terminal_width >= 120 else terminal_width

    if oov_set is None:
        oov_set = set()
    if keywords is None:
        keywords = set()
    if utt_group_map is None:
        utt_group_map = {}

    if simple_entity_accuracy:
        if simple_entity_words is None:
            simple_entity_words = _extract_simple_entity_words(ref_utts)
        process_line_simple_entity_words = set()
    else:
        simple_entity_words = set()
        process_line_simple_entity_words = set()

    group_stats = _init_group_stats(utt_group_map)
    multilines, error_stats = process_lines(
        ref_utts,
        hyp_utts,
        debug,
        use_chardiff,
        isctm,
        skip_detailed,
        terminal_width,
        oracle_wer,
        keywords,
        oov_set,
        cer,
        utt_group_map,
        group_stats,
        nocolor,
        insert_tok,
        simple_entity_words=process_line_simple_entity_words,
    )
    if simple_entity_accuracy and not oracle_wer:
        _score_simple_entities_without_spaces(ref_utts, hyp_utts, simple_entity_words, error_stats)
    return multilines, error_stats, group_stats, keywords, oov_set


def _build_output_summary(error_stats, group_stats, *, cer=False, simple_entity_accuracy=False,
                          oov_set=None, keywords=None):
    ins_count = sum(error_stats.ins.values())
    del_count = sum(error_stats.dels.values())
    sub_count = sum(error_stats.subs.values())
    summary = {
        'total_ref_words': error_stats.total_count,
        'total_utterances': len(error_stats.utts),
        'wrong_utterances': error_stats.utt_wrong,
        'ins_count': ins_count,
        'del_count': del_count,
        'sub_count': sub_count,
        'wer': _pct(ins_count + del_count + sub_count, error_stats.total_count),
        'ser': _pct(error_stats.utt_wrong, len(error_stats.utts)),
    }

    if simple_entity_accuracy:
        summary.update(
            simple_entity_accuracy=_pct(
                error_stats.simple_entity_matches,
                error_stats.simple_entity_count,
            ),
            simple_entity_matches=error_stats.simple_entity_matches,
            simple_entity_count=error_stats.simple_entity_count,
        )
    if cer:
        summary.update(
            cer=_pct(error_stats.char_error_count, error_stats.char_count),
            char_error_count=error_stats.char_error_count,
            char_count=error_stats.char_count,
        )
    if oov_set and error_stats.oov_word_count:
        summary.update(
            oov_cer=_pct(error_stats.oov_count_error, error_stats.oov_count_denom),
            oov_wer=_pct(error_stats.oov_word_error, error_stats.oov_word_count),
            oov_char_error_count=error_stats.oov_count_error,
            oov_char_count=error_stats.oov_count_denom,
            oov_word_error_count=error_stats.oov_word_error,
            oov_word_count=error_stats.oov_word_count,
        )
    if keywords:
        summary.update(
            keyword_recall=(
                error_stats.keywords_predicted / error_stats.keywords_count
                if error_stats.keywords_count else -1.
            ),
            keyword_precision=(
                error_stats.keywords_predicted / error_stats.keywords_output
                if error_stats.keywords_output else -1.
            ),
            keyword_predicted_count=error_stats.keywords_predicted,
            keyword_output_count=error_stats.keywords_output,
            keyword_count=error_stats.keywords_count,
        )
    if group_stats:
        summary['group_stats'] = {
            group: {
                'errors': stats['errors'],
                'count': stats['count'],
                'wer': _pct(stats['errors'], stats['count']),
            }
            for group, stats in group_stats.items()
        }
    return summary


def _build_count_items(counts, num_top_errors, *, sort_key=None, item_builder=None, reverse=True):
    """Convert a count mapping into a top-N JSON-friendly list with optional sorting and item shaping."""
    if sort_key is None:
        sort_key = lambda item: item[1]
    if item_builder is None:
        item_builder = lambda word, count: {'word': word, 'count': count}
    return [
        item_builder(word, count)
        for word, count in sorted(counts.items(), key=sort_key, reverse=reverse)[:num_top_errors]
    ]


def _build_output_payload(ref_file, hyp_file, summary, error_stats, *, num_top_errors,
                          freq_sort=False, simple_entity_accuracy=False):
    word_counts = error_stats.word_counts
    payload = {
        'reference_file': str(ref_file),
        'hypothesis_file': str(hyp_file),
        'summary': summary,
        'top_errors': {
            'insertions': _build_count_items(error_stats.ins, num_top_errors),
            'deletions': _build_count_items(
                error_stats.dels,
                num_top_errors,
                sort_key=lambda item: (item[1] if not freq_sort else item[1] / word_counts[item[0]]),
                item_builder=lambda word, count: {
                    'word': word,
                    'count': count,
                    'reference_count': word_counts[word],
                },
            ),
            'substitutions': _build_count_items(
                error_stats.subs,
                num_top_errors,
                sort_key=lambda item: (
                    item[1]
                    if not freq_sort else
                    (item[1] / word_counts[item[0].split('>')[0].strip()], item[1],)
                ),
                item_builder=lambda key, count: {
                    'reference': key.split('>', maxsplit=1)[0].strip(),
                    'hypothesis': key.split('>', maxsplit=1)[1],
                    'count': count,
                    'reference_count': word_counts[key.split('>', maxsplit=1)[0].strip()],
                },
            ),
        },
    }
    if simple_entity_accuracy:
        payload['simple_entities'] = {
            'missed': _build_count_items(
                error_stats.simple_entity_missed,
                num_top_errors,
                sort_key=lambda item: (-item[1], item[0]),
                reverse=False,
            ),
            'recognized': _build_count_items(
                error_stats.simple_entity_recognized,
                num_top_errors,
                sort_key=lambda item: (-item[1], item[0]),
                reverse=False,
            ),
        }
    return payload


def _write_json_payload(fh, payload):
    json.dump(payload, fh, indent=2)
    fh.write('\n')


SIMPLE_ENTITY_DETAIL_COLUMNS = [
    'reference_file',
    'hypothesis_file',
    'utt_id',
    'occurrence_index',
    'reference_entity',
    'normalized_reference_entity',
    'hypothesis_output',
    'category',
]


def _add_simple_entity_detail_file_info(ref_file, hyp_file, rows):
    return [
        {
            'reference_file': str(ref_file),
            'hypothesis_file': str(hyp_file),
            **row,
        }
        for row in rows
    ]


def _write_simple_entity_details_tsv(fh, rows):
    writer = csv.DictWriter(
        fh,
        fieldnames=SIMPLE_ENTITY_DETAIL_COLUMNS,
        delimiter='\t',
        lineterminator='\n',
    )
    writer.writeheader()
    writer.writerows(rows)


def process_lines(ref_utts, hyp_utts, debug, use_chardiff, isctm, skip_detailed,
                  terminal_width, oracle_wer, keywords, oov_set, cer, utt_group_map,
                  group_stats, nocolor, insert_tok, fullprint=False, suppress_warnings=False,
                  simple_entity_words=None):

    error_stats = ErrorStats()
    dct_char = {insert_tok: 0, 0: insert_tok}
    multilines = []
    if simple_entity_words is None:
        simple_entity_words = set()
    for utt in ref_utts.keys():
        logger.debug('%s' % utt)
        ref = ref_utts[utt]

        is_empty_reference = not len(ref.words)

        if oracle_wer:
            hyps = hyp_utts[utt]
            costs = []
            for hyp in hyps:
                _, _, cost = align_texts(ref.words, hyp.words, debug=debug, use_chardiff=use_chardiff)
                costs.append(cost)
            error_stats.total_cost += min(costs)
            error_stats.total_count += len(ref)
            continue

        hyp = hyp_utts.get(utt)
        if hyp is None:
            logger.warning(f'Missing hypothesis for utterance: {utt}')
            continue
        error_stats.utts.append(utt)
        logger.debug('ref: %s' % ref.words)
        logger.debug('hyp: %s' % hyp.words)

        if not isctm:
            ref_aligned, hyp_aligned, cost = align_texts(ref.words, hyp.words, debug=debug, use_chardiff=use_chardiff)
        else:
            ref_aligned, hyp_aligned, cost = align_texts_ctm(ref.words, hyp.words, ref.times,
                                                             hyp.times, ref.durs, hyp.durs, debug, insert_tok)
        error_stats.total_cost += cost

        # Counting errors
        error_count = 0
        ref_word_count = 0

        double_line = MultiLine(terminal_width, 2)
        for i, (ref_w, hyp_w,) in enumerate(zip(ref_aligned, hyp_aligned)):
            if ref_w in keywords:
                error_stats.keywords_count += 1
            if hyp_w in keywords:
                error_stats.keywords_output += 1
            if ref_w in oov_set:
                error_stats.oov_word_count += 1
            if ref_w in simple_entity_words:
                error_stats.simple_entity_count += 1

            if ref_w == hyp_w:
                if hyp_w in keywords:
                    error_stats.keywords_predicted += 1
                if ref_w in simple_entity_words:
                    error_stats.simple_entity_matches += 1
                    error_stats.simple_entity_recognized[ref_w] += 1
                if not fullprint:
                    double_line.add_lineelement(ref_w, '')
                else:
                    double_line.add_lineelement(ref_w, ref_w)
                error_stats.word_counts[ref_w] += 1
                ref_word_count += 1
            else:
                error_count += 1
                if ref_w in oov_set:
                    error_stats.oov_word_error += 1
                if ref_w in simple_entity_words:
                    error_stats.simple_entity_missed[ref_w] += 1
                if ref_w == '<eps>':
                    if fullprint:
                        double_line.add_lineelement('', hyp_w)
                    elif not nocolor:
                        double_line.add_lineelement('', colored(hyp_w, 'red', force_color=True))
                    else:
                        hyp_w_upper = hyp_w.upper()
                        double_line.add_lineelement('*', hyp_w_upper)
                    error_stats.ins[hyp_w] += 1
                elif hyp_w == '<eps>':
                    if fullprint:
                        double_line.add_lineelement(ref_w, '')
                    elif not nocolor:
                        double_line.add_lineelement(colored(ref_w, 'green', force_color=True), '')
                    else:
                        ref_w_upper = ref_w.upper()
                        double_line.add_lineelement(ref_w_upper, '*')
                    ref_word_count += 1
                    error_stats.dels[ref_w] += 1
                    error_stats.word_counts[ref_w] += 1
                else:
                    ref_word_count += 1
                    key = f'{ref_w}>{hyp_w}'
                    if fullprint:
                        double_line.add_lineelement(ref_w, hyp_w,)
                    elif not nocolor:
                        double_line.add_lineelement(colored(ref_w, 'green', force_color=True), colored(hyp_w, 'red', force_color=True))
                    else:
                        ref_w_upper = ref_w.upper()
                        hyp_w_upper = hyp_w.upper()
                        double_line.add_lineelement(ref_w_upper, hyp_w_upper)
                    error_stats.subs[key] += 1
                    error_stats.word_counts[ref_w] += 1
                #breakpoint()
        error_stats.total_count += ref_word_count
        #breakpoint()
        if not skip_detailed:
            multilines.append(double_line)

        if utt_group_map:
            group = utt_group_map[utt]
            group_stats[group]['count'] += ref_word_count
            group_stats[group]['errors'] += error_count

        if error_count:
            error_stats.utt_wrong += 1

        if cer:  # Calculate CER
            def convert_to_char_list(lst):
                new = []
                for i, word in enumerate(lst):
                    new.extend(list(word))
                    if i != len(lst) - 1:
                        new.append(' ')
                return new

            char_ref = convert_to_char_list(ref.words)
            char_hyp = convert_to_char_list(hyp.words)

            ref_int, hyp_int = convert_to_int(char_ref, char_hyp, dct_char)
            error_stats.char_error_count += lev_distance(ref_int, hyp_int)
            error_stats.char_count += len(ref_int)

        if is_empty_reference:
            continue

        if oov_set:  # Get OOV CER
            err, cnt = get_oov_cer(ref_aligned, hyp_aligned, oov_set)
            error_stats.oov_count_error += err
            error_stats.oov_count_denom += cnt
    # if not skip_detailed:
        # assert len(multilines) == len(error_stats.utts)
    return multilines, error_stats



def _remove_color(word):
    return re.sub(br'\x1b\[[0-9]{2}m([\p{L}\p{P}]+)\x1b\[0m', br'\1', word.encode()).decode()


def get_color_lengthoffset(word):
    if word.count('\x1b') == 2:
        return 9
    if word.count('\x1b') == 4:
        return 18
    return 0


def _merge_multilines(multilines_a, multilines_b, terminal_width, usecolor):
    multilines = []
    # print(multilines_a)
    # print()
    # print(multilines_b)
    # print()
    for multiline_a, multiline_b in zip(multilines_a, multilines_b):
        multiline = MultiLine(terminal_width, 3)
        idx_a, idx_b = 0, 0
        while idx_a < len(multiline_a) and idx_b < len(multiline_b):
            le_a = multiline_a[idx_a]
            le_b = multiline_b[idx_b]
            hyp_worda = le_a[1]
            hyp_wordb = le_b[1]
            
            if le_a[0] == le_b[0]:  # ref words match
                refword = le_a[0]
                if refword == hyp_worda and refword == hyp_wordb:  # everything correct
                    multiline.add_lineelement(refword, hyp_worda, hyp_wordb)
                elif not refword:  # double insertion
                    if usecolor:
                        hyp_worda = colored(hyp_worda, 'red', force_color=True)
                        hyp_wordb = colored(hyp_wordb, 'red', force_color=True)
                    
                    multiline.add_lineelement('', hyp_worda, hyp_wordb)
                else:  # hyp1 and/or hyp2 are wrong
                    if usecolor:
                        if refword != hyp_worda and hyp_worda:
                            hyp_worda = colored(hyp_worda, 'red', force_color=True)

                        if refword != hyp_wordb and hyp_wordb:
                            hyp_wordb = colored(hyp_wordb, 'red', force_color=True)
                        refword = colored(refword, 'green', force_color=True)

                    multiline.add_lineelement(refword, hyp_worda, hyp_wordb)
                idx_a += 1
                idx_b += 1
            elif not le_a[0]:  # ins
                if usecolor:
                    hyp_worda = colored(hyp_worda, 'red', force_color=True)
                multiline.add_lineelement('', hyp_worda, '')
                idx_a += 1
            elif not le_b[0]:  # ins
                if usecolor:
                    hyp_wordb = colored(hyp_wordb, 'red', force_color=True)
                multiline.add_lineelement('', '', hyp_wordb)
                idx_b += 1
            
            else:
                logger.warning('Weird case!! found please report')
                refword = le_a[0] + '|' + le_b[0]
                if usecolor:
                    refword = colored(refword, 'green', force_color=True)
                    if le_a[0] != hyp_worda:
                        hyp_worda = colored(hyp_worda, 'red', force_color=True)
                    if le_b[0] != hyp_wordb:
                        hyp_wordb = colored(hyp_wordb, 'red', force_color=True)
                multiline.add_lineelement(refword, hyp_worda, hyp_wordb)
                idx_a += 1
                idx_b += 1

        while idx_a < len(multiline_a):
            assert idx_b == len(multiline_b)
            le_a = multiline_a[idx_a]
            multiline.add_lineelement(le_a[0], colored(le_a[1], 'red', force_color=True), '')
            idx_a += 1
        while idx_b < len(multiline_b):
            assert idx_a == len(multiline_a)
            le_b = multiline_b[idx_b]
            multiline.add_lineelement(le_b[0], '', colored(le_b[1], 'red', force_color=True))
            idx_b += 1
        #print(multiline)
        multilines.append(multiline)
    return multilines


def process_multiple_outputs(ref_utts, hypa_utts, hypb_utts, fh, num_top_errors,
                             use_chardiff, freq_sort, ref_file, file_a, file_b, terminal_width=None, usecolor=False):
    if terminal_width is None:
        terminal_width, _ = shutil.get_terminal_size()
        terminal_width = 120 if terminal_width >= 120 else terminal_width

    multilines_ref_hypa, error_stats_ref_hypa = process_lines(ref_utts, hypa_utts, False, use_chardiff, False,
                                            False, terminal_width, False, [], [], False,
                                            None, None, nocolor=False, insert_tok='<eps>',fullprint=True)
    multilines_ref_hypb, error_stats_ref_hypb = process_lines(ref_utts, hypb_utts, False, use_chardiff, False,
                                                              False, terminal_width, False, [], [], False,
                                                              None, None, nocolor=False, insert_tok='<eps>', fullprint=True)
    _, error_stats_hypa_hypb = process_lines(hypa_utts, hypb_utts, False, use_chardiff, False,
                                                              True, terminal_width, False, [], [], False,
                                                              None, None, nocolor=True, insert_tok='<eps>')

    merged_multiline = _merge_multilines(multilines_ref_hypa, multilines_ref_hypb,
                                         terminal_width, usecolor)
    fh.write(f'Per utt details, order is \"{ref_file}\", \"{file_a}\", \"{file_b}\":\n')
    for utt, multiline in zip(error_stats_ref_hypa.utts, merged_multiline):
        fh.write(f'{utt}\n')
        for lines in multiline.iter_construct():
            for line in lines:
                fh.write(f'{line}\n')

    # Outputting metrics from gathered statistics.
    ins_count = sum(error_stats_ref_hypa.ins.values())
    del_count = sum(error_stats_ref_hypa.dels.values())
    sub_count = sum(error_stats_ref_hypa.subs.values())
    wer = (ins_count + del_count + sub_count) / float(error_stats_ref_hypa.total_count)
    fh.write(f'\nResults with file {file_a}'
        f'\nWER: {100. * wer:.1f} (ins {ins_count}, del {del_count}, sub {sub_count} / {error_stats_ref_hypa.total_count})'
        f'\nSER: {100. * error_stats_ref_hypa.utt_wrong / len(error_stats_ref_hypa.utts):.1f}\n')

    print_detailed_stats(fh, error_stats_ref_hypa.ins, error_stats_ref_hypa.dels,
                         error_stats_ref_hypa.subs, num_top_errors, freq_sort,
                         error_stats_ref_hypa.word_counts)
    fh.write(f'---\n')

    ins_count = sum(error_stats_ref_hypb.ins.values())
    del_count = sum(error_stats_ref_hypb.dels.values())
    sub_count = sum(error_stats_ref_hypb.subs.values())
    wer = (ins_count + del_count + sub_count) / float(error_stats_ref_hypb.total_count)
    fh.write(f'\nResults with file {file_b}'
             f'\nWER: {100. * wer:.1f} (ins {ins_count}, del {del_count}, sub {sub_count} / {error_stats_ref_hypb.total_count})'
             f'\nSER: {100. * error_stats_ref_hypb.utt_wrong / len(error_stats_ref_hypb.utts):.1f}\n')

    print_detailed_stats(fh, error_stats_ref_hypb.ins, error_stats_ref_hypb.dels,
                         error_stats_ref_hypb.subs, num_top_errors, freq_sort,
                         error_stats_ref_hypb.word_counts)
    fh.write(f'---\n')

    fh.write(f'\nDifference between outputs:\n')
    print_detailed_stats(fh, error_stats_hypa_hypb.ins, error_stats_hypa_hypb.dels,
                         error_stats_hypa_hypb.subs, num_top_errors, freq_sort,
                         error_stats_hypa_hypb.word_counts)


def _format_comparison_value(value, decimals=1):
    if value is None:
        return '-'
    return f'{value:.{decimals}f}'


def _write_comparison_table(fh, rows, *, include_cer=False, include_simple_entity_accuracy=False,
                            include_oov=False, include_keywords=False):
    headers = ['file', 'WER', 'SER']
    if include_cer:
        headers.append('CER')
    if include_simple_entity_accuracy:
        headers.append('SimpleEntityAcc')
    if include_oov:
        headers.extend(['OOV CER', 'OOV WER'])
    if include_keywords:
        headers.extend(['Keyword Recall', 'Keyword Precision'])

    fh.write('Comparison:\n')
    fh.write('\t'.join(headers) + '\n')
    for row in rows:
        summary = row['summary']
        values = [row['file'], _format_comparison_value(summary.get('wer')), _format_comparison_value(summary.get('ser'))]
        if include_cer:
            values.append(_format_comparison_value(summary.get('cer')))
        if include_simple_entity_accuracy:
            values.append(_format_comparison_value(summary.get('simple_entity_accuracy')))
        if include_oov:
            values.append(_format_comparison_value(summary.get('oov_cer')))
            values.append(_format_comparison_value(summary.get('oov_wer')))
        if include_keywords:
            values.append(_format_comparison_value(summary.get('keyword_recall'), decimals=2))
            values.append(_format_comparison_value(summary.get('keyword_precision'), decimals=2))
        fh.write('\t'.join(values) + '\n')


def process_multiple_hyp_outputs(ref_utts, hyp_entries, fh, ref_file, cer=False, num_top_errors=10, oov_set=None,
                                 debug=False, use_chardiff=True, isctm=False, skip_detailed=False,
                                 keywords=None, utt_group_map=None, freq_sort=False, nocolor=False,
                                 insert_tok='<eps>', terminal_width=None, simple_entity_accuracy=False,
                                 output_format=OutputFormat.text, simple_entity_details_fh=None):
    rows = []
    simple_entity_detail_rows = []
    simple_entity_words = _extract_simple_entity_words(ref_utts) if simple_entity_accuracy else None
    for hyp_file, hyp_utts in hyp_entries:
        _, error_stats, group_stats, normalized_keywords, normalized_oov_set = _score_outputs(
            ref_utts,
            hyp_utts,
            debug=debug,
            use_chardiff=use_chardiff,
            isctm=isctm,
            skip_detailed=True,
            terminal_width=terminal_width,
            oracle_wer=False,
            keywords=keywords,
            oov_set=oov_set,
            cer=cer,
            utt_group_map=utt_group_map,
            freq_sort=freq_sort,
            nocolor=nocolor,
            insert_tok=insert_tok,
            simple_entity_accuracy=simple_entity_accuracy,
            simple_entity_words=simple_entity_words,
        )
        payload = _build_output_payload(
            ref_file,
            hyp_file,
            _build_output_summary(
                error_stats,
                group_stats,
                cer=cer,
                simple_entity_accuracy=simple_entity_accuracy,
                oov_set=normalized_oov_set,
                keywords=normalized_keywords,
            ),
            error_stats,
            num_top_errors=num_top_errors,
            freq_sort=freq_sort,
            simple_entity_accuracy=simple_entity_accuracy,
        )
        if simple_entity_details_fh is not None:
            simple_entity_detail_rows.extend(
                _add_simple_entity_detail_file_info(ref_file, hyp_file, error_stats.simple_entity_details)
            )
        rows.append({
            'file': hyp_file,
            'summary': payload['summary'],
            'payload': payload,
            'hyp_utts': hyp_utts,
        })

    if simple_entity_details_fh is not None:
        _write_simple_entity_details_tsv(simple_entity_details_fh, simple_entity_detail_rows)

    if output_format == OutputFormat.json:
        _write_json_payload(
            fh,
            {
                'reference_file': str(ref_file),
                'outputs': [row['payload'] for row in rows],
            },
        )
        return

    _write_comparison_table(
        fh,
        rows,
        include_cer=cer,
        include_simple_entity_accuracy=simple_entity_accuracy,
        include_oov=bool(oov_set),
        include_keywords=bool(keywords),
    )

    if skip_detailed:
        return

    for row in rows:
        fh.write(f'\n\nResults with file {row["file"]}\n')
        process_output(
            ref_utts,
            row['hyp_utts'],
            fh,
            ref_file=ref_file,
            hyp_file=row['file'],
            cer=cer,
            num_top_errors=num_top_errors,
            oov_set=oov_set,
            debug=debug,
            use_chardiff=use_chardiff,
            isctm=isctm,
            skip_detailed=False,
            keywords=keywords,
            utt_group_map=utt_group_map,
            oracle_wer=False,
            freq_sort=freq_sort,
            nocolor=nocolor,
            insert_tok=insert_tok,
            terminal_width=terminal_width,
            simple_entity_accuracy=simple_entity_accuracy,
        )


def process_output(ref_utts, hyp_utts, fh, ref_file, hyp_file, cer=False, num_top_errors=10, oov_set=None, debug=False,
                  use_chardiff=True, isctm=False, skip_detailed=False,
                  keywords=None, utt_group_map=None, oracle_wer=False,
                  freq_sort=False, nocolor=False, insert_tok='<eps>', terminal_width=None,
                  simple_entity_accuracy=False, output_format=OutputFormat.text, simple_entity_details_fh=None):
    effective_skip_detailed = skip_detailed or output_format == OutputFormat.json
    multilines, error_stats, group_stats, keywords, oov_set = _score_outputs(
        ref_utts,
        hyp_utts,
        debug=debug,
        use_chardiff=use_chardiff,
        isctm=isctm,
        skip_detailed=effective_skip_detailed,
        terminal_width=terminal_width,
        oracle_wer=oracle_wer,
        keywords=keywords,
        oov_set=oov_set,
        cer=cer,
        utt_group_map=utt_group_map,
        freq_sort=freq_sort,
        nocolor=nocolor,
        insert_tok=insert_tok,
        simple_entity_accuracy=simple_entity_accuracy,
    )

    if not effective_skip_detailed and not oracle_wer:
        if nocolor:
            fh.write(f'\"{ref_file}\" is treated as reference, \"{hyp_file}\" as hypothesis. Errors are capitalized.\n')
        else:
            fh.write(f'\"{ref_file}\" is treated as reference (white and green), \"{hyp_file}\" as hypothesis (white and red).\n')
        fh.write(f'Per utt details:\n')
        for utt, multiline in zip(error_stats.utts, multilines):
            fh.write(f'{utt}\n')
            for upper_line, lower_line in multiline.iter_construct():
                fh.write(f'{upper_line}\n')
                fh.write(f'{lower_line}\n')

    if not use_chardiff and not oracle_wer:
        s = sum(v for v in chain(error_stats.ins.values(), error_stats.dels.values(), error_stats.subs.values()))
        assert s == error_stats.total_cost, f'{s} {error_stats.total_cost}'
    if oracle_wer:
        if output_format == OutputFormat.json:
            _write_json_payload(
                fh,
                {
                    'reference_file': str(ref_file),
                    'hypothesis_file': str(hyp_file),
                    'summary': {
                        'oracle_wer': error_stats.total_cost / error_stats.total_count,
                        'total_ref_words': error_stats.total_count,
                    },
                },
            )
            return
        fh.write(f'Oracle WER: {error_stats.total_cost / error_stats.total_count}\n')
        return

    summary = _build_output_summary(
        error_stats,
        group_stats,
        cer=cer,
        simple_entity_accuracy=simple_entity_accuracy,
        oov_set=oov_set,
        keywords=keywords,
    )
    if simple_entity_details_fh is not None:
        _write_simple_entity_details_tsv(
            simple_entity_details_fh,
            _add_simple_entity_detail_file_info(ref_file, hyp_file, error_stats.simple_entity_details),
        )
    if output_format == OutputFormat.json:
        _write_json_payload(
            fh,
            _build_output_payload(
                ref_file,
                hyp_file,
                summary,
                error_stats,
                num_top_errors=num_top_errors,
                freq_sort=freq_sort,
                simple_entity_accuracy=simple_entity_accuracy,
            ),
        )
        return

    if not effective_skip_detailed:
        fh.write('\n')
    fh.write(
        f'WER: {summary["wer"]:.1f} '
        f'(ins {summary["ins_count"]}, del {summary["del_count"]}, sub {summary["sub_count"]} / {error_stats.total_count})'
        f'\nSER: {summary["ser"]:.1f}\n'
    )

    if simple_entity_accuracy:
        fh.write(
            f'Simple Entity Accuracy: {summary["simple_entity_accuracy"]:.1f} '
            f'({error_stats.simple_entity_matches} / {error_stats.simple_entity_count})\n'
        )

    if 'cer' in summary:
        fh.write(f'CER: {summary["cer"]:.1f} ({error_stats.char_error_count} / {error_stats.char_count})\n')
    if oov_set:
        if error_stats.oov_word_count:
            fh.write(f'OOV CER: {summary["oov_cer"]:.1f}\n')
            fh.write(f'OOV WER: {summary["oov_wer"]:.1f}\n')
        else:
            logger.error('None of the words in the OOV list file were found in the reference!')
    if keywords:
        fh.write(
            f'Keyword results - recall {summary["keyword_recall"]:.2f} '
            f'- precision {summary["keyword_precision"]:.2f}\n'
        )
    if utt_group_map:
        fh.write('Group WERs:\n')
        for group, stats in summary.get('group_stats', {}).items():
            fh.write(f'{group}\t{stats["wer"]:.1f}\n')
        fh.write('\n')

    if not effective_skip_detailed:
        print_detailed_stats(fh, error_stats.ins, error_stats.dels, error_stats.subs, num_top_errors, freq_sort,
                             error_stats.word_counts)
        if simple_entity_accuracy:
            print_simple_entity_stats(
                fh,
                error_stats.simple_entity_missed,
                error_stats.simple_entity_recognized,
                num_top_errors,
            )


def main(
    ref_file: str,
    hyp_file: str,
    outf: str = '',
    oov_list_f: str = '',
    isark: bool = False,
    isctm: bool = False,
    use_chardiff: bool = False,
    cer: bool = False,
    debug: bool = False,
    skip_detailed: bool = False,
    keywords_f: str = '',
    freq_sort: bool = False,
    oracle_wer: bool = False,
    utt_group_map_f: str = '',
    usecolor: bool = False,
    num_top_errors: int = 10,
    second_hyp_f: str = '',
    simple_entity_accuracy: bool = False,
    simple_entity_details_out: str = '',
    output_format: str = 'text',
    ):
    hyp_files = [hyp_file]
    if second_hyp_f:
        hyp_files.append(second_hyp_f)
    _run_cli(
        ref_file=ref_file,
        hyp_files=hyp_files,
        outf=outf,
        oov_list_f=oov_list_f,
        isark=isark,
        isctm=isctm,
        use_chardiff=use_chardiff,
        cer=cer,
        debug=debug,
        skip_detailed=skip_detailed,
        keywords_f=keywords_f,
        freq_sort=freq_sort,
        oracle_wer=oracle_wer,
        utt_group_map_f=utt_group_map_f,
        usecolor=usecolor,
        num_top_errors=num_top_errors,
        simple_entity_accuracy=simple_entity_accuracy,
        output_format=output_format,
        simple_entity_details_out=simple_entity_details_out,
    )


def _load_oov_set(oov_list_f, use_chardiff):
    oov_set = set()
    if oov_list_f:
        if not use_chardiff:
            logger.warning('Because you are using standard alignment (not `-use_chardiff`) the alignments could be suboptimal\n'
                           ' which will lead to the OOV-CER being slightly wrong. Use `-use_chardiff` for better alignment, ctm based for the best.')
        with open(oov_list_f) as fh_oov:
            for line in fh_oov:
                oov_set.add(line.split()[0])
    return oov_set


def _old_interface_message(ref_file, hyp_files, outf_candidate):
    hyp_args = ' '.join(hyp_files[:-1]) if len(hyp_files) > 1 else hyp_files[0]
    return (
        f'Input file not found: {outf_candidate}\n'
        'It looks like you may be using the old CLI interface where the output file was positional.\n'
        f'Use -o/--out instead, for example:\n  texterrors {ref_file} {hyp_args} -o {outf_candidate}\n'
        'If this was meant to be a hypothesis file, check that the path exists.'
    )


def _validate_cli_paths(ref_file, hyp_files, outf):
    if not os.path.exists(ref_file):
        raise typer.BadParameter(f'Input file not found: {ref_file}', param_hint='ref_file')
    if len(hyp_files) > 1 and not outf and not os.path.exists(hyp_files[-1]):
        raise typer.BadParameter(_old_interface_message(ref_file, hyp_files, hyp_files[-1]), param_hint='hyp_files')
    for hyp_file in hyp_files:
        if not os.path.exists(hyp_file):
            raise typer.BadParameter(f'Input file not found: {hyp_file}', param_hint='hyp_files')


def _read_keyword_file(keywords_f):
    keywords = set()
    if keywords_f:
        for line in open(keywords_f):
            assert len(line.split()) == 1, 'A keyword must be a single word!'
            keywords.add(line.strip())
    return keywords


def _read_utt_group_map(utt_group_map_f):
    utt_group_map = {}
    if utt_group_map_f:
        for line in open(utt_group_map_f):
            uttid, group = line.split(maxsplit=1)
            utt_group_map[uttid] = group.strip()
    return utt_group_map


def _validate_cli_options(hyp_files, isark, isctm, oracle_wer, simple_entity_details_out):
    if not oracle_wer:
        return
    if len(hyp_files) != 1:
        raise typer.BadParameter('Oracle WER mode only supports a single hypothesis input.', param_hint='hyp_files')
    if not isark or isctm:
        raise typer.BadParameter('Oracle WER requires `--isark` and does not support `--isctm`.')
    if simple_entity_details_out:
        raise typer.BadParameter(
            '`--entity-details` is not supported with `--oracle-wer`.',
            param_hint='simple_entity_details_out',
        )


def _run_cli(ref_file, hyp_files, outf='', oov_list_f='', isark=False, isctm=False,
             use_chardiff=False, cer=False, debug=False, skip_detailed=False, keywords_f='',
             freq_sort=False, oracle_wer=False, utt_group_map_f='', usecolor=False,
             num_top_errors=10, simple_entity_accuracy=False, output_format=OutputFormat.text,
             simple_entity_details_out=''):
    """Execute the CLI request after argument parsing.

    This indirection keeps the Typer command thin and declarative while putting the
    actual branching and file-handling logic in one testable function.
    """
    output_format = OutputFormat(output_format)
    _validate_cli_paths(ref_file, hyp_files, outf)
    if simple_entity_details_out:
        simple_entity_accuracy = True
    _validate_cli_options(hyp_files, isark, isctm, oracle_wer, simple_entity_details_out)

    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if debug else 'INFO')

    fh = open(outf, 'w') if outf else sys.stdout
    simple_entity_details_fh = open(simple_entity_details_out, 'w') if simple_entity_details_out else None
    try:
        if oracle_wer:
            skip_detailed = True
            if use_chardiff:
                logger.warning('You probably would prefer running without `-use_chardiff`, the WER will be slightly better for the cost of a worse alignment')

        oov_set = _load_oov_set(oov_list_f, use_chardiff)

        if len(hyp_files) == 1:
            ref_utts, hyp_utts, keywords, utt_group_map = read_files(
                ref_file,
                hyp_files[0],
                isark,
                isctm,
                keywords_f,
                utt_group_map_f,
                oracle_wer,
            )
            process_output(
                ref_utts,
                hyp_utts,
                fh,
                cer=cer,
                debug=debug,
                oov_set=oov_set,
                ref_file=ref_file,
                hyp_file=hyp_files[0],
                use_chardiff=use_chardiff,
                skip_detailed=skip_detailed,
                keywords=keywords,
                utt_group_map=utt_group_map,
                freq_sort=freq_sort,
                isctm=isctm,
                oracle_wer=oracle_wer,
                nocolor=not usecolor,
                num_top_errors=num_top_errors,
                simple_entity_accuracy=simple_entity_accuracy,
                output_format=output_format,
                simple_entity_details_fh=simple_entity_details_fh,
            )
            return

        ref_utts = read_ref_file(ref_file, isark) if not isctm else read_ctm_file(ref_file)
        keywords = _read_keyword_file(keywords_f)
        utt_group_map = _read_utt_group_map(utt_group_map_f)

        hyp_entries = []
        for hyp_file in hyp_files:
            hyp_utts = read_hyp_file(hyp_file, isark, False) if not isctm else read_ctm_file(hyp_file)
            hyp_entries.append((hyp_file, hyp_utts))

        process_multiple_hyp_outputs(
            ref_utts,
            hyp_entries,
            fh,
            ref_file=ref_file,
            cer=cer,
            num_top_errors=num_top_errors,
            oov_set=oov_set,
            debug=debug,
            use_chardiff=use_chardiff,
            isctm=isctm,
            skip_detailed=skip_detailed,
            keywords=keywords,
            utt_group_map=utt_group_map,
            freq_sort=freq_sort,
            nocolor=not usecolor,
            simple_entity_accuracy=simple_entity_accuracy,
            output_format=output_format,
            simple_entity_details_fh=simple_entity_details_fh,
        )
    finally:
        if simple_entity_details_fh is not None:
            simple_entity_details_fh.close()
        if fh is not sys.stdout:
            fh.close()


app = typer.Typer(
    add_completion=False,
    context_settings={'help_option_names': ['-h', '--help']},
    pretty_exceptions_enable=False,
)


@app.command()
def run(
    ref_file: str = typer.Argument(..., help='Reference text.'),
    hyp_files: List[str] = typer.Argument(..., help='One or more hypothesis files.'),
    out: Optional[str] = typer.Option(None, '--out', '-o', help='Optional output file.'),
    oov_list_f: str = typer.Option('', '--oov-list-f', help='List of OOVs.'),
    isark: bool = typer.Option(False, '--isark', help='Text files start with utterance ID.'),
    isctm: bool = typer.Option(False, '--isctm', help='Text files start with utterance ID and end with word, time, duration.'),
    use_chardiff: bool = typer.Option(False, '--use-chardiff', help='Use character lev distance for better alignment in exchange for slightly higher WER.'),
    cer: bool = typer.Option(False, '--cer', help='Calculate CER.'),
    debug: bool = typer.Option(False, '--debug', '-d', help='Print debug messages, will write cost matrix to summedcost.'),
    skip_detailed: bool = typer.Option(False, '--skip-detailed', '-s', help='No per utterance output.'),
    keywords_f: str = typer.Option('', '--keywords-f', '--keywords-list-f', help='Will filter out non keyword reference words.'),
    freq_sort: bool = typer.Option(False, '--freq-sort', help='Turn on sorting del/sub errors by frequency (default is by count).'),
    oracle_wer: bool = typer.Option(False, '--oracle-wer', help='Hyp file should have multiple hypothesis per utterance, lowest edit distance will be used for WER.'),
    utt_group_map_f: str = typer.Option('', '--utt-group-map', '--utt-group-map-f', help='Should be a file which maps uttids to group, WER will be output per group.'),
    usecolor: bool = typer.Option(False, '--usecolor', '-c', help='Show detailed output with color (use less -R). Red/white is reference, Green/white model output.'),
    num_top_errors: int = typer.Option(10, '--num-top-errors', help='Number of errors to show per type in detailed output.'),
    simple_entity_accuracy: bool = typer.Option(False, '--simple-entity-accuracy', '-w', help='Use simple entity accuracy from reference-side casing cues, matching with whitespace ignored.'),
    simple_entity_details_out: str = typer.Option('', '--entity-details', help='Optional TSV file with one row per entity occurrence and normalized compact model output.'),
    output_format: OutputFormat = typer.Option(OutputFormat.text, '--output-format', help='Output format. JSON contains only aggregate statistics and top-error summaries.'),
):
    _run_cli(
        ref_file=ref_file,
        hyp_files=hyp_files,
        outf=out or '',
        oov_list_f=oov_list_f,
        isark=isark,
        isctm=isctm,
        use_chardiff=use_chardiff,
        cer=cer,
        debug=debug,
        skip_detailed=skip_detailed,
        keywords_f=keywords_f,
        freq_sort=freq_sort,
        oracle_wer=oracle_wer,
        utt_group_map_f=utt_group_map_f,
        usecolor=usecolor,
        num_top_errors=num_top_errors,
        simple_entity_accuracy=simple_entity_accuracy,
        output_format=output_format,
        simple_entity_details_out=simple_entity_details_out,
    )


def cli():
    app()


if __name__ == "__main__":
    cli()
