# texterrors

`texterrors` scores ASR or transcription output against a reference and helps you inspect what went wrong.

It supports:
- WER and CER
- standard and character-aware alignment
- detailed aligned error reports
- colored output for inspecting alignments
- comparison of multiple hypothesis files against the same reference
- per-group metrics such as per-speaker WER
- keyword and OOV evaluation
- OOV-CER for targeted analysis of out-of-vocabulary words, as in [this paper](https://arxiv.org/abs/2107.08091)
- oracle WER
- simple entity accuracy
- per-entity diagnostic TSV output
- aggregate JSON output for scripting

Example of colored detailed output (`--usecolor`):

![Example](docs/images/texterrors_example.png)

For more background on the motivation for the tool, see [this post](https://ruabraun.github.io/jekyll/update/2020/11/27/On-word-error-rates.html).

# Installing

Requires Python 3.9 or newer.

```bash
pip install texterrors
```

This installs both the Python package and the `texterrors` command-line tool.

# Common usage

If your files are ark-like text with an utterance ID as the first field, use `--isark`.

Compute aggregate WER only:

```bash
texterrors --isark -s ref hyp
```

Write a detailed report to a file:

```bash
texterrors --isark --cer -c -o detailed_report.txt ref hyp
```

If you use `--usecolor`, view the output with `less -R`.

Compare several systems against the same reference:

```bash
texterrors --isark -s ref hyp1 hyp2 hyp3
```

This prints a comparison table with one row per hypothesis file.

Write aggregate-only JSON instead of the normal text report:

```bash
texterrors --isark --output-format json ref hyp
```

Measure simple entity accuracy:

```bash
texterrors --isark -w ref hyp
```

Write one TSV row per entity occurrence for diagnostics:

```bash
texterrors --isark --entity-details entity_details.tsv ref hyp
```

# Input formats

By default, `texterrors` expects one reference line and one hypothesis line per utterance.

Useful input flags:

- `--isark`: each line starts with an utterance ID
- `--isctm`: input is CTM-like and includes timing fields

# Output modes

The default output is a human-readable text report. Unless you pass `--skip-detailed`, it includes per-utterance aligned detail as well as overall summary statistics.

Useful output options:

- `--skip-detailed`: show only aggregate statistics
- `--out`, `-o`: write the text report to a file
- `--output-format json`: write aggregate statistics and top-error summaries as JSON, without per-utterance detail
- `--entity-details FILE`: write a TSV with one row per simple-entity occurrence

`--entity-details` is meant for compact diagnostics. The TSV records the normalized compact hypothesis output aligned to the entity span, not the original surface form with exact spacing and casing.

# Common analysis options

- `--cer`: compute CER in addition to WER
- `--utt-group-map FILE`: report metrics by group, for example by speaker
- `--keywords-f FILE`: restrict keyword precision and recall analysis to terms in a file
- `--oov-list-f FILE`: compute OOV-CER for words in a file
- `--oracle-wer`: pick the lowest-edit-distance hypothesis when multiple hypotheses are available per utterance
- `--freq-sort`: sort error summaries by frequency instead of raw count
- `--num-top-errors N`: control how many top insertions, deletions, and substitutions are shown

# Entity scoring

`--simple-entity-accuracy` uses reference-side casing cues to identify likely entity terms and then scores whether those terms were recognized in the hypothesis.

Entity matching is case-insensitive and ignores whitespace, so examples like `GenAI` and `Gen AI` count as the same entity for this metric.

If you want to inspect misses, use `--entity-details FILE`. That TSV is useful for spotting whether an entity matched cleanly, was substituted, or was effectively deleted.

# Python usage

If you want to use the library directly from Python:

```python
from texterrors import align_texts

ref_aligned, hyp_aligned, cost = align_texts(
    ["speedbird", "eight", "six", "two"],
    ["hello", "speedbird", "six", "two"],
    use_chardiff=True,
)
```

# Benchmarking

A small benchmark harness lives in `benchmarks/alignment_benchmark.py`.

Run it from the repo root:

```bash
.venv/bin/python benchmarks/alignment_benchmark.py --repeat 7
```

You can also point it at other ark-like files:

```bash
.venv/bin/python benchmarks/alignment_benchmark.py --ref my_ref.txt --hyp my_hyp.txt
```

# Development install

If you want to build the extension locally:

```bash
uv venv
env UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$PWD/.venv/bin/python -Dnanobind_DIR=$PWD/.venv/lib/python3.12/site-packages/nanobind/cmake
.venv/bin/cmake --build build --config Release
.venv/bin/cmake --install build --config Release --prefix $PWD
```

# Note on `--use-chardiff`

`--use-chardiff` enables character-aware alignment. This often gives more intuitive alignments when words are similar, but it can also make WER slightly higher than standard token-only alignment.

If you want behavior closer to a plain token-alignment scorer, leave `--use-chardiff` off.

For example, a plain token alignment might force a one-to-one mapping:

| test | sentence | okay    | words | ending | now |
|------|----------|---------|-------|--------|-----|
| test | a        | sentenc | ok    | endin  | now |

Character-aware alignment may instead align it like this:

| test | - | sentence | okay | words | ending | now |
|------|---|----------|------|-------|--------|-----|
| test | a | sentenc  | ok   | -     | endin  | now |

That can increase WER because it exposes an insertion and deletion that the token-only alignment hides.
