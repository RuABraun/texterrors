
# texterrors  
  
For calculating WER, CER, other metrics, getting detailed statistics and comparing outputs. 

Meant to replace older tools like `sclite` by being easy to use, modify and extend.    
  
Features:
- Character aware, standard (default) and ctm based alignment
- Metrics by group (for example speaker)
- Comparing two hypothesis files to reference
- Oracle WER
- **NEW** Simple entity accuracy from reference-side casing
- Sorting most common errors by frequency or count
- Measuring performance on keywords
- Measuring OOV-CER (see [https://arxiv.org/abs/2107.08091](https://arxiv.org/abs/2107.08091) )
- Colored output to inspect errors

Example of colored output below (use `-c` flag). Read the white and green words to read the reference. Read the white and red words to read the hypothesis.  

![Example](docs/images/texterrors_example.png)   

See here for [background motivation](https://ruabraun.github.io/jekyll/update/2020/11/27/On-word-error-rates.html).  


# Installing  
Requires minimum python 3.9!  
```
pip install texterrors
```
The package will be installed as `texterrors` and there will be a `texterrors` script in your path.  

For development, create a local environment and install build/test dependencies, then build in place:
```  
$ uv venv
$ env UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python -r requirements.txt
$ .venv/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$PWD/.venv/bin/python -Dnanobind_DIR=$PWD/.venv/lib/python3.12/site-packages/nanobind/cmake
$ .venv/bin/cmake --build build --config Release
$ .venv/bin/cmake --install build --config Release --prefix $PWD
```  

# Example

The `-s` option means there will be no detailed output. Below `ref` and `hyp` are files with the first field equalling the utterance ID (therefore the `isark` flag).  
```
$ texterrors -isark -s ref hyp  
WER: 83.33 (ins 1, del 1, sub 3 / 6)  
```  
  
You can specify an output file to save the results, probably what you want if you are getting detailed output (not using `-s`). 
Here we are also calculating the CER, the OOV-CER to measure the performance on the OOV words inside the `oov_list` file, and using
colored output (therefore the `-c` flag).
```  
$ texterrors -c -isark -cer -oov-list-f oov_list ref hyp detailed_wer_output  
```  
**Use `less -R` to view the colored output. Skip the `-c` flag to not use color.**

Check `texterrors/__init__.py` to see functions that you may be interested in using from python. 

Direct Python alignment accepts plain token lists and uses keyword-only options:
```python
from texterrors import align_texts

ref_aligned, hyp_aligned, cost = align_texts(
    ["speedbird", "eight", "six", "two"],
    ["hello", "speedbird", "six", "two"],
    use_chardiff=True,
)
```

# Benchmarking

A small benchmark harness lives in `benchmarks/alignment_benchmark.py`. By default it runs:
- fast default alignment (`use_chardiff=False`, `skip_detailed=True`)
- character-aware alignment (`use_chardiff=True`, `skip_detailed=True`)
- detailed output mode (`use_chardiff=True`, `skip_detailed=False`)

Run it from the repo root like this:
```  
$ .venv/bin/python benchmarks/alignment_benchmark.py --repeat 7
```  

You can also point it at other ark-like files:
```  
$ .venv/bin/python benchmarks/alignment_benchmark.py --ref my_ref.txt --hyp my_hyp.txt
```  

# Options you might want to use
Call `texterrors -h` to see all options.  
  
`-cer`, `-isctm` - Calculate CER, Use ctms for alignment

`-utt-group-map` - Should be a file which maps uttids to group, WER will be output per group (could use  
to get per speaker WER for example).  

`-second-hyp-f` - Use to compare the outputs of two different models to the reference.

`-w` - Calculate simple entity accuracy using reference-side casing, the top 10,000 common words for sentence-start filtering, then lowercase text for scoring.
  
`-freq-sort` - Sort errors by frequency rather than count
  
`-oov-list-f` - The CER between words aligned to the OOV words will be calculated (the OOV-CER).   
  
`-keywords-list-f` - Will calculate precision & recall of words in the file.

`-oracle-wer` - Hypothesis file should have multiple entries for each utterance, oracle WER will be calculated.
  
# Why is the WER slightly higher than in kaldi if I use `-use_chardiff`?
  
**You can make it equal by not using the `-use_chardiff` argument.**

This difference is because this tool can do character aware alignment. Across a normal sized test set this should result in a small difference.
  
In the below example a normal WER calculation would do a one-to-one mapping and arrive at a WER of 66.67\%.  
  
| test | sentence | okay    | words | ending | now |  
|------|----------|---------|-------|--------|-----|  
| test | a        | sentenc | ok    | endin  | now |  
  
But character aware alignment would result in the following alignment:  
  
| test | - | sentence | okay | words | ending | now |  
|------|---|----------|------|-------|--------|-----|  
| test | a | sentenc  | ok   | -     | endin  | now |  
  
This results in a WER of 83.3\% because of the extra insertion and deletion. And I think one could argue this is the actually correct WER.

# Changelog

Recent changes:  

- 25.03.26 Added detailed simple-entity hit/miss summaries and documented the extraction heuristic.
- 25.03.26 Refined simple entity accuracy with top-10k sentence-start filtering, full-stop sentence resets, and lowercase-occurrence suppression.
- 18.03.26 Migrated the extension module from pybind11 to nanobind and moved builds to CMake/scikit-build-core.
- 24.03.26 Removed weighted WER and added simple entity accuracy from reference-side casing.
- 11.11.25 Weighted WER for English
- 26.02.25 Faster alignment, better multihyp support, fixed multihyp bug.
- 22.06.22 refactored internals to make them simpler, character aware alignment is off by default, added more explanations
- 20.05.22 fixed bug missing regex dependency
- 16.05.22 fixed bug causing wrong detailed output when there is utterance with empty reference, and utts with empty reference are not ignored
- 21.04.22 insertion errors on lower line and switching colors so green is reference
- 27.01.22 oracle WER and small bug fixes
- 26.01.22 fixed bug causing OOV-CER feature to not work
- 22.11.21 new feature to compare two outputs to reference; lots of small changes 
- 04.10.21 fixed bug, nocolor option, refactoring, keywords feature works properly, updated README
- 22.08.21 added oracle wer feature, cost matrix creation returns cost now  
- 16.07.21 improves alignment based on ctms (much stricter now).  
