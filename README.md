# texterrors

For calculating WER, CER, other metrics and getting detailed statistics.  

Does character aware alignment by default, core is C++ so is fast.

Supports scoring by group (for example by speaker) or just scoring keywords or phrases and other things. Meant to replace older tools like `sclite` by being easy to use, modify and extend.

See here for [background motivation](https://ruabraun.github.io/jekyll/update/2020/11/27/On-word-error-rates.html).

Green - Insertion, Red - Deletion, Purple - Substitution

![Example](docs/images/texterrors_example.png)

Colored output is based on the [werpp](https://github.com/nsmartinez/WERpp) package by https://github.com/nsmartinez

# Installing
Requires minimum python 3.6!
```
git clone https://github.com/RuABraun/texterrors.git
cd texterrors
pip install -r requirements.txt
pip install .
```
The package will be installed as `texterrors`.

# Example

The `texterrors.py` file will be in your path after running pip install.

## Command line

The `-s` option means there will be no detailed output. Below `ref` and `hyp` are files with the first field equalling the utterance ID.
```
$ texterrors.py -isark -s ref hyp
WER: 83.33 (ins 1, del 1, sub 3 / 6)
```

You can specify an output file to save the results, probably what you if you are getting detailed output.
```
$ texterrors.py -isark -cer -oov-list-f oov_list ref hyp detailed_wer_output
```
If you look at the output file with `less` use the `-R` flag to see color.

# Options you might want to use 

There are more options, call with `-h` to see.

`-utt-group-map` - Should be a file which maps uttids to group, WER will be output per group (could use
to get per speaker WER for example).

`-isctm` - Will use time stamps for alignment (this will give the best one), last three columns of ctm should be time, duration, word.

`-oov-list-f` - The CER between words aligned to the OOV words will be calculated (the OOV-CER). 

`-keywords-list-f` - The hypothesis is assumed to only contain keywords, the reference is filtered by them before calculating metrics like WER.

`-phrase-f` - If you just want to score a phrase inside an utterance.

# Why is the WER slightly higher than in kaldi ?

**You can make it equal by using the `-no-chardiff` argument.**

This difference is because this tool does character aware alignment. Across a normal sized test set this should result in a small difference. 

In the below example a normal WER calculation would do a one-to-one mapping and arrive at a WER of 66.67\%.

| test | sentence | okay    | words | ending | now |
|------|----------|---------|-------|--------|-----|
| test | a        | sentenc | ok    | endin  | now |

But character aware alignment would result in the following alignment:

| test | - | sentence | okay | words | ending | now |
|------|---|----------|------|-------|--------|-----|
| test | a | sentenc  | ok   | -     | endin  | now |

This results in a WER of 83.3\% because of the extra insertion and deletion. And I think one could argue this is the actually correct WER.
