# texterrors

For getting detailed WER / CER stats. 

Colored output is based on the brilliant [werpp](https://github.com/nsmartinez/WERpp) package by https://github.com/nsmartinez

# Installing
```
python -m pip install .
```
The package will be installed as `texterrors`.


# Example

The `texterrors.py` file will be in your path after running pip install.

## Command line

The `-s` option will means there will be no detailed output. Below `ref` and `hyp` are files with the first field equalling the utterance ID.
```
$ texterrors.py -isark -s ref hyp
WER: 83.33 (ins 1, del 1, sub 3 / 6)
```

You can specify an output file to save the results, probably what you if you are getting detailed output.
```
$ texterrors.py -isark -cer -oov-list-f oov_list ref hyp detailed_wer_output
```

# Why is the WER slightly higher than in kaldi ?

This difference is because this tool does character aware alignment. Across a normal sized test set this should result in a difference of ~0.1% absolute.

In the below example a normal WER calculation would do a one-to-one mapping and arrive at a WER of 66.67\%.

| test | sentence | okay    | words | ending | now |
|------|----------|---------|-------|--------|-----|
| test | a        | sentenc | ok    | endin  | now |

But character aware alignment would result in the following alignment:

| test | - | sentence | okay | words | ending | now |
|------|---|----------|------|-------|--------|-----|
| test | a | sentenc  | ok   | -     | endin  | now |

This results in a WER of 83.3\% because of the extra insertion and deletion. And I think one could argue this is the actually correct WER.
