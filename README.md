# textalign

For aligning two pieces of text with each other.

# Example

```
>>> from textalign import textalign
>>> a = ['a', 'b', 'c', 'd', 'e']
>>> b = ['z', 'a', 'b', 'd', 'e']
>>> textalign.align_texts(a, b, insert_tok='-')
(['-', 'a', 'b', 'c', 'd', 'e'], ['z', 'a', 'b', '-', 'd', 'e'])
```
