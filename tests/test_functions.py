import Levenshtein as levd
import texterrors


def test_levd():
    pairs = ['a', '', '', 'a', 'MOZILLA', 'MUSIAL', 'ARE', 'MOZILLA', 'TURNIPS', 'TENTH', 'POSTERS', 'POSTURE']
    for a, b in zip(pairs[:-1:2], pairs[1::2]):
        d1 = texterrors.lev_distance(a, b)
        print(a, b, d1)
        # d2 = texterrors.lev_distance(a, b)
        # if d1 != d2:
        #     print(a, b, d1, d2)
        #     raise RuntimeError('Assert failed!')



test_levd()