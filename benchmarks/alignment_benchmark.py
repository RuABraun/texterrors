#!/usr/bin/env python
import argparse
import io
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

from texterrors.alignment import StringVector
from texterrors.texterrors import process_output


@dataclass
class Utt:
    uid: str
    words: StringVector


def load_ark_like(path):
    utts = {}
    with open(path) as fh:
        for line in fh:
            utt, *words = line.split()
            utts[utt] = Utt(utt, StringVector(words))
    return utts


def run_case(refs, hyps, *, use_chardiff, skip_detailed):
    buffer = io.StringIO()
    start = time.perf_counter()
    process_output(
        refs,
        hyps,
        buffer,
        ref_file="ref",
        hyp_file="hyp",
        skip_detailed=skip_detailed,
        use_chardiff=use_chardiff,
        debug=False,
    )
    return time.perf_counter() - start


def benchmark_case(name, refs, hyps, *, use_chardiff, skip_detailed, warmup, repeat):
    for _ in range(warmup):
        run_case(refs, hyps, use_chardiff=use_chardiff, skip_detailed=skip_detailed)

    times = [
        run_case(refs, hyps, use_chardiff=use_chardiff, skip_detailed=skip_detailed)
        for _ in range(repeat)
    ]
    print(
        f"{name}: mean={statistics.mean(times):.6f}s "
        f"median={statistics.median(times):.6f}s min={min(times):.6f}s max={max(times):.6f}s"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark texterrors alignment/reporting cases.")
    parser.add_argument("--ref", type=Path, default=Path("tests/reftext"))
    parser.add_argument("--hyp", type=Path, default=Path("tests/hyptext"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    refs = load_ark_like(args.ref)
    hyps = load_ark_like(args.hyp)

    benchmark_case(
        "fast-default",
        refs,
        hyps,
        use_chardiff=False,
        skip_detailed=True,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    benchmark_case(
        "char-aware",
        refs,
        hyps,
        use_chardiff=True,
        skip_detailed=True,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    benchmark_case(
        "detailed-output",
        refs,
        hyps,
        use_chardiff=True,
        skip_detailed=False,
        warmup=args.warmup,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    main()
