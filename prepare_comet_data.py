"""Build line-aligned input files for COMET MBR from a JSONL dataset."""

import argparse
import json
import re
from pathlib import Path


# Python/COMET may treat CR and LF as line endings.  The remaining characters
# are Unicode-defined line separators that should not occur inside one sample.
LINE_SEPARATORS = re.compile(r"[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]+")


def one_line(value: object) -> str:
    """Convert a source or hypothesis to exactly one physical text line."""
    return LINE_SEPARATORS.sub(" ", str(value)).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--source-output", type=Path, default=Path("txt/s.txt"))
    parser.add_argument("--mt-output", type=Path, default=Path("mt.txt"))
    parser.add_argument("--reference-output", type=Path, default=Path("txt/ref.txt"))
    parser.add_argument("--num-samples", type=int, default=512)
    args = parser.parse_args()

    with args.jsonl.open(encoding="utf-8") as stream:
        dataset = [json.loads(line) for line in stream if line.strip()]

    sources: list[str] = []
    references: list[str] = []
    translations: list[str] = []
    for source_index, item in enumerate(dataset):
        hypotheses = item["result"]
        if len(hypotheses) < args.num_samples:
            raise ValueError(
                f"record {source_index} has {len(hypotheses)} hypotheses; "
                f"at least {args.num_samples} are required"
            )
        sources.append(one_line(item["src_text"]))
        references.append(one_line(item["ref"]))
        translations.extend(
            one_line(hypothesis) for hypothesis in hypotheses[: args.num_samples]
        )

    # newline="\n" makes the output format explicit on every platform.
    args.source_output.write_text("\n".join(sources), encoding="utf-8", newline="\n")
    args.reference_output.write_text(
        "\n".join(references), encoding="utf-8", newline="\n"
    )
    args.mt_output.write_text(
        "\n".join(translations), encoding="utf-8", newline="\n"
    )

    # Validate exactly as COMET does: text mode plus readlines().
    with args.source_output.open(encoding="utf-8") as stream:
        source_count = len(stream.readlines())
    with args.mt_output.open(encoding="utf-8") as stream:
        translation_count = len(stream.readlines())
    with args.reference_output.open(encoding="utf-8") as stream:
        reference_count = len(stream.readlines())

    expected = source_count * args.num_samples
    if (
        source_count != len(dataset)
        or reference_count != len(dataset)
        or translation_count != expected
    ):
        raise RuntimeError(
            f"invalid output: {source_count} sources, {reference_count} references, "
            f"{translation_count} translations; expected {len(dataset)}, "
            f"{len(dataset)}, and {expected}"
        )

    print(
        f"wrote {source_count} sources, {reference_count} references, and "
        f"{translation_count} translations ({args.num_samples} per source)"
    )


if __name__ == "__main__":
    main()
