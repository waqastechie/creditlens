"""Command-line entry point: ``python -m creditlens.cli generate --rows 50000``."""

from __future__ import annotations

import argparse
from pathlib import Path

from creditlens.data.synthetic import GeneratorConfig, generate_applications


def _generate(args: argparse.Namespace) -> None:
    cfg = GeneratorConfig(
        n_rows=args.rows,
        seed=args.seed,
        target_default_rate=args.default_rate,
        missing_rate=args.missing_rate,
        messy=args.messy,
    )
    df = generate_applications(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} applications to {out} (default rate {df['defaulted'].mean():.2%})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="creditlens")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate synthetic loan applications")
    gen.add_argument("--rows", type=int, default=10_000)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--default-rate", type=float, default=0.08)
    gen.add_argument("--missing-rate", type=float, default=0.0)
    gen.add_argument("--messy", action="store_true", help="Inject inconsistent labels and outliers")
    gen.add_argument("--out", default="data/raw/applications.csv")
    gen.set_defaults(func=_generate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
