import argparse
import sys
from pathlib import Path

from botocore.exceptions import ClientError
from requests import HTTPError

from . import fetch_data, list_datasets, list_experiments
from ._cache import ChecksumError


class _Progress:
    """Prints per-file progress to stderr and tallies a summary."""

    def __init__(self) -> None:
        self.downloaded = 0
        self.cached = 0

    def __call__(self, event: str, path: Path) -> None:
        if event == "cached":
            self.cached += 1
            print(f"cached      {path}", file=sys.stderr)
        elif event == "start":
            print(f"downloading {path} ... ", end="", flush=True, file=sys.stderr)
        elif event == "done":
            self.downloaded += 1
            print(_format_size(path.stat().st_size), file=sys.stderr)


def _format_size(n: int) -> str:
    size = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def experiments(args: argparse.Namespace) -> None:
    all_exp = list_experiments()
    if not all_exp:
        return  # print(*[], sep="\n") would emit a bare newline
    print(*all_exp, sep="\n")


def datasets(args: argparse.Namespace) -> None:
    all_datasets = list_datasets(args.experiment)
    if not all_datasets:
        return
    print(*all_datasets, sep="\n")


def fetch(args: argparse.Namespace) -> None:
    experiment = args.experiment
    dataset = args.dataset
    cache_path = args.output
    quiet = args.quiet

    progress = None if quiet else _Progress()

    result = fetch_data(
        experiment,
        dataset,
        cache_path,
        on_file=progress,
    )

    paths = result if isinstance(result, list) else [result]

    print(*paths, sep="\n")

    if progress is not None:
        print(
            f"{len(paths)} file(s): {progress.downloaded} downloaded, "
            f"{progress.cached} cached -> {cache_path}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lambdaquery", description="Tool to download data from LAMBDA"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # lambdaquery experiments
    exp_parser = subparsers.add_parser("experiments", help="List available experiments")
    exp_parser.set_defaults(func=experiments)

    # lambdaquery datasets EXPERIMENT
    datasets_parser = subparsers.add_parser(
        "datasets",
        help="List datasets for an experiment",
    )
    datasets_parser.add_argument("experiment", help="Experiment name")
    datasets_parser.set_defaults(func=datasets)

    # lambdaquery fetch EXPERIMENT DATASET [-o DIR] [-q]
    fetch_parser = subparsers.add_parser("fetch", help="Fetch a dataset")
    fetch_parser.add_argument("experiment", help="Experiment name")
    fetch_parser.add_argument("dataset", help="Dataset name")
    fetch_parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        type=Path,
        default=Path("."),
        help="Output directory",
    )
    fetch_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    fetch_parser.set_defaults(func=fetch)

    args = parser.parse_args(argv)

    try:
        args.func(args)
    except KeyError as err:
        print(f"error: {err.args[0]}", file=sys.stderr)
        return 1
    except (
        TypeError,
        HTTPError,
        ValueError,
        OSError,
        ClientError,
        ChecksumError,
    ) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    return 0
