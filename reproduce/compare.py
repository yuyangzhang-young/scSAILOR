"""Written tables against the deposited ones, cell for cell: the ``--compare`` of the reproduction commands."""
import csv
from pathlib import Path

__all__ = ["read_rows", "compare"]


def read_rows(path):
    """The rows of a CSV file, as strings."""
    with Path(path).open(newline="") as handle:
        return list(csv.reader(handle))


def compare(out, root, paths):
    """One line per table path (relative to the data root): 'equal' or 'DIFFERS', as the table written under ``out``
    is or is not the same, cell for cell, as the one at the same path under ``root``; 'not deposited' where ``root``
    has none. On another CPU the last digits of a value can differ; a 'DIFFERS' is then checked numerically."""
    lines = []
    for path in map(Path, paths):
        if not (root / path).is_file():
            lines.append(f"not deposited  {path.as_posix()}")
            continue
        same = read_rows(root / path) == read_rows(out / path)
        lines.append(f"{'equal' if same else 'DIFFERS'}  {path.as_posix()}")
    return lines
