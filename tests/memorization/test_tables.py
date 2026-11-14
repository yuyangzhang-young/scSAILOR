"""The command's comparison of written tables with saved ones, at each table's path under its screen's folder; no
study data.

    python -m pytest tests/memorization/test_tables.py
"""
import tempfile
import unittest
from pathlib import Path

from reproduce.memorization import tables


class Compare(unittest.TestCase):
    def test_each_table_reads_equal_differs_or_not_deposited_cell_for_cell(self):
        folder = tables.folder('tahoe-100m')
        with tempfile.TemporaryDirectory() as tmp:
            out, root = Path(tmp) / 'out', Path(tmp) / 'root'
            for base in (out, root):
                (base / folder).mkdir(parents=True)
            (out / folder / 'A.csv').write_text('x,y\n1,2\n')
            (root / folder / 'A.csv').write_text('x,y\r\n1,2\r\n')   # the same cells
            (out / folder / 'B.csv').write_text('x,y\n1,2\n')
            (root / folder / 'B.csv').write_text('x,y\n1,3\n')
            (out / folder / 'C.csv').write_text('x\n1\n')
            got = tables.compare(out, root, [folder / name for name in ('A.csv', 'B.csv', 'C.csv')])
        path = folder.as_posix()
        self.assertEqual(got, [f'equal  {path}/A.csv', f'DIFFERS  {path}/B.csv', f'not deposited  {path}/C.csv'])


if __name__ == '__main__':
    unittest.main()
