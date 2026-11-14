"""The command's comparison of written tables with saved ones; no study data.

    python -m pytest tests/memorization/test_tables.py
"""
import tempfile
import unittest
from pathlib import Path

from reproduce.memorization import tables


class Compare(unittest.TestCase):
    def test_each_table_reads_equal_differs_or_not_deposited_cell_for_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, root = Path(tmp) / 'out', Path(tmp) / 'root'
            for base in (out, root):
                (base / tables.FOLDER).mkdir(parents=True)
            (out / tables.FOLDER / 'A.csv').write_text('x,y\n1,2\n')
            (root / tables.FOLDER / 'A.csv').write_text('x,y\r\n1,2\r\n')   # the same cells
            (out / tables.FOLDER / 'B.csv').write_text('x,y\n1,2\n')
            (root / tables.FOLDER / 'B.csv').write_text('x,y\n1,3\n')
            (out / tables.FOLDER / 'C.csv').write_text('x\n1\n')
            got = tables.compare(out, root, ['A.csv', 'B.csv', 'C.csv'])
        folder = tables.FOLDER.as_posix()
        self.assertEqual(got, [f'equal  {folder}/A.csv', f'DIFFERS  {folder}/B.csv', f'not deposited  {folder}/C.csv'])


if __name__ == '__main__':
    unittest.main()
