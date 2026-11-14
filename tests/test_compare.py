"""The reproduction commands' ``--compare``: a table equal cell for cell reads 'equal', any changed cell 'DIFFERS', a
table missing from the deposit 'not deposited'.

    python -m pytest tests/test_compare.py
"""
import tempfile
import unittest
from pathlib import Path

from reproduce.compare import compare


class Compare(unittest.TestCase):
    def test_equal_differs_and_not_deposited(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, root = Path(tmp) / 'out', Path(tmp) / 'root'
            for base in (out, root):
                (base / 'layer').mkdir(parents=True)
                (base / 'layer/SAME.csv').write_text('a,b\n1,2.5\n')
            (out / 'layer/CHANGED.csv').write_text('a,b\n1,2.5\n')
            (root / 'layer/CHANGED.csv').write_text('a,b\n1,2.50\n')   # the same number, written differently
            (out / 'layer/NEW.csv').write_text('a\n1\n')
            lines = compare(out, root, ['layer/SAME.csv', 'layer/CHANGED.csv', 'layer/NEW.csv'])
        self.assertEqual(lines, ['equal  layer/SAME.csv', 'DIFFERS  layer/CHANGED.csv', 'not deposited  layer/NEW.csv'])


if __name__ == '__main__':
    unittest.main()
