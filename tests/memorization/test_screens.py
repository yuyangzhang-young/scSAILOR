"""The command's screens: each writes its tables under its own folder; no study data.

    python -m pytest tests/memorization/test_screens.py
"""
import unittest

from reproduce.memorization import tables


class Screens(unittest.TestCase):
    def test_each_screen_has_its_folder_under_memorization(self):
        self.assertEqual({s: tables.folder(s).as_posix() for s in tables.SCREENS},
                         {'tahoe-100m': 'memorization/tahoe-100m', 'parse-pbmc': 'memorization/parse-pbmc',
                          'genetic-screens': 'memorization/genetic-screens'})


if __name__ == '__main__':
    unittest.main()
