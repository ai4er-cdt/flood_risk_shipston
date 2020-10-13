import unittest


class TestCase(unittest.TestCase):
    def test_upper(self):
        """This is a placeholder."""
        self.assertEqual("foo".upper(), "FOO")


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
