"""No model/dependency download: test scoring and evidence acceptance boundaries."""
import copy
import unittest
from validate import wer, smoke_complete


class EvidenceTests(unittest.TestCase):
    def test_insertions_use_reference_denominator(self):
        self.assertEqual(wer('a b', 'a b c'), .5)

    def test_deletions_use_reference_denominator(self):
        self.assertAlmostEqual(wer('a b c', 'a b'), 1 / 3)

    def test_empty_reference_rejected(self):
        with self.assertRaises(ValueError):
            wer('', 'a')

    def test_empty_hypothesis_is_full_error(self):
        self.assertEqual(wer('a b', ''), 1)

    def test_normalization_preserves_diacritics(self):
        self.assertEqual(wer('Čerstvý, chlieb!', 'čerstvý chlieb'), 0)
        self.assertEqual(wer('mlieko', 'mléko'), 1)

    def test_acceptance_requires_every_language_task(self):
        result = {'errors': [], 'rows': [dict(source_lang=s, target_lang=t, status='complete')
                  for s, t in [('sk', 'sk'), ('sk', 'en'), ('cs', 'cs'), ('cs', 'en'), ('en', 'en')]]}
        self.assertTrue(smoke_complete(result))
        incomplete = copy.deepcopy(result)
        incomplete['rows'].pop()
        self.assertFalse(smoke_complete(incomplete))
        failed = copy.deepcopy(result)
        failed['rows'][2]['status'] = 'missing_reference'
        self.assertFalse(smoke_complete(failed))
        wrong_language = copy.deepcopy(result)
        wrong_language['rows'][2]['source_lang'] = 'sk'
        self.assertFalse(smoke_complete(wrong_language))
        self.assertFalse(smoke_complete({'rows': [], 'errors': []}))


if __name__ == '__main__':
    unittest.main()
