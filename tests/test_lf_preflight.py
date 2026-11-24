"""Protect the gate against silent target truncation and prompt supervision."""

import unittest
import unicodedata

from table_vlm.lf_preflight import check_encoded
from table_vlm.prompts import PROMPT


TARGET = '<table><tr><td>42</td></tr></table>'


class FixtureTokenizer:
    def __init__(self, target=TARGET):
        self.target = unicodedata.normalize('NFC', target)

    def encode(self, text, **kwargs):
        return [12, 13] if unicodedata.normalize('NFC', text) == self.target + '<|im_end|>\n' else [99]

    def convert_tokens_to_ids(self, token):
        return {'<|image_pad|>': 10, '<|im_end|>': 13}[token]

    def decode(self, ids, **kwargs):
        return ''.join({10: '<|image_pad|>', 11: PROMPT, 12: self.target, 14: '',
                        13: '' if kwargs.get('skip_special_tokens') else '<|im_end|>'}[i] for i in ids)


class LFPreflightTest(unittest.TestCase):
    def sample(self):
        return {'input_ids': [10, 11, 12, 13], 'labels': [-100, -100, 12, 13],
                'attention_mask': [1, 1, 1, 1]}

    def test_complete_answer_with_masked_prompt_passes(self):
        result = check_encoded(self.sample(), TARGET, FixtureTokenizer(), 4)
        self.assertEqual(result['tokens'], 4)
        self.assertEqual(result['supervised_tokens'], 2)

    def test_truncated_answer_and_overlength_are_rejected(self):
        row = self.sample()
        row['input_ids'].pop()
        row['labels'].pop()
        row['attention_mask'].pop()
        with self.assertRaisesRegex(ValueError, 'end token missing'):
            check_encoded(row, TARGET, FixtureTokenizer(), 4)
        with self.assertRaisesRegex(ValueError, 'exceeds training cutoff'):
            check_encoded(self.sample(), TARGET, FixtureTokenizer(), 3)

    def test_prompt_or_image_supervision_and_changed_target_are_rejected(self):
        for index in [0, 1]:
            row = self.sample()
            row['labels'][index] = row['input_ids'][index]
            with self.assertRaises(ValueError):
                check_encoded(row, TARGET, FixtureTokenizer(), 4)
        with self.assertRaisesRegex(ValueError, 'complete exported HTML'):
            check_encoded(self.sample(), TARGET.replace('42', '43'), FixtureTokenizer(), 4)

    def test_nfc_canonical_equivalence_passes_without_changing_source(self):
        target = TARGET.replace('42', 'Loe\u0300ve')
        result = check_encoded(self.sample(), target, FixtureTokenizer(target), 4)
        self.assertTrue(result['nfc_target_changed'])
        self.assertTrue(result['exact_answer_tokens'])
        self.assertIn('e\u0300', target)

    def test_nfkc_equivalence_and_hidden_extra_tokens_still_fail(self):
        with self.assertRaisesRegex(ValueError, 'complete exported HTML'):
            check_encoded(self.sample(), TARGET.replace('42', '\uff14\uff12'), FixtureTokenizer(), 4)
        row = self.sample()
        for key, value in [('input_ids', 14), ('labels', 14), ('attention_mask', 1)]:
            row[key].append(value)
        with self.assertRaisesRegex(ValueError, 'token sequence differs'):
            check_encoded(row, TARGET, FixtureTokenizer(), 5)


if __name__ == '__main__':
    unittest.main()
