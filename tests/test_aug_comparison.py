"""The augmentation ablation must not silently change optimization or initialize from Q1."""

import copy
import unittest

from table_vlm.aug_comparison import verify_config_pair


class AugComparisonTest(unittest.TestCase):
    def pair(self):
        clean = {'dataset_dir':'data/prepared/formal-clean/lf', 'media_dir':'clean',
                 'output_dir':'q1', 'num_train_epochs':2.0, 'learning_rate':1e-4}
        augmented = copy.deepcopy(clean)
        augmented.update(dataset_dir='data/prepared/formal-aug/lf', media_dir='aug', output_dir='q2')
        return clean, augmented

    def test_equal_parameters_with_only_data_paths_changed_passes(self):
        verify_config_pair(*self.pair())

    def test_changed_learning_rate_is_rejected(self):
        clean, aug = self.pair()
        aug['learning_rate'] = 2e-4
        with self.assertRaisesRegex(ValueError, 'hyperparameters differ'):
            verify_config_pair(clean, aug)

    def test_loading_clean_adapter_is_rejected(self):
        clean, aug = self.pair()
        aug['adapter_name_or_path'] = 'runs/q1-adapter'
        with self.assertRaises(ValueError):
            verify_config_pair(clean, aug)
