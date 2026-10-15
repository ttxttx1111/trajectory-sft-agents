"""Protect published evidence and check portable preparation without a model."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / 'experiments/alfworld/releases/v0.3.0/reproduce_checkpoint.py'
spec = importlib.util.spec_from_file_location('alfworld_checkpoint', FILE)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class AlfworldReleaseTests(unittest.TestCase):
    def test_output_cannot_overlap_published_evidence(self):
        for directory in [ROOT, ROOT / 'data', release.ALF, release.ALF / 'results',
                          release.ROOT, release.ROOT / 'results/new-run']:
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                release.check_output(directory)
        self.assertEqual(release.check_output(release.ALF / 'runs/new-run'), release.ALF / 'runs/new-run')

    def test_materialization_preserves_training_text_and_relative_ids(self):
        example = {'gamefile': 'json_2.1.1/train/task/game.tw-pddl',
                   'prompt': 'Observation: json_2.1.1/train/task/game.tw-pddl', 'values': [1, False]}
        result = release.materialize(example, Path('/dataset'))
        self.assertEqual(result['gamefile'], str(Path('/dataset/json_2.1.1/train/task/game.tw-pddl')))
        self.assertEqual(result['prompt'], example['prompt'])
        self.assertEqual(example['gamefile'], 'json_2.1.1/train/task/game.tw-pddl')

    def test_prepare_does_not_overwrite_an_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'new-run'
            release.prepare(path, Path('/dataset'), 'A')
            original = (path / 'dev_games.json').read_bytes()
            with self.assertRaises(FileExistsError):
                release.prepare(path, Path('/dataset'), 'A')
            self.assertEqual(original, (path / 'dev_games.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
