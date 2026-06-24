import unittest

from scripts.run_benchmark import resolve_benchmark


class ResolveBenchmarkTest(unittest.TestCase):
    def test_bare_name_resolves_to_convention(self):
        self.assertEqual(resolve_benchmark("smoke"), "benchmarks/smoke/scenario.yaml")

    def test_full_path_is_used_as_is(self):
        self.assertEqual(resolve_benchmark("benchmarks/smoke/scenario.yaml"),
                         "benchmarks/smoke/scenario.yaml")

    def test_explicit_yaml_path_is_used_as_is(self):
        self.assertEqual(resolve_benchmark("custom/x.yaml"), "custom/x.yaml")

    def test_bare_yaml_filename_is_used_as_is(self):
        # ends with .yaml -> treat as a path, not a name
        self.assertEqual(resolve_benchmark("foo.yaml"), "foo.yaml")


from scripts.run_benchmark import validate_benchmark


class ValidateBenchmarkTest(unittest.TestCase):
    def test_valid_scenario_passes(self):
        validate_benchmark({"task": {"phases": [{"id": "x"}]}, "devs": [4, 8]})  # no raise

    def test_missing_phases_raises(self):
        with self.assertRaises(SystemExit):
            validate_benchmark({"task": {"gates": []}, "devs": [4]})

    def test_missing_devs_raises(self):
        with self.assertRaises(SystemExit):
            validate_benchmark({"task": {"phases": [{"id": "x"}]}})

    def test_empty_devs_raises(self):
        with self.assertRaises(SystemExit):
            validate_benchmark({"task": {"phases": [{"id": "x"}]}, "devs": []})
