import importlib.metadata
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_environment


class ParseRequirementsTest(unittest.TestCase):
    def test_parses_exact_pins_and_comments(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text(
                "numpy==1.26.4\n\n# comment\ntorch==2.1.0+cu118\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_environment.parse_pinned_requirements(path),
                {"numpy": "1.26.4", "torch": "2.1.0+cu118"},
            )

    def test_rejects_unpinned_entries(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text("numpy>=1.26\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported requirement"):
                check_environment.parse_pinned_requirements(path)


class EnvironmentChecksTest(unittest.TestCase):
    def test_python_supported_range(self):
        self.assertTrue(check_environment.check_python((3, 8, 0))["ok"])
        self.assertTrue(check_environment.check_python((3, 11, 9))["ok"])
        self.assertFalse(check_environment.check_python((3, 7, 9))["ok"])
        self.assertFalse(check_environment.check_python((3, 12, 0))["ok"])

    def test_package_results_include_missing_and_mismatch(self):
        versions = {"installed": "1.0", "mismatch": "2.0"}

        def get_version(package):
            if package not in versions:
                raise importlib.metadata.PackageNotFoundError(package)
            return versions[package]

        results = check_environment.check_packages(
            {"installed": "1.0", "mismatch": "1.0", "missing": "1.0"},
            version_getter=get_version,
        )

        by_package = {item["package"]: item for item in results}
        self.assertTrue(by_package["installed"]["ok"])
        self.assertFalse(by_package["mismatch"]["ok"])
        self.assertEqual(by_package["mismatch"]["actual"], "2.0")
        self.assertFalse(by_package["missing"]["ok"])
        self.assertIsNone(by_package["missing"]["actual"])

    def test_command_check_reports_presence(self):
        def find_command(command):
            return "/usr/bin/nvidia-smi" if command == "nvidia-smi" else None

        results = check_environment.check_commands(
            ("nvidia-smi", "nvcc"), command_finder=find_command
        )

        self.assertTrue(results[0]["ok"])
        self.assertFalse(results[1]["ok"])


if __name__ == "__main__":
    unittest.main()
