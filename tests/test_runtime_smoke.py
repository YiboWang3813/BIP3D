import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import runtime_smoke


def make_report(*, cuda=True, devices=1, matmul_sum=120.0, extensions=True):
    return {
        "cuda": {
            "available": cuda,
            "device_count": devices,
            "matmul_sum": matmul_sum,
        },
        "extensions": {
            name: {"ok": extensions} for name in runtime_smoke.CUDA_EXTENSIONS
        },
    }


class RuntimeValidationTest(unittest.TestCase):
    def test_accepts_complete_cuda_runtime(self):
        issues = runtime_smoke.validate_report(
            make_report(),
            require_extensions=True,
        )

        self.assertEqual(issues, [])

    def test_reports_cuda_failures(self):
        issues = runtime_smoke.validate_report(
            make_report(cuda=False, devices=0, matmul_sum=None),
            require_extensions=False,
        )

        self.assertEqual(len(issues), 3)
        self.assertIn("CUDA is not available", issues[0])

    def test_extensions_are_optional_by_default(self):
        report = make_report(extensions=False)

        self.assertEqual(
            runtime_smoke.validate_report(
                report,
                require_extensions=False,
            ),
            [],
        )
        self.assertIn(
            "extensions are unavailable",
            runtime_smoke.validate_report(
                report,
                require_extensions=True,
            )[0],
        )


if __name__ == "__main__":
    unittest.main()
