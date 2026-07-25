import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.sparse_pipeline_smoke import verify_fixture_manifest


class SparsePipelineSmokeTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        payload = root / "image.bin"
        payload.write_bytes(b"fixture-data")
        manifest = {
            "files": [
                {
                    "path": payload.name,
                    "bytes": payload.stat().st_size,
                    "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                }
            ]
        }
        manifest_path = root / "fixture_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return payload

    def test_verifies_declared_fixture_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            manifest = verify_fixture_manifest(root)

        self.assertEqual(len(manifest["files"]), 1)

    def test_rejects_tampered_fixture_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.make_fixture(root)
            payload.write_bytes(b"tampered-data")

            with self.assertRaisesRegex(ValueError, "fixture size mismatch"):
                verify_fixture_manifest(root)

    def test_rejects_path_outside_fixture_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside.bin"
            outside.write_bytes(b"outside")
            manifest = {
                "files": [
                    {
                        "path": "../outside.bin",
                        "bytes": outside.stat().st_size,
                        "sha256": hashlib.sha256(
                            outside.read_bytes()
                        ).hexdigest(),
                    }
                ]
            }
            (root / "fixture_manifest.json").write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "escapes fixture root"):
                verify_fixture_manifest(root)


if __name__ == "__main__":
    unittest.main()
