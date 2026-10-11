"""Exercise startup ordering without installing packages or running CUDA."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        shutil.copyfile(SCRIPTS / "entrypoint.sh", self.scripts / "entrypoint.sh")
        for directory in ("flashinfer-bench-private", "dataset/definitions", "dataset/workloads", "acc/bin"):
            (self.root / directory).mkdir(parents=True)
        (self.root / "flashinfer-bench-private/pyproject.toml").touch()
        self.bin = self.root / "acc/bin"
        self.write(self.bin / "activate", 'export PATH="$VENV_DIR/bin:$PATH"\n')
        self.write(self.bin / "uv", '#!/bin/sh\nprintf "install\\n" >> "$TEST_LOG"\n')
        self.write(self.bin / "python", '''#!/bin/sh
if [ "$#" -eq 0 ] || [ "$1" = - ]; then cat >/dev/null; exit 0; fi
printf 'generate %s LD_LIBRARY_PATH=%s\n' "$*" "$LD_LIBRARY_PATH" >> "$TEST_LOG"
exit "${TEST_GENERATE_EXIT:-0}"
''')
        self.write(self.scripts / "restart_profiling.sh", '#!/bin/sh\nprintf "serve\\n" >> "$TEST_LOG"\n')
        self.write(self.scripts / "gpu_preflight.sh", '''profile_service_filter_cool_gpu_devices() {
    printf 'filter %s\n' "$2" >> "$TEST_LOG"
    [ "${TEST_FILTER_EXIT:-0}" = 0 ] || return 1
    printf 'cuda:1,cuda:2\n'
}
''')
        self.log = self.root / "startup.log"
        self.env = dict(os.environ, WORKDIR=str(self.root), VENV_DIR=str(self.root / "acc"),
                        FIB_DIR="flashinfer-bench-private", FIB_DATASET_DIR=str(self.root / "dataset"),
                        FIB_DEVICES="cuda:1,cuda:2", LD_LIBRARY_PATH="/unwanted/cuda",
                        TEST_LOG=str(self.log), UV_CACHE_DIR=str(self.root / "cache"))
        self.env.pop("FIB_DATASET_MODE", None)

    def write(self, path, text):
        path.write_text(text)
        path.chmod(0o755)

    def run_entrypoint(self, **overrides):
        return subprocess.run(["bash", str(self.scripts / "entrypoint.sh")],
                              env=dict(self.env, **overrides), text=True, capture_output=True)

    def bundled(self, **overrides):
        (self.root / "dataset/generate_blobs.py").touch()
        return self.run_entrypoint(FIB_DATASET_MODE="bundled", **overrides)

    def test_default_existing_serves_without_generator_or_filter(self):
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.log.read_text().splitlines(), ["install", "serve"])

    def test_bundled_generates_after_install_on_selected_gpu_before_serving(self):
        result = self.bundled()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = self.log.read_text().splitlines()
        self.assertEqual(lines[:2], ["install", "filter cuda:1,cuda:2"])
        self.assertIn("--dataset-root " + str(self.root / "dataset"), lines[2])
        self.assertTrue(lines[2].endswith("--device cuda:1 LD_LIBRARY_PATH="))
        self.assertEqual(lines[3:], ["serve"])

    def test_missing_generator_fails_before_install(self):
        result = self.run_entrypoint(FIB_DATASET_MODE="bundled")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("generate_blobs.py", result.stderr)
        self.assertFalse(self.log.exists())

    def test_invalid_mode_fails_before_install(self):
        result = self.run_entrypoint(FIB_DATASET_MODE="typo")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_failed_generation_does_not_start_service(self):
        result = self.bundled(TEST_GENERATE_EXIT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("serve", self.log.read_text().splitlines())

    def test_failed_thermal_filter_does_not_generate_or_serve(self):
        result = self.bundled(TEST_FILTER_EXIT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.log.read_text().splitlines(), ["install", "filter cuda:1,cuda:2"])


if __name__ == "__main__":
    unittest.main()
