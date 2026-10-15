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
        self.write(self.bin / "uv", '''#!/bin/sh
if [ "${3:-}" = -v ]; then
    printf 'flash-kda install %s\\n' "$*" >> "$TEST_LOG"
    exit "${TEST_FLASH_KDA_INSTALL_EXIT:-0}"
fi
printf 'install\\n' >> "$TEST_LOG"
''')
        self.write(self.bin / "git", '''#!/bin/sh
printf 'git %s\\n' "$*" >> "$TEST_LOG"
if [ "$1" = clone ]; then
    [ "${TEST_FLASH_KDA_CLONE_EXIT:-0}" = 0 ] || exit 1
    mkdir -p "$3/.git"
elif [ "$3 $4 $5 $6" = 'submodule update --init --recursive' ]; then
    [ -d "$2/.git" ] || exit 1
    exit "${TEST_FLASH_KDA_SUBMODULE_EXIT:-0}"
else
    exit 1
fi
''')
        self.write(self.bin / "python", '''#!/bin/sh
if [ "$#" -eq 0 ] || [ "$1" = - ]; then
    cat >/dev/null
    printf 'validate\\n' >> "$TEST_LOG"
    exit "${TEST_FLASH_KDA_IMPORT_EXIT:-0}"
fi
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
                        FLASH_KDA_DIR=str(self.root / "flash-kda"),
                        FLASH_KDA_REPO_URL="https://github.com/MoonshotAI/FlashKDA.git",
                        TEST_LOG=str(self.log), UV_CACHE_DIR=str(self.root / "cache"))
        self.env.pop("FIB_DATASET_MODE", None)
        self.install_steps = [
            "install",
            "git clone https://github.com/MoonshotAI/FlashKDA.git " + str(self.root / "flash-kda"),
            "git -C " + str(self.root / "flash-kda") + " submodule update --init --recursive",
            "flash-kda install pip install -v --no-build-isolation .",
            "validate",
        ]

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
        self.assertEqual(self.log.read_text().splitlines(), self.install_steps + ["serve"])

    def test_bundled_generates_after_install_on_selected_gpu_before_serving(self):
        result = self.bundled()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = self.log.read_text().splitlines()
        self.assertEqual(lines[:6], self.install_steps + ["filter cuda:1,cuda:2"])
        self.assertIn("--dataset-root " + str(self.root / "dataset"), lines[6])
        self.assertTrue(lines[6].endswith("--device cuda:1 LD_LIBRARY_PATH="))
        self.assertEqual(lines[7:], ["serve"])

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
        self.assertEqual(self.log.read_text().splitlines(), self.install_steps + ["filter cuda:1,cuda:2"])

    def test_existing_flash_kda_checkout_is_reused_without_checkout_or_pull(self):
        (self.root / "flash-kda/.git").mkdir(parents=True)
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.log.read_text().splitlines(),
                         [self.install_steps[0], *self.install_steps[2:], "serve"])

    def test_flash_kda_failures_stop_before_serving(self):
        for stage in ("CLONE", "SUBMODULE", "INSTALL", "IMPORT"):
            with self.subTest(stage=stage):
                shutil.rmtree(self.root / "flash-kda", ignore_errors=True)
                self.log.unlink(missing_ok=True)
                result = self.run_entrypoint(**{f"TEST_FLASH_KDA_{stage}_EXIT": "1"})
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("serve", self.log.read_text().splitlines())

    def test_invalid_flash_kda_directory_is_not_overwritten(self):
        path = self.root / "flash-kda"
        path.mkdir()
        (path / "keep").write_text("existing content")
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a git checkout", result.stderr)
        self.assertEqual((path / "keep").read_text(), "existing content")
        self.assertEqual(self.log.read_text().splitlines(), ["install"])


if __name__ == "__main__":
    unittest.main()
