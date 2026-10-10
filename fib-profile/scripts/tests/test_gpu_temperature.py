import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_temperature", SCRIPTS / "gpu_temperature.py")
temperature = importlib.util.module_from_spec(spec)
spec.loader.exec_module(temperature)


class TemperatureTests(unittest.TestCase):
    def cuda(self, uuids):
        return SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: len(uuids),
            get_device_properties=lambda index: SimpleNamespace(uuid=uuids[index]),
        ))

    def filter(self, devices, rows, spec="all"):
        with patch.dict(sys.modules, {"torch": self.cuda(devices)}), \
                patch.object(temperature, "query_temperatures", return_value=rows), \
                contextlib.redirect_stderr(io.StringIO()):
            return temperature.filter_cuda(spec, 45, ["nvidia-smi"], "[test]")

    def test_cuda_zero_uses_host_three_uuid(self):
        self.assertEqual(self.filter(["GPU-three"], [("0", "zero", "80"), ("3", "three", "30")]), "cuda:0")

    def test_reordered_cuda_visibility_uses_uuids(self):
        self.assertEqual(self.filter(["GPU-three", "GPU-zero"], [("0", "zero", "30"), ("3", "three", "80")]), "cuda:1")

    def test_explicit_cuda_subset(self):
        self.assertEqual(self.filter(["GPU-zero", "GPU-three"], [("0", "zero", "80"), ("3", "three", "30")], "cuda:1"), "cuda:1")

    def test_threshold_is_exclusive(self):
        with self.assertRaises(temperature.TemperatureError):
            self.filter(["GPU-three"], [("3", "three", "45")])

    def test_unknown_uuid_stops_filter(self):
        with self.assertRaisesRegex(temperature.TemperatureError, "temperature unavailable"):
            self.filter(["GPU-three"], [("0", "zero", "30")])

    def test_missing_temperature_stops_filter(self):
        with self.assertRaisesRegex(temperature.TemperatureError, "temperature unavailable"):
            self.filter(["GPU-three"], [("3", "three", "N/A")])

    def test_missing_cuda_uuid_stops_filter(self):
        with self.assertRaisesRegex(temperature.TemperatureError, "no GPU UUID"):
            self.filter([None], [("0", "zero", "30")])

    def test_uuid_byte_representation(self):
        value = uuid.UUID("12345678-1234-1234-1234-123456789abc")
        self.assertEqual(temperature.uuid_key(value.bytes), temperature.uuid_key("GPU-" + str(value)))

    def test_invalid_cuda_index(self):
        with self.assertRaisesRegex(temperature.TemperatureError, "invalid CUDA device"):
            self.filter(["GPU-three"], [("3", "three", "30")], "cuda:3")

    def test_host_selection_formats(self):
        self.assertEqual(temperature.host_selection('"device=3,4"', "docker"), ["3", "4"])
        self.assertEqual(temperature.host_selection("device=GPU-three", "docker"), ["GPU-three"])
        self.assertEqual(temperature.host_selection("3,4", "podman"), ["3", "4"])
        self.assertIsNone(temperature.host_selection("all", "docker"))

    def test_docker_count_defers_to_container(self):
        with patch.object(temperature, "query_temperatures") as query, \
                contextlib.redirect_stderr(io.StringIO()):
            temperature.check_host("1", "docker", 45, ["nvidia-smi"], "[test]")
        query.assert_not_called()

    def test_host_uuid_selection(self):
        with patch.object(temperature, "query_temperatures", return_value=[("3", "three-long", "30")]):
            temperature.check_host("device=GPU-three", "docker", 45, ["nvidia-smi"], "[test]")

    def test_unknown_host_device(self):
        with patch.object(temperature, "query_temperatures", return_value=[("0", "zero", "30")]):
            with self.assertRaisesRegex(temperature.TemperatureError, "cannot identify"):
                temperature.check_host("device=3", "docker", 45, ["nvidia-smi"], "[test]")

    def test_query_failure_stops_check(self):
        with patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(2, "nvidia-smi")):
            with self.assertRaises(temperature.TemperatureError):
                temperature.query_temperatures(["nvidia-smi"])

    def test_empty_query_stops_check(self):
        with patch.object(subprocess, "run", return_value=SimpleNamespace(stdout="")):
            with self.assertRaises(temperature.TemperatureError):
                temperature.query_temperatures(["nvidia-smi"])


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for item in ["flashinfer-bench-private/pyproject.toml", "dataset/definitions/.keep",
                     "dataset/workloads/.keep", "scripts/restart_profiling.sh"]:
            path = self.root / item
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.command("nvidia-smi", """#!/bin/sh
printf '0, GPU-zero, 80\n3, GPU-three, %s\n' "$TEST_HOST_TEMP"
""")
        self.command("docker", """#!/bin/sh
if [ "$1" = --version ]; then printf 'Docker version test\n'; exit 0; fi
printf '%s\n' "$@" >> "$TEST_DOCKER_LOG"
""")
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        HOST_FIB_ROOT=str(self.root), GPUS="device=3", FIB_DEVICES="cuda:0",
                        FIB_GPU_MAX_TEMP_C="45", TEST_HOST_TEMP="30",
                        TEST_DOCKER_LOG=str(self.root / "docker.log"))
        self.env.pop("FIB_SKIP_GPU_THERMAL_PREFLIGHT", None)

    def command(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)

    def launch(self):
        return subprocess.run(["bash", str(SCRIPTS / "run_container.sh")], env=self.env,
                              text=True, capture_output=True)

    def test_launcher_checks_host_three_and_keeps_cuda_zero(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.root / "docker.log").read_text().splitlines()
        self.assertIn("FIB_DEVICES=cuda:0", args)
        self.assertEqual(args[args.index("--gpus") + 1], "device=3")

    def test_hot_selected_host_does_not_remove_container(self):
        self.env["TEST_HOST_TEMP"] = "45"
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("host GPU 3", result.stderr)
        self.assertFalse((self.root / "docker.log").exists())

    def test_shell_cuda_filter_matches_uuid(self):
        (self.bin / "torch.py").write_text("""from types import SimpleNamespace
cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                       get_device_properties=lambda index: SimpleNamespace(uuid='GPU-three'))
""")
        self.env["PYTHONPATH"] = str(self.bin)
        result = subprocess.run([
            "bash", "-c", 'source "$1"; FIB_GPU_MAX_TEMP_C=35; '
            'profile_service_filter_cool_gpu_devices "[test]" cuda:0 nvidia-smi',
            "test", str(SCRIPTS / "gpu_preflight.sh"),
        ], env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "cuda:0")

    def test_shell_cuda_filter_stops_on_query_failure(self):
        (self.bin / "torch.py").write_text("""from types import SimpleNamespace
cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                       get_device_properties=lambda index: SimpleNamespace(uuid='GPU-three'))
""")
        self.command("nvidia-smi", "#!/bin/sh\nexit 2\n")
        self.env["PYTHONPATH"] = str(self.bin)
        result = subprocess.run([
            "bash", "-c", 'source "$1"; profile_service_filter_cool_gpu_devices "[test]" cuda:0 nvidia-smi',
            "test", str(SCRIPTS / "gpu_preflight.sh"),
        ], env=self.env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
