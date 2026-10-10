"""Check host GPU selections and filter CUDA devices by their GPU UUIDs."""

import argparse
import os
import subprocess
import sys
import uuid


class TemperatureError(RuntimeError):
    pass


def uuid_key(value):
    if isinstance(value, bytes):
        value = uuid.UUID(bytes=value)
    text = str(value).lower()
    return text[4:] if text.startswith("gpu-") else text


def query_temperatures(command):
    try:
        result = subprocess.run(
            [*command, "--query-gpu=index,uuid,temperature.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TemperatureError("unable to query GPU temperatures with nvidia-smi") from exc
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1]:
            raise TemperatureError(f"invalid GPU temperature record: {line}")
        index, gpu_uuid, temperature = fields
        rows.append((index, uuid_key(gpu_uuid), temperature))
    if not rows:
        raise TemperatureError("nvidia-smi reported no GPU temperatures")
    return rows


def host_selection(spec, runtime):
    # Docker's multi-device argument is commonly enclosed in literal quotes.
    spec = spec.strip().strip("\"'")
    if spec == "all":
        return None
    if spec.startswith("device="):
        spec = spec[len("device="):]
    elif runtime == "docker":
        # Docker count selections do not identify the host GPUs in advance.
        if spec.isdigit() or (spec.startswith("count=") and spec[6:].isdigit()):
            return []
        raise TemperatureError(f"unsupported Docker GPU selection: {spec}")
    tokens = [token.strip() for token in spec.split(",")]
    if not all(token.isdigit() or token.startswith("GPU-") for token in tokens):
        raise TemperatureError(f"unsupported host GPU selection: {spec}")
    return tokens


def check_host(spec, runtime, threshold, command, log_prefix):
    selection = host_selection(spec, runtime)
    if selection == []:
        print(f"{log_prefix} GPU count selection: temperature checks deferred to the container", file=sys.stderr)
        return
    rows = query_temperatures(command)
    selected = rows if selection is None else []
    if selection is not None:
        for token in selection:
            matches = [row for row in rows if
                       row[0] == token or
                       (token.startswith("GPU-") and row[1].startswith(uuid_key(token)))]
            if len(matches) != 1:
                raise TemperatureError(f"cannot identify selected host GPU: {token}")
            selected.append(matches[0])
    for index, gpu_uuid, temperature in selected:
        if not temperature.isdigit():
            raise TemperatureError(f"host GPU {index} (GPU-{gpu_uuid}): temperature unavailable")
        if int(temperature) >= threshold:
            raise TemperatureError(f"host GPU {index} (GPU-{gpu_uuid}) is {temperature}C; must be below {threshold}C")


def cuda_device_uuids(spec):
    import torch

    if not torch.cuda.is_available():
        raise TemperatureError("CUDA is unavailable inside the container")
    count = torch.cuda.device_count()
    tokens = [f"cuda:{index}" for index in range(count)] if spec in ("", "all") else spec.split(",")
    devices = []
    for token in tokens:
        token = token.strip()
        index_text = token[5:] if token.startswith("cuda:") else token
        if not index_text.isdigit() or int(index_text) >= count:
            raise TemperatureError(f"invalid CUDA device: {token}; CUDA sees {count} device(s)")
        index = int(index_text)
        gpu_uuid = getattr(torch.cuda.get_device_properties(index), "uuid", None)
        if not gpu_uuid:
            raise TemperatureError(f"CUDA device cuda:{index} has no GPU UUID")
        devices.append((f"cuda:{index}", uuid_key(gpu_uuid)))
    return devices


def filter_cuda(spec, threshold, command, log_prefix):
    devices = cuda_device_uuids(spec)
    temperatures = {gpu_uuid: temperature for _, gpu_uuid, temperature in query_temperatures(command)}
    kept = []
    for device, gpu_uuid in devices:
        temperature = temperatures.get(gpu_uuid, "")
        if not temperature.isdigit():
            raise TemperatureError(f"{device} (GPU-{gpu_uuid}): temperature unavailable")
        if int(temperature) >= threshold:
            print(f"{log_prefix} excluding {device} (GPU-{gpu_uuid}, {temperature}C); must be below {threshold}C", file=sys.stderr)
        else:
            kept.append(device)
    if not kept:
        raise TemperatureError(f"no requested CUDA devices are below {threshold}C")
    return ",".join(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("host", "cuda"))
    parser.add_argument("spec")
    parser.add_argument("--runtime", choices=("docker", "podman"), default="docker")
    parser.add_argument("--log-prefix", default="[gpu_preflight]")
    parser.add_argument("--nvidia-smi", nargs=argparse.REMAINDER, default=["nvidia-smi"])
    args = parser.parse_args()
    try:
        raw_threshold = os.environ.get("FIB_GPU_MAX_TEMP_C", "40")
        if not raw_threshold.isdigit():
            raise TemperatureError(f"FIB_GPU_MAX_TEMP_C must be an integer, got: {raw_threshold}")
        threshold = int(raw_threshold)
        if args.mode == "host":
            check_host(args.spec, args.runtime, threshold, args.nvidia_smi, args.log_prefix)
        else:
            print(filter_cuda(args.spec, threshold, args.nvidia_smi, args.log_prefix))
    except Exception as exc:
        print(f"{args.log_prefix} ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
