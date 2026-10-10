#!/usr/bin/env bash

# Host GPUS selections and container CUDA ordinals use different namespaces.
# Resolve container ordinals to UUIDs before looking up their temperatures.
profile_service_filter_cool_gpu_devices() {
    local log_prefix="$1"
    local device_spec="$2"
    shift 2
    if [ "${FIB_SKIP_GPU_THERMAL_PREFLIGHT:-0}" = "1" ]; then
        echo "$log_prefix WARNING: skipping GPU thermal filter because FIB_SKIP_GPU_THERMAL_PREFLIGHT=1" >&2
        printf '%s\n' "$device_spec"
        return 0
    fi
    FIB_GPU_MAX_TEMP_C="${FIB_GPU_MAX_TEMP_C:-40}" python \
        "$(dirname "${BASH_SOURCE[0]}")/gpu_temperature.py" \
        cuda "$device_spec" --log-prefix "$log_prefix" --nvidia-smi "$@"
}

profile_service_check_host_gpu_temperatures() {
    local log_prefix="$1"
    local gpu_spec="$2"
    local runtime="$3"
    shift 3
    if [ "${FIB_SKIP_GPU_THERMAL_PREFLIGHT:-0}" = "1" ]; then
        echo "$log_prefix WARNING: skipping host GPU thermal check because FIB_SKIP_GPU_THERMAL_PREFLIGHT=1" >&2
        return 0
    fi
    FIB_GPU_MAX_TEMP_C="${FIB_GPU_MAX_TEMP_C:-40}" python3 \
        "$(dirname "${BASH_SOURCE[0]}")/gpu_temperature.py" \
        host "$gpu_spec" --runtime "$runtime" --log-prefix "$log_prefix" --nvidia-smi "$@"
}
