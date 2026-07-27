#!/usr/bin/env bash

_profile_service_trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

_profile_service_gpu_indices_from_spec() {
    local spec="$1"
    local token
    local clean
    spec="${spec#device=}"
    if [ -z "$spec" ] || [ "$spec" = "all" ]; then
        return 0
    fi
    IFS=',' read -r -a _profile_service_tokens <<< "$spec"
    for token in "${_profile_service_tokens[@]}"; do
        clean="$(_profile_service_trim "$token")"
        if [[ "$clean" =~ ^cuda:([0-9]+)$ ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
        elif [[ "$clean" =~ ^([0-9]+)$ ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
        fi
    done
}

_profile_service_query_gpu_indices() {
    local -a nvidia_smi_cmd=("$@")
    "${nvidia_smi_cmd[@]}" --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
        | while IFS= read -r line; do
            _profile_service_trim "$line"
            printf '\n'
        done
}

_profile_service_query_gpu_temperature() {
    local -a nvidia_smi_cmd=("$@")
    "${nvidia_smi_cmd[@]}" \
        --query-gpu=index,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null
}

# Filters a FIB_DEVICES-style list by keeping only GPUs whose temperature is
# below FIB_GPU_MAX_TEMP_C.
profile_service_filter_cool_gpu_devices() {
    local log_prefix="$1"
    local device_spec="$2"
    shift 2
    local -a nvidia_smi_cmd=("$@")
    local -a source_tokens=()
    local -a kept_tokens=()
    local -a excluded_tokens=()
    local token
    local clean
    local index
    local temp_output
    local status
    local joined
    local threshold="${FIB_GPU_MAX_TEMP_C:-40}"
    declare -A temp_by_index=()

    if [ "${FIB_SKIP_GPU_THERMAL_PREFLIGHT:-0}" = "1" ]; then
        echo "$log_prefix WARNING: skipping GPU thermal filter because FIB_SKIP_GPU_THERMAL_PREFLIGHT=1" >&2
        printf '%s\n' "$device_spec"
        return 0
    fi
    if ! [[ "$threshold" =~ ^[0-9]+$ ]]; then
        echo "$log_prefix ERROR: FIB_GPU_MAX_TEMP_C must be an integer, got: $threshold" >&2
        return 1
    fi

    if temp_output="$(_profile_service_query_gpu_temperature "${nvidia_smi_cmd[@]}")"; then
        while IFS= read -r line; do
            [ -n "$line" ] || continue
            IFS=',' read -r index temp <<< "$line"
            index="$(_profile_service_trim "$index")"
            temp="$(_profile_service_trim "${temp:-}")"
            [ -n "$index" ] || continue
            temp_by_index["$index"]="$temp"
        done <<< "$temp_output"
    else
        status=$?
        echo "$log_prefix WARNING: unable to query GPU temperatures with nvidia-smi; keeping requested devices unchanged" >&2
        printf '%s\n' "$device_spec"
        return "$status"
    fi

    if [ -z "$device_spec" ] || [ "$device_spec" = "all" ]; then
        mapfile -t source_tokens < <(_profile_service_query_gpu_indices "${nvidia_smi_cmd[@]}" | awk 'NF {print "cuda:" $1}')
    else
        IFS=',' read -r -a source_tokens <<< "$device_spec"
    fi

    for token in "${source_tokens[@]}"; do
        clean="$(_profile_service_trim "$token")"
        if [[ "$clean" =~ ^cuda:([0-9]+)$ ]]; then
            index="${BASH_REMATCH[1]}"
            temp="${temp_by_index[$index]:-}"
            if ! [[ "$temp" =~ ^[0-9]+$ ]]; then
                excluded_tokens+=("$clean(temp-unavailable)")
                continue
            fi
            if [ "$temp" -ge "$threshold" ]; then
                excluded_tokens+=("$clean(${temp}C)")
                continue
            fi
        elif [[ "$clean" =~ ^([0-9]+)$ ]]; then
            index="${BASH_REMATCH[1]}"
            temp="${temp_by_index[$index]:-}"
            if ! [[ "$temp" =~ ^[0-9]+$ ]]; then
                excluded_tokens+=("cuda:$index(temp-unavailable)")
                continue
            fi
            if [ "$temp" -ge "$threshold" ]; then
                excluded_tokens+=("cuda:$index(${temp}C)")
                continue
            fi
        fi
        [ -n "$clean" ] && kept_tokens+=("$clean")
    done

    if [ "${#excluded_tokens[@]}" -gt 0 ]; then
        echo "$log_prefix excluding GPU(s) that are not below ${threshold}C: ${excluded_tokens[*]}" >&2
    fi
    if [ "${#kept_tokens[@]}" -eq 0 ]; then
        echo "$log_prefix ERROR: no requested CUDA devices are below ${threshold}C" >&2
        return 1
    fi

    joined="${kept_tokens[0]}"
    for token in "${kept_tokens[@]:1}"; do
        joined+=",$token"
    done
    printf '%s\n' "$joined"
    return 0
}

