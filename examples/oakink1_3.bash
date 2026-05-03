#!/usr/bin/env bash
# OakInk1 → Shadow Hand full pipeline
# 0. process_datasets/oakink1.py  — MANO trajectory 생성
# 1. decompose → contact → gen_xml → ik → retarget (per trajectory)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
HAND_TYPE="right"
ROBOT_TYPE="xhand"
DATASET_NAME="oakink1"
CONDA_ENV="spider"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${REPO_ROOT}/example_datasets"
MANO_DIR="${DATASET_DIR}/processed/${DATASET_NAME}/mano/${HAND_TYPE}"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${RESET}    $*"; }
fail() { echo -e "  ${RED}[FAIL]${RESET}  $*"; }
skip() { echo -e "  ${YELLOW}[SKIP]${RESET}  $*"; }
info() { echo -e "${CYAN}$*${RESET}"; }

# ── Activate conda environment ────────────────────────────────────────────────
info "Activating conda environment: ${CONDA_ENV}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo ""

# ── Step 0: Generate MANO trajectories ───────────────────────────────────────
# info "[0/N] Generating MANO trajectories via oakink1.py ..."
# cd "${REPO_ROOT}"
# if python spider/process_datasets/oakink1.py --dataset-dir "${DATASET_DIR}" > /tmp/spider_oakink1.log 2>&1; then
#     ok "oakink1.py"
# else
#     fail "oakink1.py"
#     tail -10 /tmp/spider_oakink1.log | sed 's/^/    /'
#     echo -e "${RED}Aborting: trajectory generation failed.${RESET}" >&2
#     exit 1
# fi
# echo ""

# ── Discover trajectories ─────────────────────────────────────────────────────
mapfile -t TRAJ_FILES < <(find "${MANO_DIR}" -name "trajectory_keypoints.npz" | sort)

if [[ ${#TRAJ_FILES[@]} -eq 0 ]]; then
    echo "No trajectories found in ${MANO_DIR}" >&2
    exit 1
fi

TASKS=()
DATA_IDS=()
for f in "${TRAJ_FILES[@]}"; do
    DATA_IDS+=("$(basename "$(dirname "$f")")")
    TASKS+=("$(basename "$(dirname "$(dirname "$f")")")")
done

TOTAL=${#TASKS[@]}
info "Found ${TOTAL} trajectories in ${MANO_DIR}"
echo ""

# ── Summary tracking ──────────────────────────────────────────────────────────
declare -A RESULTS  # key: "task|data_id|step", value: ok|fail|skip

# ── Pipeline steps ────────────────────────────────────────────────────────────
run_step() {
    local step="$1"; shift
    local task="$1";    local data_id="$2"
    printf "  %-12s " "[${step}]"
    if "${@:3}" > /tmp/spider_step.log 2>&1; then
        echo -e "${GREEN}OK${RESET}"
        RESULTS["${task}|${data_id}|${step}"]="ok"
        return 0
    else
        echo -e "${RED}FAILED${RESET}"
        tail -5 /tmp/spider_step.log | sed 's/^/    /'
        RESULTS["${task}|${data_id}|${step}"]="fail"
        return 1
    fi
}

run_step_verbose() {
    local step="$1"; shift
    local task="$1";    local data_id="$2"
    echo -e "  ${BOLD}[${step}]${RESET}"
    if "${@:3}"; then
        echo -e "  ${GREEN}[${step}] OK${RESET}"
        RESULTS["${task}|${data_id}|${step}"]="ok"
        return 0
    else
        echo -e "  ${RED}[${step}] FAILED${RESET}"
        RESULTS["${task}|${data_id}|${step}"]="fail"
        return 1
    fi
}

run_pipeline() {
    local task="$1"
    local data_id="$2"
    local base_args=(
        --task "${task}"
        --dataset-name "${DATASET_NAME}"
        --data-id "${data_id}"
        --embodiment-type "${HAND_TYPE}"
    )
    local robot_args=("${base_args[@]}" --robot-type "${ROBOT_TYPE}")

    cd "${REPO_ROOT}"

    run_step decompose  "${task}" "${data_id}" \
        python spider/preprocess/decompose.py "${base_args[@]}" || { _skip_rest "${task}" "${data_id}"; return; }

    run_step contact    "${task}" "${data_id}" \
        python spider/preprocess/detect_contact.py "${base_args[@]}" --no-show-viewer || { _skip_rest "${task}" "${data_id}"; return; }

    run_step gen_xml    "${task}" "${data_id}" \
        python spider/preprocess/generate_xml.py "${robot_args[@]}" --no-show-viewer --act-scene --hand_floor_collision || { _skip_rest "${task}" "${data_id}"; return; }

    run_step ik         "${task}" "${data_id}" \
        python spider/preprocess/ik_fast.py "${robot_args[@]}" --no-show-viewer --act-scene || { _skip_rest "${task}" "${data_id}"; return; }

    run_step_verbose retarget "${task}" "${data_id}" \
        python examples/run_mjwp.py \
            "+override=${DATASET_NAME}_act" \
            "task=${task}" \
            "data_id=${data_id}" \
            "robot_type=${ROBOT_TYPE}" \
            "embodiment_type=${HAND_TYPE}" \
            "viewer=none"
}

_skip_rest() {
    local task="$1"; local data_id="$2"
    for step in contact gen_xml ik retarget; do
        [[ -z "${RESULTS[${task}|${data_id}|${step}]+x}" ]] && \
            RESULTS["${task}|${data_id}|${step}"]="skip"
    done
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    data_id="${DATA_IDS[$i]}"
    idx=$((i + 1))

    # If task name is not started with "S", skip (already processed)
    # if [[ ! "$task" =~ ^S ]]; then
    #     echo -e "${YELLOW}Skipping [${idx}/${TOTAL}] ${task} data_id=${data_id} (already processed)${RESET}"
    #     _skip_rest "${task}" "${data_id}"
    #     continue
    # fi

    echo -e "${BOLD}[${idx}/${TOTAL}]${RESET} ${task}  data_id=${data_id}"
    run_pipeline "${task}" "${data_id}"
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
STEPS=(decompose contact gen_xml ik retarget)
STEP_W=10
HDR_FMT="%-45s %8s"
printf "\n${BOLD}${HDR_FMT}" "Task" "data_id"
for s in "${STEPS[@]}"; do printf "  %${STEP_W}s" "$s"; done
printf "${RESET}\n"
printf '%0.s-' {1..105}; echo ""

n_fail=0
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"; data_id="${DATA_IDS[$i]}"
    printf "%-45s %8s" "${task}" "${data_id}"
    for s in "${STEPS[@]}"; do
        v="${RESULTS[${task}|${data_id}|${s}]:-skip}"
        case "$v" in
            ok)   printf "  ${GREEN}%${STEP_W}s${RESET}" "OK" ;;
            fail) printf "  ${RED}%${STEP_W}s${RESET}" "FAIL"; ((n_fail++)) ;;
            skip) printf "  ${YELLOW}%${STEP_W}s${RESET}" "-" ;;
        esac
    done
    echo ""
done

echo ""
echo -e "Total: ${TOTAL} trajectories, ${n_fail} step-level failures."
[[ ${n_fail} -eq 0 ]] && echo -e "${GREEN}All steps succeeded.${RESET}" || echo -e "${RED}${n_fail} failures — check logs above.${RESET}"
