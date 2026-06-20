#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# download_models.sh — Download recommended GGUF models for AnvilAgent
#
# Usage:
#   ./download_models.sh
#   ./download_models.sh --model-name phi-4-mini-q4_k_m
#   ./download_models.sh --model-name gemma-3-4b-q4_k_m
#   ./download_models.sh --model-name qwen-3-4b-q4_k_m
#   ./download_models.sh --list
###############################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

###############################################################################
# Model registry
###############################################################################
declare -A MODELS
MODELS[phi-4-mini-q4_k_m]="https://huggingface.co/microsoft/phi-4-mini-instruct-gguf/resolve/main/Phi-4-mini-instruct-q4_k_m.gguf"
MODELS[gemma-3-4b-q4_k_m]="https://huggingface.co/google/gemma-3-4b-it-gguf/resolve/main/gemma-3-4b-it-q4_k_m.gguf"
MODELS[qwen-3-4b-q4_k_m]="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/qwen3-4b-q4_k_m.gguf"

declare -A SHA256S
SHA256S[phi-4-mini-q4_k_m]=""
SHA256S[gemma-3-4b-q4_k_m]=""
SHA256S[qwen-3-4b-q4_k_m]=""

declare -A SIZES
SIZES[phi-4-mini-q4_k_m]="~2.5 GB"
SIZES[gemma-3-4b-q4_k_m]="~2.7 GB"
SIZES[qwen-3-4b-q4_k_m]="~2.5 GB"

###############################################################################
# Helpers
###############################################################################
list_models() {
    echo -e "${BOLD}Available models:${NC}"
    echo ""
    for key in "${!MODELS[@]}"; do
        printf "  ${CYAN}%-30s${NC} %s\n" "${key}" "${SIZES[$key]}"
    done
    echo ""
    echo "Pass --model-name <name> to download a specific model."
}

detect_download_tool() {
    if command -v wget &>/dev/null; then
        DOWNLOAD_CMD="wget -q --show-progress -O"
    elif command -v curl &>/dev/null; then
        DOWNLOAD_CMD="curl -L --progress-bar -o"
    else
        log_error "Neither curl nor wget found. Install one of them first."
        exit 1
    fi
    log_ok "Using download tool: ${DOWNLOAD_CMD%% *}"
}

download_model() {
    local MODEL_NAME="$1"
    local MODEL_URL="${MODELS[${MODEL_NAME}]:-}"
    local MODEL_DIR="${HOME}/models"

    if [[ -z "${MODEL_URL}" ]]; then
        log_error "Unknown model: '${MODEL_NAME}'."
        list_models
        exit 1
    fi

    # Derive filename from URL
    local FILENAME="${MODEL_URL##*/}"
    local OUTPUT_PATH="${MODEL_DIR}/${FILENAME}"

    mkdir -p "${MODEL_DIR}"

    if [[ -f "${OUTPUT_PATH}" ]]; then
        log_info "Model already exists at ${OUTPUT_PATH}"
        log_info "Delete it first if you want to re-download."
        print_model_info "${MODEL_NAME}" "${OUTPUT_PATH}"
        return
    fi

    log_info "Downloading ${MODEL_NAME} (${SIZES[${MODEL_NAME}]:-unknown size})..."
    log_info "URL: ${MODEL_URL}"
    log_info "Target: ${OUTPUT_PATH}"
    echo ""

    ${DOWNLOAD_CMD} "${OUTPUT_PATH}" "${MODEL_URL}"

    echo ""
    log_ok "Download complete: ${OUTPUT_PATH}"

    # Verify checksum if available
    local EXPECTED_SHA="${SHA256S[${MODEL_NAME}]:-}"
    if [[ -n "${EXPECTED_SHA}" ]]; then
        log_info "Verifying SHA256 checksum..."
        local ACTUAL_SHA
        ACTUAL_SHA=$(sha256sum "${OUTPUT_PATH}" | awk '{print $1}')
        if [[ "${ACTUAL_SHA}" == "${EXPECTED_SHA}" ]]; then
            log_ok "Checksum matches."
        else
            log_error "Checksum mismatch!"
            log_error "  Expected: ${EXPECTED_SHA}"
            log_error "  Actual:   ${ACTUAL_SHA}"
            rm -f "${OUTPUT_PATH}"
            exit 1
        fi
    else
        log_warn "No checksum available for '${MODEL_NAME}'. Skipping verification."
    fi

    local FILE_SIZE
    FILE_SIZE=$(numfmt --to=iec "$(stat -c%s "${OUTPUT_PATH}")" 2>/dev/null || stat -c%s "${OUTPUT_PATH}")
    log_ok "Final size: ${FILE_SIZE}"

    print_model_info "${MODEL_NAME}" "${OUTPUT_PATH}"
}

print_model_info() {
    local MODEL_NAME="$1"
    local OUTPUT_PATH="$2"

    echo ""
    echo -e "${BOLD}============================================${NC}"
    echo -e "${BOLD}     Model Ready                           ${NC}"
    echo -e "${BOLD}============================================${NC}"
    echo ""
    echo -e "  ${CYAN}Name:${NC}     ${MODEL_NAME}"
    echo -e "  ${CYAN}Path:${NC}     ${OUTPUT_PATH}"
    echo ""
    echo -e "  ${YELLOW}Test commands:${NC}"
    echo -e "  llama-cli -m ${OUTPUT_PATH} -p 'Hello, how are you?' -n 128"
    echo -e "  llama-server -m ${OUTPUT_PATH} --host 127.0.0.1 --port 8080"
    echo ""
    echo -e "${BOLD}============================================${NC}"
    echo ""
}

###############################################################################
# Main
###############################################################################
main() {
    local MODEL_NAME="phi-4-mini-q4_k_m"

    if [[ $# -gt 0 ]]; then
        case "$1" in
            --list|-l)
                list_models
                exit 0
                ;;
            --model-name|-m)
                if [[ -z "${2:-}" ]]; then
                    log_error "--model-name requires an argument."
                    list_models
                    exit 1
                fi
                MODEL_NAME="$2"
                ;;
            *)
                log_error "Unknown argument: $1"
                echo "Usage: $0 [--model-name <name> | --list]"
                exit 1
                ;;
        esac
    fi

    detect_download_tool
    download_model "${MODEL_NAME}"
}

main "$@"
