#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# anvil_setup.sh — Full system bootstrap for AnvilAgent (llama.cpp + Vulkan)
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

detect_pkg_manager() {
    if command -v apt &>/dev/null; then
        PKG_MANAGER="apt"
        PKG_INSTALL="apt install -y"
        BUILD_DEPS="build-essential cmake git"
        VULKAN_DEPS="vulkan-tools libvulkan-dev mesa-vulkan-drivers"
    elif command -v dnf &>/dev/null; then
        PKG_MANAGER="dnf"
        PKG_INSTALL="dnf install -y"
        BUILD_DEPS="gcc gcc-c++ make cmake git"
        VULKAN_DEPS="vulkan-tools vulkan-loader-devel mesa-vulkan-drivers"
    elif command -v pacman &>/dev/null; then
        PKG_MANAGER="pacman"
        PKG_INSTALL="pacman -S --noconfirm"
        BUILD_DEPS="base-devel cmake git"
        VULKAN_DEPS="vulkan-tools vulkan-headers vulkan-radeon"
    else
        log_error "No supported package manager found (apt, dnf, pacman)."
        exit 1
    fi
    log_ok "Detected package manager: ${PKG_MANAGER}"
}

install_deps() {
    log_info "Updating package lists..."
    case "${PKG_MANAGER}" in
        apt)    apt update -y ;;
        dnf)    dnf check-update -y || true ;;
        pacman) pacman -Sy --noconfirm ;;
    esac

    log_info "Installing build dependencies..."
    # shellcheck disable=SC2086
    ${PKG_INSTALL} ${BUILD_DEPS}

    log_info "Installing Vulkan dependencies..."
    # shellcheck disable=SC2086
    ${PKG_INSTALL} ${VULKAN_DEPS}

    log_ok "Dependencies installed successfully."
}

setup_user_groups() {
    local CURRENT_USER
    CURRENT_USER="$(whoami)"

    if groups "${CURRENT_USER}" | grep -qE '\b(video|render)\b'; then
        log_ok "User '${CURRENT_USER}' already in video/render groups."
        return
    fi

    log_info "Adding '${CURRENT_USER}' to 'video' and 'render' groups..."
    sudo usermod -aG video "${CURRENT_USER}"
    sudo usermod -aG render "${CURRENT_USER}"
    log_warn "Group changes will take effect after logout/login. For now, continuing..."
}

verify_gpu() {
    log_info "Checking Vulkan GPU detection..."
    if ! command -v vulkaninfo &>/dev/null; then
        log_error "vulkaninfo not found. Vulkan tools may not be installed."
        exit 1
    fi

    local GPU_COUNT
    GPU_COUNT=$(vulkaninfo --summary 2>/dev/null | grep -c "GPU id" || true)

    if [[ "${GPU_COUNT}" -eq 0 ]]; then
        log_warn "No Vulkan-capable GPU detected. Falling back to CPU inference will work but be slow."
    else
        log_ok "Detected ${GPU_COUNT} Vulkan-capable GPU(s)."
        vulkaninfo --summary 2>/dev/null | grep -E "(deviceName|GPU id)" || true
    fi
}

clone_llamacpp() {
    local LLAMA_DIR="${HOME}/llama.cpp"

    if [[ -d "${LLAMA_DIR}" ]]; then
        log_info "llama.cpp already exists at ${LLAMA_DIR}, skipping clone."
        return
    fi

    log_info "Cloning llama.cpp (shallow) into ${LLAMA_DIR}..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${LLAMA_DIR}"
    log_ok "llama.cpp cloned successfully."
}

compile_llamacpp() {
    local LLAMA_DIR="${HOME}/llama.cpp"
    local BUILD_DIR="${LLAMA_DIR}/build"

    if [[ ! -d "${LLAMA_DIR}" ]]; then
        log_error "llama.cpp directory not found at ${LLAMA_DIR}. Clone it first."
        exit 1
    fi

    log_info "Configuring build with Vulkan support..."
    cmake -B "${BUILD_DIR}" -DGGML_VULKAN=ON "${LLAMA_DIR}"

    log_info "Compiling in Release mode using all cores..."
    cmake --build "${BUILD_DIR}" --config Release -j"$(nproc)"

    log_ok "llama.cpp compiled successfully."
}

verify_binary() {
    local LLAMA_CLI="${HOME}/llama.cpp/build/bin/llama-cli"

    if [[ ! -x "${LLAMA_CLI}" ]]; then
        log_error "llama-cli binary not found at ${LLAMA_CLI}. Build may have failed."
        exit 1
    fi

    log_info "Verifying llama-cli binary..."
    "${LLAMA_CLI}" --version
    log_ok "Binary verification passed."
}

create_dirs_and_links() {
    local CACHE_DIR="${HOME}/.cache/anvil/kv_pages"
    local CONFIG_DIR="${HOME}/.config/anvil"
    local SCRIPT_DIR
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local CONFIG_SRC="${SCRIPT_DIR}/../configs/anvil.yaml"
    local CONFIG_DST="${CONFIG_DIR}/anvil.yaml"

    log_info "Creating cache directory: ${CACHE_DIR}..."
    mkdir -p "${CACHE_DIR}"
    log_ok "Cache directory ready."

    log_info "Creating config directory: ${CONFIG_DIR}..."
    mkdir -p "${CONFIG_DIR}"

    if [[ -L "${CONFIG_DST}" && "$(readlink "${CONFIG_DST}")" == "${CONFIG_SRC}" ]]; then
        log_ok "Symlink already points to the correct config."
    elif [[ -f "${CONFIG_DST}" ]]; then
        log_warn "${CONFIG_DST} exists and is not our symlink. Skipping."
    else
        ln -s "${CONFIG_SRC}" "${CONFIG_DST}"
        log_ok "Created symlink: ${CONFIG_DST} -> ${CONFIG_SRC}"
    fi
}

print_summary() {
    echo ""
    echo -e "${BOLD}============================================${NC}"
    echo -e "${BOLD}     AnvilAgent Setup Complete              ${NC}"
    echo -e "${BOLD}============================================${NC}"
    echo ""
    echo -e "  ${CYAN}llama.cpp:${NC}     ${HOME}/llama.cpp/"
    echo -e "  ${CYAN}llama-cli:${NC}     ${HOME}/llama.cpp/build/bin/llama-cli"
    echo -e "  ${CYAN}llama-server:${NC}  ${HOME}/llama.cpp/build/bin/llama-server"
    echo -e "  ${CYAN}Config:${NC}        ${HOME}/.config/anvil/anvil.yaml"
    echo -e "  ${CYAN}Cache:${NC}         ${HOME}/.cache/anvil/kv_pages"
    echo ""
    echo -e "  ${YELLOW}Next steps:${NC}"
    echo -e "  1. Log out and back in for group changes to take effect."
    echo -e "  2. Download a model:  ./scripts/download_models.sh --model-name phi-4-mini-q4_k_m"
    echo -e "  3. Run inference:     ${HOME}/llama.cpp/build/bin/llama-cli -m ~/models/phi-4-mini-q4_k_m.gguf -p 'Hello' -n 128"
    echo -e "  4. Start server:      ${HOME}/llama.cpp/build/bin/llama-server -m ~/models/phi-4-mini-q4_k_m.gguf --host 127.0.0.1 --port 8080"
    echo ""
    echo -e "${BOLD}============================================${NC}"
    echo ""
}

main() {
    echo -e "${BOLD}${CYAN}"
    echo "  ╔═══════════════════════════════════════╗"
    echo "  ║        AnvilAgent System Setup        ║"
    echo "  ╚═══════════════════════════════════════╝"
    echo -e "${NC}"

    detect_pkg_manager
    install_deps
    setup_user_groups
    verify_gpu
    clone_llamacpp
    compile_llamacpp
    verify_binary
    create_dirs_and_links
    print_summary
}

main "$@"
