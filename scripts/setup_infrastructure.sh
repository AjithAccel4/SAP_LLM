#!/usr/bin/env bash
#
# SAP_LLM Infrastructure Setup Script
#
# This script performs a complete infrastructure setup including:
# - Dependency checks
# - Virtual environment setup
# - Package installation
# - Model downloads
# - Database initialization
# - Kubernetes setup
# - Secret generation
#

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Log file
LOG_FILE="$PROJECT_ROOT/setup.log"

# Configuration
PYTHON_VERSION_MIN="3.8"
VENV_DIR="$PROJECT_ROOT/venv"
MODELS_DIR="${MODELS_DIR:-/models}"
DATA_DIR="${DATA_DIR:-/data}"

#######################################
# Logging functions
#######################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "\n${CYAN}===================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}$*${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}===================================================${NC}" | tee -a "$LOG_FILE"
}

#######################################
# Helper functions
#######################################

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

version_compare() {
    # Compare two version strings
    # Returns 0 if $1 >= $2, 1 otherwise
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

check_python_version() {
    local python_cmd=$1
    local version
    version=$($python_cmd --version 2>&1 | awk '{print $2}')

    if version_compare "$version" "$PYTHON_VERSION_MIN"; then
        echo "$version"
        return 0
    else
        return 1
    fi
}

wait_for_user() {
    if [[ "${INTERACTIVE:-true}" == "true" ]]; then
        read -p "Press Enter to continue or Ctrl+C to abort..."
    fi
}

#######################################
# Dependency checks
#######################################

check_dependencies() {
    log_step "Step 1: Checking Dependencies"

    local missing_deps=()
    local all_ok=true

    # Python
    log_info "Checking Python..."
    local python_cmd=""

    for cmd in python3 python; do
        if command_exists "$cmd"; then
            if check_python_version "$cmd"; then
                python_cmd=$cmd
                local version=$(check_python_version "$cmd")
                log_success "Found $cmd $version"
                break
            fi
        fi
    done

    if [[ -z "$python_cmd" ]]; then
        log_error "Python $PYTHON_VERSION_MIN or higher not found"
        missing_deps+=("python3")
        all_ok=false
    fi

    export PYTHON_CMD=$python_cmd

    # pip
    log_info "Checking pip..."
    if command_exists pip3 || command_exists pip; then
        log_success "pip is installed"
    else
        log_warning "pip not found - will try to bootstrap"
    fi

    # Git
    log_info "Checking Git..."
    if command_exists git; then
        local git_version=$(git --version | awk '{print $3}')
        log_success "Git $git_version is installed"
    else
        log_error "Git not found"
        missing_deps+=("git")
        all_ok=false
    fi

    # Docker (optional but recommended)
    log_info "Checking Docker..."
    if command_exists docker; then
        local docker_version=$(docker --version | awk '{print $3}' | tr -d ',')
        log_success "Docker $docker_version is installed"

        # Check if Docker daemon is running
        if docker info >/dev/null 2>&1; then
            log_success "Docker daemon is running"
        else
            log_warning "Docker is installed but daemon is not running"
        fi
    else
        log_warning "Docker not found (optional for containerized deployment)"
    fi

    # Kubernetes (optional)
    log_info "Checking Kubernetes..."
    if command_exists kubectl; then
        local kubectl_version=$(kubectl version --client --short 2>/dev/null | awk '{print $3}')
        log_success "kubectl $kubectl_version is installed"
    else
        log_warning "kubectl not found (optional for Kubernetes deployment)"
    fi

    # Check system resources
    log_info "Checking system resources..."

    # Memory
    if command_exists free; then
        local total_mem=$(free -g | awk '/^Mem:/{print $2}')
        log_info "Total memory: ${total_mem}GB"

        if [[ $total_mem -lt 16 ]]; then
            log_warning "Less than 16GB RAM detected. Large models may require more memory."
        fi
    fi

    # Disk space
    local free_space=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | tr -d 'G')
    log_info "Free disk space: ${free_space}GB"

    if [[ $free_space -lt 100 ]]; then
        log_warning "Less than 100GB free disk space. Model downloads require significant space."
    fi

    # GPU (optional)
    log_info "Checking GPU availability..."
    if command_exists nvidia-smi; then
        log_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read -r line; do
            log_info "  GPU: $line"
        done
    else
        log_warning "No NVIDIA GPU detected (will use CPU mode)"
    fi

    # Summary
    if [[ $all_ok == true ]]; then
        log_success "All required dependencies are installed"
        return 0
    else
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_error "Please install missing dependencies and run this script again"
        return 1
    fi
}

#######################################
# Virtual environment setup
#######################################

setup_virtualenv() {
    log_step "Step 2: Setting Up Virtual Environment"

    if [[ -d "$VENV_DIR" ]]; then
        log_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
        else
            log_info "Using existing virtual environment"
            return 0
        fi
    fi

    log_info "Creating virtual environment at $VENV_DIR..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"

    log_success "Virtual environment created"

    # Activate virtual environment
    log_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel

    log_success "Virtual environment setup complete"
}

#######################################
# Package installation
#######################################

install_packages() {
    log_step "Step 3: Installing Python Packages"

    # Activate virtual environment if not already activated
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        source "$VENV_DIR/bin/activate"
    fi

    local requirements_file="$PROJECT_ROOT/requirements.txt"

    if [[ ! -f "$requirements_file" ]]; then
        log_error "requirements.txt not found at $requirements_file"
        return 1
    fi

    log_info "Installing packages from requirements.txt..."
    log_info "This may take several minutes..."

    # Install with progress
    pip install -r "$requirements_file" --progress-bar on

    # Install the package itself in development mode
    log_info "Installing SAP_LLM package in development mode..."
    pip install -e "$PROJECT_ROOT"

    log_success "Package installation complete"

    # Verify key packages
    log_info "Verifying key packages..."
    local key_packages=("torch" "transformers" "fastapi" "redis" "pymongo")

    for package in "${key_packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            local version=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
            log_success "$package $version"
        else
            log_error "$package not installed correctly"
        fi
    done
}

#######################################
# Model downloads
#######################################

download_models() {
    log_step "Step 4: Downloading Models"

    # Activate virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        source "$VENV_DIR/bin/activate"
    fi

    # Create models directory
    mkdir -p "$MODELS_DIR"

    log_info "Models will be downloaded to: $MODELS_DIR"
    log_warning "This will download approximately 100GB of model files"

    # Ask for confirmation unless non-interactive
    if [[ "${INTERACTIVE:-true}" == "true" ]]; then
        read -p "Do you want to download models now? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            log_warning "Skipping model download. You can download them later using:"
            log_info "  python scripts/download_models.py --cache-dir $MODELS_DIR"
            return 0
        fi
    fi

    # Check for HuggingFace token
    if [[ -z "${HF_TOKEN:-}" ]]; then
        log_warning "HF_TOKEN not set. Some models may require authentication."
        log_info "Set HF_TOKEN environment variable or use --token option"
    fi

    # Run download script
    log_info "Starting model download..."

    local download_script="$SCRIPT_DIR/download_models.py"

    if [[ -f "$download_script" ]]; then
        if python "$download_script" --cache-dir "$MODELS_DIR"; then
            log_success "Model download complete"
        else
            log_warning "Some models may have failed to download"
            log_info "You can retry with: python $download_script --cache-dir $MODELS_DIR"
        fi
    else
        log_error "Download script not found: $download_script"
        return 1
    fi
}

#######################################
# Database initialization
#######################################

init_databases() {
    log_step "Step 5: Initializing Databases"

    # Activate virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        source "$VENV_DIR/bin/activate"
    fi

    # Check for .env file
    local env_file="$PROJECT_ROOT/.env"

    if [[ ! -f "$env_file" ]]; then
        log_warning ".env file not found"

        if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
            log_info "Creating .env from .env.example..."
            cp "$PROJECT_ROOT/.env.example" "$env_file"
            log_warning "Please edit .env and configure your database credentials"

            if command_exists nano; then
                read -p "Do you want to edit .env now? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    nano "$env_file"
                fi
            fi
        else
            log_error ".env.example not found"
            return 1
        fi
    fi

    # Load environment variables
    if [[ -f "$env_file" ]]; then
        log_info "Loading environment variables from .env..."
        set -a
        source "$env_file"
        set +a
    fi

    # Run database initialization script
    log_info "Initializing databases..."

    local init_script="$SCRIPT_DIR/init_databases.py"

    if [[ -f "$init_script" ]]; then
        if python "$init_script"; then
            log_success "Database initialization complete"
        else
            log_warning "Database initialization completed with warnings"
            log_info "You can retry with: python $init_script"
        fi
    else
        log_error "Init script not found: $init_script"
        return 1
    fi
}

#######################################
# Kubernetes setup
#######################################

setup_kubernetes() {
    log_step "Step 6: Setting Up Kubernetes Resources"

    if ! command_exists kubectl; then
        log_warning "kubectl not found - skipping Kubernetes setup"
        return 0
    fi

    # Check if kubectl can connect
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_warning "Cannot connect to Kubernetes cluster - skipping setup"
        log_info "You can setup Kubernetes later using deployments/kubernetes/"
        return 0
    fi

    log_info "Kubernetes cluster detected"

    local k8s_dir="$PROJECT_ROOT/deployments/kubernetes"

    if [[ ! -d "$k8s_dir" ]]; then
        log_warning "Kubernetes deployment files not found at $k8s_dir"
        return 0
    fi

    read -p "Do you want to deploy to Kubernetes now? (y/N): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Skipping Kubernetes deployment"
        return 0
    fi

    # Create namespace
    log_info "Creating namespace..."
    kubectl apply -f "$k8s_dir/namespace.yaml"

    # Create ConfigMap
    log_info "Creating ConfigMap..."
    kubectl apply -f "$k8s_dir/configmap.yaml"

    # Create PVC
    log_info "Creating Persistent Volume Claims..."
    kubectl apply -f "$k8s_dir/pvc.yaml"

    # Deploy services
    log_info "Deploying services..."
    kubectl apply -f "$k8s_dir/redis-deployment.yaml"
    kubectl apply -f "$k8s_dir/mongo-deployment.yaml"
    kubectl apply -f "$k8s_dir/deployment.yaml"
    kubectl apply -f "$k8s_dir/service.yaml"

    # Setup HPA
    log_info "Setting up Horizontal Pod Autoscaler..."
    kubectl apply -f "$k8s_dir/hpa.yaml"

    log_success "Kubernetes resources deployed"

    # Show status
    log_info "Current pod status:"
    kubectl get pods -n sap-llm
}

#######################################
# Secret generation
#######################################

generate_secrets() {
    log_step "Step 7: Generating API Keys and Secrets"

    local keys_dir="$PROJECT_ROOT/keys"
    mkdir -p "$keys_dir"

    # Generate API secret key
    log_info "Generating API secret key..."
    local api_secret=$(openssl rand -hex 32)

    # Generate APOP key pair
    log_info "Generating APOP key pair..."

    if command_exists openssl; then
        # Generate private key
        openssl ecparam -name secp256k1 -genkey -noout -out "$keys_dir/apop_private_key.pem" 2>/dev/null

        # Generate public key
        openssl ec -in "$keys_dir/apop_private_key.pem" -pubout -out "$keys_dir/apop_public_key.pem" 2>/dev/null

        log_success "APOP key pair generated"
        log_info "Private key: $keys_dir/apop_private_key.pem"
        log_info "Public key: $keys_dir/apop_public_key.pem"
    else
        log_warning "OpenSSL not found - skipping key generation"
    fi

    # Update .env file
    local env_file="$PROJECT_ROOT/.env"

    if [[ -f "$env_file" ]]; then
        log_info "Updating .env with generated secrets..."

        # Backup .env
        cp "$env_file" "$env_file.backup"

        # Update API_SECRET_KEY
        if grep -q "^API_SECRET_KEY=" "$env_file"; then
            sed -i.tmp "s|^API_SECRET_KEY=.*|API_SECRET_KEY=$api_secret|" "$env_file"
        else
            echo "API_SECRET_KEY=$api_secret" >> "$env_file"
        fi

        # Update APOP key paths
        if grep -q "^APOP_PRIVATE_KEY_PATH=" "$env_file"; then
            sed -i.tmp "s|^APOP_PRIVATE_KEY_PATH=.*|APOP_PRIVATE_KEY_PATH=$keys_dir/apop_private_key.pem|" "$env_file"
        fi

        if grep -q "^APOP_PUBLIC_KEY_PATH=" "$env_file"; then
            sed -i.tmp "s|^APOP_PUBLIC_KEY_PATH=.*|APOP_PUBLIC_KEY_PATH=$keys_dir/apop_public_key.pem|" "$env_file"
        fi

        # Clean up temp files
        rm -f "$env_file.tmp"

        log_success "Secrets updated in .env"
    fi

    # Set secure permissions
    chmod 600 "$keys_dir"/*.pem 2>/dev/null || true

    log_warning "IMPORTANT: Keep your keys secure and never commit them to version control"
}

#######################################
# Health check
#######################################

run_health_check() {
    log_step "Step 8: Running Health Check"

    # Activate virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        source "$VENV_DIR/bin/activate"
    fi

    local health_script="$SCRIPT_DIR/health_check.py"

    if [[ -f "$health_script" ]]; then
        log_info "Running system health check..."
        python "$health_script" --verbose
    else
        log_warning "Health check script not found: $health_script"
    fi
}

#######################################
# Summary
#######################################

print_summary() {
    log_step "Setup Complete!"

    cat << EOF

${GREEN}✓ SAP_LLM infrastructure setup completed successfully!${NC}

${CYAN}Next Steps:${NC}

1. Activate the virtual environment:
   ${YELLOW}source $VENV_DIR/bin/activate${NC}

2. Review and update configuration:
   ${YELLOW}nano $PROJECT_ROOT/.env${NC}
   ${YELLOW}nano $PROJECT_ROOT/configs/default_config.yaml${NC}

3. Start the API server:
   ${YELLOW}python -m sap_llm.api.server${NC}

4. Run tests:
   ${YELLOW}./scripts/run_tests.sh${NC}

5. View documentation:
   ${YELLOW}cat $PROJECT_ROOT/README.md${NC}

${CYAN}Useful Commands:${NC}

  - Health check: ${YELLOW}python scripts/health_check.py${NC}
  - Download models: ${YELLOW}python scripts/download_models.py${NC}
  - Initialize databases: ${YELLOW}python scripts/init_databases.py${NC}

${CYAN}Documentation:${NC}
  - README: $PROJECT_ROOT/README.md
  - Deployment: $PROJECT_ROOT/DEPLOYMENT.md
  - API Docs: http://localhost:8000/docs (when server is running)

${CYAN}Support:${NC}
  - Check logs: $LOG_FILE
  - Report issues: Create a GitHub issue

EOF
}

#######################################
# Main execution
#######################################

main() {
    # Print banner
    cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              SAP_LLM Infrastructure Setup                    ║
║                                                              ║
║  This script will set up the complete SAP_LLM environment   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

EOF

    # Initialize log file
    echo "SAP_LLM Infrastructure Setup - $(date)" > "$LOG_FILE"
    log_info "Starting infrastructure setup..."
    log_info "Log file: $LOG_FILE"

    # Parse command line arguments
    INTERACTIVE=true
    SKIP_MODELS=false
    SKIP_K8S=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --non-interactive)
                INTERACTIVE=false
                shift
                ;;
            --skip-models)
                SKIP_MODELS=true
                shift
                ;;
            --skip-k8s)
                SKIP_K8S=true
                shift
                ;;
            --help)
                cat << EOF
Usage: $0 [OPTIONS]

Options:
  --non-interactive    Run without user prompts
  --skip-models        Skip model download
  --skip-k8s          Skip Kubernetes setup
  --help              Show this help message

EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Run setup steps
    check_dependencies || exit 1
    setup_virtualenv || exit 1
    install_packages || exit 1

    if [[ $SKIP_MODELS == false ]]; then
        download_models || true  # Don't fail if models don't download
    else
        log_info "Skipping model download (--skip-models flag)"
    fi

    init_databases || true  # Don't fail if databases aren't ready

    if [[ $SKIP_K8S == false ]]; then
        setup_kubernetes || true  # Don't fail if K8s isn't available
    else
        log_info "Skipping Kubernetes setup (--skip-k8s flag)"
    fi

    generate_secrets || exit 1
    run_health_check || true  # Don't fail on health check warnings

    # Print summary
    print_summary

    log_success "Setup completed successfully!"
}

# Run main function
main "$@"
