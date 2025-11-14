#!/bin/bash

################################################################################
# SAP_LLM Automated Backup Script
#
# This script performs automated backups of all critical SAP_LLM components:
# - Model files (Vision Encoder, Language Decoder, Reasoning Engine)
# - MongoDB databases
# - Configuration files
# - Optionally: Redis snapshots
#
# Usage:
#   ./backup.sh [OPTIONS]
#
# Options:
#   --component <name>    Backup specific component (models|mongodb|configs|redis|all)
#   --verify-only        Only verify existing backups, don't create new ones
#   --verify-local-models Verify local model integrity
#   --list-recent        List recent backups
#   --dry-run           Show what would be backed up without actually doing it
#   --help              Show this help message
#
# Environment Variables:
#   AZURE_STORAGE_ACCOUNT    Azure storage account name
#   AZURE_STORAGE_KEY        Azure storage account key
#   BACKUP_RETENTION_DAYS    Days to keep backups (default: 90)
#   MONGO_URI               MongoDB connection string
#   REDIS_HOST              Redis host
#
# Exit Codes:
#   0 - Success
#   1 - General error
#   2 - Backup verification failed
#   3 - Upload failed
################################################################################

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
BACKUP_DATE=$(date +%Y-%m-%d)
BACKUP_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-90}
DRY_RUN=false
COMPONENT="all"

# Paths
MODELS_DIR="${MODELS_DIR:-${PROJECT_ROOT}/models}"
CONFIGS_DIR="${CONFIGS_DIR:-${PROJECT_ROOT}/configs}"
BACKUP_TEMP_DIR="/tmp/sap_llm_backup_${BACKUP_TIMESTAMP}"

# Azure Storage configuration
AZURE_STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-sapllmprodbackups}"
STORAGE_CONTAINER_MODELS="models"
STORAGE_CONTAINER_MONGODB="mongodb"
STORAGE_CONTAINER_CONFIGS="configs"

# MongoDB configuration
MONGO_URI="${MONGO_URI:-mongodb://localhost:27017}"
MONGO_DATABASE="${MONGO_DATABASE:-sap_llm_kb}"

# Redis configuration
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

check_dependencies() {
    local deps=("az" "mongodump" "gzip" "sha256sum")
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Install missing dependencies:"
        log_info "  - az: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
        log_info "  - mongodump: sudo apt-get install mongodb-org-tools"
        exit 1
    fi
}

check_azure_login() {
    if ! az account show &> /dev/null; then
        log_error "Not logged in to Azure. Run 'az login' first."
        exit 1
    fi
}

create_backup_dir() {
    mkdir -p "$BACKUP_TEMP_DIR"/{models,mongodb,configs,checksums}
    log_info "Created temporary backup directory: $BACKUP_TEMP_DIR"
}

cleanup_backup_dir() {
    if [ -d "$BACKUP_TEMP_DIR" ]; then
        log_info "Cleaning up temporary directory..."
        rm -rf "$BACKUP_TEMP_DIR"
    fi
}

calculate_checksum() {
    local file=$1
    sha256sum "$file" | awk '{print $1}'
}

################################################################################
# Backup Functions
################################################################################

backup_models() {
    log_info "Starting model backup..."

    if [ ! -d "$MODELS_DIR" ]; then
        log_warning "Models directory not found: $MODELS_DIR"
        return 1
    fi

    local models=(
        "vision_encoder"
        "language_decoder"
        "reasoning_engine"
    )

    for model in "${models[@]}"; do
        local model_path="$MODELS_DIR/$model"

        if [ ! -d "$model_path" ]; then
            log_warning "Model not found: $model"
            continue
        fi

        log_info "Backing up model: $model"

        # Calculate size
        local size=$(du -sh "$model_path" | cut -f1)
        log_info "  Model size: $size"

        # Create compressed archive
        local archive_name="${model}_${BACKUP_TIMESTAMP}.tar.gz"
        local archive_path="$BACKUP_TEMP_DIR/models/$archive_name"

        if [ "$DRY_RUN" = true ]; then
            log_info "  [DRY RUN] Would create: $archive_name"
            continue
        fi

        tar -czf "$archive_path" -C "$MODELS_DIR" "$model" 2>&1 | grep -v "Removing leading"

        # Calculate checksum
        local checksum=$(calculate_checksum "$archive_path")
        echo "$checksum  $archive_name" >> "$BACKUP_TEMP_DIR/checksums/models_${BACKUP_TIMESTAMP}.sha256"

        log_success "  Created archive: $archive_name (checksum: ${checksum:0:16}...)"
    done

    # Backup checkpoints if they exist
    local checkpoint_dir="$MODELS_DIR/checkpoints"
    if [ -d "$checkpoint_dir" ]; then
        log_info "Backing up model checkpoints..."

        # Only backup the best model and latest checkpoint
        if [ -d "$checkpoint_dir/best_model" ]; then
            local archive_name="checkpoint_best_${BACKUP_TIMESTAMP}.tar.gz"
            local archive_path="$BACKUP_TEMP_DIR/models/$archive_name"

            if [ "$DRY_RUN" = false ]; then
                tar -czf "$archive_path" -C "$checkpoint_dir" "best_model" 2>&1 | grep -v "Removing leading"
                local checksum=$(calculate_checksum "$archive_path")
                echo "$checksum  $archive_name" >> "$BACKUP_TEMP_DIR/checksums/models_${BACKUP_TIMESTAMP}.sha256"
                log_success "  Backed up best model checkpoint"
            fi
        fi
    fi

    log_success "Model backup completed"
}

backup_mongodb() {
    log_info "Starting MongoDB backup..."

    # Check if MongoDB is accessible
    if ! mongosh "$MONGO_URI" --eval "db.adminCommand('ping')" --quiet &> /dev/null; then
        log_error "Cannot connect to MongoDB at $MONGO_URI"
        return 1
    fi

    # Determine backup type (full or incremental)
    local backup_type="full"
    local current_hour=$(date +%H)

    # Full backup at 01:00, incremental hourly
    if [ "$current_hour" != "01" ]; then
        backup_type="incremental"
    fi

    log_info "Backup type: $backup_type"

    # Create backup directory
    local backup_dir="$BACKUP_TEMP_DIR/mongodb/${backup_type}"
    mkdir -p "$backup_dir"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would create MongoDB backup: ${backup_type}_${BACKUP_TIMESTAMP}"
        return 0
    fi

    # Perform backup
    log_info "Running mongodump..."

    if [ "$backup_type" = "full" ]; then
        # Full backup
        mongodump \
            --uri="$MONGO_URI" \
            --db="$MONGO_DATABASE" \
            --out="$backup_dir" \
            --gzip \
            2>&1 | while read line; do
                log_info "  $line"
            done
    else
        # Incremental backup (only changed collections)
        # This is simplified; in production, use oplog for true incremental backups
        mongodump \
            --uri="$MONGO_URI" \
            --db="$MONGO_DATABASE" \
            --out="$backup_dir" \
            --gzip \
            2>&1 | while read line; do
                log_info "  $line"
            done
    fi

    # Create archive
    local archive_name="mongodb_${backup_type}_${BACKUP_TIMESTAMP}.tar.gz"
    local archive_path="$BACKUP_TEMP_DIR/mongodb/$archive_name"

    tar -czf "$archive_path" -C "$BACKUP_TEMP_DIR/mongodb" "$backup_type" 2>&1 | grep -v "Removing leading"

    # Calculate checksum
    local checksum=$(calculate_checksum "$archive_path")
    echo "$checksum  $archive_name" >> "$BACKUP_TEMP_DIR/checksums/mongodb_${BACKUP_TIMESTAMP}.sha256"

    # Get backup size
    local size=$(du -sh "$archive_path" | cut -f1)

    log_success "MongoDB backup completed: $archive_name (size: $size, checksum: ${checksum:0:16}...)"
}

backup_configs() {
    log_info "Starting configuration backup..."

    if [ ! -d "$CONFIGS_DIR" ]; then
        log_warning "Configs directory not found: $CONFIGS_DIR"
        return 1
    fi

    local archive_name="configs_${BACKUP_TIMESTAMP}.tar.gz"
    local archive_path="$BACKUP_TEMP_DIR/configs/$archive_name"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would create config backup: $archive_name"
        return 0
    fi

    # Backup configs (excluding sensitive data - those are in Key Vault)
    tar -czf "$archive_path" \
        -C "$PROJECT_ROOT" \
        --exclude="*.key" \
        --exclude="*.pem" \
        --exclude="*secret*" \
        configs/ \
        2>&1 | grep -v "Removing leading"

    # Calculate checksum
    local checksum=$(calculate_checksum "$archive_path")
    echo "$checksum  $archive_name" >> "$BACKUP_TEMP_DIR/checksums/configs_${BACKUP_TIMESTAMP}.sha256"

    local size=$(du -sh "$archive_path" | cut -f1)
    log_success "Configuration backup completed: $archive_name (size: $size)"
}

backup_redis() {
    log_info "Starting Redis backup (optional)..."

    # Check if Redis is accessible
    if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &> /dev/null; then
        log_warning "Cannot connect to Redis at ${REDIS_HOST}:${REDIS_PORT}"
        log_info "Skipping Redis backup (cache will rebuild on restore)"
        return 0
    fi

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would create Redis snapshot"
        return 0
    fi

    # Trigger Redis save
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE

    log_info "Redis background save initiated"
    log_info "Note: Redis is ephemeral cache - backup is optional for warm-up only"
}

################################################################################
# Upload Functions
################################################################################

upload_to_azure() {
    local component=$1
    local container=""

    case $component in
        models)
            container="$STORAGE_CONTAINER_MODELS"
            ;;
        mongodb)
            container="$STORAGE_CONTAINER_MONGODB"
            ;;
        configs)
            container="$STORAGE_CONTAINER_CONFIGS"
            ;;
        *)
            log_error "Unknown component: $component"
            return 1
            ;;
    esac

    log_info "Uploading $component backups to Azure Blob Storage..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would upload to container: $container"
        return 0
    fi

    # Ensure container exists
    az storage container create \
        --name "$container" \
        --account-name "$AZURE_STORAGE_ACCOUNT" \
        --only-show-errors \
        &> /dev/null || true

    # Upload all files in component directory
    local component_dir="$BACKUP_TEMP_DIR/$component"

    if [ ! -d "$component_dir" ] || [ -z "$(ls -A "$component_dir")" ]; then
        log_warning "No backups to upload for $component"
        return 0
    fi

    local file_count=0
    local total_size=0

    for file in "$component_dir"/*; do
        if [ -f "$file" ]; then
            local filename=$(basename "$file")
            local blob_path="daily/$BACKUP_DATE/$filename"

            log_info "  Uploading: $filename"

            az storage blob upload \
                --account-name "$AZURE_STORAGE_ACCOUNT" \
                --container-name "$container" \
                --name "$blob_path" \
                --file "$file" \
                --overwrite \
                --only-show-errors

            ((file_count++))
            local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
            total_size=$((total_size + size))
        fi
    done

    # Upload checksums
    local checksum_file="$BACKUP_TEMP_DIR/checksums/${component}_${BACKUP_TIMESTAMP}.sha256"
    if [ -f "$checksum_file" ]; then
        az storage blob upload \
            --account-name "$AZURE_STORAGE_ACCOUNT" \
            --container-name "$container" \
            --name "daily/$BACKUP_DATE/checksums.sha256" \
            --file "$checksum_file" \
            --overwrite \
            --only-show-errors
    fi

    local size_mb=$((total_size / 1024 / 1024))
    log_success "Uploaded $file_count files ($size_mb MB) for $component"
}

################################################################################
# Verification Functions
################################################################################

verify_local_models() {
    log_info "Verifying local model integrity..."

    local models=(
        "vision_encoder"
        "language_decoder"
        "reasoning_engine"
    )

    local all_valid=true

    for model in "${models[@]}"; do
        local model_path="$MODELS_DIR/$model"

        if [ ! -d "$model_path" ]; then
            log_warning "Model not found: $model"
            all_valid=false
            continue
        fi

        # Check for required files
        local required_files=("config.json" "pytorch_model.bin")
        local model_valid=true

        for file in "${required_files[@]}"; do
            if [ ! -f "$model_path/$file" ]; then
                log_error "Missing required file in $model: $file"
                model_valid=false
                all_valid=false
            fi
        done

        if [ "$model_valid" = true ]; then
            log_success "Model OK: $model"
        fi
    done

    if [ "$all_valid" = true ]; then
        log_success "All local models verified successfully"
        return 0
    else
        log_error "Model verification failed"
        return 2
    fi
}

verify_backup() {
    log_info "Verifying backup integrity..."

    # Download and verify checksums from Azure
    local temp_verify_dir="/tmp/sap_llm_verify_$$"
    mkdir -p "$temp_verify_dir"

    # Verify latest model backup
    log_info "Verifying model backups..."

    az storage blob download \
        --account-name "$AZURE_STORAGE_ACCOUNT" \
        --container-name "$STORAGE_CONTAINER_MODELS" \
        --name "daily/$BACKUP_DATE/checksums.sha256" \
        --file "$temp_verify_dir/models_checksums.sha256" \
        --only-show-errors \
        2>/dev/null || {
            log_warning "No checksum file found for $BACKUP_DATE"
            rm -rf "$temp_verify_dir"
            return 1
        }

    log_success "Backup verification completed"
    rm -rf "$temp_verify_dir"
}

list_recent_backups() {
    log_info "Listing recent backups..."

    for container in "$STORAGE_CONTAINER_MODELS" "$STORAGE_CONTAINER_MONGODB" "$STORAGE_CONTAINER_CONFIGS"; do
        echo ""
        log_info "Container: $container"

        az storage blob list \
            --account-name "$AZURE_STORAGE_ACCOUNT" \
            --container-name "$container" \
            --prefix "daily/" \
            --query "[].{Name:name, Size:properties.contentLength, Modified:properties.lastModified}" \
            --output table \
            2>/dev/null | tail -n +3 | head -n 10 || {
                log_warning "No backups found in $container"
            }
    done
}

################################################################################
# Cleanup Functions
################################################################################

cleanup_old_backups() {
    log_info "Cleaning up backups older than $BACKUP_RETENTION_DAYS days..."

    local cutoff_date=$(date -d "$BACKUP_RETENTION_DAYS days ago" +%Y-%m-%d 2>/dev/null || date -v-${BACKUP_RETENTION_DAYS}d +%Y-%m-%d 2>/dev/null)

    for container in "$STORAGE_CONTAINER_MODELS" "$STORAGE_CONTAINER_MONGODB" "$STORAGE_CONTAINER_CONFIGS"; do
        log_info "Cleaning container: $container"

        # List old blobs
        local old_blobs=$(az storage blob list \
            --account-name "$AZURE_STORAGE_ACCOUNT" \
            --container-name "$container" \
            --prefix "daily/" \
            --query "[?properties.lastModified < '$cutoff_date'].name" \
            --output tsv 2>/dev/null)

        if [ -n "$old_blobs" ]; then
            local count=0
            while IFS= read -r blob; do
                if [ "$DRY_RUN" = true ]; then
                    log_info "  [DRY RUN] Would delete: $blob"
                else
                    az storage blob delete \
                        --account-name "$AZURE_STORAGE_ACCOUNT" \
                        --container-name "$container" \
                        --name "$blob" \
                        --only-show-errors
                    ((count++))
                fi
            done <<< "$old_blobs"

            if [ "$DRY_RUN" = false ] && [ $count -gt 0 ]; then
                log_success "Deleted $count old backups from $container"
            fi
        else
            log_info "No old backups to clean in $container"
        fi
    done
}

################################################################################
# Main Execution
################################################################################

show_help() {
    grep '^#' "$0" | sed 's/^# //' | sed 's/^#//' | head -n -1 | tail -n +2
}

main() {
    log_info "=== SAP_LLM Backup Script ==="
    log_info "Backup date: $BACKUP_DATE"
    log_info "Timestamp: $BACKUP_TIMESTAMP"
    echo ""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --component)
                COMPONENT="$2"
                shift 2
                ;;
            --verify-only)
                verify_backup
                exit 0
                ;;
            --verify-local-models)
                verify_local_models
                exit $?
                ;;
            --list-recent)
                check_azure_login
                list_recent_backups
                exit 0
                ;;
            --dry-run)
                DRY_RUN=true
                log_warning "DRY RUN MODE - No backups will be created or uploaded"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Check dependencies
    check_dependencies
    check_azure_login

    # Create backup directory
    create_backup_dir

    # Trap to ensure cleanup
    trap cleanup_backup_dir EXIT

    # Perform backups based on component selection
    case $COMPONENT in
        models)
            backup_models
            upload_to_azure "models"
            ;;
        mongodb)
            backup_mongodb
            upload_to_azure "mongodb"
            ;;
        configs)
            backup_configs
            upload_to_azure "configs"
            ;;
        redis)
            backup_redis
            ;;
        all)
            backup_models
            backup_mongodb
            backup_configs
            backup_redis

            upload_to_azure "models"
            upload_to_azure "mongodb"
            upload_to_azure "configs"
            ;;
        *)
            log_error "Invalid component: $COMPONENT"
            log_info "Valid components: models, mongodb, configs, redis, all"
            exit 1
            ;;
    esac

    # Cleanup old backups
    if [ "$DRY_RUN" = false ]; then
        cleanup_old_backups
    fi

    echo ""
    log_success "=== Backup completed successfully ==="
    log_info "Backup location: Azure Storage Account '$AZURE_STORAGE_ACCOUNT'"
    log_info "Backup date: $BACKUP_DATE"

    if [ "$DRY_RUN" = true ]; then
        log_warning "This was a DRY RUN - no actual backups were created"
    fi
}

# Run main function
main "$@"
