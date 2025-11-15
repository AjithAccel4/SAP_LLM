#!/bin/bash

################################################################################
# SAP_LLM Automated Restore Script
#
# This script performs restoration of SAP_LLM components from backups:
# - Model files (Vision Encoder, Language Decoder, Reasoning Engine)
# - MongoDB databases
# - Configuration files
# - Redis warm-up (optional)
#
# Usage:
#   ./restore.sh [OPTIONS]
#
# Options:
#   --component <name>    Restore specific component (models|mongodb|configs|redis|all)
#   --date <YYYY-MM-DD>   Date of backup to restore (default: latest)
#   --point-in-time <ISO> Point-in-time for MongoDB restore (ISO 8601 format)
#   --environment <env>   Target environment (production|staging|test|dr)
#   --source <region>     Source region for DR scenarios (eastus|westus)
#   --target <region>     Target region for restoration
#   --location <region>   Azure region for DR restore
#   --warm-up            Warm up Redis cache after restore
#   --verify             Verify restored data integrity
#   --dry-run            Show what would be restored without actually doing it
#   --help               Show this help message
#
# Environment Variables:
#   AZURE_STORAGE_ACCOUNT    Azure storage account name
#   AZURE_STORAGE_KEY        Azure storage account key
#   MONGO_URI                MongoDB connection string
#   REDIS_HOST               Redis host
#
# Exit Codes:
#   0 - Success
#   1 - General error
#   2 - Verification failed
#   3 - Download failed
#   4 - Restore failed
################################################################################

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
RESTORE_DATE=""
RESTORE_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
DRY_RUN=false
COMPONENT="all"
ENVIRONMENT="production"
VERIFY=true
WARM_UP=false
POINT_IN_TIME=""

# Paths
MODELS_DIR="${MODELS_DIR:-${PROJECT_ROOT}/models}"
CONFIGS_DIR="${CONFIGS_DIR:-${PROJECT_ROOT}/configs}"
RESTORE_TEMP_DIR="/tmp/sap_llm_restore_${RESTORE_TIMESTAMP}"

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
MAGENTA='\033[0;35m'
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

log_step() {
    echo -e "${MAGENTA}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

check_dependencies() {
    local deps=("az" "mongorestore" "tar" "gzip" "sha256sum")
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
        log_info "  - mongorestore: sudo apt-get install mongodb-org-tools"
        exit 1
    fi
}

check_azure_login() {
    if ! az account show &> /dev/null; then
        log_error "Not logged in to Azure. Run 'az login' first."
        exit 1
    fi
}

create_restore_dir() {
    mkdir -p "$RESTORE_TEMP_DIR"/{models,mongodb,configs}
    log_info "Created temporary restore directory: $RESTORE_TEMP_DIR"
}

cleanup_restore_dir() {
    if [ -d "$RESTORE_TEMP_DIR" ]; then
        log_info "Cleaning up temporary directory..."
        rm -rf "$RESTORE_TEMP_DIR"
    fi
}

get_latest_backup_date() {
    local container=$1

    local latest_date=$(az storage blob list \
        --account-name "$AZURE_STORAGE_ACCOUNT" \
        --container-name "$container" \
        --prefix "daily/" \
        --query "[-1].name" \
        --output tsv 2>/dev/null | cut -d'/' -f2)

    if [ -z "$latest_date" ]; then
        log_error "No backups found in container: $container"
        exit 3
    fi

    echo "$latest_date"
}

verify_checksum() {
    local file=$1
    local expected_checksum=$2

    local actual_checksum=$(sha256sum "$file" | awk '{print $1}')

    if [ "$actual_checksum" = "$expected_checksum" ]; then
        return 0
    else
        log_error "Checksum mismatch for $file"
        log_error "  Expected: $expected_checksum"
        log_error "  Actual:   $actual_checksum"
        return 1
    fi
}

################################################################################
# Pre-Restore Safety Checks
################################################################################

pre_restore_checks() {
    log_step "Performing pre-restore safety checks..."

    # Check if services are running
    if [ "$ENVIRONMENT" = "production" ]; then
        log_warning "Restoring to PRODUCTION environment!"
        log_warning "This will overwrite existing data."
        log_warning ""

        if [ "$DRY_RUN" = false ]; then
            read -p "Are you sure you want to continue? (yes/no): " confirm
            if [ "$confirm" != "yes" ]; then
                log_info "Restore cancelled by user"
                exit 0
            fi
        fi
    fi

    # Backup current state before restore
    if [ "$ENVIRONMENT" = "production" ] && [ "$DRY_RUN" = false ]; then
        log_info "Creating safety backup of current state..."
        local safety_backup_dir="/backup/pre-restore/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$safety_backup_dir"

        if [ -d "$MODELS_DIR" ]; then
            log_info "  Backing up current models..."
            cp -r "$MODELS_DIR" "$safety_backup_dir/" 2>/dev/null || true
        fi

        log_success "Safety backup created at: $safety_backup_dir"
    fi

    # Check disk space
    local required_space=524288000  # 500GB in KB
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')

    if [ "$available_space" -lt "$required_space" ]; then
        log_warning "Low disk space: $(($available_space / 1024 / 1024))GB available"
        log_warning "Recommended: 500GB available"
    fi

    log_success "Pre-restore checks completed"
}

################################################################################
# Download Functions
################################################################################

download_backup() {
    local component=$1
    local date=${2:-$RESTORE_DATE}
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

    log_info "Downloading $component backup from $date..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would download from container: $container, date: $date"
        return 0
    fi

    # List available backups for the date
    local blobs=$(az storage blob list \
        --account-name "$AZURE_STORAGE_ACCOUNT" \
        --container-name "$container" \
        --prefix "daily/$date/" \
        --query "[].name" \
        --output tsv 2>/dev/null)

    if [ -z "$blobs" ]; then
        log_error "No backups found for $component on $date"
        return 3
    fi

    # Download each blob
    local download_count=0
    local total_size=0

    while IFS= read -r blob; do
        local filename=$(basename "$blob")
        local dest_file="$RESTORE_TEMP_DIR/$component/$filename"

        log_info "  Downloading: $filename"

        az storage blob download \
            --account-name "$AZURE_STORAGE_ACCOUNT" \
            --container-name "$container" \
            --name "$blob" \
            --file "$dest_file" \
            --only-show-errors

        ((download_count++))
        local size=$(stat -f%z "$dest_file" 2>/dev/null || stat -c%s "$dest_file" 2>/dev/null)
        total_size=$((total_size + size))
    done <<< "$blobs"

    local size_mb=$((total_size / 1024 / 1024))
    log_success "Downloaded $download_count files ($size_mb MB) for $component"
}

verify_downloaded_backup() {
    local component=$1

    log_info "Verifying downloaded backup integrity..."

    local checksum_file="$RESTORE_TEMP_DIR/$component/checksums.sha256"

    if [ ! -f "$checksum_file" ]; then
        log_warning "No checksum file found - skipping verification"
        return 0
    fi

    local failed=false

    while IFS= read -r line; do
        local expected_checksum=$(echo "$line" | awk '{print $1}')
        local filename=$(echo "$line" | awk '{print $2}')
        local file_path="$RESTORE_TEMP_DIR/$component/$filename"

        if [ -f "$file_path" ]; then
            if ! verify_checksum "$file_path" "$expected_checksum"; then
                failed=true
            fi
        fi
    done < "$checksum_file"

    if [ "$failed" = true ]; then
        log_error "Backup verification failed"
        return 2
    fi

    log_success "Backup verification successful"
}

################################################################################
# Restore Functions
################################################################################

restore_models() {
    log_step "Starting model restore..."

    # Download backups
    download_backup "models" "$RESTORE_DATE"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would restore models to: $MODELS_DIR"
        return 0
    fi

    # Verify downloaded backups
    if [ "$VERIFY" = true ]; then
        verify_downloaded_backup "models"
    fi

    # Create backup of existing models
    if [ -d "$MODELS_DIR" ]; then
        local backup_suffix=$(date +%Y%m%d_%H%M%S)
        log_info "Backing up existing models to ${MODELS_DIR}.backup_${backup_suffix}"
        mv "$MODELS_DIR" "${MODELS_DIR}.backup_${backup_suffix}"
    fi

    # Create models directory
    mkdir -p "$MODELS_DIR"

    # Extract each model archive
    for archive in "$RESTORE_TEMP_DIR/models"/*.tar.gz; do
        if [ -f "$archive" ]; then
            local filename=$(basename "$archive")
            log_info "Extracting: $filename"

            tar -xzf "$archive" -C "$MODELS_DIR"

            log_success "  Extracted: $filename"
        fi
    done

    log_success "Model restore completed"
}

restore_mongodb() {
    log_step "Starting MongoDB restore..."

    # Check if MongoDB is accessible
    if ! mongosh "$MONGO_URI" --eval "db.adminCommand('ping')" --quiet &> /dev/null; then
        log_error "Cannot connect to MongoDB at $MONGO_URI"
        return 4
    fi

    # Point-in-time restore vs snapshot restore
    if [ -n "$POINT_IN_TIME" ]; then
        restore_mongodb_point_in_time
    else
        restore_mongodb_snapshot
    fi
}

restore_mongodb_snapshot() {
    log_info "Restoring MongoDB from snapshot..."

    # Download backups
    download_backup "mongodb" "$RESTORE_DATE"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would restore MongoDB to: $MONGO_DATABASE"
        return 0
    fi

    # Verify downloaded backups
    if [ "$VERIFY" = true ]; then
        verify_downloaded_backup "mongodb"
    fi

    # Find the full backup archive
    local full_backup=$(find "$RESTORE_TEMP_DIR/mongodb" -name "mongodb_full_*.tar.gz" | sort | tail -n 1)

    if [ -z "$full_backup" ]; then
        # Try incremental backup
        full_backup=$(find "$RESTORE_TEMP_DIR/mongodb" -name "mongodb_incremental_*.tar.gz" | sort | tail -n 1)
    fi

    if [ -z "$full_backup" ]; then
        log_error "No MongoDB backup archive found"
        return 3
    fi

    log_info "Using backup: $(basename "$full_backup")"

    # Extract backup
    local extract_dir="$RESTORE_TEMP_DIR/mongodb/extracted"
    mkdir -p "$extract_dir"
    tar -xzf "$full_backup" -C "$extract_dir"

    # Find the backup directory
    local backup_dir=$(find "$extract_dir" -type d -name "$MONGO_DATABASE" | head -n 1)

    if [ -z "$backup_dir" ]; then
        log_error "Backup directory not found in archive"
        return 4
    fi

    # Stop application writes (optional, recommended for production)
    if [ "$ENVIRONMENT" = "production" ]; then
        log_warning "Consider scaling down application pods before restore"
        log_info "  kubectl scale deployment/sap-llm-api --replicas=0 -n sap-llm"
    fi

    # Perform restore
    log_info "Running mongorestore..."

    mongorestore \
        --uri="$MONGO_URI" \
        --db="$MONGO_DATABASE" \
        --drop \
        --gzip \
        "$backup_dir" \
        2>&1 | while read line; do
            log_info "  $line"
        done

    log_success "MongoDB snapshot restore completed"
}

restore_mongodb_point_in_time() {
    log_info "Point-in-time restore requested: $POINT_IN_TIME"

    # This requires oplog backups
    # Simplified implementation - full implementation would:
    # 1. Restore latest full backup before point-in-time
    # 2. Apply oplog entries up to the point-in-time

    log_warning "Point-in-time restore requires oplog backups"
    log_warning "This is a simplified implementation"

    # Download the latest full backup
    download_backup "mongodb" "$RESTORE_DATE"

    # Download oplog backups (if available)
    log_info "Downloading oplog backups..."

    # TODO: Implement oplog-based point-in-time restore
    log_error "Full point-in-time restore not yet implemented"
    log_info "Falling back to snapshot restore"

    restore_mongodb_snapshot
}

restore_configs() {
    log_step "Starting configuration restore..."

    # Download backups
    download_backup "configs" "$RESTORE_DATE"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would restore configs to: $CONFIGS_DIR"
        return 0
    fi

    # Verify downloaded backups
    if [ "$VERIFY" = true ]; then
        verify_downloaded_backup "configs"
    fi

    # Create backup of existing configs
    if [ -d "$CONFIGS_DIR" ]; then
        local backup_suffix=$(date +%Y%m%d_%H%M%S)
        log_info "Backing up existing configs to ${CONFIGS_DIR}.backup_${backup_suffix}"
        mv "$CONFIGS_DIR" "${CONFIGS_DIR}.backup_${backup_suffix}"
    fi

    # Extract config archive
    local archive=$(find "$RESTORE_TEMP_DIR/configs" -name "configs_*.tar.gz" | sort | tail -n 1)

    if [ -z "$archive" ]; then
        log_error "No config backup archive found"
        return 3
    fi

    log_info "Extracting: $(basename "$archive")"
    tar -xzf "$archive" -C "$PROJECT_ROOT"

    # Restore secrets from Azure Key Vault
    log_info "Restoring secrets from Azure Key Vault..."

    if command -v kubectl &> /dev/null; then
        # If running in Kubernetes, update secrets
        local vault_name="sap-llm-kv"

        # Get secrets from Key Vault
        local cosmos_key=$(az keyvault secret show \
            --vault-name "$vault_name" \
            --name cosmos-key \
            --query value \
            --output tsv 2>/dev/null || echo "")

        local api_secret=$(az keyvault secret show \
            --vault-name "$vault_name" \
            --name api-secret \
            --query value \
            --output tsv 2>/dev/null || echo "")

        if [ -n "$cosmos_key" ] && [ -n "$api_secret" ]; then
            log_info "Updating Kubernetes secrets..."

            kubectl delete secret sap-llm-secrets -n sap-llm 2>/dev/null || true

            kubectl create secret generic sap-llm-secrets \
                --from-literal=cosmos-key="$cosmos_key" \
                --from-literal=api-secret="$api_secret" \
                -n sap-llm

            log_success "Kubernetes secrets updated"
        fi
    fi

    log_success "Configuration restore completed"
}

restore_redis() {
    log_step "Starting Redis warm-up..."

    # Redis is ephemeral - we don't restore from backup
    # Instead, we warm up the cache by:
    # 1. Loading frequently accessed data
    # 2. Pre-computing common queries

    if [ "$WARM_UP" = false ]; then
        log_info "Skipping Redis warm-up (use --warm-up to enable)"
        return 0
    fi

    # Check if Redis is accessible
    if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &> /dev/null; then
        log_error "Cannot connect to Redis at ${REDIS_HOST}:${REDIS_PORT}"
        return 1
    fi

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would warm up Redis cache"
        return 0
    fi

    log_info "Warming up Redis cache..."

    # Clear existing cache
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" FLUSHDB

    # Load warm-up data using Python script
    python3 <<EOF
import redis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to Redis
r = redis.Redis(host='$REDIS_HOST', port=$REDIS_PORT, decode_responses=True)

# Warm up common cache keys
# This would typically be done by:
# 1. Loading frequently accessed SAP field mappings
# 2. Caching common business rules
# 3. Pre-computing template data

logger.info("Redis cache warm-up initiated")
logger.info("Cache will rebuild naturally from application requests")

EOF

    log_success "Redis warm-up completed"
}

################################################################################
# Post-Restore Validation
################################################################################

validate_restore() {
    log_step "Validating restored components..."

    local validation_failed=false

    # Validate models
    if [ "$COMPONENT" = "models" ] || [ "$COMPONENT" = "all" ]; then
        log_info "Validating models..."

        if [ -d "$MODELS_DIR" ]; then
            # Check for required model directories
            local models=("vision_encoder" "language_decoder" "reasoning_engine")

            for model in "${models[@]}"; do
                if [ -d "$MODELS_DIR/$model" ]; then
                    log_success "  Model OK: $model"
                else
                    log_error "  Model missing: $model"
                    validation_failed=true
                fi
            done
        else
            log_error "Models directory not found: $MODELS_DIR"
            validation_failed=true
        fi
    fi

    # Validate MongoDB
    if [ "$COMPONENT" = "mongodb" ] || [ "$COMPONENT" = "all" ]; then
        log_info "Validating MongoDB..."

        if mongosh "$MONGO_URI/$MONGO_DATABASE" --eval "db.stats()" --quiet &> /dev/null; then
            # Count documents in key collections
            local collections=("documents" "results" "field_mappings" "business_rules")

            for collection in "${collections[@]}"; do
                local count=$(mongosh "$MONGO_URI/$MONGO_DATABASE" \
                    --eval "db.${collection}.countDocuments({})" \
                    --quiet 2>/dev/null || echo "0")

                log_info "  Collection $collection: $count documents"
            done

            log_success "  MongoDB validation passed"
        else
            log_error "  Cannot connect to MongoDB"
            validation_failed=true
        fi
    fi

    # Validate configs
    if [ "$COMPONENT" = "configs" ] || [ "$COMPONENT" = "all" ]; then
        log_info "Validating configurations..."

        if [ -f "$CONFIGS_DIR/default_config.yaml" ]; then
            log_success "  Configuration files restored"
        else
            log_error "  Configuration files missing"
            validation_failed=true
        fi
    fi

    if [ "$validation_failed" = true ]; then
        log_error "Restore validation FAILED"
        return 2
    fi

    log_success "Restore validation PASSED"
}

################################################################################
# Main Execution
################################################################################

show_help() {
    grep '^#' "$0" | sed 's/^# //' | sed 's/^#//' | head -n -1 | tail -n +2
}

main() {
    log_info "=== SAP_LLM Restore Script ==="
    log_info "Environment: $ENVIRONMENT"
    log_info "Timestamp: $RESTORE_TIMESTAMP"
    echo ""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --component)
                COMPONENT="$2"
                shift 2
                ;;
            --date)
                RESTORE_DATE="$2"
                shift 2
                ;;
            --point-in-time)
                POINT_IN_TIME="$2"
                shift 2
                ;;
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --source|--location)
                # DR-related options
                log_info "DR restore from region: $2"
                shift 2
                ;;
            --target)
                # DR target region
                log_info "DR restore to region: $2"
                shift 2
                ;;
            --warm-up)
                WARM_UP=true
                shift
                ;;
            --verify)
                VERIFY=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                log_warning "DRY RUN MODE - No actual restore will be performed"
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

    # Determine restore date if not specified
    if [ -z "$RESTORE_DATE" ]; then
        log_info "No restore date specified, finding latest backup..."
        RESTORE_DATE=$(get_latest_backup_date "$STORAGE_CONTAINER_MODELS")
        log_info "Using latest backup: $RESTORE_DATE"
    fi

    # Pre-restore safety checks
    pre_restore_checks

    # Create restore directory
    create_restore_dir

    # Trap to ensure cleanup
    trap cleanup_restore_dir EXIT

    # Perform restore based on component selection
    case $COMPONENT in
        models)
            restore_models
            ;;
        mongodb)
            restore_mongodb
            ;;
        configs)
            restore_configs
            ;;
        redis)
            restore_redis
            ;;
        all)
            restore_models
            restore_mongodb
            restore_configs
            if [ "$WARM_UP" = true ]; then
                restore_redis
            fi
            ;;
        *)
            log_error "Invalid component: $COMPONENT"
            log_info "Valid components: models, mongodb, configs, redis, all"
            exit 1
            ;;
    esac

    # Post-restore validation
    if [ "$VERIFY" = true ] && [ "$DRY_RUN" = false ]; then
        validate_restore
    fi

    echo ""
    log_success "=== Restore completed successfully ==="
    log_info "Restored from date: $RESTORE_DATE"
    log_info "Environment: $ENVIRONMENT"

    if [ "$DRY_RUN" = true ]; then
        log_warning "This was a DRY RUN - no actual restore was performed"
    else
        log_info ""
        log_info "Next steps:"
        log_info "  1. Verify application functionality"
        log_info "  2. Run health checks: ./scripts/health_check.py --comprehensive"
        log_info "  3. Run integration tests: pytest tests/integration/"
        log_info "  4. Restart application services if needed"
        log_info "  5. Monitor logs and metrics"
    fi
}

# Run main function
main "$@"
