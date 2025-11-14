#!/bin/bash

# SAP_LLM Deployment Script
# Deploys SAP_LLM to Kubernetes cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if we can connect to cluster
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

print_info "Connected to Kubernetes cluster"
kubectl cluster-info

# Parse arguments
DEPLOY_TYPE="${1:-all}"
NAMESPACE="sap-llm"

print_info "Deployment type: $DEPLOY_TYPE"
print_info "Namespace: $NAMESPACE"

# Function to deploy infrastructure
deploy_infrastructure() {
    print_info "Deploying infrastructure components..."

    # Create namespace
    print_info "Creating namespace..."
    kubectl apply -f kubernetes/namespace.yaml

    # Create configmaps
    print_info "Creating configmaps..."
    kubectl apply -f kubernetes/configmap.yaml

    # Create PVCs
    print_info "Creating persistent volume claims..."
    kubectl apply -f kubernetes/pvc.yaml

    # Check if secrets exist
    if kubectl get secret sap-llm-secrets -n $NAMESPACE &> /dev/null; then
        print_info "Secrets already exist, skipping creation"
    else
        print_warning "Secrets not found! Please create secrets manually:"
        print_warning "kubectl apply -f kubernetes/secrets.yaml"
        print_warning "Or use: kubectl create secret generic sap-llm-secrets --from-env-file=.env -n $NAMESPACE"
    fi
}

# Function to deploy database services
deploy_databases() {
    print_info "Deploying database services..."

    # Deploy Redis
    print_info "Deploying Redis..."
    kubectl apply -f kubernetes/redis-deployment.yaml

    # Deploy MongoDB
    print_info "Deploying MongoDB..."
    kubectl apply -f kubernetes/mongo-deployment.yaml

    # Wait for databases to be ready
    print_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s || true
    kubectl wait --for=condition=ready pod -l app=mongo -n $NAMESPACE --timeout=300s || true
}

# Function to deploy main application
deploy_application() {
    print_info "Deploying SAP_LLM application..."

    # Deploy main application
    print_info "Deploying SAP_LLM API..."
    kubectl apply -f kubernetes/deployment.yaml

    # Create services
    print_info "Creating services..."
    kubectl apply -f kubernetes/service.yaml

    # Deploy ingress
    print_info "Deploying ingress..."
    kubectl apply -f kubernetes/ingress.yaml

    # Deploy HPA
    print_info "Deploying horizontal pod autoscaler..."
    kubectl apply -f kubernetes/hpa.yaml

    # Wait for deployment to be ready
    print_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available deployment/sap-llm-api -n $NAMESPACE --timeout=600s || true
}

# Function to check deployment status
check_status() {
    print_info "Checking deployment status..."

    echo ""
    print_info "Namespace:"
    kubectl get namespace $NAMESPACE

    echo ""
    print_info "Pods:"
    kubectl get pods -n $NAMESPACE -o wide

    echo ""
    print_info "Services:"
    kubectl get services -n $NAMESPACE

    echo ""
    print_info "Ingress:"
    kubectl get ingress -n $NAMESPACE

    echo ""
    print_info "PVCs:"
    kubectl get pvc -n $NAMESPACE

    echo ""
    print_info "HPAs:"
    kubectl get hpa -n $NAMESPACE
}

# Function to get logs
get_logs() {
    POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=sap-llm-api -o jsonpath='{.items[0].metadata.name}')
    print_info "Getting logs from pod: $POD_NAME"
    kubectl logs -n $NAMESPACE $POD_NAME --tail=100 -f
}

# Function to run database migrations
run_migrations() {
    print_info "Running database migrations..."
    # TODO: Implement migrations if needed
    print_info "No migrations to run"
}

# Function to undeploy everything
undeploy() {
    print_warning "Undeploying SAP_LLM..."
    read -p "Are you sure you want to delete all resources? (yes/no): " -r
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        kubectl delete -f kubernetes/hpa.yaml || true
        kubectl delete -f kubernetes/ingress.yaml || true
        kubectl delete -f kubernetes/service.yaml || true
        kubectl delete -f kubernetes/deployment.yaml || true
        kubectl delete -f kubernetes/mongo-deployment.yaml || true
        kubectl delete -f kubernetes/redis-deployment.yaml || true
        kubectl delete -f kubernetes/pvc.yaml || true
        kubectl delete -f kubernetes/configmap.yaml || true
        kubectl delete -f kubernetes/namespace.yaml || true
        print_info "Undeployment complete"
    else
        print_info "Undeployment cancelled"
    fi
}

# Main deployment logic
case $DEPLOY_TYPE in
    all)
        print_info "Deploying full stack..."
        deploy_infrastructure
        deploy_databases
        sleep 10  # Wait for databases to initialize
        deploy_application
        check_status
        print_info "Deployment complete!"
        ;;

    infrastructure)
        deploy_infrastructure
        ;;

    databases)
        deploy_databases
        ;;

    application)
        deploy_application
        ;;

    status)
        check_status
        ;;

    logs)
        get_logs
        ;;

    migrations)
        run_migrations
        ;;

    undeploy)
        undeploy
        ;;

    *)
        print_error "Unknown deployment type: $DEPLOY_TYPE"
        echo ""
        echo "Usage: $0 [all|infrastructure|databases|application|status|logs|migrations|undeploy]"
        echo ""
        echo "Options:"
        echo "  all            - Deploy everything (default)"
        echo "  infrastructure - Deploy namespace, configmaps, PVCs"
        echo "  databases      - Deploy Redis and MongoDB"
        echo "  application    - Deploy SAP_LLM API"
        echo "  status         - Check deployment status"
        echo "  logs           - Get application logs"
        echo "  migrations     - Run database migrations"
        echo "  undeploy       - Remove all resources"
        exit 1
        ;;
esac

print_info "Done!"
