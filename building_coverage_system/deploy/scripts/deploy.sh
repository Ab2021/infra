#!/bin/bash

# Building Coverage System Deployment Script
# This script handles deployment to various environments

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DEPLOY_DIR="$PROJECT_ROOT/deploy"

# Default values
ENVIRONMENT="development"
BUILD_ARGS=""
DOCKER_REGISTRY=""
IMAGE_TAG="latest"
NAMESPACE="building-coverage"
KUBECTL_CONTEXT=""
DRY_RUN=false
SKIP_BUILD=false
SKIP_TESTS=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Building Coverage System Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV       Target environment (development, staging, production)
    -t, --tag TAG              Docker image tag (default: latest)
    -r, --registry REGISTRY    Docker registry URL
    -n, --namespace NAMESPACE  Kubernetes namespace (default: building-coverage)
    -c, --context CONTEXT      Kubernetes context to use
    -d, --dry-run              Perform dry run without actual deployment
    -s, --skip-build           Skip Docker image build
    --skip-tests               Skip running tests
    -v, --verbose              Enable verbose output
    -h, --help                 Show this help message

EXAMPLES:
    # Deploy to development environment
    $0 --environment development

    # Deploy to production with specific tag
    $0 --environment production --tag v1.2.3 --registry myregistry.com

    # Dry run deployment
    $0 --environment staging --dry-run

    # Deploy with custom Kubernetes context
    $0 --environment production --context prod-cluster

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -c|--context)
                KUBECTL_CONTEXT="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -s|--skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
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
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        development|staging|production)
            log_info "Deploying to $ENVIRONMENT environment"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl context
    if [[ -n "$KUBECTL_CONTEXT" ]]; then
        if ! kubectl config use-context "$KUBECTL_CONTEXT" &> /dev/null; then
            log_error "Invalid kubectl context: $KUBECTL_CONTEXT"
            exit 1
        fi
    fi
    
    # Check if Kubernetes cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Load environment configuration
load_environment_config() {
    local env_file="$DEPLOY_DIR/config/${ENVIRONMENT}.env"
    
    if [[ -f "$env_file" ]]; then
        log_info "Loading environment configuration from $env_file"
        set -a
        source "$env_file"
        set +a
    else
        log_warning "Environment file not found: $env_file"
    fi
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" = true ]]; then
        log_warning "Skipping tests"
        return 0
    fi
    
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run unit tests
    if ! python -m pytest tests/ -v; then
        log_error "Unit tests failed"
        exit 1
    fi
    
    # Run linting
    if command -v flake8 &> /dev/null; then
        if ! flake8 building_coverage_system/; then
            log_error "Linting failed"
            exit 1
        fi
    fi
    
    # Run type checking
    if command -v mypy &> /dev/null; then
        if ! mypy building_coverage_system/; then
            log_warning "Type checking failed (non-blocking)"
        fi
    fi
    
    log_success "Tests passed"
}

# Build Docker image
build_docker_image() {
    if [[ "$SKIP_BUILD" = true ]]; then
        log_warning "Skipping Docker build"
        return 0
    fi
    
    log_info "Building Docker image..."
    
    local image_name="building-coverage-system"
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        image_name="$DOCKER_REGISTRY/building-coverage-system"
    fi
    
    local full_image_name="$image_name:$IMAGE_TAG"
    
    # Build arguments based on environment
    case $ENVIRONMENT in
        development)
            BUILD_ARGS="--build-arg BUILD_ENV=development"
            ;;
        staging)
            BUILD_ARGS="--build-arg BUILD_ENV=staging"
            ;;
        production)
            BUILD_ARGS="--build-arg BUILD_ENV=production"
            ;;
    esac
    
    cd "$PROJECT_ROOT"
    
    if [[ "$VERBOSE" = true ]]; then
        docker build $BUILD_ARGS -t "$full_image_name" -f deploy/Dockerfile .
    else
        docker build $BUILD_ARGS -t "$full_image_name" -f deploy/Dockerfile . > /dev/null 2>&1
    fi
    
    log_success "Docker image built: $full_image_name"
    
    # Push to registry if specified
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        log_info "Pushing image to registry..."
        docker push "$full_image_name"
        log_success "Image pushed to registry"
    fi
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    local kubectl_args=""
    if [[ "$DRY_RUN" = true ]]; then
        kubectl_args="--dry-run=client"
        log_warning "Performing dry run - no actual changes will be made"
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" $kubectl_args --dry-run=client -o yaml | kubectl apply -f - $kubectl_args
    
    # Apply Kubernetes manifests
    local k8s_dir="$DEPLOY_DIR/kubernetes"
    
    # Apply in order
    local manifests=(
        "namespace.yaml"
        "configmap.yaml"
        "secret.yaml"
        "pvc.yaml"
        "deployment.yaml"
        "service.yaml"
        "ingress.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        local manifest_path="$k8s_dir/$manifest"
        if [[ -f "$manifest_path" ]]; then
            log_info "Applying $manifest..."
            
            # Replace placeholders in manifests
            local temp_manifest="/tmp/$(basename "$manifest")"
            envsubst < "$manifest_path" > "$temp_manifest"
            
            kubectl apply -f "$temp_manifest" $kubectl_args -n "$NAMESPACE"
            rm -f "$temp_manifest"
        else
            log_warning "Manifest not found: $manifest_path"
        fi
    done
    
    if [[ "$DRY_RUN" = false ]]; then
        log_info "Waiting for deployment to be ready..."
        kubectl rollout status deployment/building-coverage-app -n "$NAMESPACE" --timeout=300s
        log_success "Deployment completed successfully"
    else
        log_success "Dry run completed successfully"
    fi
}

# Deploy using Docker Compose (for development)
deploy_with_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "$DEPLOY_DIR"
    
    local compose_args=""
    if [[ "$VERBOSE" = true ]]; then
        compose_args="--verbose"
    fi
    
    # Set environment variables
    export BUILD_ENV="$ENVIRONMENT"
    export APP_VERSION="$IMAGE_TAG"
    
    if [[ "$DRY_RUN" = true ]]; then
        log_warning "Dry run mode - showing Docker Compose configuration"
        docker-compose config
    else
        # Build and start services
        docker-compose build $compose_args
        docker-compose up -d $compose_args
        
        log_info "Waiting for services to be ready..."
        sleep 10
        
        # Check service health
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "Application is healthy and ready"
        else
            log_warning "Application health check failed"
        fi
    fi
}

# Post-deployment verification
verify_deployment() {
    if [[ "$DRY_RUN" = true ]]; then
        return 0
    fi
    
    log_info "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n "$NAMESPACE" -l app=building-coverage-system
    
    # Check service endpoints
    kubectl get services -n "$NAMESPACE"
    
    # Perform health check
    local app_service="building-coverage-app-service"
    if kubectl get service "$app_service" -n "$NAMESPACE" &> /dev/null; then
        log_info "Performing health check..."
        
        # Port forward for health check
        kubectl port-forward service/"$app_service" 8080:8000 -n "$NAMESPACE" &
        local port_forward_pid=$!
        
        sleep 5
        
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Health check passed"
        else
            log_warning "Health check failed"
        fi
        
        kill $port_forward_pid 2>/dev/null || true
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup logic here
}

# Main deployment function
main() {
    log_info "Starting Building Coverage System deployment..."
    
    parse_args "$@"
    validate_environment
    check_prerequisites
    load_environment_config
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    run_tests
    build_docker_image
    
    # Choose deployment method based on environment
    case $ENVIRONMENT in
        development)
            deploy_with_docker_compose
            ;;
        staging|production)
            deploy_to_kubernetes
            verify_deployment
            ;;
    esac
    
    log_success "Deployment completed successfully!"
    
    # Show deployment information
    cat << EOF

Deployment Summary:
  Environment: $ENVIRONMENT
  Image Tag: $IMAGE_TAG
  Namespace: $NAMESPACE
  Registry: ${DOCKER_REGISTRY:-"local"}
  Dry Run: $DRY_RUN

EOF
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi