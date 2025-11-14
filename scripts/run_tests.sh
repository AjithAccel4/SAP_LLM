#!/bin/bash

# Test runner script for SAP_LLM
# Provides convenient test execution with various options

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
TEST_TYPE="${1:-all}"

print_info "Running tests: $TEST_TYPE"

# Ensure we're in the project root
cd "$(dirname "$0")/.."

case $TEST_TYPE in
    all)
        print_info "Running all tests..."
        pytest tests/ -v
        ;;

    unit)
        print_info "Running unit tests..."
        pytest tests/ -v -m unit
        ;;

    integration)
        print_info "Running integration tests..."
        pytest tests/ -v -m integration
        ;;

    api)
        print_info "Running API tests..."
        pytest tests/ -v -m api
        ;;

    performance)
        print_info "Running performance benchmarks..."
        pytest tests/ -v -m performance --benchmark-only
        ;;

    fast)
        print_info "Running fast tests (excluding slow tests)..."
        pytest tests/ -v -m "not slow"
        ;;

    coverage)
        print_info "Running tests with coverage report..."
        pytest tests/ -v --cov=sap_llm --cov-report=html --cov-report=term
        print_info "Coverage report generated in htmlcov/index.html"
        ;;

    watch)
        print_info "Running tests in watch mode..."
        pytest-watch tests/ -- -v
        ;;

    ci)
        print_info "Running CI test suite..."
        pytest tests/ -v \
            -m "not slow and not gpu and not requires_models" \
            --cov=sap_llm \
            --cov-report=xml \
            --junitxml=test-results.xml
        ;;

    specific)
        if [ -z "$2" ]; then
            print_error "Please specify test file or function"
            print_info "Usage: $0 specific tests/test_file.py::test_function"
            exit 1
        fi
        print_info "Running specific test: $2"
        pytest "$2" -v
        ;;

    clean)
        print_info "Cleaning test artifacts..."
        rm -rf .pytest_cache htmlcov .coverage test-results.xml
        find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        print_info "Clean complete"
        ;;

    help)
        echo ""
        echo "SAP_LLM Test Runner"
        echo ""
        echo "Usage: $0 [TEST_TYPE]"
        echo ""
        echo "Test Types:"
        echo "  all           - Run all tests (default)"
        echo "  unit          - Run unit tests only"
        echo "  integration   - Run integration tests only"
        echo "  api           - Run API tests only"
        echo "  performance   - Run performance benchmarks"
        echo "  fast          - Run fast tests (exclude slow tests)"
        echo "  coverage      - Run tests with coverage report"
        echo "  watch         - Run tests in watch mode (requires pytest-watch)"
        echo "  ci            - Run CI test suite (fast tests with coverage)"
        echo "  specific      - Run specific test file/function"
        echo "  clean         - Clean test artifacts"
        echo "  help          - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 unit"
        echo "  $0 specific tests/test_utils.py::TestHash"
        echo "  $0 coverage"
        echo ""
        ;;

    *)
        print_error "Unknown test type: $TEST_TYPE"
        echo ""
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

print_info "Tests complete!"
