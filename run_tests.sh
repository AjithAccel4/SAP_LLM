#!/bin/bash
# Test runner script for SAP_LLM

set -e

echo "========================================"
echo "SAP_LLM Test Suite Runner"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
RUN_ALL=false
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_PERFORMANCE=false
GENERATE_COVERAGE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --unit)
            RUN_UNIT=true
            shift
            ;;
        --integration)
            RUN_INTEGRATION=true
            shift
            ;;
        --performance)
            RUN_PERFORMANCE=true
            shift
            ;;
        --no-coverage)
            GENERATE_COVERAGE=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--all|--unit|--integration|--performance] [--no-coverage]"
            exit 1
            ;;
    esac
done

# If no specific test type specified, run all
if [ "$RUN_ALL" = false ] && [ "$RUN_UNIT" = false ] && [ "$RUN_INTEGRATION" = false ] && [ "$RUN_PERFORMANCE" = false ]; then
    RUN_ALL=true
fi

# Prepare coverage options
COVERAGE_OPTS=""
if [ "$GENERATE_COVERAGE" = true ]; then
    COVERAGE_OPTS="--cov=sap_llm --cov-report=html --cov-report=term-missing --cov-report=xml"
fi

# Run unit tests
if [ "$RUN_ALL" = true ] || [ "$RUN_UNIT" = true ]; then
    echo -e "${GREEN}Running Unit Tests...${NC}"
    pytest tests/unit/ -m unit $COVERAGE_OPTS -v
    echo ""
fi

# Run integration tests
if [ "$RUN_ALL" = true ] || [ "$RUN_INTEGRATION" = true ]; then
    echo -e "${GREEN}Running Integration Tests...${NC}"
    pytest tests/integration/ -m integration $COVERAGE_OPTS -v
    echo ""
fi

# Run performance tests
if [ "$RUN_ALL" = true ] || [ "$RUN_PERFORMANCE" = true ]; then
    echo -e "${YELLOW}Running Performance Tests...${NC}"
    pytest tests/performance/ -m performance -v
    echo ""
fi

# Generate coverage report
if [ "$GENERATE_COVERAGE" = true ]; then
    echo -e "${GREEN}Generating Coverage Report...${NC}"
    coverage report
    echo ""
    echo -e "${GREEN}HTML coverage report generated at: htmlcov/index.html${NC}"
    echo -e "${GREEN}XML coverage report generated at: coverage.xml${NC}"
fi

echo ""
echo -e "${GREEN}========================================"
echo "Test Suite Complete!"
echo "========================================${NC}"
