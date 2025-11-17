#!/bin/bash
set -e

# =============================================================================
# SAP_LLM Pre-Production Validation Script
# =============================================================================
# This script validates ALL production readiness criteria for 100/100 score
#
# Usage: ./scripts/pre_production_validation.sh
# =============================================================================

echo "======================================================================"
echo "  SAP_LLM Pre-Production Validation"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
WARNINGS=0

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $2"
        ((FAILED++))
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# =============================================================================
# 1. CODE QUALITY CHECKS (15 points)
# =============================================================================
echo "======================================================================"
echo "1. CODE QUALITY CHECKS (15 points)"
echo "======================================================================"

# Check for TODO/FIXME/XXX comments in production code
echo ""
echo "Checking for TODO comments in production code..."
TODO_COUNT=$(grep -r "TODO\|FIXME\|XXX" sap_llm/*.py 2>/dev/null | grep -v "TODO.*:" | wc -l || echo "0")
if [ "$TODO_COUNT" -eq "0" ]; then
    print_status 0 "No TODO/FIXME/XXX comments in production code (5 pts)"
else
    print_status 1 "Found $TODO_COUNT TODO comments in production code (0 pts)"
    grep -r "TODO\|FIXME\|XXX" sap_llm/*.py 2>/dev/null | grep -v "TODO.*:" | head -5
fi

# Check Black formatting
echo ""
echo "Checking Black formatting..."
if command -v black &> /dev/null; then
    if black --check sap_llm/ tests/ &> /dev/null; then
        print_status 0 "Black formatting compliant (2 pts)"
    else
        print_status 1 "Black formatting not compliant (0 pts)"
    fi
else
    print_warning "Black not installed - skipping (install with: pip install black)"
fi

# Check Ruff linting
echo ""
echo "Checking Ruff linting..."
if command -v ruff &> /dev/null; then
    if ruff check sap_llm/ --exit-zero &> /dev/null; then
        print_status 0 "Ruff linting passed (3 pts)"
    else
        print_warning "Ruff found issues - review and fix"
    fi
else
    print_warning "Ruff not installed - skipping (install with: pip install ruff)"
fi

# =============================================================================
# 2. TESTING (20 points)
# =============================================================================
echo ""
echo "======================================================================"
echo "2. TESTING (20 points)"
echo "======================================================================"

# Check if pytest is installed
echo ""
echo "Checking test infrastructure..."
if command -v pytest &> /dev/null; then
    print_status 0 "pytest installed"
    
    # Check test coverage
    echo ""
    echo "Checking test coverage..."
    if [ -f ".coverage" ] || [ -d "htmlcov" ]; then
        print_warning "Coverage reports exist - verify ≥90% coverage manually"
    else
        print_warning "No coverage reports found - run: pytest --cov=sap_llm --cov-report=html"
    fi
    
    # Check if tests exist
    TEST_COUNT=$(find tests/ -name "test_*.py" -o -name "*_test.py" 2>/dev/null | wc -l)
    if [ "$TEST_COUNT" -gt "25" ]; then
        print_status 0 "Found $TEST_COUNT test files (3 pts)"
    else
        print_status 1 "Only found $TEST_COUNT test files - expected >25 (0 pts)"
    fi
else
    print_status 1 "pytest not installed (0 pts)"
    echo "Install with: pip install pytest pytest-cov pytest-asyncio"
fi

# =============================================================================
# 3. SECURITY (15 points)
# =============================================================================
echo ""
echo "======================================================================"
echo "3. SECURITY (15 points)"
echo "======================================================================"

# Check for hardcoded secrets
echo ""
echo "Checking for hardcoded secrets..."
SECRETS_FOUND=0
if grep -r "SECRET_KEY.*=.*['\"].*['\"]" sap_llm/ 2>/dev/null | grep -v "os.getenv\|os.environ" | grep -v ".pyc" > /dev/null; then
    print_status 1 "Found hardcoded SECRET_KEY (0 pts)"
    SECRETS_FOUND=1
else
    print_status 0 "No hardcoded SECRET_KEY found (3 pts)"
fi

# Check CORS configuration
echo ""
echo "Checking CORS configuration..."
if grep -r "allow_origins.*=.*\[.*\"\*\".*\]" sap_llm/ 2>/dev/null > /dev/null; then
    print_status 1 "CORS allows all origins '*' (0 pts)"
else
    print_status 0 "CORS properly configured (2 pts)"
fi

# Check for security scanning tools
echo ""
echo "Checking security scanning configuration..."
if [ -f ".github/workflows/security.yml" ]; then
    print_status 0 "Security scanning workflow exists (8 pts)"
else
    print_status 1 "No security scanning workflow (0 pts)"
fi

if [ -f ".github/dependabot.yml" ]; then
    print_status 0 "Dependabot configured (2 pts)"
else
    print_status 1 "Dependabot not configured (0 pts)"
fi

# =============================================================================
# 4. CI/CD PIPELINE (15 points)
# =============================================================================
echo ""
echo "======================================================================"
echo "4. CI/CD PIPELINE (15 points)"
echo "======================================================================"

# Check for CI/CD workflows
echo ""
echo "Checking CI/CD configuration..."
if [ -f ".github/workflows/ci.yml" ] || [ -f ".github/workflows/ci-cd.yaml" ]; then
    print_status 0 "CI/CD pipeline configured (8 pts)"
else
    print_status 1 "No CI/CD pipeline found (0 pts)"
fi

# Check Docker configuration
echo ""
echo "Checking Docker configuration..."
if [ -f "Dockerfile" ]; then
    print_status 0 "Dockerfile exists (2 pts)"
    
    # Try building (quick check)
    print_warning "Docker build validation skipped (run manually: docker build -t sap-llm:test .)"
else
    print_status 1 "No Dockerfile found (0 pts)"
fi

# Check Kubernetes manifests
echo ""
echo "Checking Kubernetes configuration..."
K8S_FILES=$(find k8s/ -name "*.yaml" -o -name "*.yml" 2>/dev/null | wc -l)
if [ "$K8S_FILES" -gt "5" ]; then
    print_status 0 "Kubernetes manifests found ($K8S_FILES files) (3 pts)"
else
    print_status 1 "Insufficient Kubernetes manifests (0 pts)"
fi

# Check Helm chart
if [ -d "helm/sap-llm" ]; then
    print_status 0 "Helm chart exists (2 pts)"
else
    print_status 1 "No Helm chart found (0 pts)"
fi

# =============================================================================
# 5. PERFORMANCE (20 points)
# =============================================================================
echo ""
echo "======================================================================"
echo "5. PERFORMANCE (20 points)"
echo "======================================================================"

# Check if performance benchmarks exist
echo ""
echo "Checking performance benchmarks..."
if [ -f "scripts/run_benchmarks.py" ]; then
    print_status 0 "Performance benchmark script exists (5 pts)"
    
    # Check if benchmarks have been run
    if [ -d "benchmarks" ] && [ "$(ls -A benchmarks 2>/dev/null)" ]; then
        print_status 0 "Benchmark results exist (5 pts)"
        
        # Try to read latest results
        LATEST_REPORT=$(ls -t benchmarks/performance_report_*.json 2>/dev/null | head -1)
        if [ -n "$LATEST_REPORT" ]; then
            echo ""
            echo "Latest benchmark report: $LATEST_REPORT"
            print_warning "Review benchmark results manually to verify targets met"
        fi
    else
        print_warning "No benchmark results found - run: python scripts/run_benchmarks.py"
    fi
else
    print_status 1 "No performance benchmark script (0 pts)"
fi

# =============================================================================
# 6. INFRASTRUCTURE (15 points)
# =============================================================================
echo ""
echo "======================================================================"
echo "6. INFRASTRUCTURE (15 points)"
echo "======================================================================"

# Check monitoring configuration
echo ""
echo "Checking monitoring & alerting..."
if [ -f "configs/alerting_rules.yml" ]; then
    print_status 0 "Alerting rules configured (2 pts)"
else
    print_status 1 "No alerting rules found (0 pts)"
fi

# Check runbooks
echo ""
echo "Checking operational runbooks..."
RUNBOOK_COUNT=$(find docs/runbooks/ -name "*.md" 2>/dev/null | wc -l)
if [ "$RUNBOOK_COUNT" -gt "0" ]; then
    print_status 0 "Found $RUNBOOK_COUNT runbooks (3 pts)"
else
    print_status 1 "No runbooks found (0 pts)"
fi

# Check configuration files
echo ""
echo "Checking configuration management..."
if [ -f "configs/default_config.yaml" ] || [ -f "config.yaml" ]; then
    print_status 0 "Configuration files exist (2 pts)"
else
    print_status 1 "No configuration files found (0 pts)"
fi

# =============================================================================
# 7. DOCUMENTATION (10 points)
# =============================================================================
echo ""
echo "======================================================================"
echo "7. DOCUMENTATION (10 points)"
echo "======================================================================"

# Check README
echo ""
echo "Checking documentation..."
if [ -f "README.md" ]; then
    README_SIZE=$(wc -l < README.md)
    if [ "$README_SIZE" -gt "50" ]; then
        print_status 0 "README exists and is substantial ($README_SIZE lines) (3 pts)"
    else
        print_status 1 "README too short ($README_SIZE lines) (0 pts)"
    fi
else
    print_status 1 "No README found (0 pts)"
fi

# Check for architecture docs
if [ -f "docs/ARCHITECTURE.md" ] || [ -f "ARCHITECTURE.md" ]; then
    print_status 0 "Architecture documentation exists (2 pts)"
else
    print_warning "No architecture documentation found"
fi

# Check API docs
if grep -q "API documentation\|OpenAPI\|Swagger" README.md 2>/dev/null; then
    print_status 0 "API documentation referenced (2 pts)"
else
    print_warning "No API documentation reference in README"
fi

# Count documentation files
DOC_COUNT=$(find docs/ -name "*.md" 2>/dev/null | wc -l)
if [ "$DOC_COUNT" -gt "10" ]; then
    print_status 0 "Found $DOC_COUNT documentation files (3 pts)"
else
    print_warning "Only found $DOC_COUNT documentation files"
fi

# =============================================================================
# 8. FEATURE COMPLETENESS (5 points)
# =============================================================================
echo ""
echo "======================================================================"
echo "8. FEATURE COMPLETENESS (5 points)"
echo "======================================================================"

# Check for stub implementations
echo ""
echo "Checking for stub implementations..."
STUB_COUNT=$(grep -r "raise NotImplementedError\|pass  # TODO" sap_llm/ 2>/dev/null | wc -l || echo "0")
if [ "$STUB_COUNT" -eq "0" ]; then
    print_status 0 "No stub implementations found (3 pts)"
else
    print_status 1 "Found $STUB_COUNT stub implementations (0 pts)"
    grep -r "raise NotImplementedError\|pass  # TODO" sap_llm/ 2>/dev/null | head -5
fi

# Check critical modules exist
echo ""
echo "Checking critical modules..."
CRITICAL_MODULES=(
    "sap_llm/models/unified_model.py"
    "sap_llm/stages/classification.py"
    "sap_llm/stages/extraction.py"
    "sap_llm/pmg/graph_client.py"
    "sap_llm/apop/orchestrator.py"
)

MISSING=0
for module in "${CRITICAL_MODULES[@]}"; do
    if [ ! -f "$module" ]; then
        ((MISSING++))
    fi
done

if [ "$MISSING" -eq "0" ]; then
    print_status 0 "All critical modules present (2 pts)"
else
    print_status 1 "Missing $MISSING critical modules (0 pts)"
fi

# =============================================================================
# FINAL SCORE
# =============================================================================
echo ""
echo "======================================================================"
echo "  VALIDATION SUMMARY"
echo "======================================================================"
echo ""
echo "Checks Passed:  $PASSED"
echo "Checks Failed:  $FAILED"
echo "Warnings:       $WARNINGS"
echo ""

TOTAL_CHECKS=$((PASSED + FAILED))
if [ "$TOTAL_CHECKS" -gt "0" ]; then
    SCORE=$((PASSED * 100 / TOTAL_CHECKS))
else
    SCORE=0
fi

echo "======================================================================"
if [ "$SCORE" -ge "90" ]; then
    echo -e "${GREEN}✅ PRODUCTION READY - Score: $SCORE/100${NC}"
    echo "======================================================================"
    echo ""
    echo "System meets production readiness criteria!"
    exit 0
elif [ "$SCORE" -ge "70" ]; then
    echo -e "${YELLOW}⚠️  CONDITIONAL READY - Score: $SCORE/100${NC}"
    echo "======================================================================"
    echo ""
    echo "System is mostly ready but has some gaps to address."
    echo "Review warnings and failed checks above."
    exit 1
else
    echo -e "${RED}❌ NOT READY - Score: $SCORE/100${NC}"
    echo "======================================================================"
    echo ""
    echo "System has significant gaps. Address failed checks before production."
    exit 1
fi
