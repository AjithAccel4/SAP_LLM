#!/usr/bin/env python3
"""
Verify test suite completeness for SAP_LLM.
"""

import os
from pathlib import Path

# Expected test files
EXPECTED_TESTS = {
    "unit": [
        "test_models.py",
        "test_stages.py",
        "test_apop.py",
        "test_pmg.py",
        "test_shwl.py",
        "test_knowledge_base.py",
        "test_optimization.py",
    ],
    "integration": [
        "test_end_to_end.py",
    ],
    "performance": [
        "test_throughput.py",
        "test_latency.py",
        "test_memory.py",
        "test_gpu_utilization.py",
    ],
    "fixtures": [
        "sample_documents.py",
        "mock_data.py",
        "__init__.py",
    ]
}

def verify_tests():
    """Verify all expected test files exist."""
    tests_dir = Path("/home/user/SAP_LLM/tests")
    
    print("=" * 60)
    print("SAP_LLM Test Suite Verification")
    print("=" * 60)
    print()
    
    all_present = True
    
    for category, files in EXPECTED_TESTS.items():
        print(f"{category.upper()} Tests:")
        category_dir = tests_dir / category
        
        for test_file in files:
            test_path = category_dir / test_file
            if test_path.exists():
                size = test_path.stat().st_size
                print(f"  ✓ {test_file} ({size:,} bytes)")
            else:
                print(f"  ✗ {test_file} MISSING")
                all_present = False
        print()
    
    # Check configuration files
    print("Configuration Files:")
    config_files = [
        "pytest.ini",
        "run_tests.sh",
        "TESTING_SUMMARY.md",
    ]
    
    for config_file in config_files:
        config_path = Path("/home/user/SAP_LLM") / config_file
        if config_path.exists():
            size = config_path.stat().st_size
            print(f"  ✓ {config_file} ({size:,} bytes)")
        else:
            print(f"  ✗ {config_file} MISSING")
            all_present = False
    
    print()
    print("=" * 60)
    if all_present:
        print("✓ All test files present!")
    else:
        print("✗ Some test files are missing!")
    print("=" * 60)
    
    return all_present

if __name__ == "__main__":
    verify_tests()
