#!/usr/bin/env python3
"""
Dependency Verification Script

This script verifies that all dependencies are correctly installed and compatible.
It checks:
1. All imports resolve correctly
2. Version compatibility
3. Conflicting dependencies
4. License compatibility (optional)
5. Critical package functionality

Usage:
    python scripts/verify_dependencies.py [--verbose] [--check-licenses]
"""

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Critical packages that must be importable
CRITICAL_PACKAGES = {
    # Core ML frameworks
    "torch": "2.1.0",
    "transformers": "4.35.2",
    "accelerate": "0.25.0",
    "datasets": "2.15.0",

    # API frameworks
    "fastapi": "0.105.0",
    "uvicorn": "0.25.0",
    "pydantic": "2.5.2",

    # Databases
    "redis": "5.0.1",
    "pymongo": "4.6.1",
    "motor": "3.3.2",

    # Azure SDKs
    "azure.cosmos": "4.5.1",
    "azure.storage.blob": "12.19.0",

    # Utilities
    "numpy": "1.26.2",
    "pandas": "2.1.4",
}

# Import mappings for packages with different import names
IMPORT_MAPPINGS = {
    "pillow": "PIL",
    "pyyaml": "yaml",
    "opencv-python": "cv2",
    "opencv-contrib-python": "cv2",
    "beautifulsoup4": "bs4",
    "scikit-learn": "sklearn",
    "python-dotenv": "dotenv",
    "azure-cosmos": "azure.cosmos",
    "azure-storage-blob": "azure.storage.blob",
    "azure-servicebus": "azure.servicebus",
    "azure-identity": "azure.identity",
    "openai-whisper": "whisper",
    "sentence-transformers": "sentence_transformers",
}

# Restricted licenses (require manual review)
RESTRICTED_LICENSES = {
    "GPL-2.0",
    "GPL-3.0",
    "AGPL-3.0",
    "LGPL-2.0",
    "LGPL-3.0",
}


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{text.center(80)}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓{RESET} {text}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{RED}✗{RESET} {text}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}⚠{RESET} {text}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{BLUE}ℹ{RESET} {text}")


def get_package_name(import_name: str) -> str:
    """Convert import name to package name."""
    # Reverse mapping
    for pkg_name, imp_name in IMPORT_MAPPINGS.items():
        if imp_name == import_name:
            return pkg_name
    return import_name.replace("_", "-")


def check_import(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package can be imported.

    Args:
        package_name: The package name as listed in requirements
        import_name: The actual import name (if different)

    Returns:
        Tuple of (success, message)
    """
    if import_name is None:
        import_name = IMPORT_MAPPINGS.get(package_name, package_name)

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, f"{import_name} (version: {version})"
    except ImportError as e:
        return False, f"{import_name}: {str(e)}"
    except Exception as e:
        return False, f"{import_name}: Unexpected error - {str(e)}"


def check_version(package_name: str, expected_version: str) -> Tuple[bool, str]:
    """
    Check if installed version matches expected version.

    Args:
        package_name: The package name
        expected_version: The expected version

    Returns:
        Tuple of (matches, message)
    """
    try:
        installed_version = importlib.metadata.version(package_name)
        matches = installed_version == expected_version

        if matches:
            return True, f"{package_name}: {installed_version}"
        else:
            return False, f"{package_name}: Expected {expected_version}, got {installed_version}"
    except importlib.metadata.PackageNotFoundError:
        return False, f"{package_name}: Not installed"


def check_conflicts() -> Tuple[bool, List[str]]:
    """
    Check for dependency conflicts.

    Returns:
        Tuple of (has_conflicts, conflict_messages)
    """
    try:
        import pkg_resources

        conflicts = []
        for dist in pkg_resources.working_set:
            try:
                dist.check_version_conflict()
            except pkg_resources.VersionConflict as e:
                conflicts.append(str(e))

        return len(conflicts) > 0, conflicts
    except Exception as e:
        return True, [f"Error checking conflicts: {str(e)}"]


def check_licenses(verbose: bool = False) -> Tuple[List[str], Dict[str, str]]:
    """
    Check package licenses.

    Args:
        verbose: If True, print all licenses

    Returns:
        Tuple of (restricted_packages, all_licenses)
    """
    try:
        import pkg_resources

        restricted = []
        all_licenses = {}

        for dist in pkg_resources.working_set:
            try:
                metadata = dist.get_metadata("METADATA")
                license_info = "Unknown"

                for line in metadata.split("\n"):
                    if line.startswith("License:"):
                        license_info = line.split(":", 1)[1].strip()
                        break

                all_licenses[dist.project_name] = license_info

                # Check if license is restricted
                for restricted_license in RESTRICTED_LICENSES:
                    if restricted_license.lower() in license_info.lower():
                        restricted.append(f"{dist.project_name}: {license_info}")
            except Exception:
                all_licenses[dist.project_name] = "Unknown"

        return restricted, all_licenses
    except Exception as e:
        return [], {"error": str(e)}


def test_critical_functionality() -> Tuple[bool, List[str]]:
    """
    Test critical package functionality.

    Returns:
        Tuple of (success, messages)
    """
    messages = []
    all_success = True

    # Test PyTorch
    try:
        import torch

        # Test tensor creation
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert tensor.shape == (3,), "PyTorch tensor creation failed"

        # Check CUDA availability (not required, just informational)
        if torch.cuda.is_available():
            messages.append(f"PyTorch: CUDA available (devices: {torch.cuda.device_count()})")
        else:
            messages.append("PyTorch: CPU only")

        messages.append("PyTorch: Basic functionality OK")
    except Exception as e:
        all_success = False
        messages.append(f"PyTorch: Functionality test failed - {str(e)}")

    # Test Transformers
    try:
        from transformers import AutoTokenizer

        # Just verify we can import the class
        messages.append("Transformers: Import OK")
    except Exception as e:
        all_success = False
        messages.append(f"Transformers: Import failed - {str(e)}")

    # Test FastAPI
    try:
        from fastapi import FastAPI

        app = FastAPI()
        messages.append("FastAPI: App creation OK")
    except Exception as e:
        all_success = False
        messages.append(f"FastAPI: App creation failed - {str(e)}")

    # Test NumPy
    try:
        import numpy as np

        arr = np.array([1, 2, 3])
        assert arr.sum() == 6, "NumPy array operation failed"
        messages.append("NumPy: Basic operations OK")
    except Exception as e:
        all_success = False
        messages.append(f"NumPy: Operations failed - {str(e)}")

    # Test Pandas
    try:
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        assert len(df) == 3, "Pandas DataFrame creation failed"
        messages.append("Pandas: DataFrame operations OK")
    except Exception as e:
        all_success = False
        messages.append(f"Pandas: Operations failed - {str(e)}")

    return all_success, messages


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(
        description="Verify SAP LLM dependencies"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--check-licenses",
        action="store_true",
        help="Check package licenses"
    )
    args = parser.parse_args()

    print_header("SAP LLM Dependency Verification")

    # Track overall success
    overall_success = True

    # 1. Check Python version
    print_info("Checking Python version...")
    py_version = sys.version_info
    if 3 <= py_version.major and 9 <= py_version.minor <= 11:
        print_success(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print_error(f"Python {py_version.major}.{py_version.minor} - Expected 3.9, 3.10, or 3.11")
        overall_success = False

    # 2. Check critical package imports
    print_header("Checking Critical Package Imports")

    import_failures = []
    for package_name in CRITICAL_PACKAGES.keys():
        import_name = IMPORT_MAPPINGS.get(package_name, package_name)
        success, message = check_import(package_name, import_name)

        if success:
            if args.verbose:
                print_success(message)
        else:
            print_error(message)
            import_failures.append(package_name)
            overall_success = False

    if not import_failures:
        print_success(f"All {len(CRITICAL_PACKAGES)} critical packages importable")
    else:
        print_error(f"{len(import_failures)} package(s) failed to import")

    # 3. Check version compatibility
    print_header("Checking Version Compatibility")

    version_mismatches = []
    for package_name, expected_version in CRITICAL_PACKAGES.items():
        matches, message = check_version(package_name, expected_version)

        if matches:
            if args.verbose:
                print_success(message)
        else:
            print_warning(message)
            version_mismatches.append(package_name)

    if not version_mismatches:
        print_success(f"All {len(CRITICAL_PACKAGES)} versions match requirements")
    else:
        print_warning(f"{len(version_mismatches)} version mismatch(es) found")

    # 4. Check for conflicts
    print_header("Checking Dependency Conflicts")

    has_conflicts, conflicts = check_conflicts()

    if not has_conflicts:
        print_success("No dependency conflicts detected")
    else:
        print_error("Dependency conflicts detected:")
        for conflict in conflicts:
            print(f"  - {conflict}")
        overall_success = False

    # 5. Check licenses (if requested)
    if args.check_licenses:
        print_header("Checking License Compatibility")

        restricted, all_licenses = check_licenses(args.verbose)

        if restricted:
            print_warning("Restricted licenses found (require manual review):")
            for item in restricted:
                print(f"  - {item}")
        else:
            print_success("No restricted licenses detected")

        if args.verbose:
            print_info("All licenses:")
            for package, license_info in sorted(all_licenses.items()):
                print(f"  - {package}: {license_info}")

    # 6. Test critical functionality
    print_header("Testing Critical Functionality")

    func_success, func_messages = test_critical_functionality()

    for message in func_messages:
        if "failed" in message.lower() or "error" in message.lower():
            print_error(message)
        elif "OK" in message:
            print_success(message)
        else:
            print_info(message)

    if not func_success:
        overall_success = False

    # 7. Summary
    print_header("Verification Summary")

    if overall_success:
        print_success("All dependency checks passed!")
        print_info("Dependencies are correctly installed and compatible.")
        return 0
    else:
        print_error("Some dependency checks failed!")
        print_warning("Please review the errors above and fix any issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
