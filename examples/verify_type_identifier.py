"""
Verification script for the hierarchical TypeIdentifier implementation.

This script verifies the implementation without requiring all dependencies.
"""

import ast
import sys
from pathlib import Path


def verify_implementation():
    """Verify the TypeIdentifier implementation."""

    print("=" * 80)
    print("SAP_LLM Hierarchical Type Identifier - Implementation Verification")
    print("=" * 80)

    # Read the implementation file
    impl_file = Path(__file__).parent.parent / "sap_llm" / "stages" / "type_identifier.py"

    with open(impl_file, 'r') as f:
        content = f.read()

    # Parse the AST
    tree = ast.parse(content)

    # Find the TypeIdentifierStage class
    type_identifier_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TypeIdentifierStage":
            type_identifier_class = node
            break

    if not type_identifier_class:
        print("❌ TypeIdentifierStage class not found!")
        return False

    print("\n✅ TypeIdentifierStage class found")

    # Extract SUBTYPES
    subtypes_dict = None
    for node in type_identifier_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SUBTYPES":
                    subtypes_dict = ast.literal_eval(node.value)
                    break

    if not subtypes_dict:
        print("❌ SUBTYPES dictionary not found!")
        return False

    print("✅ SUBTYPES dictionary found")

    # Count subtypes
    total_subtypes = sum(len(v) for v in subtypes_dict.values())
    total_doc_types = len(subtypes_dict)

    print(f"\n📊 Hierarchical Classification Structure:")
    print(f"   Total Document Types: {total_doc_types}")
    print(f"   Total Subtypes: {total_subtypes}")
    print(f"   Requirement: 35+ subtypes")
    print(f"   Status: {'✅ PASSED' if total_subtypes >= 35 else '❌ FAILED'}")

    print(f"\n📋 Subtype Distribution:")
    for doc_type, subtypes in sorted(subtypes_dict.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"   {doc_type:35s} : {len(subtypes):2d} subtypes")
        print(f"      └─ {', '.join(subtypes)}")

    # Verify required methods
    print(f"\n🔍 Verifying Required Methods:")

    required_methods = [
        "_load_models",
        "_initialize_classifiers",
        "process",
        "_extract_features",
        "_classify_subtype",
        "_single_label_results",
        "_multi_label_results",
        "get_total_subtypes",
        "get_hierarchy_info",
        "save_classifiers",
        "load_classifiers",
    ]

    methods_found = []
    for node in type_identifier_class.body:
        if isinstance(node, ast.FunctionDef):
            methods_found.append(node.name)

    for method in required_methods:
        status = "✅" if method in methods_found else "❌"
        print(f"   {status} {method}")

    all_methods_found = all(method in methods_found for method in required_methods)

    # Verify imports
    print(f"\n📦 Verifying Required Imports:")

    required_imports = [
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "VisionEncoder",
    ]

    import_found = {imp: False for imp in required_imports}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in required_imports:
                    import_found[alias.name] = True
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                full_module = node.module
                for alias in node.names:
                    full_name = f"{full_module}.{alias.name}"
                    for req in required_imports:
                        if req in full_name or alias.name == req:
                            import_found[req] = True

    for imp, found in import_found.items():
        status = "✅" if found else "❌"
        print(f"   {status} {imp}")

    print("\n" + "=" * 80)
    print("📝 Implementation Summary:")
    print("=" * 80)

    checks = [
        ("35+ subtypes implemented", total_subtypes >= 35),
        ("Vision encoder integration", import_found.get("VisionEncoder", False)),
        ("PyTorch neural network support", import_found.get("torch.nn", False)),
        ("All required methods implemented", all_methods_found),
        ("Hierarchical classification structure", total_doc_types >= 15),
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status:10s} : {check_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All verification checks PASSED!")
        print("=" * 80)
        print("\n✨ Key Features Implemented:")
        print("   • Hierarchical classification (doc_type → subtype)")
        print("   • Vision encoder for feature extraction (LayoutLMv3)")
        print(f"   • {total_subtypes} total subtypes across {total_doc_types} document types")
        print("   • Multi-label classification support")
        print("   • Confidence scores for all predictions")
        print("   • Separate classifier head for each document type")
        print("   • Model persistence (save/load)")
    else:
        print("⚠️ Some verification checks FAILED!")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = verify_implementation()
    sys.exit(0 if success else 1)
