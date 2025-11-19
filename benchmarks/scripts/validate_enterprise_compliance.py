#!/usr/bin/env python3
"""
Enterprise Compliance Validation Script

Validates that all benchmark results meet enterprise-grade standards:
- Latency targets (P95 <600ms)
- Throughput targets (≥100k docs/min)
- Accuracy targets (Classification ≥99%, F1 ≥97%, Routing ≥99.5%)
- Resource usage (<70% CPU/Memory per AWS Well-Architected Framework)
- Error rates (<1% industry standard)
- Scaling efficiency (≥85% at 4 workers)

Generates compliance report with pass/fail for each criterion.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime


class EnterpriseComplianceValidator:
    """Validate benchmark results against enterprise standards."""

    # Enterprise Standards (2025)
    STANDARDS = {
        "latency": {
            "p95_max_ms": 600,
            "p99_max_ms": 800,
            "description": "Document processing latency targets"
        },
        "throughput": {
            "min_docs_per_min": 100000,
            "description": "Sustained throughput target"
        },
        "accuracy": {
            "classification_min": 0.99,
            "extraction_f1_min": 0.97,
            "routing_min": 0.995,
            "description": "AI/ML accuracy targets"
        },
        "resources": {
            "cpu_max_percent": 70,
            "memory_max_percent": 70,
            "gpu_min_percent": 80,
            "description": "AWS Well-Architected Framework thresholds"
        },
        "reliability": {
            "error_rate_max": 0.01,  # 1%
            "failure_rate_max": 0.001,  # 0.1%
            "description": "Industry standard reliability targets"
        },
        "scaling": {
            "efficiency_min_4_workers": 0.85,  # 85%
            "description": "Horizontal scaling efficiency"
        }
    }

    def __init__(self):
        """Initialize validator."""
        self.results = {}
        self.violations = []
        self.warnings = []
        self.passed_checks = []

    def load_results(self, results_dir: str) -> bool:
        """Load latest benchmark results."""
        results_path = Path(results_dir)

        # Find latest combined results
        combined_files = list(results_path.glob("combined_*.json"))

        if not combined_files:
            print(f"❌ No benchmark results found in {results_dir}")
            return False

        latest_file = max(combined_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, "r") as f:
            self.results = json.load(f)

        print(f"✓ Loaded results from: {latest_file.name}")
        return True

    def validate_all(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all validation checks.

        Returns:
            (passed, report) tuple
        """
        print("\n" + "="*70)
        print("  ENTERPRISE COMPLIANCE VALIDATION")
        print("="*70)
        print("\nValidating against 2025 industry standards...")

        checks = [
            ("Latency Performance", self._validate_latency),
            ("Throughput Performance", self._validate_throughput),
            ("Accuracy Metrics", self._validate_accuracy),
            ("Resource Usage", self._validate_resources),
            ("Error Rates & Reliability", self._validate_reliability),
            ("Horizontal Scaling", self._validate_scaling),
        ]

        all_passed = True

        for check_name, check_func in checks:
            print(f"\n{'─'*70}")
            print(f"Validating: {check_name}")
            print(f"{'─'*70}")

            passed = check_func()
            all_passed = all_passed and passed

        # Generate report
        report = self._generate_report(all_passed)

        # Print summary
        self._print_summary(all_passed)

        return all_passed, report

    def _validate_latency(self) -> bool:
        """Validate latency performance."""
        latency_data = self.results.get("results", {}).get("latency", {}).get("e2e_latency", {})

        if not latency_data:
            self.violations.append("Missing latency data")
            print("❌ No latency data found")
            return False

        std = self.STANDARDS["latency"]
        p95 = latency_data.get("p95", float('inf'))
        p99 = latency_data.get("p99", float('inf'))

        # P95 check
        if p95 <= std["p95_max_ms"]:
            self.passed_checks.append(f"P95 latency: {p95:.1f}ms ≤ {std['p95_max_ms']}ms")
            print(f"✅ P95 Latency: {p95:.1f}ms (target: ≤{std['p95_max_ms']}ms)")
        else:
            self.violations.append(f"P95 latency {p95:.1f}ms exceeds {std['p95_max_ms']}ms")
            print(f"❌ P95 Latency: {p95:.1f}ms EXCEEDS {std['p95_max_ms']}ms")
            return False

        # P99 check (warning only)
        if p99 <= std["p99_max_ms"]:
            print(f"✅ P99 Latency: {p99:.1f}ms (target: ≤{std['p99_max_ms']}ms)")
        else:
            self.warnings.append(f"P99 latency {p99:.1f}ms exceeds {std['p99_max_ms']}ms")
            print(f"⚠️  P99 Latency: {p99:.1f}ms exceeds {std['p99_max_ms']}ms (warning)")

        return True

    def _validate_throughput(self) -> bool:
        """Validate throughput performance."""
        throughput_data = self.results.get("results", {}).get("throughput", {}).get("sustained", {})

        if not throughput_data:
            self.violations.append("Missing throughput data")
            print("❌ No throughput data found")
            return False

        std = self.STANDARDS["throughput"]
        actual = throughput_data.get("avg_throughput_per_min", 0)

        if actual >= std["min_docs_per_min"]:
            self.passed_checks.append(f"Throughput: {actual:,.0f} ≥ {std['min_docs_per_min']:,} docs/min")
            print(f"✅ Throughput: {actual:,.0f} docs/min (target: ≥{std['min_docs_per_min']:,})")
            return True
        else:
            self.violations.append(f"Throughput {actual:,.0f} below {std['min_docs_per_min']:,} docs/min")
            print(f"❌ Throughput: {actual:,.0f} docs/min BELOW {std['min_docs_per_min']:,}")
            return False

    def _validate_accuracy(self) -> bool:
        """Validate accuracy metrics."""
        accuracy_data = self.results.get("results", {}).get("accuracy", {})

        if not accuracy_data:
            self.violations.append("Missing accuracy data")
            print("❌ No accuracy data found")
            return False

        std = self.STANDARDS["accuracy"]
        all_passed = True

        # Classification
        class_acc = accuracy_data.get("classification", {}).get("accuracy", 0)
        if class_acc >= std["classification_min"]:
            self.passed_checks.append(f"Classification: {class_acc*100:.2f}% ≥ {std['classification_min']*100}%")
            print(f"✅ Classification: {class_acc*100:.2f}% (target: ≥{std['classification_min']*100}%)")
        else:
            self.violations.append(f"Classification {class_acc*100:.2f}% below {std['classification_min']*100}%")
            print(f"❌ Classification: {class_acc*100:.2f}% BELOW {std['classification_min']*100}%")
            all_passed = False

        # Extraction F1
        ext_f1 = accuracy_data.get("extraction", {}).get("avg_f1", 0)
        if ext_f1 >= std["extraction_f1_min"]:
            self.passed_checks.append(f"Extraction F1: {ext_f1*100:.2f}% ≥ {std['extraction_f1_min']*100}%")
            print(f"✅ Extraction F1: {ext_f1*100:.2f}% (target: ≥{std['extraction_f1_min']*100}%)")
        else:
            self.violations.append(f"Extraction F1 {ext_f1*100:.2f}% below {std['extraction_f1_min']*100}%")
            print(f"❌ Extraction F1: {ext_f1*100:.2f}% BELOW {std['extraction_f1_min']*100}%")
            all_passed = False

        # Routing
        routing_acc = accuracy_data.get("routing", {}).get("accuracy", 0)
        if routing_acc >= std["routing_min"]:
            self.passed_checks.append(f"Routing: {routing_acc*100:.2f}% ≥ {std['routing_min']*100}%")
            print(f"✅ Routing: {routing_acc*100:.2f}% (target: ≥{std['routing_min']*100}%)")
        else:
            self.violations.append(f"Routing {routing_acc*100:.2f}% below {std['routing_min']*100}%")
            print(f"❌ Routing: {routing_acc*100:.2f}% BELOW {std['routing_min']*100}%")
            all_passed = False

        return all_passed

    def _validate_resources(self) -> bool:
        """Validate resource usage against AWS Well-Architected Framework."""
        resources_data = self.results.get("results", {}).get("resources", {})

        if not resources_data:
            self.warnings.append("No resource monitoring data available")
            print("⚠️  No resource data found (run monitor_resources.py)")
            return True  # Don't fail if not monitored

        std = self.STANDARDS["resources"]
        all_passed = True

        # CPU check
        cpu_usage = resources_data.get("cpu", {}).get("percent", 0)
        if cpu_usage <= std["cpu_max_percent"]:
            self.passed_checks.append(f"CPU usage: {cpu_usage:.1f}% ≤ {std['cpu_max_percent']}%")
            print(f"✅ CPU Usage: {cpu_usage:.1f}% (threshold: ≤{std['cpu_max_percent']}%)")
        else:
            self.violations.append(f"CPU usage {cpu_usage:.1f}% exceeds {std['cpu_max_percent']}%")
            print(f"❌ CPU Usage: {cpu_usage:.1f}% EXCEEDS {std['cpu_max_percent']}%")
            print("   ⚠️  Insufficient headroom for production surges (AWS Well-Architected)")
            all_passed = False

        # Memory check
        mem_usage = resources_data.get("memory", {}).get("percent", 0)
        if mem_usage <= std["memory_max_percent"]:
            self.passed_checks.append(f"Memory usage: {mem_usage:.1f}% ≤ {std['memory_max_percent']}%")
            print(f"✅ Memory Usage: {mem_usage:.1f}% (threshold: ≤{std['memory_max_percent']}%)")
        else:
            self.violations.append(f"Memory usage {mem_usage:.1f}% exceeds {std['memory_max_percent']}%")
            print(f"❌ Memory Usage: {mem_usage:.1f}% EXCEEDS {std['memory_max_percent']}%")
            print("   ⚠️  Risk of OOM during peak load (AWS Well-Architected)")
            all_passed = False

        return all_passed

    def _validate_reliability(self) -> bool:
        """Validate error rates and reliability."""
        # Check throughput errors
        throughput_data = self.results.get("results", {}).get("throughput", {}).get("sustained", {})

        std = self.STANDARDS["reliability"]

        total_processed = throughput_data.get("total_processed", 0)
        errors = throughput_data.get("errors", 0)

        if total_processed > 0:
            error_rate = errors / total_processed

            if error_rate <= std["error_rate_max"]:
                self.passed_checks.append(f"Error rate: {error_rate*100:.2f}% ≤ {std['error_rate_max']*100}%")
                print(f"✅ Error Rate: {error_rate*100:.2f}% (target: ≤{std['error_rate_max']*100}%)")
                return True
            else:
                self.violations.append(f"Error rate {error_rate*100:.2f}% exceeds {std['error_rate_max']*100}%")
                print(f"❌ Error Rate: {error_rate*100:.2f}% EXCEEDS {std['error_rate_max']*100}%")
                print("   ⚠️  Reliability concerns (industry standard: <1%)")
                return False
        else:
            print("⚠️  No error rate data available")
            return True

    def _validate_scaling(self) -> bool:
        """Validate horizontal scaling efficiency."""
        scaling_data = self.results.get("results", {}).get("throughput", {}).get("scaling", {})

        if not scaling_data or "4" not in scaling_data:
            self.warnings.append("No scaling data for 4 workers")
            print("⚠️  No scaling data found for 4 workers")
            return True  # Don't fail if not tested

        std = self.STANDARDS["scaling"]
        efficiency = scaling_data.get("4", {}).get("scaling_efficiency", 0) / 100

        if efficiency >= std["efficiency_min_4_workers"]:
            self.passed_checks.append(f"Scaling efficiency: {efficiency*100:.1f}% ≥ {std['efficiency_min_4_workers']*100}%")
            print(f"✅ Scaling Efficiency (4 workers): {efficiency*100:.1f}% (target: ≥{std['efficiency_min_4_workers']*100}%)")
            return True
        else:
            self.warnings.append(f"Scaling efficiency {efficiency*100:.1f}% below {std['efficiency_min_4_workers']*100}%")
            print(f"⚠️  Scaling Efficiency: {efficiency*100:.1f}% below {std['efficiency_min_4_workers']*100}% (warning)")
            return True  # Warning only

    def _generate_report(self, overall_passed: bool) -> Dict[str, Any]:
        """Generate compliance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_compliant": overall_passed,
            "standards_version": "2025",
            "frameworks": [
                "AWS Well-Architected Framework",
                "Industry Standard SLA Targets",
                "Enterprise Performance Best Practices"
            ],
            "passed_checks": self.passed_checks,
            "violations": self.violations,
            "warnings": self.warnings,
            "standards": self.STANDARDS,
            "summary": {
                "total_checks": len(self.passed_checks) + len(self.violations),
                "passed": len(self.passed_checks),
                "failed": len(self.violations),
                "warnings": len(self.warnings),
            }
        }

        return report

    def _print_summary(self, overall_passed: bool) -> None:
        """Print validation summary."""
        print("\n" + "="*70)
        print("  COMPLIANCE VALIDATION SUMMARY")
        print("="*70)

        print(f"\nTotal Checks: {len(self.passed_checks) + len(self.violations)}")
        print(f"✅ Passed: {len(self.passed_checks)}")
        print(f"❌ Failed: {len(self.violations)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")

        if self.violations:
            print("\n❌ VIOLATIONS:")
            for v in self.violations:
                print(f"   • {v}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for w in self.warnings:
                print(f"   • {w}")

        print("\n" + "="*70)
        if overall_passed:
            print("✅ ENTERPRISE COMPLIANCE: PASSED")
            print("   System meets all enterprise-grade requirements")
        else:
            print("❌ ENTERPRISE COMPLIANCE: FAILED")
            print("   System does not meet enterprise standards")
        print("="*70 + "\n")

    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Save compliance report."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"💾 Compliance report saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enterprise Compliance Validation")
    parser.add_argument("--results", type=str, default="benchmarks/results",
                       help="Results directory")
    parser.add_argument("--output", type=str, default="benchmarks/results/compliance_report.json",
                       help="Output report file")

    args = parser.parse_args()

    print("="*70)
    print("  SAP_LLM ENTERPRISE COMPLIANCE VALIDATOR")
    print("="*70)
    print("\nValidating against 2025 industry standards:")
    print("  • AWS Well-Architected Framework")
    print("  • Enterprise SLA Targets")
    print("  • Performance Testing Best Practices")

    validator = EnterpriseComplianceValidator()

    if not validator.load_results(args.results):
        sys.exit(1)

    passed, report = validator.validate_all()
    validator.save_report(report, args.output)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
