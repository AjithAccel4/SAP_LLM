#!/usr/bin/env python3
"""
Check for performance regressions by comparing current vs baseline results.

Fails if performance degrades by more than threshold (default 10%).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


class RegressionChecker:
    """Check for performance regressions."""

    def __init__(self, threshold: float = 0.10):
        """Initialize with regression threshold."""
        self.threshold = threshold  # 10% by default
        self.regressions = []

    def check(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> bool:
        """
        Check for regressions.

        Returns:
            True if no regressions found, False otherwise
        """
        print("="*70)
        print("  Performance Regression Check")
        print("="*70)
        print(f"\nThreshold: {self.threshold*100:.0f}% degradation\n")

        # Check latency
        self._check_latency(current, baseline)

        # Check throughput
        self._check_throughput(current, baseline)

        # Check accuracy
        self._check_accuracy(current, baseline)

        # Print summary
        if self.regressions:
            print("\n" + "="*70)
            print("❌ PERFORMANCE REGRESSIONS DETECTED")
            print("="*70)
            for regression in self.regressions:
                print(f"\n{regression}")
            return False
        else:
            print("\n" + "="*70)
            print("✅ NO REGRESSIONS DETECTED")
            print("="*70)
            return True

    def _check_latency(self, current: Dict, baseline: Dict) -> None:
        """Check latency regressions."""
        curr_latency = current.get("results", {}).get("latency", {}).get("e2e_latency", {})
        base_latency = baseline.get("results", {}).get("latency", {}).get("e2e_latency", {})

        if not curr_latency or not base_latency:
            print("⚠️  Latency: No baseline data for comparison")
            return

        curr_p95 = curr_latency.get("p95", 0)
        base_p95 = base_latency.get("p95", 0)

        if base_p95 > 0:
            change = (curr_p95 - base_p95) / base_p95

            if change > self.threshold:
                msg = (f"Latency P95 Regression: {base_p95:.1f}ms → {curr_p95:.1f}ms "
                      f"({change*100:+.1f}%)")
                self.regressions.append(msg)
                print(f"❌ {msg}")
            else:
                print(f"✅ Latency P95: {base_p95:.1f}ms → {curr_p95:.1f}ms ({change*100:+.1f}%)")

    def _check_throughput(self, current: Dict, baseline: Dict) -> None:
        """Check throughput regressions."""
        curr_tput = current.get("results", {}).get("throughput", {}).get("sustained", {})
        base_tput = baseline.get("results", {}).get("throughput", {}).get("sustained", {})

        if not curr_tput or not base_tput:
            print("⚠️  Throughput: No baseline data for comparison")
            return

        curr_rate = curr_tput.get("avg_throughput_per_min", 0)
        base_rate = base_tput.get("avg_throughput_per_min", 0)

        if base_rate > 0:
            change = (curr_rate - base_rate) / base_rate

            if change < -self.threshold:  # Negative change is bad for throughput
                msg = (f"Throughput Regression: {base_rate:,.0f} → {curr_rate:,.0f} docs/min "
                      f"({change*100:+.1f}%)")
                self.regressions.append(msg)
                print(f"❌ {msg}")
            else:
                print(f"✅ Throughput: {base_rate:,.0f} → {curr_rate:,.0f} docs/min ({change*100:+.1f}%)")

    def _check_accuracy(self, current: Dict, baseline: Dict) -> None:
        """Check accuracy regressions."""
        curr_acc = current.get("results", {}).get("accuracy", {})
        base_acc = baseline.get("results", {}).get("accuracy", {})

        if not curr_acc or not base_acc:
            print("⚠️  Accuracy: No baseline data for comparison")
            return

        # Classification
        curr_class = curr_acc.get("classification", {}).get("accuracy", 0)
        base_class = base_acc.get("classification", {}).get("accuracy", 0)

        if base_class > 0:
            change = curr_class - base_class  # Absolute change for accuracy

            if change < -self.threshold:
                msg = (f"Classification Regression: {base_class*100:.2f}% → {curr_class*100:.2f}% "
                      f"({change*100:+.2f}%)")
                self.regressions.append(msg)
                print(f"❌ {msg}")
            else:
                print(f"✅ Classification: {base_class*100:.2f}% → {curr_class*100:.2f}% ({change*100:+.2f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check for performance regressions")
    parser.add_argument("--current", type=str, required=True,
                       help="Directory with current results")
    parser.add_argument("--baseline", type=str, required=True,
                       help="Directory with baseline results")
    parser.add_argument("--threshold", type=float, default=0.10,
                       help="Regression threshold (default: 0.10 = 10%%)")

    args = parser.parse_args()

    # Load current results
    current_dir = Path(args.current)
    current_files = list(current_dir.glob("combined_*.json"))

    if not current_files:
        print(f"❌ No current results found in {current_dir}")
        sys.exit(1)

    current_file = max(current_files, key=lambda p: p.stat().st_mtime)
    with open(current_file, "r") as f:
        current = json.load(f)

    # Load baseline results
    baseline_dir = Path(args.baseline)
    if not baseline_dir.exists():
        print(f"⚠️  No baseline directory found at {baseline_dir}")
        print("   Skipping regression check")
        sys.exit(0)

    baseline_files = list(baseline_dir.glob("combined_*.json"))

    if not baseline_files:
        print(f"⚠️  No baseline results found in {baseline_dir}")
        print("   Skipping regression check")
        sys.exit(0)

    baseline_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
    with open(baseline_file, "r") as f:
        baseline = json.load(f)

    # Run regression check
    checker = RegressionChecker(threshold=args.threshold)
    passed = checker.check(current, baseline)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
