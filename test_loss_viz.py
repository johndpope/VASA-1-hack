#!/usr/bin/env python3
"""
Test script to demonstrate loss monitoring visualization.
"""

import numpy as np
from loss_monitor import LossRangeMonitor

def simulate_loss_convergence():
    """Simulate a training run with losses converging to target values."""

    monitor = LossRangeMonitor(history_size=50)

    # Simulate expression_cosine loss improving over time
    print("Simulating expression_cosine loss (should decrease from 0.8 to < 0.1)")
    for i in range(50):
        # Start high, converge to healthy range
        value = 0.8 * np.exp(-i / 15.0) + 0.05
        noise = np.random.normal(0, 0.02)
        value = max(0.0, value + noise)

        monitor.check_loss('expression_cosine', value, step=i)

    # Simulate theta_loss improving
    print("\nSimulating theta_loss (should decrease toward 0.001-0.05)")
    for i in range(50):
        value = 0.5 * np.exp(-i / 20.0) + 0.02
        noise = np.random.normal(0, 0.005)
        value = max(0.001, value + noise)

        monitor.check_loss('theta_loss', value, step=i)

    # Simulate a loss that's not converging (warning case)
    print("\nSimulating control_blink loss (stays elevated)")
    for i in range(50):
        value = 0.15 + np.sin(i / 5.0) * 0.05
        noise = np.random.normal(0, 0.01)
        value = max(0.0, value + noise)

        monitor.check_loss('control_blink', value, step=i)

    # Print visualization summary
    print("\n" + "="*80)
    print("VISUALIZATION TEST")
    print("="*80)

    # Show individual loss graph
    print("\n\nExample: expression_cosine loss graph")
    print(monitor.visualize_loss('expression_cosine'))

    # Show summary for all losses
    print("\n\nFull summary with graphs:")
    print(monitor.visualize_summary(
        loss_names=['expression_cosine', 'theta_loss', 'control_blink'],
        show_graphs=True,
        graph_width=50,
        graph_height=6
    ))

    # Show statistics
    stats = monitor.get_statistics()
    print("\n\nMonitoring Statistics:")
    print(f"  Total warnings: {stats['total_warnings']}")
    print(f"  Total criticals: {stats['total_criticals']}")
    print(f"  Warning counts: {stats['warning_counts']}")
    print(f"  Critical counts: {stats['critical_counts']}")

if __name__ == '__main__':
    simulate_loss_convergence()
