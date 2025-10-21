#!/usr/bin/env python3
"""
Test lambda weight display in loss monitor.
"""

from omegaconf import OmegaConf
from loss_monitor import LossRangeMonitor

# Create a test config
config = OmegaConf.create({
    'loss': {
        'nonlip': 1.0,
        'lips': 2.0,
        'sync': 0.5,
        'flow_dpo': 25,
        'control': 1.0,
    }
})

# Create monitor with config
monitor = LossRangeMonitor(config=config)

# Test with critical losses
print("\n" + "="*80)
print("Testing Lambda Weight Display in Loss Monitor")
print("="*80)

# Test nonlip_total (critical)
print("\n1. Testing nonlip_total (critical level):")
result = monitor.check_loss('nonlip_total', 58.36, step=0)
print(result['message'])

# Test lips_total (warning)
print("\n2. Testing lips_total (high warning level):")
result = monitor.check_loss('lips_total', 1.5, step=0)
print(result['message'])

# Test flow_dpo (critical)
print("\n3. Testing flow_dpo (critical level):")
result = monitor.check_loss('flow_dpo', 10.0, step=0)
print(result['message'])

# Test sync_loss (warning)
print("\n4. Testing sync_loss (warning level):")
result = monitor.check_loss('sync_loss', 0.7, step=0)
print(result['message'])

print("\n" + "="*80)
print("✅ All tests completed - lambda weights should be displayed above!")
print("="*80)
