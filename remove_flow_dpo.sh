#!/bin/bash
# Script to remove all Flow-DPO related code from VASA-1 codebase

echo "🗑️  Removing Flow-DPO from VASA-1 codebase..."
echo "This script will comment out Flow-DPO code with '# FLOW-DPO REMOVED:' markers"
echo ""

# Backup files first
echo "📦 Creating backups..."
cp vasa_dataset.py vasa_dataset.py.backup_before_flow_dpo_removal
cp vasa_model.py vasa_model.py.backup_before_flow_dpo_removal
cp vasa_losses.py vasa_losses.py.backup_before_flow_dpo_removal
cp vasa_sampler.py vasa_sampler.py.backup_before_flow_dpo_removal
cp overfit_config.yaml overfit_config.yaml.backup_before_flow_dpo_removal

echo "✅ Backups created with .backup_before_flow_dpo_removal extension"
echo ""
echo "Files to be modified:"
echo "  - vasa_dataset.py (remove velocity computation)"
echo "  - vasa_model.py (remove reward/ref models, compute_velocity)"
echo "  - vasa_losses.py (remove flow_dpo loss)"
echo "  - vasa_sampler.py (remove velocity assertions)"
echo "  - overfit_config.yaml (disable flow_dpo)"
echo ""
echo "⚠️  You will need to manually apply the changes using the Python script"
echo "   Run: python apply_flow_dpo_removal.py"
