#!/bin/bash
# Convenience script to upsert phoneme_gt to existing cache

set -e

CACHE_DIR="${1:-cache_per_video}"
NUM_QUERIES="${2:-8}"
DRY_RUN="${3:-}"

echo "=================================="
echo "UPSERT PHONEME_GT TO CACHE"
echo "=================================="
echo "Cache directory: $CACHE_DIR"
echo "Number of queries: $NUM_QUERIES"
echo ""

if [ "$DRY_RUN" == "--dry-run" ]; then
    echo "⚠️  DRY RUN MODE - No changes will be saved"
    python upsert_phoneme_gt.py \
        --cache_dir "$CACHE_DIR" \
        --num_queries "$NUM_QUERIES" \
        --dry_run
else
    echo "🚀 Running upsert (changes will be saved)"
    python upsert_phoneme_gt.py \
        --cache_dir "$CACHE_DIR" \
        --num_queries "$NUM_QUERIES"
fi

echo ""
echo "✅ Done!"
