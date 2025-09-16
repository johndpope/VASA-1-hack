#!/bin/bash
# Monitor cache file growth

CACHE_FILE="cache_overfit/all_windows_cache.h5"

while true; do
    if [ -f "$CACHE_FILE" ]; then
        SIZE=$(ls -lh "$CACHE_FILE" | awk '{print $5}')
        DATE=$(date '+%H:%M:%S')
        echo "[$DATE] Cache size: $SIZE"
    else
        echo "Cache file not created yet..."
    fi
    sleep 10
done