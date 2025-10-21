#!/bin/bash
# Build expression embedding database from H5 cache files
# Usage: ./build_expr_db.sh [cache_dir_or_file] [frame_stride]
# Output will be saved in the same directory as input with name expression_embeddings.h5

CACHE_PATH="${1:-./test_cache}"
STRIDE="1"

# Auto-determine output path - save in same directory as input
if [ -d "$CACHE_PATH" ]; then
    OUTPUT="$CACHE_PATH/expression_embeddings.h5"
elif [ -f "$CACHE_PATH" ]; then
    # Get directory of the file
    CACHE_DIR=$(dirname "$CACHE_PATH")
    OUTPUT="$CACHE_DIR/expression_embeddings.h5"
else
    OUTPUT="expression_embeddings.h5"
fi

echo "🚀 Building expression database from H5 cache"
echo "Cache path: $CACHE_PATH"
echo "Output: $OUTPUT"
echo "Frame stride: $STRIDE"
echo ""

# Check if input is a directory or file
if [ -d "$CACHE_PATH" ]; then
    echo "📁 Processing directory of H5 cache files..."
    python3 -c "
import h5py
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

cache_dir = Path('$CACHE_PATH')
output_file = '$OUTPUT'
stride = $STRIDE

print(f'Scanning cache directory: {cache_dir}')

# Find all .h5 files in the directory
h5_files = list(cache_dir.glob('*.h5'))
print(f'Found {len(h5_files)} H5 cache files')

all_embeddings = []
total_windows = 0

for h5_file in tqdm(h5_files, desc='Processing cache files'):
    try:
        with h5py.File(h5_file, 'r') as f:
            # Find all window keys
            window_keys = sorted([k for k in f.keys() if k.startswith('window_')])

            for window_key in window_keys:
                window = f[window_key]

                # Check if expression_embed exists
                if 'expression_embed' not in window:
                    continue

                # Get expression embeddings [T, 128]
                expr_data = np.array(window['expression_embed'])

                # Sample frames with stride
                sampled_frames = expr_data[::stride]

                # Add to collection
                for frame_embed in sampled_frames:
                    if frame_embed.shape == (128,):
                        all_embeddings.append(frame_embed)
                    else:
                        print(f'Warning: Invalid shape in {h5_file.name}/{window_key}: {frame_embed.shape}')

                total_windows += 1

    except Exception as e:
        print(f'Error processing {h5_file}: {e}')
        continue

print(f'Collected {len(all_embeddings)} embeddings from {total_windows} windows')

if len(all_embeddings) == 0:
    print('❌ No embeddings found!')
    sys.exit(1)

# Save to output file
print(f'Saving to {output_file}...')
embeddings_array = np.stack(all_embeddings, axis=0)

with h5py.File(output_file, 'w') as f:
    f.create_dataset(
        'expression_embeddings',
        data=embeddings_array,
        compression='gzip',
        compression_opts=4
    )

    f.attrs['num_embeddings'] = len(all_embeddings)
    f.attrs['embedding_dim'] = 128
    f.attrs['frame_stride'] = stride
    f.attrs['num_files'] = len(h5_files)
    f.attrs['num_windows'] = total_windows

print(f'✅ Database saved: {output_file}')
print(f'   Shape: {embeddings_array.shape}')
print(f'   Size: {embeddings_array.nbytes / 1024**2:.2f} MB')
print(f'   Files processed: {len(h5_files)}')
print(f'   Windows processed: {total_windows}')
"

elif [ -f "$CACHE_PATH" ]; then
    echo "📄 Processing single H5 cache file..."
    python3 -c "
import h5py
import numpy as np
from tqdm import tqdm
import sys

cache_file = '$CACHE_PATH'
output_file = '$OUTPUT'
stride = $STRIDE

print(f'Loading cache file: {cache_file}')

all_embeddings = []

with h5py.File(cache_file, 'r') as f:
    # Find all window keys
    window_keys = sorted([k for k in f.keys() if k.startswith('window_')])
    print(f'Found {len(window_keys)} windows')

    for window_key in tqdm(window_keys, desc='Extracting embeddings'):
        window = f[window_key]

        # Check if expression_embed exists
        if 'expression_embed' not in window:
            print(f'Warning: No expression_embed in {window_key}')
            continue

        # Get expression embeddings [T, 128]
        expr_data = np.array(window['expression_embed'])
        print(f'{window_key}: expression_embed shape = {expr_data.shape}')

        # Sample frames with stride
        sampled_frames = expr_data[::stride]

        # Add to collection
        for frame_embed in sampled_frames:
            if frame_embed.shape == (128,):
                all_embeddings.append(frame_embed)
            else:
                print(f'Warning: Invalid shape in {window_key}: {frame_embed.shape}')

print(f'Collected {len(all_embeddings)} embeddings from {len(window_keys)} windows')

if len(all_embeddings) == 0:
    print('❌ No embeddings found!')
    sys.exit(1)

# Save to output file
print(f'Saving to {output_file}...')
embeddings_array = np.stack(all_embeddings, axis=0)

with h5py.File(output_file, 'w') as f:
    f.create_dataset(
        'expression_embeddings',
        data=embeddings_array,
        compression='gzip',
        compression_opts=4
    )

    f.attrs['num_embeddings'] = len(all_embeddings)
    f.attrs['embedding_dim'] = 128
    f.attrs['frame_stride'] = stride
    f.attrs['num_files'] = 1
    f.attrs['num_windows'] = len(window_keys)

print(f'✅ Database saved: {output_file}')
print(f'   Shape: {embeddings_array.shape}')
print(f'   Size: {embeddings_array.nbytes / 1024**2:.2f} MB')
"
else
    echo "❌ Error: Path not found: $CACHE_PATH"
    exit 1
fi

echo ""
echo "✅ Done! Verifying database..."
python3 -c "
import h5py
import numpy as np

output_file = '$OUTPUT'

with h5py.File(output_file, 'r') as f:
    embeddings = f['expression_embeddings'][:]

    print('📊 Database Statistics:')
    print(f'   Total embeddings: {f.attrs[\"num_embeddings\"]}')
    print(f'   Embedding dimension: {f.attrs[\"embedding_dim\"]}')
    print(f'   Frame stride: {f.attrs[\"frame_stride\"]}')
    print(f'   Source files: {f.attrs[\"num_files\"]}')
    print(f'   Source windows: {f.attrs[\"num_windows\"]}')
    print(f'   Array shape: {embeddings.shape}')
    print(f'   Mean: {embeddings.mean():.6f}')
    print(f'   Std: {embeddings.std():.6f}')
    print(f'   Min: {embeddings.min():.6f}')
    print(f'   Max: {embeddings.max():.6f}')
"
