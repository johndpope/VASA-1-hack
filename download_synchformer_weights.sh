#!/bin/bash

# Script to download Synchformer pre-trained weights for VASA

echo "Downloading Synchformer weights for lip-sync..."

# Create directory for pretrained models
mkdir -p Synchformer/pretrained

# Download LRS3 model (best for lip-sync)
echo "Downloading LRS3 model (recommended for lip-sync)..."
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/23-12-23T18-33-57/23-12-23T18-33-57.pt \
     -O Synchformer/pretrained/synchformer.pt

# Download config
echo "Downloading config file..."
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/23-12-23T18-33-57/cfg-23-12-23T18-33-57.yaml \
     -O Synchformer/pretrained/config_lrs3.yaml

# Optional: Download AudioSet model for comparison
read -p "Do you also want to download the AudioSet model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Downloading AudioSet model..."
    wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt \
         -O Synchformer/pretrained/synchformer_audioset.pt

    wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml \
         -O Synchformer/pretrained/config_audioset.yaml
fi

echo "✅ Download complete!"
echo ""
echo "Weights saved to:"
echo "  - Synchformer/pretrained/synchformer.pt (LRS3 model)"
if [ -f "Synchformer/pretrained/synchformer_audioset.pt" ]; then
    echo "  - Synchformer/pretrained/synchformer_audioset.pt (AudioSet model)"
fi
echo ""
echo "The wrapper will automatically find and use these weights."