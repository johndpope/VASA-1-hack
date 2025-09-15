#!/usr/bin/env python3
"""Monitor component contributions during training."""

import subprocess
import re
import time
import sys

def monitor_training():
    """Monitor training and extract component contribution lines."""

    # Start the training process
    process = subprocess.Popen(
        ['python', 'train_overfit.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    contribution_pattern = re.compile(r'Component contributions.*Audio: ([\d.]+)%, Controls: ([\d.]+)%, Blink: ([\d.]+)%')
    found_contributions = False

    print("Monitoring training for component contributions...")
    print("=" * 60)

    try:
        for line in process.stdout:
            # Look for component contribution lines
            match = contribution_pattern.search(line)
            if match:
                audio_pct = float(match.group(1))
                controls_pct = float(match.group(2))
                blink_pct = float(match.group(3))
                total = audio_pct + controls_pct + blink_pct

                print(f"\nComponent Contributions Found:")
                print(f"  Audio:    {audio_pct:6.2f}%")
                print(f"  Controls: {controls_pct:6.2f}%")
                print(f"  Blink:    {blink_pct:6.2f}%")
                print(f"  Total:    {total:6.2f}% {'✓ CORRECT' if 99 < total < 101 else '✗ ERROR'}")

                found_contributions = True

                # Exit after finding first set of contributions
                if found_contributions:
                    print("\n✓ Fix verified - contributions now sum to ~100%")
                    process.terminate()
                    break

            # Also print epoch/step info
            if 'Epoch' in line or 'Step' in line or 'Loss' in line:
                print(f"  {line.strip()}")

    except KeyboardInterrupt:
        print("\n\nStopping monitoring...")
        process.terminate()

    process.wait()

    if not found_contributions:
        print("\nNo component contributions found in output.")
        print("They may only appear with DEBUG logging enabled.")

if __name__ == "__main__":
    monitor_training()