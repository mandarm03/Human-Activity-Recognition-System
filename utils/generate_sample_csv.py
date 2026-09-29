"""
generate_sample_csv.py — Generate a test sensor CSV for API testing.

Creates a synthetic 128×9 CSV that the /predict endpoint will accept.
The signal is a random-walk to simulate realistic-looking sensor data.

Usage (from project root):
    python utils/generate_sample_csv.py
    python utils/generate_sample_csv.py --output my_test.csv
    python utils/generate_sample_csv.py --activity walking
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SIGNAL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

# Approximate signal profiles per activity (mean, std) per channel
SIGNAL_PROFILES = {
    "walking": [
        (0.1, 0.8),
        (0.05, 0.5),
        (0.9, 0.6),
        (0.0, 0.4),
        (0.02, 0.3),
        (0.01, 0.2),
        (0.1, 0.8),
        (0.05, 0.5),
        (1.0, 0.5),
    ],
    "walking_upstairs": [
        (0.2, 0.9),
        (0.1, 0.6),
        (1.1, 0.7),
        (0.01, 0.5),
        (0.03, 0.4),
        (0.02, 0.3),
        (0.2, 0.9),
        (0.1, 0.6),
        (1.1, 0.6),
    ],
    "walking_downstairs": [
        (0.15, 0.85),
        (0.08, 0.55),
        (0.85, 0.65),
        (0.0, 0.45),
        (0.02, 0.35),
        (0.01, 0.25),
        (0.15, 0.85),
        (0.08, 0.55),
        (0.95, 0.55),
    ],
    "sitting": [
        (0.0, 0.05),
        (0.0, 0.04),
        (0.98, 0.03),
        (0.0, 0.03),
        (0.0, 0.02),
        (0.0, 0.02),
        (0.0, 0.05),
        (0.0, 0.04),
        (1.0, 0.03),
    ],
    "standing": [
        (0.0, 0.04),
        (0.0, 0.03),
        (0.99, 0.02),
        (0.0, 0.02),
        (0.0, 0.02),
        (0.0, 0.01),
        (0.0, 0.04),
        (0.0, 0.03),
        (1.0, 0.02),
    ],
    "laying": [
        (0.0, 0.03),
        (0.98, 0.03),
        (0.0, 0.03),
        (0.0, 0.02),
        (0.0, 0.01),
        (0.0, 0.01),
        (0.0, 0.03),
        (1.0, 0.03),
        (0.0, 0.02),
    ],
}


def generate_csv(activity: str = "walking", output: str = "test_sample.csv") -> Path:
    """
    Generate a synthetic 128×9 sensor CSV.

    Args:
        activity: One of walking / walking_upstairs / walking_downstairs /
                  sitting / standing / laying.
        output:   Output file path.

    Returns:
        Path to the written CSV file.
    """
    np.random.seed(42)
    activity_key = activity.lower().replace(" ", "_")
    profile = SIGNAL_PROFILES.get(activity_key, SIGNAL_PROFILES["walking"])

    data = np.zeros((128, 9))
    for ch, (mean, std) in enumerate(profile):
        # Random walk with drift toward expected mean
        noise = np.random.normal(0, std * 0.3, 128)
        walk = np.cumsum(noise) * 0.05
        data[:, ch] = mean + np.random.normal(0, std * 0.7, 128) + walk

    df = pd.DataFrame(data, columns=SIGNAL_NAMES)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Generated test CSV → {out_path.resolve()}")
    print(f"   Shape : {df.shape}")
    print(f"   Activity profile: {activity_key}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a test sensor CSV for the HAR API."
    )
    parser.add_argument(
        "--activity",
        default="walking",
        choices=list(SIGNAL_PROFILES.keys()),
        help="Activity profile to simulate (default: walking)",
    )
    parser.add_argument(
        "--output",
        default="test_sample.csv",
        help="Output CSV file path (default: test_sample.csv)",
    )
    args = parser.parse_args()
    generate_csv(activity=args.activity, output=args.output)
