"""
Script to generate and save the engagement behavior dataset.

This script uses the simulate_behaviors module to create a realistic dataset
with multiple user profiles and attack types.
"""

import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
from src.data.simulate_behaviors import generate_dataset


def main():
    """Generate and save the dataset."""
    parser = argparse.ArgumentParser(
        description="Generate realistic engagement behavior dataset"
    )
    parser.add_argument(
        "--n_users",
        type=int,
        default=1000,
        help="Number of users to generate (default: 1000)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=24 * 7,
        help="Length of each series in time units (default: 168 = 1 week hourly)",
    )
    parser.add_argument(
        "--fake_ratio",
        type=float,
        default=0.3,
        help="Proportion of fake engagement series (default: 0.3)",
    )
    parser.add_argument(
        "--start_timestamp",
        type=str,
        default="2024-01-01",
        help="Start timestamp (default: 2024-01-01)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="H",
        choices=["H", "D"],
        help="Frequency: H (hourly) or D (daily) (default: H)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/raw/engagement_behaviors.parquet",
        help="Output file path (default: data/raw/engagement_behaviors.parquet)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output format: parquet or csv (default: parquet)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING ENGAGEMENT BEHAVIOR DATASET")
    print("=" * 80)
    print(f"Number of users: {args.n_users}")
    print(f"Series length: {args.length} ({args.freq})")
    print(f"Fake ratio: {args.fake_ratio:.1%}")
    print(f"Start timestamp: {args.start_timestamp}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 80)
    print("\nGenerating dataset...")

    # generate dataset
    df = generate_dataset(
        n_users=args.n_users,
        length=args.length,
        fake_ratio=args.fake_ratio,
        start_timestamp=args.start_timestamp,
        freq=args.freq,
        random_seed=args.random_seed,
    )

    print(f"\nDataset generated successfully!")
    print(f"Total rows: {len(df):,}")
    print(f"Number of users: {df['user_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # print statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"\nFake series: {df['is_fake_series'].sum():,} ({df['is_fake_series'].mean():.1%})")
    print(f"Anomaly windows: {df['is_anomaly_window'].sum():,} ({df['is_anomaly_window'].mean():.1%})")
    
    print("\nProfile distribution:")
    print(df['profile'].value_counts().sort_index())
    
    print("\nAttack type distribution (fake series only):")
    fake_df = df[df['is_fake_series']]
    if len(fake_df) > 0:
        print(fake_df['attack_type'].value_counts().sort_index())
    else:
        print("No fake series in dataset")

    print("\nEngagement metrics summary:")
    print(df[['views', 'likes', 'comments', 'shares']].describe())

    # save dataset
    print("\n" + "=" * 80)
    print("SAVING DATASET")
    print("=" * 80)
    
    if args.output_format == "parquet":
        df.to_parquet(args.output_path, index=False)
    elif args.output_format == "csv":
        df.to_csv(args.output_path, index=False)
    
    print(f"Dataset saved to: {args.output_path}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()

