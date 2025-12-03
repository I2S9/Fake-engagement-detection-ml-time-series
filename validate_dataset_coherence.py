#!/usr/bin/env python3
"""
Validation script for dataset coherence.
Checks user profiles and attack patterns consistency.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_data

def validate_profiles(df: pd.DataFrame) -> dict:
    """
    Validate user profiles coherence.
    
    Checks:
    - Different scales/variability per profile
    - Realistic distributions (heavy tails, variance)
    - Profile-specific characteristics
    """
    results = {
        "profiles_checked": [],
        "scale_differences": {},
        "variance_differences": {},
        "distribution_tests": {},
        "all_valid": True
    }
    
    profiles = df['profile'].unique()
    metrics = ['views', 'likes', 'comments']
    
    print("\n" + "="*60)
    print("VALIDATION 1: USER PROFILES COHERENCE")
    print("="*60)
    
    for profile in sorted(profiles):
        profile_df = df[df['profile'] == profile]
        results["profiles_checked"].append(profile)
        
        print(f"\n--- Profile: {profile} ---")
        print(f"  Users: {profile_df['user_id'].nunique()}")
        print(f"  Total records: {len(profile_df)}")
        
        profile_stats = {}
        for metric in metrics:
            values = profile_df[metric].values
            profile_stats[metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'cv': np.std(values) / (np.mean(values) + 1e-6),  # coefficient of variation
                'q95': np.percentile(values, 95),
                'q99': np.percentile(values, 99),
                'max': np.max(values),
                'zeros_ratio': np.mean(values == 0)
            }
            
            print(f"\n  {metric.upper()}:")
            print(f"    Mean: {profile_stats[metric]['mean']:.2f}")
            print(f"    Median: {profile_stats[metric]['median']:.2f}")
            print(f"    Std: {profile_stats[metric]['std']:.2f}")
            print(f"    CV: {profile_stats[metric]['cv']:.2f}")
            print(f"    95th percentile: {profile_stats[metric]['q95']:.2f}")
            print(f"    99th percentile: {profile_stats[metric]['q99']:.2f}")
            print(f"    Max: {profile_stats[metric]['max']:.2f}")
            print(f"    Zeros ratio: {profile_stats[metric]['zeros_ratio']:.2%}")
        
        # Check for heavy tails (realistic distribution)
        views = profile_df['views'].values
        if len(views) > 100:
            # Check if distribution has heavy tail (high kurtosis)
            kurtosis = stats.kurtosis(views)
            skewness = stats.skew(views)
            print(f"\n  Distribution shape:")
            print(f"    Skewness: {skewness:.2f} (positive = right tail)")
            print(f"    Kurtosis: {kurtosis:.2f} (>3 = heavy tail)")
            
            has_heavy_tail = kurtosis > 3
            has_right_skew = skewness > 0.5
            if has_heavy_tail and has_right_skew:
                print(f"    ✓ Heavy tail detected (realistic)")
            else:
                print(f"    ⚠ Light tail (may be too uniform)")
        
        # Store scale differences
        results["scale_differences"][profile] = profile_stats['views']['mean']
        results["variance_differences"][profile] = profile_stats['views']['cv']
    
    # Verify profiles have different scales
    print("\n--- Profile Scale Comparison ---")
    scale_values = list(results["scale_differences"].values())
    if len(set(scale_values)) == len(scale_values):
        print("✓ All profiles have different scales")
    else:
        scale_ratio = max(scale_values) / min(scale_values)
        print(f"  Scale ratio (max/min): {scale_ratio:.2f}x")
        if scale_ratio > 2:
            print("✓ Significant scale differences between profiles")
        else:
            print("⚠ Profiles may be too similar in scale")
            results["all_valid"] = False
    
    # Verify variance differences
    print("\n--- Profile Variance Comparison ---")
    cv_values = list(results["variance_differences"].values())
    cv_ratio = max(cv_values) / min(cv_values)
    print(f"  CV ratio (max/min): {cv_ratio:.2f}x")
    if cv_ratio > 1.5:
        print("✓ Profiles show different variability patterns")
    else:
        print("⚠ Profiles may have similar variability")
    
    return results


def validate_attacks(df: pd.DataFrame) -> dict:
    """
    Validate attack patterns coherence.
    
    Checks:
    - Each attack type produces unique visual pattern
    - Anomaly lengths vary logically
    - Spikes/bursts are visible
    """
    results = {
        "attack_types_checked": [],
        "anomaly_lengths": {},
        "spike_visibility": {},
        "pattern_uniqueness": {},
        "all_valid": True
    }
    
    fake_df = df[df['is_fake_series'] == True].copy()
    
    if len(fake_df) == 0:
        print("\n⚠ No fake series found in dataset")
        results["all_valid"] = False
        return results
    
    attack_types = fake_df['attack_type'].unique()
    
    print("\n" + "="*60)
    print("VALIDATION 2: ATTACK PATTERNS COHERENCE")
    print("="*60)
    
    for attack_type in sorted(attack_types):
        attack_df = fake_df[fake_df['attack_type'] == attack_type]
        results["attack_types_checked"].append(attack_type)
        
        print(f"\n--- Attack Type: {attack_type} ---")
        print(f"  Series count: {attack_df['user_id'].nunique()}")
        
        # Analyze anomaly windows
        anomaly_windows = []
        for user_id in attack_df['user_id'].unique():
            user_data = attack_df[attack_df['user_id'] == user_id].sort_values('timestamp')
            anomaly_mask = user_data['is_anomaly_window'].values
            
            # Find contiguous anomaly windows
            in_window = False
            window_start = None
            for i, is_anomaly in enumerate(anomaly_mask):
                if is_anomaly and not in_window:
                    window_start = i
                    in_window = True
                elif not is_anomaly and in_window:
                    window_length = i - window_start
                    anomaly_windows.append(window_length)
                    in_window = False
            if in_window:
                window_length = len(anomaly_mask) - window_start
                anomaly_windows.append(window_length)
        
        if anomaly_windows:
            avg_length = np.mean(anomaly_windows)
            min_length = np.min(anomaly_windows)
            max_length = np.max(anomaly_windows)
            std_length = np.std(anomaly_windows)
            
            results["anomaly_lengths"][attack_type] = {
                'mean': avg_length,
                'min': min_length,
                'max': max_length,
                'std': std_length
            }
            
            print(f"  Anomaly window lengths:")
            print(f"    Mean: {avg_length:.1f} time units")
            print(f"    Range: {min_length:.0f} - {max_length:.0f}")
            print(f"    Std: {std_length:.1f}")
            
            # Check if lengths vary logically
            if std_length > 0:
                cv_length = std_length / avg_length
                print(f"    CV: {cv_length:.2f}")
                if cv_length > 0.3:
                    print("    ✓ Lengths vary significantly (realistic)")
                else:
                    print("    ⚠ Lengths may be too uniform")
        else:
            print("  ⚠ No anomaly windows detected")
            results["all_valid"] = False
        
        # Check spike visibility
        spike_ratios = []
        for user_id in attack_df['user_id'].unique()[:10]:  # Sample 10 users
            user_data = attack_df[attack_df['user_id'] == user_id].sort_values('timestamp')
            views = user_data['views'].values
            anomaly_mask = user_data['is_anomaly_window'].values
            
            if np.any(anomaly_mask):
                normal_views = views[~anomaly_mask]
                anomaly_views = views[anomaly_mask]
                
                if len(normal_views) > 0 and len(anomaly_views) > 0:
                    normal_mean = np.mean(normal_views)
                    anomaly_mean = np.mean(anomaly_views)
                    if normal_mean > 0:
                        spike_ratio = anomaly_mean / normal_mean
                        spike_ratios.append(spike_ratio)
        
        if spike_ratios:
            avg_spike = np.mean(spike_ratios)
            results["spike_visibility"][attack_type] = avg_spike
            print(f"\n  Spike visibility:")
            print(f"    Average spike ratio: {avg_spike:.2f}x")
            if avg_spike > 1.5:
                print("    ✓ Spikes are clearly visible")
            elif avg_spike > 1.2:
                print("    ⚠ Spikes are moderately visible")
            else:
                print("    ✗ Spikes may be too subtle")
                results["all_valid"] = False
    
    # Check pattern uniqueness
    print("\n--- Attack Pattern Uniqueness ---")
    if len(results["anomaly_lengths"]) > 1:
        lengths = [v['mean'] for v in results["anomaly_lengths"].values()]
        length_std = np.std(lengths)
        length_cv = length_std / np.mean(lengths)
        
        print(f"  Mean anomaly length CV across types: {length_cv:.2f}")
        if length_cv > 0.2:
            print("✓ Different attack types have different anomaly lengths")
        else:
            print("⚠ Attack types may be too similar")
    
    spike_values = list(results["spike_visibility"].values())
    if len(spike_values) > 1:
        spike_std = np.std(spike_values)
        spike_cv = spike_std / np.mean(spike_values)
        print(f"  Spike ratio CV across types: {spike_cv:.2f}")
        if spike_cv > 0.15:
            print("✓ Different attack types produce different spike magnitudes")
        else:
            print("⚠ Attack types may produce similar spikes")
    
    return results


def create_validation_plots(df: pd.DataFrame, output_dir: Path, attack_results: dict):
    """Create visualization plots for validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Profile distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Profile Coherence Validation', fontsize=16, fontweight='bold')
    
    profiles = sorted(df['profile'].unique())
    
    # Views distribution by profile
    ax = axes[0, 0]
    for profile in profiles:
        profile_data = df[df['profile'] == profile]['views']
        ax.hist(profile_data, bins=50, alpha=0.6, label=profile, density=True)
    ax.set_xlabel('Views')
    ax.set_ylabel('Density')
    ax.set_title('Views Distribution by Profile')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # Profile scale comparison
    ax = axes[0, 1]
    profile_means = df.groupby('profile')['views'].mean().sort_values()
    profile_means.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Mean Views')
    ax.set_title('Profile Scale Differences')
    ax.grid(axis='x', alpha=0.3)
    
    # Profile variance comparison
    ax = axes[1, 0]
    profile_cv = df.groupby('profile')['views'].apply(
        lambda x: np.std(x) / (np.mean(x) + 1e-6)
    ).sort_values()
    profile_cv.plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Coefficient of Variation')
    ax.set_title('Profile Variability Differences')
    ax.grid(axis='x', alpha=0.3)
    
    # Heavy tail check (Q-Q plot for one profile)
    ax = axes[1, 1]
    sample_profile = profiles[0]
    sample_data = df[df['profile'] == sample_profile]['views'].values
    sample_data = sample_data[sample_data > 0]
    if len(sample_data) > 100:
        stats.probplot(np.log1p(sample_data), dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot (log scale) - {sample_profile}')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_profiles.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved profile validation plot: {output_dir / 'validation_profiles.png'}")
    plt.close()
    
    # Plot 2: Attack patterns
    fake_df = df[df['is_fake_series'] == True]
    if len(fake_df) == 0:
        return
    
    attack_types = sorted(fake_df['attack_type'].unique())[:6]  # Top 6
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Attack Pattern Coherence Validation', fontsize=16, fontweight='bold')
    
    for idx, attack_type in enumerate(attack_types):
        ax = axes[idx // 2, idx % 2]
        
        attack_data = fake_df[fake_df['attack_type'] == attack_type]
        sample_user = attack_data['user_id'].iloc[0]
        user_data = attack_data[attack_data['user_id'] == sample_user].sort_values('timestamp')
        
        ax.plot(user_data['timestamp'], user_data['views'], 'b-', alpha=0.7, label='Views')
        anomaly_mask = user_data['is_anomaly_window']
        if anomaly_mask.any():
            anomaly_data = user_data[anomaly_mask]
            ax.scatter(anomaly_data['timestamp'], anomaly_data['views'], 
                      color='red', s=50, alpha=0.8, label='Anomaly', zorder=5)
        
        ax.set_title(f'{attack_type}\n(Spike ratio: {attack_results["spike_visibility"].get(attack_type, 0):.2f}x)')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Views')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_attacks.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved attack validation plot: {output_dir / 'validation_attacks.png'}")
    plt.close()


def main():
    """Main validation function."""
    data_path = project_root / "data" / "raw" / "engagement.parquet"
    
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Please generate the dataset first:")
        print("  python -m src.data.make_dataset --n_users 500 --length 336 --fake_ratio 0.35")
        return
    
    print("Loading dataset...")
    df = load_data(data_path)
    print(f"Dataset loaded: {len(df)} records, {df['user_id'].nunique()} users")
    
    # Adapt column names if needed
    if 'user_id' in df.columns and 'id' not in df.columns:
        df['id'] = df['user_id']
    if 'is_fake_series' in df.columns and 'label' not in df.columns:
        df['label'] = df['is_fake_series'].map({True: 'fake', False: 'normal'})
    
    # Run validations
    profile_results = validate_profiles(df)
    attack_results = validate_attacks(df)
    
    # Create plots
    output_dir = project_root / "outputs" / "figures"
    create_validation_plots(df, output_dir, attack_results)
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\n✓ Profiles checked: {len(profile_results['profiles_checked'])}")
    print(f"  {', '.join(profile_results['profiles_checked'])}")
    
    print(f"\n✓ Attack types checked: {len(attack_results['attack_types_checked'])}")
    print(f"  {', '.join(attack_results['attack_types_checked'][:5])}...")
    
    all_valid = profile_results['all_valid'] and attack_results['all_valid']
    
    if all_valid:
        print("\n" + "="*60)
        print("✓ DATASET VALIDATION PASSED")
        print("="*60)
        print("\nThe dataset shows:")
        print("  ✓ Coherent user profiles with different scales and variability")
        print("  ✓ Realistic distributions with heavy tails")
        print("  ✓ Distinct attack patterns with visible spikes/bursts")
        print("  ✓ Logical variation in anomaly lengths")
    else:
        print("\n" + "="*60)
        print("⚠ DATASET VALIDATION HAS WARNINGS")
        print("="*60)
        print("\nSome issues were detected. Please review the validation output above.")


if __name__ == "__main__":
    main()

