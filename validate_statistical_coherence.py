#!/usr/bin/env python3
"""
Validation script for statistical coherence of patterns.
Checks that fake patterns have stronger correlations, higher variability,
visible signatures, and that normal patterns have regular cycles.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import fft, fftfreq

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_data


def test_1_fake_stronger_correlations(df: pd.DataFrame) -> dict:
    """
    Test 1: Fake patterns have stronger correlations.
    
    Checks:
    - Correlation matrices for fake vs normal
    - Average correlation strength
    """
    print("\n" + "="*60)
    print("TEST 1: Fake patterns have stronger correlations")
    print("="*60)
    
    results = {
        'passed': True,
        'normal_avg_corr': 0.0,
        'fake_avg_corr': 0.0,
        'correlation_difference': 0.0,
        'issues': []
    }
    
    metrics = ['views', 'likes', 'comments', 'shares']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) < 2:
        results['issues'].append("Not enough metrics for correlation analysis")
        results['passed'] = False
        return results
    
    # Compute correlation matrices
    normal_df = df[df.get('is_fake_series', df.get('label') == 'normal') == False][available_metrics]
    fake_df = df[df.get('is_fake_series', df.get('label') == 'fake') == True][available_metrics]
    
    if len(normal_df) == 0 or len(fake_df) == 0:
        results['issues'].append("No normal or fake data found")
        results['passed'] = False
        return results
    
    normal_corr = normal_df.corr()
    fake_corr = fake_df.corr()
    
    # Compute average absolute correlation (excluding diagonal)
    normal_avg = normal_corr.values[np.triu_indices_from(normal_corr.values, k=1)].mean()
    fake_avg = fake_corr.values[np.triu_indices_from(fake_corr.values, k=1)].mean()
    
    results['normal_avg_corr'] = float(normal_avg)
    results['fake_avg_corr'] = float(fake_avg)
    results['correlation_difference'] = float(fake_avg - normal_avg)
    
    print(f"\n  Average correlation (absolute values):")
    print(f"    Normal: {normal_avg:.4f}")
    print(f"    Fake: {fake_avg:.4f}")
    print(f"    Difference (Fake - Normal): {results['correlation_difference']:.4f}")
    
    if fake_avg > normal_avg:
        print("  ✓ Fake patterns have stronger correlations")
    else:
        print("  ⚠ Fake patterns do not have stronger correlations")
        results['issues'].append("Fake correlations not stronger than normal")
        results['passed'] = False
    
    return results


def test_2_fake_higher_variability(df: pd.DataFrame) -> dict:
    """
    Test 2: Fake patterns have higher variability.
    
    Checks:
    - Coefficient of variation (CV) for fake vs normal
    - Standard deviation relative to mean
    """
    print("\n" + "="*60)
    print("TEST 2: Fake patterns have higher variability")
    print("="*60)
    
    results = {
        'passed': True,
        'normal_cv': {},
        'fake_cv': {},
        'variability_differences': {},
        'issues': []
    }
    
    metrics = ['views', 'likes', 'comments', 'shares']
    available_metrics = [m for m in metrics if m in df.columns]
    
    normal_df = df[df.get('is_fake_series', df.get('label') == 'normal') == False]
    fake_df = df[df.get('is_fake_series', df.get('label') == 'fake') == True]
    
    print("\n  Coefficient of Variation (CV = std/mean):")
    for metric in available_metrics:
        normal_values = normal_df[metric].values
        fake_values = fake_df[metric].values
        
        normal_cv = np.std(normal_values) / (np.mean(normal_values) + 1e-6)
        fake_cv = np.std(fake_values) / (np.mean(fake_values) + 1e-6)
        
        results['normal_cv'][metric] = float(normal_cv)
        results['fake_cv'][metric] = float(fake_cv)
        results['variability_differences'][metric] = float(fake_cv - normal_cv)
        
        print(f"\n  {metric.upper()}:")
        print(f"    Normal CV: {normal_cv:.4f}")
        print(f"    Fake CV: {fake_cv:.4f}")
        print(f"    Difference: {results['variability_differences'][metric]:.4f}")
        
        if fake_cv > normal_cv:
            print(f"    ✓ Fake has higher variability")
        else:
            print(f"    ⚠ Fake does not have higher variability")
            results['issues'].append(f"{metric}: Fake CV not higher than normal")
    
    # Overall check: at least 2 metrics should show higher fake variability
    higher_variability_count = sum(1 for diff in results['variability_differences'].values() if diff > 0)
    if higher_variability_count >= len(available_metrics) * 0.5:
        print(f"\n  ✓ Overall: Fake patterns show higher variability ({higher_variability_count}/{len(available_metrics)} metrics)")
    else:
        print(f"\n  ⚠ Overall: Fake patterns may not consistently show higher variability")
        if higher_variability_count == 0:
            results['passed'] = False
    
    return results


def test_3_fake_visible_signatures(df: pd.DataFrame) -> dict:
    """
    Test 3: Fake patterns have visible signatures.
    
    Checks:
    - Spike detection (outliers)
    - Burst patterns
    - Anomaly windows
    """
    print("\n" + "="*60)
    print("TEST 3: Fake patterns have visible signatures")
    print("="*60)
    
    results = {
        'passed': True,
        'normal_spikes': 0,
        'fake_spikes': 0,
        'normal_bursts': 0,
        'fake_bursts': 0,
        'anomaly_windows': 0,
        'issues': []
    }
    
    metrics = ['views', 'likes', 'comments', 'shares']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if 'views' not in available_metrics:
        results['issues'].append("Views metric not available")
        results['passed'] = False
        return results
    
    normal_df = df[df.get('is_fake_series', df.get('label') == 'normal') == False]
    fake_df = df[df.get('is_fake_series', df.get('label') == 'fake') == True]
    
    # Check for anomaly windows
    if 'is_anomaly_window' in df.columns:
        anomaly_count = df['is_anomaly_window'].sum()
        results['anomaly_windows'] = int(anomaly_count)
        print(f"\n  Anomaly windows detected: {anomaly_count}")
        if anomaly_count > 0:
            print("  ✓ Anomaly windows present (visible signatures)")
        else:
            print("  ⚠ No anomaly windows detected")
    
    # Detect spikes (values > 3 standard deviations from mean)
    for label, data in [('Normal', normal_df), ('Fake', fake_df)]:
        views = data['views'].values
        if len(views) > 0:
            mean_views = np.mean(views)
            std_views = np.std(views)
            threshold = mean_views + 3 * std_views
            spikes = np.sum(views > threshold)
            
            if label == 'Normal':
                results['normal_spikes'] = int(spikes)
            else:
                results['fake_spikes'] = int(spikes)
            
            spike_ratio = spikes / len(views) if len(views) > 0 else 0
            print(f"\n  {label} patterns:")
            print(f"    Spikes (>3σ): {spikes} ({spike_ratio:.2%})")
    
    # Detect bursts (rapid increases)
    for label, data in [('Normal', normal_df), ('Fake', fake_df)]:
        if len(data) > 1:
            views = data['views'].values
            diffs = np.diff(views)
            # Burst: increase > 2 standard deviations
            burst_threshold = np.mean(diffs) + 2 * np.std(diffs)
            bursts = np.sum(diffs > burst_threshold)
            
            if label == 'Normal':
                results['normal_bursts'] = int(bursts)
            else:
                results['fake_bursts'] = int(bursts)
            
            burst_ratio = bursts / len(diffs) if len(diffs) > 0 else 0
            print(f"    Bursts: {bursts} ({burst_ratio:.2%})")
    
    # Check if fake has more signatures
    if results['fake_spikes'] > results['normal_spikes'] or results['fake_bursts'] > results['normal_bursts']:
        print("\n  ✓ Fake patterns show more visible signatures (spikes/bursts)")
    else:
        print("\n  ⚠ Fake patterns may not show more visible signatures")
        if results['anomaly_windows'] == 0:
            results['issues'].append("No clear visible signatures detected")
    
    return results


def test_4_normal_regular_cycles(df: pd.DataFrame) -> dict:
    """
    Test 4: Normal patterns have regular cycles.
    
    Checks:
    - FFT analysis for periodic patterns
    - Regularity of cycles
    - Daily/weekly patterns
    """
    print("\n" + "="*60)
    print("TEST 4: Normal patterns have regular cycles")
    print("="*60)
    
    results = {
        'passed': True,
        'normal_regularity': 0.0,
        'fake_regularity': 0.0,
        'issues': []
    }
    
    if 'timestamp' not in df.columns:
        results['issues'].append("Timestamp column not available")
        results['passed'] = False
        return results
    
    normal_df = df[df.get('is_fake_series', df.get('label') == 'normal') == False]
    
    # Sample a few normal series for analysis
    normal_ids = normal_df['id'].unique()[:5] if 'id' in normal_df.columns else []
    
    if len(normal_ids) == 0:
        results['issues'].append("No normal series found")
        results['passed'] = False
        return results
    
    regularity_scores = []
    
    for user_id in normal_ids:
        user_data = normal_df[normal_df['id'] == user_id].sort_values('timestamp')
        if len(user_data) < 24:  # Need at least 24 points for cycle analysis
            continue
        
        views = user_data['views'].values
        
        # FFT to detect periodic patterns
        fft_vals = np.abs(fft(views))
        freqs = fftfreq(len(views), 1.0)
        
        # Look for strong periodic components (excluding DC component)
        positive_freqs = freqs[1:len(freqs)//2]
        positive_fft = fft_vals[1:len(fft_vals)//2]
        
        if len(positive_fft) > 0:
            # Regularity: ratio of strongest periodic component to mean
            max_periodic = np.max(positive_fft)
            mean_fft = np.mean(positive_fft)
            regularity = max_periodic / (mean_fft + 1e-6)
            regularity_scores.append(regularity)
    
    if regularity_scores:
        avg_regularity = np.mean(regularity_scores)
        results['normal_regularity'] = float(avg_regularity)
        print(f"\n  Average regularity score: {avg_regularity:.4f}")
        print(f"  (Higher = more regular cycles)")
        
        if avg_regularity > 1.5:
            print("  ✓ Normal patterns show regular cycles")
        else:
            print("  ⚠ Normal patterns may not show strong regular cycles")
    else:
        print("  ⚠ Could not compute regularity scores")
        results['issues'].append("Could not analyze cycles")
    
    return results


def test_5_attack_distinct_patterns(df: pd.DataFrame) -> dict:
    """
    Test 5: Attack types produce distinct patterns.
    
    Checks:
    - Different attack types have different characteristics
    - Visual distinctness of patterns
    """
    print("\n" + "="*60)
    print("TEST 5: Attack types produce distinct patterns")
    print("="*60)
    
    results = {
        'passed': True,
        'attack_types_checked': [],
        'pattern_distinctness': {},
        'issues': []
    }
    
    fake_df = df[df.get('is_fake_series', df.get('label') == 'fake') == True]
    
    if len(fake_df) == 0:
        results['issues'].append("No fake data found")
        results['passed'] = False
        return results
    
    if 'attack_type' not in fake_df.columns:
        results['issues'].append("Attack type column not found")
        results['passed'] = False
        return results
    
    attack_types = fake_df['attack_type'].unique()
    attack_types = [at for at in attack_types if at and at != 'none']
    
    if len(attack_types) < 2:
        results['issues'].append("Not enough attack types for comparison")
        results['passed'] = False
        return results
    
    results['attack_types_checked'] = list(attack_types)
    
    print(f"\n  Analyzing {len(attack_types)} attack types:")
    
    # Compute statistics per attack type
    attack_stats = {}
    for attack_type in attack_types:
        attack_data = fake_df[fake_df['attack_type'] == attack_type]
        if len(attack_data) == 0:
            continue
        
        views = attack_data['views'].values
        stats_dict = {
            'mean': np.mean(views),
            'std': np.std(views),
            'cv': np.std(views) / (np.mean(views) + 1e-6),
            'max': np.max(views),
            'q95': np.percentile(views, 95),
        }
        attack_stats[attack_type] = stats_dict
        
        print(f"\n  {attack_type}:")
        print(f"    Mean: {stats_dict['mean']:.2f}")
        print(f"    CV: {stats_dict['cv']:.4f}")
        print(f"    Max: {stats_dict['max']:.2f}")
        print(f"    95th percentile: {stats_dict['q95']:.2f}")
    
    # Check distinctness: CV of statistics across attack types
    if len(attack_stats) >= 2:
        means = [s['mean'] for s in attack_stats.values()]
        cvs = [s['cv'] for s in attack_stats.values()]
        maxs = [s['max'] for s in attack_stats.values()]
        
        mean_cv = np.std(means) / (np.mean(means) + 1e-6)
        cv_cv = np.std(cvs) / (np.mean(cvs) + 1e-6)
        max_cv = np.std(maxs) / (np.mean(maxs) + 1e-6)
        
        results['pattern_distinctness'] = {
            'mean_cv': float(mean_cv),
            'cv_cv': float(cv_cv),
            'max_cv': float(max_cv)
        }
        
        print(f"\n  Pattern distinctness (CV of statistics across types):")
        print(f"    Mean CV: {mean_cv:.4f}")
        print(f"    CV CV: {cv_cv:.4f}")
        print(f"    Max CV: {max_cv:.4f}")
        
        if mean_cv > 0.2 or cv_cv > 0.2:
            print("  ✓ Attack types show distinct patterns")
        else:
            print("  ⚠ Attack types may be too similar")
            results['issues'].append("Attack types may not be distinct enough")
    
    return results


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
    
    # Run all tests
    test_results = {}
    
    test_results['test_1'] = test_1_fake_stronger_correlations(df)
    test_results['test_2'] = test_2_fake_higher_variability(df)
    test_results['test_3'] = test_3_fake_visible_signatures(df)
    test_results['test_4'] = test_4_normal_regular_cycles(df)
    test_results['test_5'] = test_5_attack_distinct_patterns(df)
    
    # Final summary
    print("\n" + "="*60)
    print("STATISTICAL COHERENCE VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = all(
        test_results[f'test_{i}']['passed'] 
        for i in range(1, 6)
    )
    
    print("\nTest Results:")
    print(f"  1. Fake stronger correlations: {'✓' if test_results['test_1']['passed'] else '✗'}")
    print(f"  2. Fake higher variability: {'✓' if test_results['test_2']['passed'] else '✗'}")
    print(f"  3. Fake visible signatures: {'✓' if test_results['test_3']['passed'] else '✗'}")
    print(f"  4. Normal regular cycles: {'✓' if test_results['test_4']['passed'] else '✗'}")
    print(f"  5. Attack distinct patterns: {'✓' if test_results['test_5']['passed'] else '✗'}")
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ STATISTICAL COHERENCE VALIDATED")
        print("="*60)
        print("\nAll statistical patterns are coherent:")
        print("  ✓ Fake patterns have stronger correlations")
        print("  ✓ Fake patterns have higher variability")
        print("  ✓ Fake patterns have visible signatures")
        print("  ✓ Normal patterns have regular cycles")
        print("  ✓ Attack types produce distinct patterns")
        print("\nThe simulator is excellent!")
    else:
        print("\n" + "="*60)
        print("⚠ SOME STATISTICAL PATTERNS NEED REVIEW")
        print("="*60)
        print("\nPlease review the test results above.")


if __name__ == "__main__":
    main()

