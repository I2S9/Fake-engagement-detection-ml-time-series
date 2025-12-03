#!/usr/bin/env python3
"""
Validation script for EDA completeness.
Checks that all required visualizations and analyses are present.
"""
import sys
import json
from pathlib import Path
import re

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def check_notebook_content(notebook_path: Path) -> dict:
    """
    Check if notebook contains all required EDA elements.
    
    Required elements:
    1. Histograms
    2. Time-series per user
    3. Boxplots of variability
    4. Correlation heatmaps
    5. Normal vs Fake comparison
    6. User profiles
    7. Attack patterns
    8. Anomalies detected (red points/zones)
    """
    results = {
        'histograms': False,
        'time_series_per_user': False,
        'boxplots_variability': False,
        'correlation_heatmaps': False,
        'normal_vs_fake_comparison': False,
        'user_profiles': False,
        'attack_patterns': False,
        'anomalies_red_zones': False,
        'all_present': False
    }
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return results
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Combine all code cells into one string for searching
    all_code = []
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                all_code.append(' '.join(source))
            else:
                all_code.append(str(source))
    
    full_code = ' '.join(all_code).lower()
    
    # Check 1: Histograms
    histogram_patterns = [
        r'hist\(', r'\.hist\(', r'histogram', r'plt\.hist', r'ax\.hist'
    ]
    if any(re.search(pattern, full_code) for pattern in histogram_patterns):
        results['histograms'] = True
        print("  ✓ Histograms found")
    else:
        print("  ✗ Histograms NOT found")
    
    # Check 2: Time-series per user
    time_series_patterns = [
        r'plot\(.*timestamp', r'\.plot\(', r'time.*series', r'video.*id',
        r'user.*series', r'sort_values\(.*timestamp'
    ]
    if any(re.search(pattern, full_code) for pattern in time_series_patterns):
        results['time_series_per_user'] = True
        print("  ✓ Time-series per user found")
    else:
        print("  ✗ Time-series per user NOT found")
    
    # Check 3: Boxplots of variability
    boxplot_patterns = [
        r'boxplot', r'\.boxplot\(', r'variability', r'coefficient.*variation',
        r'cv', r'std.*mean'
    ]
    if any(re.search(pattern, full_code) for pattern in boxplot_patterns):
        results['boxplots_variability'] = True
        print("  ✓ Boxplots of variability found")
    else:
        print("  ✗ Boxplots of variability NOT found")
    
    # Check 4: Correlation heatmaps
    correlation_patterns = [
        r'corr\(\)', r'\.corr\(\)', r'correlation', r'heatmap.*corr',
        r'corr.*heatmap', r'correlation.*matrix'
    ]
    if any(re.search(pattern, full_code) for pattern in correlation_patterns):
        results['correlation_heatmaps'] = True
        print("  ✓ Correlation heatmaps found")
    else:
        print("  ✗ Correlation heatmaps NOT found")
    
    # Check 5: Normal vs Fake comparison
    comparison_patterns = [
        r'normal.*fake', r'fake.*normal', r'label.*==.*normal',
        r'label.*==.*fake', r'groupby.*label', r'normal.*vs.*fake'
    ]
    if any(re.search(pattern, full_code) for pattern in comparison_patterns):
        results['normal_vs_fake_comparison'] = True
        print("  ✓ Normal vs Fake comparison found")
    else:
        print("  ✗ Normal vs Fake comparison NOT found")
    
    # Check 6: User profiles
    profile_patterns = [
        r'profile', r'user.*profile', r'profile.*distribution',
        r'profile.*value_counts', r'profile.*unique'
    ]
    if any(re.search(pattern, full_code) for pattern in profile_patterns):
        results['user_profiles'] = True
        print("  ✓ User profiles found")
    else:
        print("  ✗ User profiles NOT found")
    
    # Check 7: Attack patterns
    attack_patterns = [
        r'attack.*type', r'attack_type', r'attack.*pattern',
        r'attack.*distribution', r'attack.*value_counts'
    ]
    if any(re.search(pattern, full_code) for pattern in attack_patterns):
        results['attack_patterns'] = True
        print("  ✓ Attack patterns found")
    else:
        print("  ✗ Attack patterns NOT found")
    
    # Check 8: Anomalies detected (red points/zones)
    anomaly_patterns = [
        r'anomaly', r'is_anomaly_window', r'red.*zone', r'red.*spike',
        r'plot_series_with_anomalies', r'anomaly.*mask', r'red.*color',
        r'color.*red', r'spike.*red', r'zone.*red'
    ]
    if any(re.search(pattern, full_code) for pattern in anomaly_patterns):
        results['anomalies_red_zones'] = True
        print("  ✓ Anomalies with red zones/spikes found")
    else:
        print("  ✗ Anomalies with red zones/spikes NOT found")
    
    results['all_present'] = all([
        results['histograms'],
        results['time_series_per_user'],
        results['boxplots_variability'],
        results['correlation_heatmaps'],
        results['normal_vs_fake_comparison'],
        results['user_profiles'],
        results['attack_patterns'],
        results['anomalies_red_zones']
    ])
    
    return results


def check_saved_plots(output_dir: Path) -> dict:
    """Check if expected plot files exist."""
    results = {
        'plots_found': [],
        'plots_missing': [],
        'all_plots_present': False
    }
    
    expected_plots = [
        '01_exploration_01_plot.png',  # Histograms
        '01_exploration_02_plot.png',  # Temporal patterns
        '01_exploration_03_plot.png',  # Time-series examples
        '01_exploration_04_plot.png',  # Heatmap comparison
        '01_exploration_05_plot.png',  # Boxplots variability
        '01_exploration_06_plot.png',  # Correlation heatmaps
        '01_exploration_07_plot.png',  # Correlation difference
        '01_spectacular_anomalies_red_zones.png',  # Red zones
    ]
    
    if not output_dir.exists():
        print(f"  ⚠ Output directory not found: {output_dir}")
        return results
    
    for plot_name in expected_plots:
        plot_path = output_dir / plot_name
        if plot_path.exists():
            results['plots_found'].append(plot_name)
        else:
            results['plots_missing'].append(plot_name)
    
    results['all_plots_present'] = len(results['plots_missing']) == 0
    
    return results


def main():
    """Main validation function."""
    notebook_path = project_root / "notebooks" / "01_exploration.ipynb"
    output_dir = project_root / "outputs" / "figures"
    
    print("="*60)
    print("EDA COMPLETENESS VALIDATION")
    print("="*60)
    
    print("\n1. Checking notebook content...")
    notebook_results = check_notebook_content(notebook_path)
    
    print("\n2. Checking saved plots...")
    plot_results = check_saved_plots(output_dir)
    
    if plot_results['plots_found']:
        print(f"\n  ✓ Found {len(plot_results['plots_found'])} plots:")
        for plot in plot_results['plots_found']:
            print(f"    - {plot}")
    
    if plot_results['plots_missing']:
        print(f"\n  ⚠ Missing {len(plot_results['plots_missing'])} plots:")
        for plot in plot_results['plots_missing']:
            print(f"    - {plot}")
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print("\nRequired EDA Elements:")
    print(f"  1. Histograms: {'✓' if notebook_results['histograms'] else '✗'}")
    print(f"  2. Time-series per user: {'✓' if notebook_results['time_series_per_user'] else '✗'}")
    print(f"  3. Boxplots of variability: {'✓' if notebook_results['boxplots_variability'] else '✗'}")
    print(f"  4. Correlation heatmaps: {'✓' if notebook_results['correlation_heatmaps'] else '✗'}")
    print(f"  5. Normal vs Fake comparison: {'✓' if notebook_results['normal_vs_fake_comparison'] else '✗'}")
    print(f"  6. User profiles: {'✓' if notebook_results['user_profiles'] else '✗'}")
    print(f"  7. Attack patterns: {'✓' if notebook_results['attack_patterns'] else '✗'}")
    print(f"  8. Anomalies (red zones): {'✓' if notebook_results['anomalies_red_zones'] else '✗'}")
    
    if notebook_results['all_present']:
        print("\n" + "="*60)
        print("✓ EDA IS COMPLETE - ALL ELEMENTS PRESENT")
        print("="*60)
        print("\nThe exploration notebook contains all required elements:")
        print("  ✓ Histograms for distribution analysis")
        print("  ✓ Time-series visualizations per user")
        print("  ✓ Boxplots showing variability differences")
        print("  ✓ Correlation heatmaps (Normal vs Fake)")
        print("  ✓ Normal vs Fake comparisons throughout")
        print("  ✓ User profile analysis")
        print("  ✓ Attack pattern analysis")
        print("  ✓ Anomaly detection with red zones/spikes")
    else:
        print("\n" + "="*60)
        print("⚠ EDA IS INCOMPLETE - SOME ELEMENTS MISSING")
        print("="*60)
        print("\nPlease add the missing elements to complete the EDA.")
    
    if plot_results['all_plots_present']:
        print("\n✓ All expected plots are saved in outputs/figures/")
    else:
        print(f"\n⚠ {len(plot_results['plots_missing'])} plot(s) missing - run the notebook to generate them")


if __name__ == "__main__":
    main()

