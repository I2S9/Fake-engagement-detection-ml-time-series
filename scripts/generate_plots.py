"""
Script to generate all plots from notebooks.
This script executes notebook cells directly to generate plots.
"""
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# create output directory
output_dir = project_root / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# set plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

print("="*80)
print("GENERATING PLOTS FROM NOTEBOOKS")
print("="*80)
print(f"Output directory: {output_dir}")

# import project modules
from src.data.load_data import load_data
from src.data.make_dataset import generate_dataset

# ensure dataset exists
data_path = project_root / "data" / "raw" / "engagement.parquet"
if not data_path.exists():
    print("\nGenerating dataset...")
    df = generate_dataset(n_users=500, length=336, fake_ratio=0.35)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path, index=False)
    print(f"Dataset generated: {data_path}")
else:
    print(f"\nDataset exists: {data_path}")

# load dataset
print("\nLoading dataset...")
df = load_data(data_path)

# adapt column names
if 'user_id' in df.columns and 'id' not in df.columns:
    df['id'] = df['user_id']
if 'is_fake_series' in df.columns and 'label' not in df.columns:
    df['label'] = df['is_fake_series'].map({True: 'fake', False: 'normal'})

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================================
# NOTEBOOK 01: EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("NOTEBOOK 01: EXPLORATION")
print("="*80)

metrics = ['views', 'likes', 'comments', 'shares']

# Plot 1: Histograms
print("\nGenerating histograms...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    normal_data = df[df['label'] == 'normal'][metric].dropna()
    fake_data = df[df['label'] == 'fake'][metric].dropna()
    
    if len(normal_data) > 0 and len(fake_data) > 0:
        ax.hist(normal_data, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        ax.hist(fake_data, bins=50, alpha=0.6, label='Fake', color='red', density=True)
        ax.set_yscale('log')
    
    ax.set_xlabel(metric.capitalize(), fontsize=12)
    ax.set_ylabel('Density (log scale)', fontsize=12)
    ax.set_title(f'Distribution of {metric.capitalize()}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "01_exploration_01_histograms.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_exploration_01_histograms.png")

# Plot 2: Average temporal patterns
print("\nGenerating average temporal patterns...")
df_normalized_time = df.copy()
df_normalized_time['time_normalized'] = df_normalized_time.groupby('id')['timestamp'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
)
n_bins = 50
df_normalized_time['time_bin'] = pd.cut(df_normalized_time['time_normalized'], bins=n_bins, labels=False)
avg_curves = df_normalized_time.groupby(['label', 'time_bin'])[metrics].mean().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    normal_curve = avg_curves[avg_curves['label'] == 'normal']
    fake_curve = avg_curves[avg_curves['label'] == 'fake']
    
    ax.plot(normal_curve['time_bin'], normal_curve[metric], 
            label='Normal', linewidth=2, color='blue', marker='o', markersize=4)
    ax.plot(fake_curve['time_bin'], fake_curve[metric], 
            label='Fake', linewidth=2, color='red', marker='s', markersize=4)
    
    ax.set_xlabel('Normalized Time (0 = start, 1 = end)', fontsize=12)
    ax.set_ylabel(f'Average {metric.capitalize()}', fontsize=12)
    ax.set_title(f'Average {metric.capitalize()} Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "01_exploration_02_temporal_patterns.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_exploration_02_temporal_patterns.png")

# Plot 3: Heatmap comparison
print("\nGenerating heatmap comparison...")
mean_comparison = df.groupby('label')[metrics].mean().T
mean_comparison.columns = ['Normal', 'Fake']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(mean_comparison, annot=True, fmt='.0f', cmap='YlOrRd', 
            cbar_kws={'label': 'Mean Value'}, ax=axes[0], linewidths=0.5)
axes[0].set_title('Mean Engagement Metrics by Label', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Metric', fontsize=12)

ratio_comparison = (mean_comparison['Fake'] / (mean_comparison['Normal'] + 1e-6)).to_frame('Fake/Normal Ratio')
sns.heatmap(ratio_comparison, annot=True, fmt='.2f', cmap='RdYlGn_r', 
            center=1, vmin=0.5, vmax=2, cbar_kws={'label': 'Ratio'}, 
            ax=axes[1], linewidths=0.5)
axes[1].set_title('Fake/Normal Ratio by Metric', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Metric', fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / "01_exploration_03_heatmap_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_exploration_03_heatmap_comparison.png")

# Plot 4: Correlation heatmaps
print("\nGenerating correlation heatmaps...")
normal_df = df[df['label'] == 'normal'][metrics]
fake_df = df[df['label'] == 'fake'][metrics]
normal_corr = normal_df.corr()
fake_corr = fake_df.corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(normal_corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, square=True, ax=axes[0],
            cbar_kws={'label': 'Correlation'})
axes[0].set_title('Normal Engagement - Correlation Matrix', fontsize=14, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

sns.heatmap(fake_corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, square=True, ax=axes[1],
            cbar_kws={'label': 'Correlation'})
axes[1].set_title('Fake Engagement - Correlation Matrix', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(output_dir / "01_exploration_04_correlation_heatmaps.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_exploration_04_correlation_heatmaps.png")

# ============================================================================
# NOTEBOOK 02: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("NOTEBOOK 02: FEATURE ENGINEERING")
print("="*80)

from src.features.temporal_features import extract_temporal_features

print("\nExtracting temporal features...")
features_df = extract_temporal_features(
    df,
    id_column="id",
    timestamp_column="timestamp",
    window_sizes=[6, 12, 24],
    autocorr_lags=[1, 6, 12, 24],
    aggregate_per_id=True,
)
print(f"Features extracted: {features_df.shape}")

# Plot 1: Rolling features panel
print("\nGenerating rolling features panel...")
sample_user_id = df['id'].unique()[0]
sample_series = df[df['id'] == sample_user_id].sort_values('timestamp')

rolling_mean_6 = sample_series['views'].rolling(window=6, min_periods=1).mean()
rolling_std_6 = sample_series['views'].rolling(window=6, min_periods=1).std()
rolling_mean_24 = sample_series['views'].rolling(window=24, min_periods=1).mean()
rolling_std_24 = sample_series['views'].rolling(window=24, min_periods=1).std()

autocorr_lag_1 = sample_series['views'].rolling(window=12, min_periods=2).apply(
    lambda x: x.corr(x.shift(1)) if len(x.dropna()) > 1 else 0, raw=False
)

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

ax1 = axes[0]
ax1.plot(sample_series['timestamp'], sample_series['views'], 
         label='Views', linewidth=2, color='blue', alpha=0.7)
ax1.plot(sample_series['timestamp'], rolling_mean_6, 
         label='Rolling Mean (6h)', linewidth=1.5, color='green', linestyle='--')
ax1.plot(sample_series['timestamp'], rolling_mean_24, 
         label='Rolling Mean (24h)', linewidth=1.5, color='orange', linestyle='--')
ax1.fill_between(sample_series['timestamp'], 
                 rolling_mean_6 - rolling_std_6, 
                 rolling_mean_6 + rolling_std_6,
                 alpha=0.2, color='green', label='Rolling Std (6h)')
ax1.set_ylabel('Views', fontsize=12)
ax1.set_title(f'Time Series with Rolling Statistics - User: {sample_user_id}', 
              fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(sample_series['timestamp'], rolling_std_6, 
         label='Rolling Std (6h)', linewidth=1.5, color='red')
ax2.plot(sample_series['timestamp'], rolling_std_24, 
         label='Rolling Std (24h)', linewidth=1.5, color='purple')
ax2_twin = ax2.twinx()
ax2_twin.plot(sample_series['timestamp'], autocorr_lag_1, 
              label='Autocorr (lag=1)', linewidth=1.5, color='brown', linestyle=':')
ax2_twin.set_ylabel('Autocorrelation', fontsize=12, color='brown')
ax2_twin.tick_params(axis='y', labelcolor='brown')
ax2.set_xlabel('Timestamp', fontsize=12)
ax2.set_ylabel('Rolling Std', fontsize=12)
ax2.set_title('Feature Scores: Rolling Variance and Autocorrelation', 
              fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "02_feature_engineering_01_rolling_features.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_feature_engineering_01_rolling_features.png")

# Plot 2: PCA visualization
print("\nGenerating PCA visualization...")
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    feature_cols = [c for c in features_df.columns if c not in ['id', 'label']]
    X = features_df[feature_cols].fillna(0).values
    y = features_df['label'].map({'normal': 0, 'fake': 1}).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    normal_mask = y == 0
    fake_mask = y == 1
    
    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
               alpha=0.6, label='Normal', color='blue', s=30)
    ax.scatter(X_pca[fake_mask, 0], X_pca[fake_mask, 1], 
               alpha=0.6, label='Fake', color='red', s=30)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA Visualization - Feature Space', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_feature_engineering_02_pca.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_feature_engineering_02_pca.png")
except Exception as e:
    print(f"  Error generating PCA: {e}")

# Plot 3: Feature distributions
print("\nGenerating feature distributions...")
feature_cols = [c for c in features_df.columns if c not in ['id', 'label']]
key_features_viz = [
    'views_max_mean_ratio',
    'views_n_peaks',
    'views_entropy',
    'views_regularity',
    'views_autocorr_lag_1',
    'ratio_likes_views',
]
key_features_viz = [f for f in key_features_viz if f in features_df.columns]

n_features = len(key_features_viz)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten() if n_features > 1 else [axes]

for idx, feature in enumerate(key_features_viz):
    ax = axes[idx]
    normal_data = features_df[features_df['label'] == 'normal'][feature].dropna()
    fake_data = features_df[features_df['label'] == 'fake'][feature].dropna()
    
    ax.hist(normal_data, bins=30, alpha=0.6, label='Normal', color='blue', density=True)
    ax.hist(fake_data, bins=30, alpha=0.6, label='Fake', color='red', density=True)
    
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Distribution: {feature}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "02_feature_engineering_03_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_feature_engineering_03_distributions.png")

# Plot 4: Feature importance heatmap (if available)
print("\nGenerating feature importance heatmap...")
try:
    from scipy.stats import mannwhitneyu
    
    feature_importance = []
    for feature in feature_cols:
        normal_values = features_df[features_df['label'] == 'normal'][feature].dropna()
        fake_values = features_df[features_df['label'] == 'fake'][feature].dropna()
        
        if len(normal_values) > 0 and len(fake_values) > 0:
            try:
                stat, p_value = mannwhitneyu(normal_values, fake_values, alternative='two-sided')
                mean_diff = fake_values.mean() - normal_values.mean()
                pooled_std = np.sqrt((normal_values.std()**2 + fake_values.std()**2) / 2)
                effect_size = mean_diff / (pooled_std + 1e-6)
                
                feature_importance.append({
                    'feature': feature,
                    'normal_mean': normal_values.mean(),
                    'fake_mean': fake_values.mean(),
                    'effect_size': abs(effect_size),
                    'p_value': p_value,
                })
            except:
                pass
    
    if len(feature_importance) > 0:
        importance_df = pd.DataFrame(feature_importance).sort_values('effect_size', ascending=False)
        top_30 = importance_df.head(30)
        
        heatmap_data = top_30[['normal_mean', 'fake_mean', 'effect_size', 'p_value']].copy()
        heatmap_data['-log10(p_value)'] = -np.log10(heatmap_data['p_value'] + 1e-10)
        heatmap_data = heatmap_data[['normal_mean', 'fake_mean', 'effect_size', '-log10(p_value)']]
        heatmap_data.index = top_30['feature']
        
        fig, ax = plt.subplots(1, 1, figsize=(12, max(10, len(top_30) * 0.4)))
        sns.heatmap(heatmap_data.T, annot=False, fmt='.2f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Value'}, ax=ax, linewidths=0.5)
        ax.set_title('Top 30 Feature Importance Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / "02_feature_engineering_04_importance_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: 02_feature_engineering_04_importance_heatmap.png")
except Exception as e:
    print(f"  Error generating importance heatmap: {e}")

# Plot 5: Series with anomalies
print("\nGenerating series with anomalies...")
fake_users = df[df.get('is_fake_series', df.get('label') == 'fake')]['id'].unique()[:3]
normal_users = df[df.get('label', pd.Series(['normal'] * len(df))) == 'normal']['id'].unique()[:3]

if len(fake_users) > 0 and len(normal_users) > 0:
    fig, axes = plt.subplots(max(len(fake_users), len(normal_users)), 2, figsize=(16, 4 * max(len(fake_users), len(normal_users))))
    if len(fake_users) == 1 and len(normal_users) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, user_id in enumerate(normal_users[:3]):
        if idx < len(axes):
            ax = axes[idx, 0] if len(axes.shape) > 1 else axes[0]
            sample = df[df['id'] == user_id].sort_values('timestamp')
            anomaly_mask = sample.get('is_anomaly_window', pd.Series([False] * len(sample))).values
            
            ax.plot(sample['timestamp'], sample['views'], label='Views', linewidth=1.5, color='blue')
            if anomaly_mask.any():
                ax.scatter(sample['timestamp'][anomaly_mask], sample['views'].values[anomaly_mask],
                          marker='o', s=30, color='red', label='Anomaly', zorder=5)
            ax.set_title(f"Normal User: {user_id}", fontsize=12, fontweight='bold')
            ax.set_xlabel('Timestamp', fontsize=10)
            ax.set_ylabel('Views', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    for idx, user_id in enumerate(fake_users[:3]):
        if idx < len(axes):
            ax = axes[idx, 1] if len(axes.shape) > 1 else axes[1]
            sample = df[df['id'] == user_id].sort_values('timestamp')
            anomaly_mask = sample.get('is_anomaly_window', pd.Series([False] * len(sample))).values
            attack_type = sample.get('attack_type', pd.Series(['unknown'] * len(sample))).iloc[0]
            
            ax.plot(sample['timestamp'], sample['views'], label='Views', linewidth=1.5, color='blue')
            if anomaly_mask.any():
                ax.scatter(sample['timestamp'][anomaly_mask], sample['views'].values[anomaly_mask],
                          marker='o', s=30, color='red', label='Anomaly Window', zorder=5)
            ax.set_title(f"Fake User: {user_id} ({attack_type})", fontsize=12, fontweight='bold')
            ax.set_xlabel('Timestamp', fontsize=10)
            ax.set_ylabel('Views', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_exploration_05_series_with_anomalies.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_exploration_05_series_with_anomalies.png")

# Plot 6: Profile distributions
print("\nGenerating profile distributions...")
if 'profile' in df.columns:
    profiles = df['profile'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for profile in profiles:
        profile_data = df[df['profile'] == profile]['views']
        axes[0].hist(profile_data, bins=50, alpha=0.6, label=profile, density=True)
    axes[0].set_xlabel('Views', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Views Distribution by Profile', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    profile_order = sorted(profiles)
    views_data = [df[df['profile'] == p]['views'].values for p in profile_order]
    axes[1].boxplot(views_data, labels=profile_order)
    axes[1].set_ylabel('Views', fontsize=12)
    axes[1].set_title('Views Distribution by Profile (Box Plot)', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_exploration_06_profile_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_exploration_06_profile_distributions.png")

# Plot 7: Attack type distributions
print("\nGenerating attack type distributions...")
if 'attack_type' in df.columns:
    fake_df = df[df.get('is_fake_series', df.get('label') == 'fake')]
    if len(fake_df) > 0:
        attack_types = fake_df['attack_type'].unique()
        
        # calculate anomaly lengths
        anomaly_lengths = []
        for user_id in fake_df['id'].unique()[:100]:  # limit for performance
            user_data = fake_df[fake_df['id'] == user_id].sort_values('timestamp')
            attack_type = user_data['attack_type'].iloc[0]
            anomaly_mask = user_data.get('is_anomaly_window', pd.Series([False] * len(user_data))).values
            
            if anomaly_mask.any():
                in_anomaly = False
                current_length = 0
                for is_anom in anomaly_mask:
                    if is_anom:
                        current_length += 1
                        in_anomaly = True
                    elif in_anomaly:
                        anomaly_lengths.append({'attack_type': attack_type, 'length': current_length})
                        current_length = 0
                        in_anomaly = False
                if in_anomaly:
                    anomaly_lengths.append({'attack_type': attack_type, 'length': current_length})
        
        if len(anomaly_lengths) > 0:
            anomaly_df = pd.DataFrame(anomaly_lengths)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            for attack_type in attack_types:
                lengths = anomaly_df[anomaly_df['attack_type'] == attack_type]['length']
                if len(lengths) > 0:
                    axes[0].hist(lengths, bins=20, alpha=0.6, label=attack_type, density=True)
            axes[0].set_xlabel('Anomaly Length (time steps)', fontsize=12)
            axes[0].set_ylabel('Density', fontsize=12)
            axes[0].set_title('Anomaly Length Distribution by Attack Type', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            length_data = [anomaly_df[anomaly_df['attack_type'] == at]['length'].values for at in attack_types]
            axes[1].boxplot(length_data, labels=attack_types)
            axes[1].set_ylabel('Anomaly Length (time steps)', fontsize=12)
            axes[1].set_title('Anomaly Length by Attack Type (Box Plot)', fontsize=14, fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / "01_exploration_07_attack_type_distributions.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  Saved: 01_exploration_07_attack_type_distributions.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

plot_files = list(output_dir.glob("*.png"))
print(f"\nGenerated {len(plot_files)} PNG files in {output_dir}:")
for plot_file in sorted(plot_files):
    size_kb = plot_file.stat().st_size / 1024
    print(f"  - {plot_file.name} ({size_kb:.1f} KB)")

# ============================================================================
# NOTEBOOK 03: MODELING (if models exist)
# ============================================================================
print("\n" + "="*80)
print("NOTEBOOK 03: MODELING")
print("="*80)

models_dir = project_root / "models"
if models_dir.exists():
    print("\nModels directory found. Generating modeling plots...")
    
    # Check for training history files
    training_history_files = list(models_dir.glob("**/training_history*.json"))
    if len(training_history_files) > 0:
        print("  Found training history files")
        # Could generate loss curves here if needed
    
    # Check for model comparison results
    comparison_file = models_dir / "model_comparison_results.csv"
    if comparison_file.exists():
        print("  Found model comparison results")
        try:
            comparison_df = pd.read_csv(comparison_file, index_col=0)
            
            # Create metrics heatmap
            metrics_to_plot = ['auc', 'precision', 'recall', 'f1', 'false_positive_rate']
            metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
            
            if len(metrics_to_plot) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(max(10, len(comparison_df) * 1.5), 6))
                sns.heatmap(comparison_df[metrics_to_plot].T, annot=True, fmt='.3f', 
                            cmap='YlOrRd', cbar_kws={'label': 'Score'}, ax=ax, linewidths=0.5)
                ax.set_title('Model Performance Heatmap', fontsize=16, fontweight='bold')
                ax.set_xlabel('Model', fontsize=12)
                ax.set_ylabel('Metric', fontsize=12)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(output_dir / "03_modeling_01_metrics_heatmap.png", dpi=150, bbox_inches='tight')
                plt.close()
                print("  Saved: 03_modeling_01_metrics_heatmap.png")
        except Exception as e:
            print(f"  Error generating comparison heatmap: {e}")
else:
    print("\nModels directory not found. Skipping modeling plots.")
    print("  (Train models first to generate modeling visualizations)")

# ============================================================================
# NOTEBOOK 04: EVALUATION (if models exist)
# ============================================================================
print("\n" + "="*80)
print("NOTEBOOK 04: EVALUATION")
print("="*80)

if models_dir.exists():
    print("\nModels directory found. Generating evaluation plots...")
    
    # Generate dummy ROC/PR curves for demonstration
    print("\nGenerating ROC/PR curves (example)...")
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    # Create example data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.binomial(1, 0.35, n_samples)
    y_proba = np.random.beta(2, 5, n_samples)
    y_proba[y_true == 1] = np.random.beta(5, 2, y_true.sum())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = np.trapz(tpr, fpr)
    axes[0].plot(fpr, tpr, linewidth=2, label=f'Example Model (AUC={roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve - Example', fontsize=16, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = np.trapz(precision, recall)
    axes[1].plot(recall, precision, linewidth=2, label=f'Example Model (AUC={pr_auc:.3f})')
    axes[1].axhline(y=y_true.mean(), color='k', linestyle='--', linewidth=1, label='Baseline')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve - Example', fontsize=16, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_evaluation_01_roc_pr_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_evaluation_01_roc_pr_curves.png")
    
    # Score distributions
    print("\nGenerating score distributions...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    normal_scores = y_proba[y_true == 0]
    fake_scores = y_proba[y_true == 1]
    ax.hist(normal_scores, bins=50, alpha=0.6, label="Normal", color="blue", density=True, 
            histtype="step", linewidth=2)
    ax.hist(fake_scores, bins=50, alpha=0.6, label="Fake", color="red", density=True, 
            histtype="step", linewidth=2, linestyle="--")
    ax.set_xlabel("Prediction Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution - Example Model", fontsize=14, fontweight="bold")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1)
    plt.tight_layout()
    plt.savefig(output_dir / "04_evaluation_02_score_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_evaluation_02_score_distributions.png")
else:
    print("\nModels directory not found. Skipping evaluation plots.")
    print("  (Train models first to generate evaluation visualizations)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

plot_files = list(output_dir.glob("*.png"))
print(f"\nGenerated {len(plot_files)} PNG files in {output_dir}:")
for plot_file in sorted(plot_files):
    size_kb = plot_file.stat().st_size / 1024
    print(f"  - {plot_file.name} ({size_kb:.1f} KB)")

print("\n" + "="*80)
print("SUCCESS: All plots generated!")
print("="*80)
print(f"\nAll PNG files are saved in: {output_dir}")
print("\nTo view the plots:")
print(f"  1. Open the directory: {output_dir}")
print("  2. Or use the notebooks to see interactive plots")
print("\nNote: Some plots require trained models.")
print("      Run training scripts first to generate model-specific visualizations.")

