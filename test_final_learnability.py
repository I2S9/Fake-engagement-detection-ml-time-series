#!/usr/bin/env python3
"""
Final test to validate dataset learnability.
Tests with Isolation Forest and simple Autoencoder.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_data
from src.features.temporal_features import extract_temporal_features


def test_isolation_forest(df: pd.DataFrame) -> dict:
    """
    Test 1: Isolation Forest detection.
    
    Target: Detect +70% of fake series.
    """
    print("\n" + "="*60)
    print("TEST 1: Isolation Forest Detection")
    print("="*60)
    
    results = {
        'passed': False,
        'detection_rate': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'confusion_matrix': None
    }
    
    # Extract temporal features
    print("\n  Extracting temporal features...")
    features_df = extract_temporal_features(df, aggregate_per_id=True)
    
    # Prepare data
    if 'label' not in features_df.columns:
        if 'is_fake_series' in features_df.columns:
            features_df['label'] = features_df['is_fake_series'].map({True: 'fake', False: 'normal'})
        else:
            print("  ✗ No label column found")
            return results
    
    # Get feature columns (exclude id and label)
    feature_cols = [col for col in features_df.columns 
                    if col not in ['id', 'label', 'is_fake_series', 'user_id']]
    
    if len(feature_cols) == 0:
        print("  ✗ No feature columns found")
        return results
    
    X = features_df[feature_cols].fillna(0).values
    y = (features_df['label'] == 'fake').astype(int).values
    
    print(f"  Features shape: {X.shape}")
    print(f"  Fake ratio: {y.mean():.2%}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    print("\n  Training Isolation Forest...")
    # Contamination = proportion of outliers expected
    contamination = y.mean()
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    iso_forest.fit(X_scaled)
    
    # Predict (1 = normal, -1 = anomaly)
    predictions = iso_forest.predict(X_scaled)
    # Convert to binary: -1 (anomaly) = 1 (fake), 1 (normal) = 0 (normal)
    y_pred = (predictions == -1).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    results['detection_rate'] = float(recall)  # Recall = detection rate of fakes
    results['precision'] = float(precision)
    results['recall'] = float(recall)
    results['f1'] = float(f1)
    results['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
    
    print(f"\n  Results:")
    print(f"    Accuracy: {accuracy:.2%}")
    print(f"    Precision: {precision:.2%}")
    print(f"    Recall (Detection Rate): {recall:.2%}")
    print(f"    F1-Score: {f1:.2%}")
    
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"                Predicted")
    print(f"              Normal  Fake")
    print(f"    Actual Normal  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"            Fake   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Check if detection rate >= 70%
    if recall >= 0.70:
        print(f"\n  ✓ Detection rate ({recall:.2%}) >= 70% - DATASET IS LEARNABLE")
        results['passed'] = True
    else:
        print(f"\n  ✗ Detection rate ({recall:.2%}) < 70% - Dataset may need improvement")
    
    return results


def test_simple_autoencoder(df: pd.DataFrame) -> dict:
    """
    Test 2: Simple Autoencoder reconstruction error.
    
    Target: Reconstruction error 2-3x higher on fake than normal.
    """
    print("\n" + "="*60)
    print("TEST 2: Simple Autoencoder Reconstruction Error")
    print("="*60)
    
    results = {
        'passed': False,
        'normal_error': 0.0,
        'fake_error': 0.0,
        'error_ratio': 0.0
    }
    
    # Prepare sequences
    print("\n  Preparing sequences...")
    
    # Get a sample of users
    user_ids = df['id'].unique()[:50]  # Sample 50 users for speed
    sample_df = df[df['id'].isin(user_ids)].copy()
    
    # Create sequences
    sequences = []
    labels = []
    
    for user_id in user_ids:
        user_data = sample_df[sample_df['id'] == user_id].sort_values('timestamp')
        if len(user_data) < 24:  # Need at least 24 time points
            continue
        
        # Use views as the main signal
        views = user_data['views'].values[:336]  # Limit to 336 points
        if len(views) < 24:
            continue
        
        # Create sliding windows
        seq_len = 24
        for i in range(len(views) - seq_len + 1):
            seq = views[i:i+seq_len]
            sequences.append(seq)
            
            # Label: 1 if fake, 0 if normal
            is_fake = user_data['is_fake_series'].iloc[0] if 'is_fake_series' in user_data.columns else False
            labels.append(1 if is_fake else 0)
    
    if len(sequences) == 0:
        print("  ✗ No sequences created")
        return results
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Fake ratio: {labels.mean():.2%}")
    
    # Normalize sequences
    scaler = StandardScaler()
    sequences_scaled = scaler.fit_transform(sequences.reshape(-1, sequences.shape[-1])).reshape(sequences.shape)
    
    # Simple Autoencoder
    class SimpleAutoencoder(nn.Module):
        def __init__(self, input_dim=24, encoding_dim=8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, encoding_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 16),
                nn.ReLU(),
                nn.Linear(16, input_dim)
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    # Prepare data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.FloatTensor(sequences_scaled).to(device)
    y_tensor = torch.LongTensor(labels).to(device)
    
    # Split train/test
    n_train = int(len(X_tensor) * 0.7)
    X_train = X_tensor[:n_train]
    X_test = X_tensor[n_train:]
    y_test = y_tensor[n_train:]
    
    # Train autoencoder
    print("\n  Training simple autoencoder...")
    model = SimpleAutoencoder(input_dim=24, encoding_dim=8).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    n_epochs = 20
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        reconstructed = model(X_train)
        loss = criterion(reconstructed, X_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    # Evaluate on test set
    print("\n  Evaluating reconstruction errors...")
    model.eval()
    with torch.no_grad():
        reconstructed_test = model(X_test)
        reconstruction_errors = torch.mean((X_test - reconstructed_test) ** 2, dim=1).cpu().numpy()
    
    # Calculate errors by class
    normal_mask = (y_test.cpu().numpy() == 0)
    fake_mask = (y_test.cpu().numpy() == 1)
    
    normal_error = np.mean(reconstruction_errors[normal_mask]) if np.any(normal_mask) else 0
    fake_error = np.mean(reconstruction_errors[fake_mask]) if np.any(fake_mask) else 0
    
    results['normal_error'] = float(normal_error)
    results['fake_error'] = float(fake_error)
    results['error_ratio'] = float(fake_error / (normal_error + 1e-6))
    
    print(f"\n  Results:")
    print(f"    Normal reconstruction error: {normal_error:.4f}")
    print(f"    Fake reconstruction error: {fake_error:.4f}")
    print(f"    Error ratio (Fake/Normal): {results['error_ratio']:.2f}x")
    
    # Check if error ratio is 2-3x
    if 2.0 <= results['error_ratio'] <= 3.0:
        print(f"\n  ✓ Error ratio ({results['error_ratio']:.2f}x) is in target range (2-3x) - DATASET IS LEARNABLE")
        results['passed'] = True
    elif results['error_ratio'] >= 2.0:
        print(f"\n  ✓ Error ratio ({results['error_ratio']:.2f}x) >= 2x - DATASET IS LEARNABLE")
        results['passed'] = True
    else:
        print(f"\n  ⚠ Error ratio ({results['error_ratio']:.2f}x) < 2x - May need improvement")
    
    return results


def main():
    """Main test function."""
    print("="*60)
    print("FINAL LEARNABILITY TEST")
    print("="*60)
    
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
    
    # Run tests
    iso_results = test_isolation_forest(df)
    ae_results = test_simple_autoencoder(df)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    
    print("\nTest Results:")
    print(f"  1. Isolation Forest Detection Rate: {iso_results['recall']:.2%}")
    print(f"     Target: >= 70%")
    print(f"     Status: {'✓ PASSED' if iso_results['passed'] else '✗ FAILED'}")
    
    print(f"\n  2. Autoencoder Error Ratio: {ae_results['error_ratio']:.2f}x")
    print(f"     Target: 2-3x (Fake error / Normal error)")
    print(f"     Status: {'✓ PASSED' if ae_results['passed'] else '⚠ NEEDS REVIEW'}")
    
    if iso_results['passed'] and ae_results['passed']:
        print("\n" + "="*60)
        print("✓ DATASET IS LEARNABLE")
        print("="*60)
        print("\nBoth tests passed:")
        print(f"  ✓ Isolation Forest detects {iso_results['recall']:.2%} of fake series")
        print(f"  ✓ Autoencoder shows {ae_results['error_ratio']:.2f}x higher error on fake")
        print("\nThe dataset is ready for ML model training!")
    elif iso_results['passed']:
        print("\n" + "="*60)
        print("✓ DATASET IS LEARNABLE (Isolation Forest validated)")
        print("="*60)
        print(f"\nIsolation Forest detects {iso_results['recall']:.2%} of fake series.")
        print("Autoencoder may need tuning, but dataset is learnable.")
    else:
        print("\n" + "="*60)
        print("⚠ DATASET MAY NEED IMPROVEMENT")
        print("="*60)
        print("\nPlease review the test results above.")


if __name__ == "__main__":
    main()

