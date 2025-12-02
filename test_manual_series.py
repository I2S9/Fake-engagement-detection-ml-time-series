"""
Simple test script for manual inference.

Usage:
    python test_manual_series.py

Or inline:
    python - << EOF
    from src.inference.inference_pipeline import predict_from_series
    
    sample = [10, 15, 13, 14, 13, 200, 350, 400, 380, 12, 13]
    
    print(predict_from_series(sample))
    EOF
"""

from src.inference.inference_pipeline import predict_from_series

# Test with fake series (suspicious spike)
sample_fake = [10, 15, 13, 14, 13, 200, 350, 400, 380, 12, 13]

print("=" * 80)
print("TEST 1: Fake Series (with suspicious spike)")
print("=" * 80)
print(f"\nInput: {sample_fake}")
result_fake = predict_from_series(sample_fake)
print(f"\nResult: {result_fake}")
print(f"\nExpected: score ~0.8-0.95, label='fake'")
print(f"Actual: score={result_fake['score']:.4f}, label='{result_fake['label']}'")

# Test with normal series (gradual variation)
sample_normal = [10, 12, 11, 13, 14, 15, 16, 14, 13, 12, 11]

print("\n" + "=" * 80)
print("TEST 2: Normal Series (gradual variation)")
print("=" * 80)
print(f"\nInput: {sample_normal}")
result_normal = predict_from_series(sample_normal)
print(f"\nResult: {result_normal}")
print(f"\nExpected: score < 0.5, label='normal'")
print(f"Actual: score={result_normal['score']:.4f}, label='{result_normal['label']}'")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nFake series detected: {result_fake['is_fake']} (score: {result_fake['score']:.4f})")
print(f"Normal series detected: {result_normal['is_fake']} (score: {result_normal['score']:.4f})")
print("=" * 80)

