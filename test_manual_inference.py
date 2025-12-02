"""
Test manual inference with simple series.

This script tests the predict_from_series function with:
1. A fake series (with suspicious spike)
2. A normal series (gradual variation)
"""

import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.inference.inference_pipeline import predict_from_series


def test_fake_series():
    """Test with a series containing a suspicious spike."""
    print("=" * 80)
    print("TEST 1: Fake Series (with suspicious spike)")
    print("=" * 80)
    
    # series with sudden spike (suspicious pattern)
    sample_fake = [10, 15, 13, 14, 13, 200, 350, 400, 380, 12, 13]
    
    print(f"\nInput series: {sample_fake}")
    print(f"Series length: {len(sample_fake)}")
    print("\nThis series shows a suspicious pattern:")
    print("  - Normal values: 10-15")
    print("  - Sudden spike: 200-400 (unrealistic jump)")
    print("  - Returns to normal: 12-13")
    
    try:
        result = predict_from_series(sample_fake)
        
        print("\n" + "-" * 80)
        print("PREDICTION RESULT:")
        print("-" * 80)
        print(f"  Score (anomaly probability): {result['score']:.4f}")
        print(f"  Label: {result['label']}")
        print(f"  Is Fake: {result['is_fake']}")
        print("-" * 80)
        
        # validate result
        assert result['score'] >= 0.0 and result['score'] <= 1.0, "Score should be in [0, 1]"
        assert result['label'] in ['normal', 'fake'], "Label should be 'normal' or 'fake'"
        assert isinstance(result['is_fake'], bool), "is_fake should be boolean"
        
        # check if it detected the fake pattern
        if result['score'] >= 0.5:
            print("\n[OK] Model correctly identified suspicious pattern (score >= 0.5)")
        else:
            print(f"\n[WARNING] Model score is low ({result['score']:.4f}). "
                  "This might indicate the model needs more training or a different threshold.")
        
        return result
        
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_normal_series():
    """Test with a normal series with gradual variation."""
    print("\n" + "=" * 80)
    print("TEST 2: Normal Series (gradual variation)")
    print("=" * 80)
    
    # series with gradual variation (normal pattern)
    sample_normal = [10, 12, 11, 13, 14, 15, 16, 14, 13, 12, 11]
    
    print(f"\nInput series: {sample_normal}")
    print(f"Series length: {len(sample_normal)}")
    print("\nThis series shows a normal pattern:")
    print("  - Gradual variation: 10-16")
    print("  - No sudden spikes")
    print("  - Natural progression")
    
    try:
        result = predict_from_series(sample_normal)
        
        print("\n" + "-" * 80)
        print("PREDICTION RESULT:")
        print("-" * 80)
        print(f"  Score (anomaly probability): {result['score']:.4f}")
        print(f"  Label: {result['label']}")
        print(f"  Is Fake: {result['is_fake']}")
        print("-" * 80)
        
        # validate result
        assert result['score'] >= 0.0 and result['score'] <= 1.0, "Score should be in [0, 1]"
        assert result['label'] in ['normal', 'fake'], "Label should be 'normal' or 'fake'"
        assert isinstance(result['is_fake'], bool), "is_fake should be boolean"
        
        # check if it detected the normal pattern
        if result['score'] < 0.5:
            print("\n[OK] Model correctly identified normal pattern (score < 0.5)")
        else:
            print(f"\n[WARNING] Model score is high ({result['score']:.4f}) for normal series. "
                  "This might indicate a false positive.")
        
        return result
        
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MANUAL INFERENCE TEST")
    print("=" * 80)
    print("\nThis script tests the predict_from_series function with:")
    print("  1. A fake series (with suspicious spike)")
    print("  2. A normal series (gradual variation)")
    print("\n" + "=" * 80)
    
    # run tests
    result_fake = test_fake_series()
    result_normal = test_normal_series()
    
    # summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if result_fake and result_normal:
        print("\nBoth tests completed successfully!")
        print(f"\nFake series: score={result_fake['score']:.4f}, label={result_fake['label']}")
        print(f"Normal series: score={result_normal['score']:.4f}, label={result_normal['label']}")
        
        # check if results make sense
        if result_fake['score'] > result_normal['score']:
            print("\n[OK] Model correctly assigns higher score to fake series")
        else:
            print("\n[WARNING] Model assigns higher score to normal series. "
                  "This might indicate the model needs more training.")
    else:
        print("\nSome tests failed. Please check the errors above.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

