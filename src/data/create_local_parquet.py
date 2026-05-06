"""Script to generate a lightweight realistic parquet for local testing."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import build_synthetic_parquet
from src.utils.logging import setup_unified_logging

def main():
    setup_unified_logging(level="INFO")
    
    out_path = Path("data/processed/local_test.parquet")
    
    build_synthetic_parquet(
        out_path=out_path,
        num_samples=200,      # Lightweight: Only 200 samples
        frame_length=200,     # Lightweight: Shorter sequences (real is ~2000)
        num_regimes=4,        # Enough regimes to be interesting
        feature_dtype="float16" # Keep it lightweight
    )
    
    print(f"\nSuccess! Created lightweight parquet at {out_path}")
    print("You can now run: make run CONFIG=local_test")

if __name__ == "__main__":
    main()
