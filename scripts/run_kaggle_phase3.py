"""
Kaggle / Google Colab Execution Script for Phase 3:
Dense Semantic Retrieval, Multi-Model Comparison, Chunk Pooling,
FAISS Indexing, Session Checkpointing, and Dev-Tuned Hybrid Fusion.

Usage in Kaggle / Colab:
    !python scripts/run_kaggle_phase3.py --dataset il_pcr --batch_size 32
"""

import os
import sys
import subprocess
import argparse
import torch

# Ensure environment flags prevent TensorFlow DLL conflicts
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def check_and_install_dependencies():
    """Ensures all necessary packages are installed in the cloud environment."""
    required = ["rank_bm25", "sentence_transformers", "faiss-cpu", "tabulate", "pydantic", "pyyaml", "scipy"]
    installed = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            installed.append(pkg)
    if installed:
        print(f"Installing missing dependencies: {installed}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + installed)


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 on Kaggle / Colab GPU")
    parser.add_argument("--dataset", type=str, default="il_pcr", help="Dataset name: il_pcr or sample")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dense encoding on GPU")
    parser.add_argument("--top_k", type=int, default=50, help="Retrieval candidate pool top-k")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Checkpoint frequency")
    parser.add_argument("--max_cases", type=int, default=None, help="Subsample cases (optional)")
    parser.add_argument("--max_queries", type=int, default=None, help="Subsample queries (optional)")
    args = parser.parse_args()

    check_and_install_dependencies()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"STARTING PHASE 3 DENSE & HYBRID BENCHMARK")
    print(f"Hardware Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Allocated VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Dataset: {args.dataset} | Batch Size: {args.batch_size} | Top-K: {args.top_k}")
    print("=" * 80)

    # Launch Phase 3 Runner with exact parameters
    cmd = [
        sys.executable, "run_phase3_dense_hybrid.py",
        "--dataset", args.dataset,
        "--device", device,
        "--batch_size", str(args.batch_size),
        "--top_k", str(args.top_k),
        "--checkpoint_interval", str(args.checkpoint_interval)
    ]
    if args.max_cases:
        cmd.extend(["--max_cases", str(args.max_cases)])
    if args.max_queries:
        cmd.extend(["--max_queries", str(args.max_queries)])

    print(f"Executing: {' '.join(cmd)}\n")
    subprocess.check_call(cmd)

    print("\n" + "=" * 80)
    print("PHASE 3 EXECUTION COMPLETED SUCCESSFULLY!")
    print("Artifacts generated:")
    print("  - Summary Report: experiments/results/phase3_dense_hybrid_summary.md")
    print("  - Metrics JSON:   experiments/results/phase3_dense_hybrid_summary.json")
    print("  - Significance:   experiments/results/statistical_significance_phase3.csv")
    print("  - Tuning Grid:    experiments/results/hybrid_fusion_dev_tuning_results.csv")
    print("  - Run Files:      experiments/runs/")
    print("=" * 80)


if __name__ == "__main__":
    main()
