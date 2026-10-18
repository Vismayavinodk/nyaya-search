"""
Unit tests for core utilities: seeds, logger, and configuration parser.
"""

import os
import random
import numpy as np
import torch
import pytest

from src.utils.seed import seed_everything
from src.utils.logger import get_logger
from src.utils.config_parser import load_config, Config


def test_seed_reproducibility():
    """Verify that setting the seed produces identical random sequences."""
    seed_everything(42)
    val1_py = random.random()
    val1_np = np.random.rand(5)
    val1_th = torch.rand(5)

    seed_everything(42)
    val2_py = random.random()
    val2_np = np.random.rand(5)
    val2_th = torch.rand(5)

    assert val1_py == val2_py, "Python random values must match for identical seed"
    np.testing.assert_array_equal(val1_np, val2_np, err_msg="NumPy random arrays must match")
    assert torch.equal(val1_th, val2_th), "PyTorch random tensors must match"


def test_config_loading_and_attributes():
    """Verify config loading and hierarchical dot-attribute access."""
    config_path = "configs/config.yaml"
    assert os.path.exists(config_path), f"Master config not found at {config_path}"

    cfg = load_config(config_path)
    assert isinstance(cfg, Config)
    assert cfg.project.seed == 42
    assert cfg.sparse_retrieval.k1 == 1.5
    assert cfg.dense_retrieval.chunk_size == 400
    assert cfg.data.dataset_name in ["sample", "il_pcr", "fire_aila"]

    # Test dictionary-style access works identically
    assert cfg["project"]["seed"] == 42
    assert cfg["sparse_retrieval"]["b"] == 0.75


def test_logger_creation(tmp_path):
    """Verify logger creation and file output."""
    test_log_dir = str(tmp_path / "test_logs")
    logger = get_logger(name="TestLogger", log_dir=test_log_dir)
    logger.info("Verification test log message")

    log_file = os.path.join(test_log_dir, "run.log")
    assert os.path.exists(log_file), "Log file should be generated"
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Verification test log message" in content
