import json
import subprocess
import sys

import pytest

from python import bench, config, cpp_runner


def test_stats():
    s = bench.stats([1.0, 2.0, 3.0, 4.0])
    assert s["median"] == 2.5 and s["mean"] == 2.5
    assert s["min"] == 1.0 and s["max"] == 4.0
    assert s["std"] == pytest.approx(1.2909944487358056)  # sample std (ddof=1)


def test_block_order_is_abba_and_balanced():
    assert bench.BLOCK_ORDER.count("C") == bench.BLOCK_ORDER.count("S")
    assert bench.BLOCK_ORDER == "CSSCCSSC"


def test_rss_unit_matches_platform():
    assert bench.rss_unit() == (1 if sys.platform == "darwin" else 1024)


def test_loss_check_rejects_wrong_loss():
    bench._check_loss(0.5, 0.5 + 1e-12, "x")
    with pytest.raises(RuntimeError, match="differs"):
        bench._check_loss(0.5, 0.6, "x")


@pytest.fixture
def binary():
    try:
        return cpp_runner.binary_path()
    except cpp_runner.CppError:
        pytest.skip("C++ binary not built")


def test_cpp_bench_rejects_wrong_reference(binary):
    proc = subprocess.run([str(binary), "--bench", "--repeat", "1", "--warmup", "0",
                           "--ref-loss", "0.5"], capture_output=True, text=True)
    assert proc.returncode != 0 and "differs from the reference" in proc.stderr


def test_cpp_mem_prints_single_json_line(binary):
    out = subprocess.run([str(binary), "--mem"], capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    assert data["peak_after_train_bytes"] >= data["peak_before_train_bytes"] > 0
    assert data["train_increment_bytes"] == (
        data["peak_after_train_bytes"] - data["peak_before_train_bytes"])
