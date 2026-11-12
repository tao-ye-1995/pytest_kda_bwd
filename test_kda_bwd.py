"""同一 CPU 数学缓存验证各 backward；运行和选择用例见 README.md。"""
import importlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import pytest
import torch

from cases import make_inputs
from implementations import device_batch
from implementations.cpu import GRAD_NAMES, cpu_forward_cache_and_backward, kda_recurrence, mathematical_value_and_grad

REL_L2_LIMIT, PEAK_REL_LIMIT, MATH_ZERO_ATOL = .03, .08, 1e-7


@pytest.fixture(scope="module", autouse=True)
def cpu_threads(pytestconfig):
    previous = torch.get_num_threads()
    torch.set_num_threads(pytestconfig.getoption("--cpu-threads"))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def sample(case):
    inputs, do, options = make_inputs(case)
    start = time.perf_counter()
    cache, _, gradients = cpu_forward_cache_and_backward(
        inputs, do, **{key: options[key] for key in ("scale", "cu_seqlens", "normalize", "fused_beta", "chunk_size")})
    return case, cache, gradients, options, time.perf_counter() - start


@pytest.fixture(scope="module")
def device(report, pytestconfig):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend A3 NPU required")
    from triton.backends.ascend.utils import _get_npucompiler_path
    assert os.environ.get("ASCEND_HOME_PATH"), "Activate CANN set_env.sh before running device implementations"
    home = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    os.environ["PATH"] = str(home / "tools/bishengir/bin") + os.pathsep + os.environ["PATH"]
    compiler, _ = _get_npucompiler_path()
    assert Path(compiler).resolve().is_relative_to(home), f"Compiler/CANN mismatch: {compiler}, {home}"
    index = pytestconfig.getoption("--device")
    assert 0 <= index < torch.npu.device_count()
    if not torch.npu.get_device_name(index).startswith(("Ascend910_93", "Ascend910C")):
        pytest.skip("This suite targets A3")
    if not os.environ.get("TRITON_CACHE_DIR"):
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="kda_pytest_")
    report["environment"]["triton_cache_dir"] = os.environ["TRITON_CACHE_DIR"]
    version = subprocess.run([compiler, "--version"], text=True, capture_output=True, check=True)
    report["environment"].update(
        ascend_home_path=str(home), compiler=str(Path(compiler).resolve()),
        compiler_version=version.stdout.strip(), target=torch.npu.get_device_name(index))
    print(f"\nCANN: {home}\nCompiler: {compiler}\n{version.stdout.strip()}"
          f"\nTarget: {report['environment']['target']}\nCache: {os.environ.get('TRITON_CACHE_DIR')}")
    torch.npu.set_device(index)
    yield torch.device(f"npu:{index}")
    torch.npu.synchronize(index)


@pytest.fixture(scope="module")
def shared_device_batch(sample, device, pytestconfig):
    _, cache, _, options, _ = sample
    return device_batch(cache, dict(options, fla_root=pytestconfig.getoption("--fla-root")), device)


def _error(value, reference):
    assert value.shape == reference.shape
    value = value.detach().to("cpu", torch.float64)
    assert torch.isfinite(value).all(), "Non-finite gradient"
    diff, norm, peak = value - reference, reference.norm().item(), reference.abs().max().item()
    return dict(relative_l2=diff.norm().item() / norm if norm else None,
                peak_relative=diff.abs().max().item() / peak if peak else None,
                max_abs=diff.abs().max().item(), reference_l2=norm, reference_peak=peak)


@pytest.fixture(scope="module")
def report(pytestconfig):
    result = dict(
        reference="CPU FP64 mathematical forward cache + autograd backward",
        shared_cache=True, shape_file=pytestconfig.getoption("--shape-file"),
        implementations=pytestconfig._kda["implementations"],
        environment=dict(ascend_home_path=os.environ.get("ASCEND_HOME_PATH"),
                         triton_cache_dir=os.environ.get("TRITON_CACHE_DIR"),
                         cpu_threads=torch.get_num_threads(), fla_root=pytestconfig.getoption("--fla-root")),
        limits=dict(relative_l2=REL_L2_LIMIT, peak_relative=PEAK_REL_LIMIT,
                    mathematical_zero_fp32_atol=MATH_ZERO_ATOL, zero_do_atol=0.), cases=[])
    yield result
    directory = pytestconfig._kda["report_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "pytest_kda_math_metrics.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    pytestconfig._kda["report_path"] = target


def test_shared_cache_backward(sample, implementation, request, report, record_property):
    case, _, expected, options, cpu_seconds = sample
    row = dict(case=case.name, implementation=implementation, total=sum(case.lengths),
               cu_seqlens=options["cu_seqlens"], cpu_reference_seconds=cpu_seconds, device="cpu", metrics={})
    report["cases"].append(row)
    try:
        if implementation == "cpu":
            values = expected
        else:
            batch = request.getfixturevalue("shared_device_batch")
            row["device"] = "npu"
            versions = {name: value._version for name, value in batch.items() if isinstance(value, torch.Tensor)}
            module = importlib.import_module(f"implementations.{implementation}")
            values = module.backward(batch)
            assert all(batch[name]._version == version for name, version in versions.items()), "Backend mutated shared inputs"
        assert len(values) == len(GRAD_NAMES)
        failures = []
        for name, value, gold in zip(GRAD_NAMES, values, expected):
            assert value.device.type == ("cpu" if implementation == "cpu" else "npu")
            dtype = torch.float64 if implementation == "cpu" else (
                torch.bfloat16 if name in ("dq", "dk", "dv") else torch.float32)
            assert value.dtype == dtype, f"{name}: expected {dtype}, got {value.dtype}"
            metric = _error(value, gold)
            row["metrics"][name] = metric
            record_property(f"{name}_max_abs", metric["max_abs"])
            if metric["reference_l2"] == 0:
                metric["zero_atol"] = 0. if case.zero_do else MATH_ZERO_ATOL
                passed = metric["max_abs"] <= metric["zero_atol"]
            else:
                passed = metric["relative_l2"] <= REL_L2_LIMIT and metric["peak_relative"] <= PEAK_REL_LIMIT
            if not passed:
                failures.append(f"{name}: {metric}")
        row["passed"] = not failures
        assert not failures, f"{implementation}/{case.name}:\n" + "\n".join(failures)
    except pytest.skip.Exception as error:
        row.update(passed=None, skipped=True, reason=str(error))
        raise
    except Exception as error:
        row.update(passed=False, error=str(error))
        raise


@pytest.mark.parametrize("normalize,fused_beta", [(True, False), (False, False), (True, True)])
def test_reference_chunk_cache_matches_recurrence(normalize, fused_beta):
    generator = torch.Generator().manual_seed(123)
    shape = (1, 9, 2, 3)
    rand = lambda s: torch.randn(s, generator=generator, dtype=torch.float64) * .2
    inputs = dict(q=rand(shape), k=rand(shape), v=rand(shape), beta=rand(shape[:-1]),
                  g=rand(shape) - 1, A_log=rand((2,)), dt_bias=rand((2, 3)))
    do = rand(shape)
    options = dict(scale=3 ** -.5, cu_seqlens=[0, 1, 6, 9], normalize=normalize, fused_beta=fused_beta)
    _, output, gradients = cpu_forward_cache_and_backward(inputs, do, chunk_size=4, **options)
    gold_output, gold_gradients = mathematical_value_and_grad(inputs, do, **options)
    torch.testing.assert_close(output, gold_output, rtol=1e-11, atol=1e-12)
    for value, gold in zip(gradients, gold_gradients):
        torch.testing.assert_close(value, gold, rtol=1e-10, atol=1e-12)


def test_reference_gradcheck():
    generator = torch.Generator().manual_seed(123)
    shapes = ((1, 3, 2, 3),) * 3 + ((1, 3, 2), (1, 3, 2, 3), (2,), (2, 3))
    leaves = tuple((torch.randn(shape, generator=generator, dtype=torch.float64) * .2).requires_grad_()
                   for shape in shapes)
    fn = lambda *args: kda_recurrence(*args, scale=3 ** -.5, cu_seqlens=[0, 1, 3], normalize=True, fused_beta=True)
    assert torch.autograd.gradcheck(fn, leaves, eps=1e-6, atol=1e-5, rtol=1e-4)
