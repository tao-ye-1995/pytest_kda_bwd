"""精确选择实现和数据用例。使用方式见 README.md。"""
from pathlib import Path
import time

import pytest

from cases import load_cases

ROOT = Path(__file__).resolve().parent


def pytest_addoption(parser):
    group = parser.getgroup("kda", "KDA backward accuracy")
    group.addoption("--impl", default="all", help="Implementation names, comma separated; default: all")
    group.addoption("--case", default="all", help="Exact case names/numbers or all/shape/edge/reference; comma separated")
    group.addoption("--device", type=int, default=0, help="Logical NPU index; default: 0")
    group.addoption("--cpu-threads", type=int, default=4, help="CPU reference threads; default: 4")
    group.addoption("--fla-root", default="/workspace/flash-linear-attention", help="Local FLA repository")
    group.addoption("--shape-file", default=str(ROOT / "shape_demo.txt"), help="Original cumulative boundaries file")
    group.addoption("--report-dir", default=None, help="Output directory; default: a new directory per run")
    group.addoption("--list-cases", action="store_true", help="List exact case names and sizes without computing")


def pytest_configure(config):
    available = sorted(p.stem for p in (ROOT / "implementations").glob("*.py") if p.stem != "__init__")
    names = [name.strip() for name in config.getoption("--impl").split(",")]
    selected_impls = available if names == ["all"] else list(dict.fromkeys(names))
    unknown = set(selected_impls) - set(available)
    if unknown:
        raise pytest.UsageError(f"Unknown implementations: {', '.join(sorted(unknown))}. Available: {', '.join(available)}")
    if config.getoption("--cpu-threads") < 1:
        raise pytest.UsageError("--cpu-threads must be positive")
    if config.getoption("--device") < 0:
        raise pytest.UsageError("--device must be nonnegative")
    try:
        cases = load_cases(config.getoption("--shape-file"))
    except (OSError, ValueError) as error:
        raise pytest.UsageError(str(error)) from error
    if config.getoption("--list-cases"):
        print("Implementations: " + ", ".join(available))
        print(f"{'case':<26} {'tokens':>8} {'sequences':>9}  layout / options")
        for case in cases:
            options = "packed" if case.packed else "dense"
            if not case.normalize:
                options += ", no L2norm"
            if case.fused_beta:
                options += ", beta sigmoid"
            if case.zero_do:
                options += ", zero dO"
            print(f"{case.name:<26} {sum(case.lengths):>8} {len(case.lengths):>9}  {options}")
        print("reference: 4 short CPU mathematical checks; no shape_demo data")
        pytest.exit("Cases listed; no computation performed", returncode=0)
    lookup = {case.name: case for case in cases}
    selected_names, reference = set(), False
    for token in config.getoption("--case").split(","):
        token = token.strip()
        if token in ("all", "shape", "edge"):
            selected_names.update(case.name for case in cases if token == "all" or
                                  case.name.startswith("shape_demo_" if token == "shape" else "edge_"))
            reference |= token == "all" and "cpu" in selected_impls
        elif token == "reference":
            if "cpu" not in selected_impls:
                raise pytest.UsageError("--case reference requires --impl cpu or all")
            reference = True
        else:
            name = f"shape_demo_{token}" if token.isdecimal() else token
            if name not in lookup:
                raise pytest.UsageError(f"Unknown case: {token!r}. Use --list-cases; numbers select shape_demo lines exactly.")
            selected_names.add(name)
    directory = config.getoption("--report-dir")
    location = "cpu" if selected_impls == ["cpu"] else f"device_{config.getoption('--device')}"
    config._kda = dict(implementations=selected_impls, cases=[case for case in cases if case.name in selected_names],
                       reference=reference, report_dir=Path(directory) if directory else
                       ROOT / "results" / location / f"run_{time.time_ns()}")


def pytest_generate_tests(metafunc):
    if "sample" in metafunc.fixturenames:
        metafunc.parametrize("case", metafunc.config._kda["cases"], ids=lambda case: case.name, scope="module")
        metafunc.parametrize("implementation", metafunc.config._kda["implementations"])


def pytest_collection_modifyitems(config, items):
    selected, deselected = [], []
    for item in items:
        keep = (config._kda["reference"] if item.name.startswith("test_reference")
                else bool(config._kda["cases"]))
        (selected if keep else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def pytest_terminal_summary(terminalreporter):
    target = terminalreporter.config._kda.get("report_path")
    if target is not None:
        terminalreporter.write_line(f"Accuracy report: {target}")
