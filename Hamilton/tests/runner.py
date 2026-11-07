"""
Minimal test runner: no pytest needed.

    python tests/runner.py              # run everything
    python tests/runner.py phase0       # run tests whose module or name contains 'phase0'
    python tests/runner.py -k xmode     # run tests whose name contains 'xmode'

Each test is a function named test_* in a module tests/test_*.py. A test passes
if it returns without raising. The same files also run under pytest.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def collect(selector: str = "", keyword: str = ""):
    tests = []
    for path in sorted(HERE.glob("test_*.py")):
        if selector and selector not in path.stem:
            continue
        module = importlib.import_module(path.stem)
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and fn.__module__ == module.__name__ and keyword in name:
                tests.append((path.stem, name, fn))
    return tests


def main(argv):
    selector, keyword = "", ""
    args = list(argv)
    if "-k" in args:
        i = args.index("-k")
        keyword = args[i + 1]
        del args[i:i + 2]
    if args:
        selector = args[0]
    tests = collect(selector, keyword)
    failures = []
    t_total = time.time()
    for module_name, name, fn in tests:
        t0 = time.time()
        try:
            fn()
            status = "PASS"
        except Exception as exc:       # noqa: BLE001 - report every failure
            status = "FAIL"
            failures.append((module_name, name, exc, traceback.format_exc()))
        dt = time.time() - t0
        print(f"{status}  {module_name}::{name}  ({dt:.1f} s)", flush=True)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed in {time.time() - t_total:.1f} s")
    for module_name, name, exc, tb in failures:
        print(f"\n--- {module_name}::{name}\n{tb}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
