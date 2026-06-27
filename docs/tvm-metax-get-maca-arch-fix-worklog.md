# Competition Work Log — TVM MetaX MACA `get_maca_arch` Fix

## Current PR

- **Repo:** [https://github.com/zhanma666/mcTVM](https://github.com/MetaX-MACA/mcTVM/pull/30)
- **Branch:** `fix/duplicate-maca-error-message`
- **Head SHA:** `485c4d14c`
- **Base:** `apache/tvm:main`
- **Status:** ready for review

### Commits

| SHA | Description |
|---|---|
| `485c4d14c` | fix: address PR review feedback for get_maca_arch |

---

## Fix Summary

The `get_maca_arch()` function in `python/tvm/contrib/mxcc.py` had three issues addressed based on upstream PR review feedback:

### 1. Missing `OSError` catch (crash fix)

`subprocess.check_output([f"{maca_path}/bin/macainfo"])` raises `FileNotFoundError` (a subclass of `OSError`) when the `macainfo` binary does not exist at the given path. The original code only caught `subprocess.CalledProcessError`, causing an unhandled crash.

**Fix:** Changed `except subprocess.CalledProcessError` to `except (subprocess.CalledProcessError, OSError)`.

### 2. `MACA_ARCH` environment variable support

For cross-compilation and non-GPU CI/container environments, there was no way to specify the MACA architecture without running `macainfo` (which requires physical GPU access).

**Fix:** Added early check for `MACA_ARCH` env var. If set (e.g. `MACA_ARCH=xcore2000`), the function returns the value immediately, skipping `macainfo` entirely. Also added `MACA_PATH` env var fallback for the installation path.

### 3. Test pollution fix (anti-pattern rewrite)

The original test file used `importlib.util.spec_from_file_location` with manual `sys.modules` injection to create fake `tvm` module stubs. This permanently pollutes `sys.modules`, causing `AttributeError` in other tests running in the same process.

**Fix:** Rewrote tests to use `from tvm.contrib import mxcc` with standard `unittest.mock.patch` for scoped mocking.

---

## Files Changed

| File | Status | Lines |
|---|---|---|
| `python/tvm/contrib/mxcc.py` | modified | +7, -1 |
| `tests/python/contrib/test_mxcc_arch.py` | new | +106 |

---

## Validation

### Environment

- **GPU:** MetaX C500
- **MACA:** 3.5.3.20
- **Python:** 3.10
- **TVM:** 0.23.dev0

### Syntax & Import Checks

| Check | Result |
|---|---|
| `python -m py_compile python/tvm/contrib/mxcc.py` | ✅ passed |
| `python -m py_compile tests/python/contrib/test_mxcc_arch.py` | ✅ passed |
| `from tvm.contrib import mxcc` (with PYTHONPATH) | ✅ passed |

### Unit Test Results (inline, all 7 tests passed)

| Test | Description | Result |
|---|---|---|
| `test_env_maca_arch` | MACA_ARCH env var returns value directly | ✅ |
| `test_env_maca_arch_overrides_maca_path` | MACA_ARCH takes priority over parameter | ✅ |
| `test_maca_not_installed_default` | MACA not installed → default xcore1000 | ✅ |
| `test_oserror_is_caught` | OSError (missing binary) → graceful fallback | ✅ |
| `test_calledprocesserror_is_caught` | CalledProcessError → graceful fallback | ✅ |
| `test_macainfo_parsing` | macainfo output parsed correctly | ✅ |
| `test_maca_path_env_var` | MACA_PATH env var used as path fallback | ✅ |

---

## Code Changes Detail

### `python/tvm/contrib/mxcc.py` — `get_maca_arch()` function (lines 204-231)

```python
# Check MACA_ARCH environment variable first
gpu_arch = os.environ.get("MACA_ARCH")
if gpu_arch:
    return gpu_arch.lower()

gpu_arch = "xcore1000"
# resolve maca path from parameter or environment variable
maca_path = maca_path or os.environ.get("MACA_PATH", "/opt/maca")
# check if maca is installed
if not os.path.exists(maca_path):
    print("MACA not detected, using default xcore1000")
    return gpu_arch
try:
    # Execute macainfo command
    macainfo_output = subprocess.check_output(
        [f"{maca_path}/bin/macainfo"]
    ).decode("utf-8")
    # Use regex to match the "Name" field
    match = re.search(r"Name:\s+(XCORE\d+[a-zA-Z]*)", macainfo_output)
    if match:
        gpu_arch = match.group(1)
    return gpu_arch.lower()
except (subprocess.CalledProcessError, OSError):
    print(...)
    return gpu_arch
```

### `tests/python/contrib/test_mxcc_arch.py`

- 8 test cases in `TestMacaArchDetection` class
- Uses `unittest.mock.patch` for `os.environ`, `os.path.exists`, and `subprocess.check_output`
- Direct import: `from tvm.contrib import mxcc`
- No `sys.modules` pollution

---

## Next Steps

1. Watch maintainer feedback and merge status
2. After merge, close the associated issue
