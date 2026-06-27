# Competition Work Log for MetaX-MACA/vLLM-metax Issue #261

## Current PR

- **Upstream PR:** https://github.com/MetaX-MACA/vLLM-metax/pull/294
- **Status:** ready for review, opened from `xzh25:fix-qwen3-mtp261-compat`
- **Local source commits:**
  - `045198c` Fix MetaX vLLM 0.23 MTP import compatibility
  - `ff67715` Handle missing torch.cuda in MetaX compat hooks
  - `0466e6f` Test compat import without torch.cuda
  - `82fccb2` Handle missing MoE disable_inplace config
- **Remote validation worktree:** `/data/vllm-metax-contribution/src/vLLM-metax-mtp261`

---

## Fix Summary

The MTP path in vLLM 0.23 imports quantization and fusion modules before the MetaX plugin has installed compatibility shims. This can fail before generation on:

- `_C::scaled_fp4_quant.out`
- `_C::silu_and_mul_per_block_quant`
- Missing `torch.accelerator` helpers in spawned EngineCore workers

The PR adds:

1. An early `vllm_metax.compat` hook, imported at package import time
2. Keeps the legacy `triton_support` import path active
3. Adds a focused regression test for the `act_quant_fusion` import path

### 35BA3B GPTQ MoE WNA16 Compatibility

The `35BA3B GPTQ` validation exposed an additional vLLM 0.23 compatibility issue in the MetaX `moe_wna16` quantization path: `MoeWNA16Method.apply()` assumed `FusedMoEConfig.disable_inplace` always exists. The PR now treats the field as optional and preserves the existing default in-place behavior when it is absent.

---

## Before Fix

Pre-fix runs failed before generation on the same MetaX validation machine:

| Log | Error |
|---|---|
| `079_qwen35_9b_mtp261_smoke.txt` | `RuntimeError: operator _C::scaled_fp4_quant.out does not exist` |
| `084_qwen35_9b_mtp261_mtp_only_after_schema_fix.txt` | `AttributeError: module 'torch.accelerator' has no attribute 'empty_cache'` |
| `088_qwen35_27b_w8a8_mtp261_mtp_only_9b_tokenizer.txt` | `RuntimeError: operator _C::scaled_fp4_quant.out does not exist` |

---

## Validation

### Environment

- **GPU:** MetaX C500 64GB (dual for 35BA3B)
- **MACA:** 3.5.3.20
- **torch:** 2.8.0+metax3.5.3.9
- **vLLM:** 0.23.0
- **Python:** 3.10

### Unit Tests

```text
python -m pytest tests/patch/test_triton_custom_op_schemas.py -q --confcutdir=tests/patch: 2 passed
```

Import smoke checks:

- `import vllm_metax.quant_config; import vllm.compilation.passes.fusion.act_quant_fusion`: passed
- `python -m py_compile ...`: passed
- `git diff --check`: passed

### MTP Smoke Results

| Case | Model | MTP | Load (s) | Generate (s) | Output tok/s | Accepted Tokens |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 9B | no | 90.289 | 3.995 | 24.0311 | N/A |
| Qwen3.5-9B | 9B | yes | 93.406 | 3.352 | 28.6370 | 50 |
| Qwen3.5-27B-W8A8 | 27B | no | 99.703 | 9.338 | 10.2801 | N/A |
| Qwen3.5-27B-W8A8 | 27B | yes | 102.532 | 4.434 | 21.6529 | 60 |
| FlagRelease 35BA3B (TP=2, 128 tok) | 35BA3B | no | 150.576 | 12.054 | 10.619 | N/A |
| FlagRelease 35BA3B (TP=2, 128 tok) | 35BA3B | yes | 150.358 | 6.293 | 20.3389 | 79 |

### Key Validation Matrix (same prompt, max_tokens=96)

| Case | MTP | Load s | Generate s | Output tok/s | Repeated bigram ratio | Spec decode |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | no | 90.289 | 3.995 | 24.0311 | 0.0125 | N/A |
| Qwen3.5-9B | yes | 93.406 | 3.352 | 28.6370 | 0.0125 | drafts=45, draft_tokens=90, accepted=50, per_pos=[30,20] |
| Qwen3.5-27B-W8A8 | no | 99.703 | 9.338 | 10.2801 | 0.0444 | N/A |
| Qwen3.5-27B-W8A8 | yes | 102.532 | 4.434 | 21.6529 | 0.0000 | drafts=35, draft_tokens=70, accepted=60, per_pos=[31,29] |

### Long-output MTP Stress (Qwen3.5-27B-W8A8, max_tokens=256)

- Generated 256 tokens normally at **23.2837 tok/s**
- Spec decode: drafts=99, draft_tokens=198, accepted=158, per_pos=[85,73]
- Repetition: bigram 0.0853, trigram 0.0391, 4-gram 0.0157
- Most repeated 4-gram was "Genshin Impact Version 5.0" (count 3) — attributable to the requested topic, not looped output

### 35BA3B FlagRelease Dual C500 TP=2 Results

| Config | max_tokens | Load s | Generate s | Output tok/s | Accepted tokens |
|---|---|---|---|---|---|
| No MTP | 128 | 150.576 | 12.054 | 10.619 | N/A |
| MTP | 128 | 150.358 | 6.293 | 20.3389 | 79 |
| No MTP | 32 | 194.923 | 7.104 | 4.5042 | N/A |
| MTP | 32 | 296.123 | 7.929 | 4.0358 | 20 |

**Acceleration:** 20.3389 / 10.619 = **1.915x** output-token throughput for 128-token run.

Peak memory: ~36.0 GiB per C500 (with or without MTP) — dual 64GB cards sufficient.

### 35BA3B GPTQ MTP Results (after moe_wna16 fix)

| Check | max_tokens | Output tok/s | Accepted tokens | Notes |
|---|---|---|---|---|
| Smoke | 32 | — | 0 | Generated normally, no looped output from #261 |
| Issue-like | 128 | 9.2379 | 0 | repeat ratios: bigram 0.0806, trigram 0.0492, 4-gram 0.0167 |
| One-token MTP | 64 | 7.3182 | 0 | num_speculative_tokens=1 |

The GPTQ output did not show the corrupted repeated-output loop from issue #261. Accepted-token count is explicitly zero — this is functional MTP-path coverage rather than a speedup claim for this checkpoint/tokenizer workaround.

### OpenAI-compatible Server API

- **Model:** Qwen3.5-9B
- **Command:** `vllm serve ... --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`
- `/v1/models`: returned served model `qwen35-9b-mtp`
- `/v1/completions`: returned 48 completion tokens
- `/metrics`: exposed spec-decode counters with drafts=18, draft_tokens=36, accepted_tokens=31, per_pos=[17,14]

This closes the server-path validation gap for a known-good MTP model.

---

## Docker PoC

For reviewers who want to reproduce the MTP path in a container, the repo includes a thin Docker PoC:

| File | Path |
|---|---|
| Dockerfile | `docker/poc-qwen3-mtp261.Dockerfile` |
| Reproduction notes | `docs/reproduce_docker.md` |
| Smoke entrypoint | `scripts/run_qwen3_mtp261_smoke.sh` |

The image is intended to be built on top of an official MetaX/vLLM-MetaX runtime image with MACA 3.5.3.x and torch 2.8. It does not bundle model weights; mount the model directory at runtime.

---

## Notes

1. The available 30B-A3B model configs do not include MTP metadata; they were not used as #261 MTP substitutes.
2. The Qwen3.5-27B-W8A8 tokenizer config uses `TokenizersBackend`, which the Transformers environment cannot instantiate. The 27B validation uses the local Qwen3.5-9B tokenizer as a smoke-test workaround.
3. The downloaded Qwen3.6-35B-A3B GPTQ tokenizer has the same `TokenizersBackend` compatibility problem, so the 35BA3B GPTQ runs also use the local Qwen3.5-9B tokenizer workaround.
4. Official Qwen3.6-35B-A3B-FP8 was rejected by MetaX/vLLM: `fp8 quantization is currently not supported in maca`.

---

## Remote Logs

```
/data/vllm-metax-contribution/ops/logs/079_qwen35_9b_mtp261_smoke.txt
/data/vllm-metax-contribution/ops/logs/084_qwen35_9b_mtp261_mtp_only_after_schema_fix.txt
/data/vllm-metax-contribution/ops/logs/088_qwen35_27b_w8a8_mtp261_mtp_only_9b_tokenizer.txt
/data/vllm-metax-contribution/ops/logs/086_qwen35_9b_mtp261_mtp_only_after_accel_fix.txt
/data/vllm-metax-contribution/ops/logs/089_test_top_level_compat_import_order.txt
/data/vllm-metax-contribution/ops/logs/090_qwen35_27b_w8a8_mtp261_after_top_level_compat.txt
/data/vllm-metax-contribution/ops/logs/091_qwen35_27b_w8a8_mtp261_issue_like_prompt.txt
/data/vllm-metax-contribution/ops/logs/092_collect_env_mtp261.txt
/data/vllm-metax-contribution/ops/logs/093_mtp261_validation_matrix.txt
/data/vllm-metax-contribution/ops/logs/094_qwen35_27b_w8a8_mtp_long_output_check.txt
/data/vllm-metax-contribution/ops/logs/095_qwen36_35ba3b_fp8_mtp_smoke.txt
/data/vllm-metax-contribution/ops/logs/096_qwen36_35ba3b_w8a8_mtp_smoke.txt
/data/vllm-metax-contribution/ops/logs/097_qwen36_35ba3b_w8a8_experts_int8_mtp_smoke.txt
/data/vllm-metax-contribution/ops/logs/098_qwen36_35ba3b_w8a8_quark_mtp_smoke.txt
/data/vllm-metax-contribution/ops/logs/099_qwen36_35ba3b_w8a8_int8_per_channel_mtp_smoke.txt
/data/vllm-metax-contribution/ops/logs/100_qwen36_35ba3b_gptq_mtp_smoke.txt
/data/vllm-metax-contribution/ops/logs/101_qwen36_35ba3b_gptq_mtp_9b_tokenizer_smoke.txt
/data/vllm-metax-contribution/ops/logs/102_qwen36_35ba3b_gptq_mtp_9b_tokenizer_file_smoke.txt
/data/vllm-metax-contribution/ops/logs/103_qwen36_35ba3b_gptq_mtp_after_moe_wna16_fix.txt
/data/vllm-metax-contribution/ops/logs/104_qwen36_35ba3b_gptq_mtp_issue_like_check.txt
/data/vllm-metax-contribution/ops/logs/105_qwen35_9b_mtp_openai_server_api.txt
/data/vllm-metax-contribution/ops/logs/106_qwen36_35ba3b_gptq_mtp_one_token_check.txt
/data/vllm-metax-contribution/ops/logs/200_dual_c500_download_flagrelease_35ba3b.log
/data/vllm-metax-contribution/ops/logs/201_cpu_resume_flagrelease_35ba3b_lfs_pull.log
/data/vllm-metax-contribution/ops/logs/202_dual_c500_flagrelease_35ba3b_mtp_smoke.log
/data/vllm-metax-contribution/ops/logs/203_dual_c500_flagrelease_35ba3b_no_mtp_smoke.log
/data/vllm-metax-contribution/ops/logs/204_dual_c500_flagrelease_35ba3b_mtp_128.log
/data/vllm-metax-contribution/ops/logs/205_dual_c500_flagrelease_35ba3b_no_mtp_128.log
```

---

## Next Steps

1. Watch PR #294 maintainer feedback and merge status.
2. If maintainers require a narrower fix, split the compatibility hook into the exact import paths they prefer.
3. GPUApps PR record has been submitted: https://www.gitlink.org.cn/ccf-ai-infra/GPUApps/issues/213
4. After merge, append the merge commit and final merge evidence to GPUApps issue #213.
