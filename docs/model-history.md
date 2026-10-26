# Model release history

These are recorded model-release measurements. The default desktop download is T16 with IQ4_NL Engram;
the MI325X has a separate MXFP4 + 3-bit package. Older packages remain documented for reproducibility.

| Package | Hardware | Download | Engram table | KL / top-1 vs FP8 |
|---|---|---|---|---|
| [T16](../cards/dsx-t16/README.md) | NVIDIA desktop | 252 GB, one GGUF | IQ4_NL, included | 0.196 / 80.9% |
| [T16, q6k-engram branch](../cards/dsx-t16/README.md) | NVIDIA desktop | 302 GB, four shards | Q6_K, included | 0.194 / 81.4% |
| [dsx-ud](../cards/dsx-ud/README.md) | NVIDIA desktop | 305 GB, five shards | Q6_K, included | 0.269 / 78.9% |
| [dsx-2bit](../cards/dsx-2bit/README.md) | NVIDIA desktop | 148 GB plus 203 GB Engram | official FP8, separate | 0.373 / 74.4% |
| [dsx-mxfp4](../cards/dsx-mxfp4/README.md) | AMD MI325X | 270.5 GB plus official Engram | official FP8, separate | 0.049 / 90.6% |

The values come from the model cards' quality tables on the Lucebox 1017 evaluation, 8×1,024 tokens against
FP8 top-256 log probabilities. Quality was measured without expert deferral. T16 output gains were selected on
that evaluation, so those values are not an independent quality estimate. T16 release quality uses the prefill path, while the AMD card describes teacher-forced decode. The table does not
isolate cross-engine numerical differences, rank current agentic task quality or compare current runtime speed.

The former repeated comparison tables listed dsx-ud's 305 GB Q6_K package as KL 0.272 / top-1 79.0%, while its own
quality table reports 0.269 / 78.9%. The table above uses the package-specific quality table. Earlier experimental
values remain in the dated findings rather than being relabeled as measurements of this package.

For current runtime configuration and benchmark conditions, see [NVIDIA status](current-state.md) and
[the AMD serving guide](../amd/README.md). Model cards in this repository are local source copies; changing them
here does not publish them to Hugging Face.
