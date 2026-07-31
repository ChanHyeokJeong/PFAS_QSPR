# Raw Data Placement

This branch includes the LogP raw inputs required for continuing the Panel 3
masking benchmark:

```text
data/raw/LogP_descriptor_original.xlsx
data/raw/PFAS_embedding_numbers.xlsx
data/raw/masking_input_cache_logp_endpoint_col4.npz
```

The benchmark script first looks beside `masking_benchmark_complete_descriptors.py`
and then falls back to `data/raw/`, so no manual copy is needed for LogP runs
when this branch is cloned.

The BCF files are not needed for the current Panel 3 task. To reproduce BCF
runs, place these files in the repository root or under `data/raw/`:

```text
BCF_descriptor.xlsx
BCF_PFAS_embedding_numbers.xlsx
```

Expected SHA256 values for the included LogP files:

```text
92CF62BC1F73E84D8C5F176DAC6CED599793607BAE78CBC905DCE2DE91AE4258  LogP_descriptor_original.xlsx
D62A4D13F07A8F72CE443701E3223336B40F0F4125BAAAA8F4D1621210707B2F  PFAS_embedding_numbers.xlsx
064469FCD9220FCDCDFFB76289E316A755A279114B9621112EBDCE7EC6226148  masking_input_cache_logp_endpoint_col4.npz
```

The original working folder was:

```text
Z:\backup\Chanhyeok Jeong\Manuscript\4. PFAS QSPR (JHM)\Code & Data\Misstransformer
```
