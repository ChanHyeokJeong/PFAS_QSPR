# Partial Office-PC Screen Summary

The office-PC screen was interrupted because PaDEL Java processes created high
CPU load. Before interruption, `partial_raw_screen_42_status.jsonl` recorded:

- Status rows: 42
- RDKit-valid rows: 42
- Raw PaDEL timeout rows: 42
- Raw PaDEL timeout threshold: 45 seconds
- RDKit atom-count range among recorded rows: 42 to 150 atoms

Interpretation: the observed all-descriptor-missing cases are not explained by
RDKit SMILES parsing failure in the recorded subset. They are better treated as
PaDEL molecule-level calculation timeout/failure candidates pending controlled
one-by-one rerun.
