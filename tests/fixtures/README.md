# Test fixtures

Synthetic responses written in the exact formats that NCBI E-utilities, Europe PMC REST and
ClinicalTrials.gov API v2 return. The PMIDs, titles and text are made up for tests and do not
refer to real papers. Unit tests never touch the network; `scripts/smoke_live.py` checks the
real APIs.
