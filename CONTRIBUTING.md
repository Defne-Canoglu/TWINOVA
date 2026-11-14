# Contributing

Use Python 3.12 and install `requirements.txt`. Preserve person-disjoint roles, availability-time checks, units, origin records and calibration evidence. Run `python -m v136.release --integrity` and `python -m v136.release --verify` before proposing a change.

Do not overwrite the bundled default model or locked dataset to make a test pass. Generate a separate versioned study and model for research changes. Record configuration, random seed, protocol, source hashes and assessment scope with each result. Keep explanatory support scores distinct from calibrated probabilities and causal claims.

Report the actual commands and outcomes in a pull request. The Windows workflow is a proposed CI check until its run is visible in the repository.
