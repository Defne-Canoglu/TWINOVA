# GitHub distribution preparation

## Scope

This distribution contains the maintained v13.6 application, its v13.2–v13.5 compatibility dependencies, saved models, bounded example cohort, current recorded research outputs, firmware source and compiled AVR artifacts. Versioned Python package names remain stable because serialized models refer to them.

The original delivery archive remains unchanged. Historical application versions, duplicate delivery archives, screen-capture collections, local browser logs, environment-specific build logs, old acceptance narratives, generated dependency maps and draft documents are excluded from this distribution. This is a focused current repository, not a historical archive of every earlier experiment.

## Changes

- English default interface, actions, explanations, documentation and launchers. Legacy language transport keys remain accepted for compatibility; their maintained display content is English.
- Standalone v13.6 HTTP handler, shared application routing for GET and POST, strict JSON parsing, bounded request size, connection timeout, explicit errors and guarded artifact paths.
- Updated report and hardware links; an English recorded-results renderer and maintained documentation routes.
- English release entry point, publication instructions and Linux/Windows GitHub Actions configuration.
- An example-request fixture replaces a dependency on an omitted historical UI-review directory. Research commands no longer emit a historical browser observation as if it were a current check.
- Runtime dependencies separated from optional serial hardware capture dependencies.

The source origin and validation scope are retained. Engineering data are not presented as human observations. Recorded virtual hardware results are not presented as physical device measurements. Author attribution, source references and third-party notices are not removed as production residue.

## Evidence boundaries

The default model and included data/model files were compared byte for byte with the supplied archive. Ten paired default-model cases compare original and prepared numerical predictions, intervals, probabilities, state, features and physiology. See `validation/numerical_equivalence.json`.

The included frozen-file integrity map covers 110 maintained files, rather than the broader historical delivery inventory. The original 48-case virtual bench is preserved as recorded evidence; this preparation did not rerun a physical experiment or the full research training pipeline.

The release check record states the actual locally executed test count. CI configuration is provided; a Windows run, fresh internet dependency installation, browser visual inspection, physical flashing and biological validation are not claimed by this preparation. Numerical tests and source checks reduce the risk of defects; they do not establish an absolute absence of errors.
