# Architecture

The maintained request path is `v136.service.Handler` → `v136.application.Application` → `v135.application.Application` → `v135.replay.run` → `v135.engine.Engine.decide`.

The current service implements GET and POST through the same v136 application. Legacy HTTP routes are intentionally not exposed. Engine output is enriched after numerical inference with evidence, routine progress, hardware context and provenance.

`v133.physiology` integrates the Forger99 oscillator and sleep homeostat. `v135.trials` summarizes acquired trial prefixes. Labels are delivered only at their availability times. `v132.ledger` stores immutable application-level records. `v136.evidence` is associative explanation, not causal probability.

The research workflow compares feature groups with person-disjoint roles, performs calibration sensitivity resampling and numerical phase checks, and runs day-length scenarios. Recorded plots are not new experiments conducted during repository preparation.

The interface uses a 15-second deadline covering response headers and JSON body, abort controllers and sequence guards. No extra JavaScript package installation is required for the normal interface.
