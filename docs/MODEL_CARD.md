# Model card

**Intended use:** research demonstrations of personal sleep/attention monitoring and measurement suitability.

**Bundled data:** synthetic engineering cohort; five person-disjoint roles: train, probability, policy, calibration and test. Source origin is not a predictor. The manifest defines units, identities and the two-hour forecast horizon.

**Default:** scaled Ridge regression, calibrated event head, acquired short protocol, normalized prediction intervals, personal history features and the always-measure fallback. The candidate physiological wake-rate update remains a separate research model.

**Reported result:** the common-protocol H versus H+P comparison uses 20 existing test identities and 556 suitable decisions per stage. H MAE is 13.3285/12.4757 ms before/after; H+P MAE is 12.0226/11.4191 ms. This is a retrospective fixed-long-acquisition study, not an improvement claim over the default short-protocol model.

**Validation scope:** numerical checks, recorded engineering experiments, virtual hardware and software tests. Biological phase matching, new real people, physical hardware, clinical/astronaut validation and Windows execution require their own evidence. Test counts in this distribution are measured again; historical test totals are not reused.

**Limitations:** the generator and model share physiological assumptions. A narrow error distribution in this cohort is not evidence of real-world physiological validity. Workload explanation requires a validated workload input. Exposure integrals are not organ-risk probabilities.
