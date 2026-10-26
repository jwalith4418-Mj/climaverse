# Publication review — 2026-10-26

Selected the supplied CLIMAVERSE 0.1.0 backend proof of concept from
`hackathon/ISRO/CLIMAVERSE.zip`. Inventoried all four approved source roots,
including archive entries: this is the only available CLIMAVERSE source version.
The archive contains an implemented engine, API, IMD-format reader, simulation,
walk-forward validation and tests. The frontend remains a design specification;
no working UI is claimed. All original architecture and research/design notes
are preserved in their supplied locations.

This is an offline synthetic-data demonstration. No real IMD datasets are bundled
or downloaded. Synthetic benchmark results do not establish real-world forecast
skill. The residual model can perform worse than persistence despite an overly
strong claim in its original docstring; inspect measured backtest results instead.
Scenario outputs are first-order proxies. Real-data ingestion, operational service
deployment, national scaling and a browser UI have not been validated.

Original source files are preserved. Publication adds this review, CI and ignore
rules only. No original license file was supplied; no new license is assigned.
Excluded the adjacent personal participation certificate from this source release.
No credentials, personal datasets, dependencies or generated caches are included.

Compared the implementation with all 28 accessible GitHub repositories, including
private and collaborator repositories: no nonempty source files matched. Unrelated
third-party dependency forks do not contain this project and do not constitute
one of the user's project uploads.

All seven supplied tests passed locally on Python 3.14, including the synthetic
model-versus-climatology check, IMD-format round trip, ledger tamper detection,
scenario direction and health/recalibration invariants. API validation is recorded
in the publication ledger. All six API endpoints passed with full ConvLSTM startup;
the synthetic-data flag, valid ledger and scenario direction were verified.
Training uses random initialization, so benchmark values may vary between runs.
