# Release Notes

## Rust extension 0.50.0

- `s2and-rust>=0.50.0` is required for Rust-backed incremental linking.
- Native extension load failures now surface as import errors instead of silently falling back to Python. Missing extension modules still use the Python fallback path.
- Incremental linking uses the current bucketed score and margin gate artifact format only; flat legacy gate thresholds are not supported.
- Incremental name compatibility now accepts joined and first-token aliases in addition to exact first-name tuples.
- Artifact cache entries are keyed by validator type. Existing raw-ETag cache filenames are still probed as a read-only fallback.
