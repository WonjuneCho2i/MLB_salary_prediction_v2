# 3-6 final feature selection

The source model-ready CSV is preserved. Modeling columns are selected through feature_manifest_final.json.

## Final comparison sets

- baseline_previous_salary: 2 numeric + 0 categorical
- performance_only_reduced: 16 numeric + 1 categorical
- linear_reduced_main: 24 numeric + 2 categorical
- tree_reduced_main: 32 numeric + 2 categorical
- full_main_rate_safe_sensitivity: 46 numeric + 2 categorical
- full_expanded_sensitivity: 83 numeric + 3 categorical

## Linear reduced-set diagnostic

- VIF >= 10 or infinite after reduction: 2
- Age and age squared are intentionally retained together for a quadratic age effect.
- Regularized linear models remain the primary linear specification.
- Tree models use a broader but still de-duplicated snapshot-safe set.
- Raw cumulative totals remain sensitivity-only because 2026 is a partial-season snapshot.

## Leakage safeguards

- No next-season contract group, next-season age, future AAV, or contract text is included.
- Numeric imputation and categorical encoding must be fitted inside each training fold.
- 2026 features are used only for the final 2027 prediction after model selection.
