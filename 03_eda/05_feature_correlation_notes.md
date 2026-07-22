# 3-5 correlation and multicollinearity notes

- Numeric features checked: 46
- Feature pairs with |Spearman r| >= 0.85: 15
- Features with VIF >= 10 or infinite: 28
- VIF is diagnostic only; missing numeric values were median-filled for this calculation.
- The actual model must fit all imputers inside each training fold.

## Modeling decisions to test

- Keep current_contract_stage as the primary contract representation and avoid duplicating all stage flags in linear models.
- Do not use WAR/600 and Dollars/600 together when they encode essentially the same value scale.
- Avoid combining component rates with their exact or near-exact totals without regularization.
- Age and age squared are intentionally related; retain both only for models that need an explicit nonlinear age term.
- Ridge/Elastic Net and tree models can be compared with a reduced-feature linear model.
