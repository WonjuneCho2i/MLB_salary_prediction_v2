# 4-1 temporal validation design

Random splitting is not used. Every validation season occurs strictly after all training seasons in its fold.

| Fold | Training salary years | Validation salary year | Train rows | Validation rows |
|---:|---|---:|---:|---:|
| 1 | 2020-2022 | 2023 | 948 | 301 |
| 2 | 2020-2023 | 2024 | 1,249 | 314 |
| 3 | 2020-2024 | 2025 | 1,563 | 327 |
| 4 | 2020-2025 | 2026 | 1,890 | 338 |

After model selection, all 2,228 labeled rows from salary years 2020-2026 are refitted.
The final model then predicts 2027 salary for 368 PA>=50 hitters using 2026 features.

Repeated players may appear in both earlier training years and a later validation year. This is intentional: the real task predicts a player's future salary from information already available at prediction time.

All imputers, scalers, and encoders must be fitted inside each training fold to prevent leakage.
