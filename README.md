# Analytics_AB_testing
Here will be shared different options to build analytics to A/B-test for license activation.

## Overview
This project analyzes a 15,000-user A/B experiment on IDE, with a focus on understanding what drives license activation. Group B received a modified onboarding flow; Group A was the control. The analysis reveals a single, dominant pattern that explains nearly all the variance in activation outcomes.

**Dataset:** `License_activation_ab_test_dataset_3ver.csv`  
**Goal:** Understand what drives user activation across experiment groups A and B  
**Key question:** Why does Group B activate at a higher rate than Group A?

### Columns of dataset
| Column | Description |
|---|---|
| `user_id` | Unique user identifier |
| `experiment_group` | A or B |
| `activated` | 1 = activated, 0 = not activated |
| `time_to_first_run_min` | Minutes from install to first IDE run |
| `used_autocomplete_day1` | Used autocomplete feature on day 1 (0/1) |
| `used_refactoring_day1` | Used refactoring feature on day 1 (0/1) |



### Key findings
![picture alt](https://github.com/torgart-dacx/Analytics_AB_testing/blob/main/plot1_ab_comparison.png)

![picture alt](https://github.com/torgart-dacx/Analytics_AB_testing/blob/main/plot2_30min_cliff.png)

![picture alt](https://github.com/torgart-dacx/Analytics_AB_testing/blob/main/plot3_time_distribution.png)

![picture alt](https://github.com/torgart-dacx/Analytics_AB_testing/blob/main/plot4_group_time_comparison.png)

![picture alt](https://github.com/torgart-dacx/Analytics_AB_testing/blob/main/plot5_feature_usage.png)

## Key Findings

| # | Finding | Statistical Strength |
|---|---------|----------------------|
| 1 | Group B activates **12.8% more** than Group A (68.4% vs 60.7%) | χ²=99.4, p<0.001 |
| 2 | **Time to first run is the #1 predictor** — users under 15 min activate at 97%; 30+ min drops to 3.4% | t=−123.6, p<0.001 |
| 3 | Group B reaches first run **4.7 min faster** (median 17.7m vs 22.4m) — this explains B's lift | t=18.5, p<0.001 |
| 4 | Feature usage barely predicts activation — all combos within a 2% band | No signal |

## Recommendations

- 🎯 Make **"first run under 15 minutes"** your North Star metric.
- 🚀 **Ship Group B's onboarding broadly** — the evidence is conclusive.
- 🔍 Investigate what Group B does differently in the **first 17 minutes**.
- ⛔ Do not delay first run to showcase features — feature exposure on day 1 does not move activation.
- 🔧 Focus friction reduction on the **~25% of users taking 30+ minutes** (only 3.4% activate).
