# Analytics_AB_testing
Here will be shared different options to build analytics to A/B-test for license activation.


**Dataset:** `License_activation_ab_test_dataset_3ver.csv`  
**Goal:** Understand what drives user activation across experiment groups A and B  
**Key question:** Why does Group B activate at a higher rate than Group A?

### Columns
| Column | Description |
|---|---|
| `user_id` | Unique user identifier |
| `experiment_group` | A or B |
| `activated` | 1 = activated, 0 = not activated |
| `time_to_first_run_min` | Minutes from install to first IDE run |
| `used_autocomplete_day1` | Used autocomplete feature on day 1 (0/1) |
| `used_refactoring_day1` | Used refactoring feature on day 1 (0/1) |

