## Churn models (validation split)

| model | subset | accuracy | precision | recall | f1 | roc_auc | tn | fp | fn | tp | search_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | recency<=180 | 0.702 | 0.584 | 0.716 | 0.643 | 0.785 | 179 | 79 | 44 | 111 | 14.7 |
| random_forest | recency<=180 | 0.741 | 0.767 | 0.445 | 0.563 | 0.78 | 237 | 21 | 86 | 69 | 37.0 |
| lightgbm | recency<=180 | 0.731 | 0.657 | 0.594 | 0.624 | 0.779 | 210 | 48 | 63 | 92 | 13.0 |
| decision_tree | recency<=180 | 0.726 | 0.706 | 0.465 | 0.56 | 0.777 | 228 | 30 | 83 | 72 | 0.0 |
| logistic_regression | recency<=180 | 0.709 | 0.761 | 0.329 | 0.459 | 0.758 | 242 | 16 | 104 | 51 | 0.0 |
| knn | recency<=180 | 0.714 | 0.628 | 0.587 | 0.607 | 0.746 | 204 | 54 | 64 | 91 | 0.8 |
| xgboost | all | 0.736 | 0.712 | 0.897 | 0.794 | 0.814 | 181 | 162 | 46 | 400 | 14.7 |
| random_forest | all | 0.749 | 0.787 | 0.762 | 0.774 | 0.813 | 251 | 92 | 106 | 340 | 37.0 |
| lightgbm | all | 0.752 | 0.753 | 0.834 | 0.791 | 0.807 | 221 | 122 | 74 | 372 | 13.0 |
| logistic_regression | all | 0.725 | 0.781 | 0.713 | 0.746 | 0.796 | 254 | 89 | 128 | 318 | 0.0 |
| decision_tree | all | 0.736 | 0.778 | 0.747 | 0.762 | 0.793 | 248 | 95 | 113 | 333 | 0.0 |
| knn | all | 0.73 | 0.742 | 0.8 | 0.77 | 0.782 | 219 | 124 | 89 | 357 | 0.8 |

## CLV models (validation split)

| model | subset | scale | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| ridge | all | log | 2.221 | 2.624 | 0.333 |
| ridge | all | original | 458.712 | 1777.085 | 0.326 |
| ridge | recency<=180 | log | 2.429 | 2.718 | 0.296 |
| ridge | recency<=180 | original | 743.828 | 2411.763 | 0.312 |
| random_forest_reg | all | log | 2.152 | 2.565 | 0.363 |
| random_forest_reg | all | original | 446.536 | 1841.723 | 0.276 |
| random_forest_reg | recency<=180 | log | 2.328 | 2.653 | 0.329 |
| random_forest_reg | recency<=180 | original | 721.201 | 2502.624 | 0.26 |
| two_stage | all | log | 3.072 | 3.671 | -0.305 |
| two_stage | all | original | 438.141 | 1651.089 | 0.418 |
| two_stage | recency<=180 | log | 2.426 | 3.251 | -0.008 |
| two_stage | recency<=180 | original | 658.947 | 2243.103 | 0.405 |
