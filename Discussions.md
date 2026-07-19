# Fish Classification Discussion

For a deeper look at the metrics, look at reports/final_model_comparison.json

## Qualitative Analysis
#### Data Augmentation
Data augmentation improved generalization by reducing overfitting. Training and validation accuracy increased together, and validation performance remained stable throughout training.

#### Hyperparameter Impact
Learning rate had the greatest effect on performance. Of the options attempted, learning rate 0.001 produced the lowest validation loss (0.563). Batch size 32 consistently outperformed 64, while increasing dropout to 0.5 improved generalization without slowing convergence.

#### Overfitting
The baseline model showed a widening gap between training and validation performance during later epochs, indicating mild overfitting. The optimized model reduced this effect through improved regularization while maintaining strong validation accuracy.

#### Baseline vs. Optimized
The optimized model improved overall performance from 88.8% to 90.8% test accuracy. Precision, recall, and F1-score also increased, demonstrating that the selected hyperparameters produced a modest but consistent improvement over the baseline.


## Quantitative Analysis
#### Baseline Model
The baseline CNN achieved 88.8% test accuracy. Macro-averaged precision, recall, and F1-score were 0.877, 0.870, and 0.872, respectively. The strongest performance was on the Gold and Discus classes, while Cray and Oscar were the most difficult to classify.

#### Optimized Model
The optimized CNN achieved 90.8% test accuracy. Macro-averaged precision, recall, and F1-score improved to 0.904, 0.894, and 0.896, respectively. The largest improvement occurred for the Oscar class, with smaller gains in Bete and Cray, while performance on the remaining classes remained consistently high.

#### Model Comparison
The optimized model improved test accuracy by 2.0 percentage points (88.8% → 90.8%). Macro precision increased by 2.7 percentage points, macro recall by 2.4 percentage points, and macro F1-score by 2.4 percentage points. These results indicate that the selected hyperparameters produced a modest but consistent improvement over the baseline model.
