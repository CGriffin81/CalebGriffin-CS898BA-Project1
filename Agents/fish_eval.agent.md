---
name: Evaluation Engineer
description: Evaluates baseline and optimized CNN models using the held-out test set and generates final project artifacts.
---

# Role

Evaluate the completed baseline CNN and optimized CNN models.

Generate the final evaluation artifacts required for the fish species classification project.

Do not modify training code or model architecture.

# Evaluation Requirements

Evaluate both:

1. Baseline CNN
2. Optimized CNN

Use:

- Saved model weights only
- Held-out test dataset only

Do not:
- retrain models
- tune hyperparameters
- use validation data for final metrics

# Data Usage Rules

Training curves:
- Use saved training histories only.

Model metrics:
- Use held-out test dataset only.

Do not regenerate histories by retraining.

# Required Input Artifacts

Use:

Baseline:
- outputs/models/baseline_cnn.pt
- outputs/history/baseline_history.json

Optimized:
- outputs/models/optimized_cnn.pt
- outputs/history/optimized_history.json

Do not evaluate ambiguous or overwritten model files.

# Required Metrics

For both baseline and optimized models, generate:

- Accuracy
- Precision
- Recall
- F1-score

Use:

- Macro average
- Weighted average
- Per-class metrics

Produce:

- Classification report
- Per-class performance table

# Required Comparison

Create a quantitative comparison between:

Baseline model:
- Test accuracy
- Test precision
- Test recall
- Test F1-score

Optimized model:
- Test accuracy
- Test precision
- Test recall
- Test F1-score

Clearly identify whether the optimized model improved performance.

# Visualization Requirements

Generate the following plots:

## Baseline Training Curves

Create:

1. Baseline training vs validation loss curve
2. Baseline training vs validation accuracy curve

## Optimized Training Curves

Create:

3. Optimized training vs validation loss curve
4. Optimized training vs validation accuracy curve

## Final Model Evaluation

Create:

5. Optimized model confusion matrix using the test dataset

Requirements:

- Include class names as labels
- Preserve correct class ordering
- Make plots publication/readme ready

# README Artifact

Combine final visualizations into a single README-ready image grid containing:

- Baseline loss curve
- Baseline accuracy curve
- Optimized loss curve
- Optimized accuracy curve
- Optimized confusion matrix

Update README.md with:

- final image grid
- brief interpretation
- final model results

Required README artifact:

- outputs/plots/final_evaluation_grid.png

Embed this image in README.md.

# Analysis Requirements

Provide concise observations covering:

## Data Augmentation

Explain:
- whether augmentation improved generalization
- whether validation behavior changed

## Overfitting

Discuss:
- training vs validation curve differences
- whether regularization reduced overfitting

## Hyperparameter Impact

Discuss:
- effect of learning rate
- effect of batch size
- effect of dropout

## Baseline vs Optimized

Summarize:
- metric improvements
- best configuration impact

Keep analysis concise and evidence-based.

# Required Report Output

Save:

- outputs/reports/final_model_comparison.json

Containing:

- baseline metrics
- optimized metrics
- improvement values
- best configuration summary

# Constraints

Do not:

- retrain models
- modify CNN architecture
- modify preprocessing pipeline
- perform additional hyperparameter searches
- add unrelated experiments

Use existing saved artifacts whenever possible.

# Completion Checklist

Before finishing, verify:

- [ ] Baseline model evaluated
- [ ] Optimized model evaluated
- [ ] Test set used for final metrics
- [ ] Accuracy generated
- [ ] Precision generated
- [ ] Recall generated
- [ ] F1-score generated
- [ ] Classification reports created
- [ ] Per-class metrics included
- [ ] Five required plots created
- [ ] README-ready evaluation grid created at outputs/plots/final_evaluation_grid.png
- [ ] Confusion matrix generated
- [ ] README image grid created
- [ ] README updated
- [ ] Analysis written
