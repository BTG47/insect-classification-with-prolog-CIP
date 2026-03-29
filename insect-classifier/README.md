# Insect Classifier

This module contains the visual component of the project. Its purpose is to process insect images and produce predictions or useful visual evidence that can later be consumed by the symbolic reasoning layer.

# Purpose

The classifier is the perception component of the system.It is responsible for:

- loading and organizing image data
- preprocessing images
- training and evaluating a vision model
- generating predictions from insect images
- producing outputs that may later support symbolic inference

## Folder Structure

```text
insect-classifier/
├── configs/       # Configuration files for experiments, training, and parameters
├── data/          # Local data references, lightweight metadata, and data notes
├── docs/          # Technical notes and supporting documentation
├── notebooks/     # Exploratory notebooks and early experiments
├── outputs/       # Saved results, plots, logs, and lightweight model outputs
├── src/           # Main source code for the classifier
├── README.md
└── requirements.txt
```

# Expected Responsibilities of This Module

This module may include:

* dataset loading
* train/validation/test split logic
* image transformations and preprocessing
* model definition
* training loop
* evaluation metrics
* inference pipeline

## Suggested Development Flow

1. Explore and inspect the dataset.
2. Define the target classes for the first MVP.
3. Implement preprocessing and data loading.
4. Train a simple baseline model.
5. Evaluate performance.
6. Export predictions or confidence values for downstream reasoning.

## Scope for the First Iteration

To keep the project feasible, the initial version should prioritize:

* static images instead of real-time input
* a reduced number of classes
* a simple and understandable baseline
* clear outputs that can be reused by the symbolic module

## Possible Outputs

Examples of outputs from this module:

* predicted class
* confidence scores
* top-k predictions
* extracted attributes or intermediate evidence
* saved evaluation results

## Notes

* Avoid storing large raw datasets directly in Git.
* Keep notebooks readable and progressively move stable code into `src/`.
* Use `configs/` to avoid hardcoding experimental settings.

## Dependencies

See `requirements.txt`.

## Status

This module is under active development and will evolve as the project baseline becomes more stable.
