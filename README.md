# CIP Insect Classification with Neuro-Symbolic Reasoning

This repository contains the development of a neuro-symbolic insect classification project that combines two main components:

1. **Visual classification** through an image-based model.
2. **Symbolic reasoning** through rules and logical inference.

The goal is not only to classify insect categories from images, but also to provide a more interpretable reasoning process behind the final prediction.

## Repository Structure

```text
.
├── insect-classifier/
│   ├── configs/
│   ├── data/
│   ├── docs/
│   ├── notebooks/
│   ├── outputs/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
│
├── symbolic-reasoning/
│   ├── docs/
│   ├── examples/
│   ├── notebooks/
│   ├── rules/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

## Project Overview

This project is organized into two major modules:

### 1. `insect-classifier`

This module focuses on the visual pipeline:

* dataset exploration
* preprocessing
* model training
* evaluation
* inference from insect images

### 2. `symbolic-reasoning`

This module focuses on the symbolic and neuro-symbolic pipeline:

* rule design
* logical reasoning
* class approximation through expert knowledge
* experimentation with DeepProbLog / symbolic logic integration

## Current Scope

This first iteration is designed as a  **reduced academic MVP** .

The main focus is to:

* build a working image classification baseline
* connect predictions or extracted evidence to a symbolic reasoning layer
* produce a classification process that is more explainable than a black-box model alone

## Main Objectives

* Learn and apply a practical computer vision workflow for insect images.
* Explore a neuro-symbolic approach using logical rules.
* Keep the project modular so each component can evolve independently.
* Build a foundation for future improvements such as:
* more advanced models
* better fine-grained classification
* prototype-based explanations
* real-time demo support

## Development Strategy

The project is being developed in stages:

1. establish a visual baseline
2. define a small symbolic reasoning layer
3. integrate both components
4. evaluate results and explainability

## Notes

* Large datasets and heavy artifacts should **not** be stored directly in the repository.
* Code, notebooks, rules, and lightweight metadata should be versioned here.
* Additional documentation and project management may be maintained externally.

## Status

This repository is currently under active development.

## Authors

Project developed by a 4-member team as part of an Artificial Intelligence final project, with an emphasis on insect classification and explainable neuro-symbolic reasoning.
