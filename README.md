# Machine Learning Final Project

## Overview

This project was completed as part of the final requirements for the Machine Learning course in the Department of Information Management at National Cheng Kung University.

## Datasets Used

### 1. Glass Identification Dataset

- **Description:**
  - Contains 124 instances
  - Includes 9 attributes: ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
  - Has 6 classes: [1, 2, 3, 5, 6, 7]
- **Steps:**
  1. Data overview and preprocessing
  2. 10-bin discretization
  3. Prior probability calculation
  4. Five-fold cross-validation
  5. Naive Bayes Classifier (NBC) and Selective Naive Bayes (SNB) model implementation

### 2. Hepatitis Dataset

- **Description:**
  - Initially contains 155 instances, reduced to 80 after preprocessing
  - Includes 19 attributes
  - Has 2 classes: [1 (DIE), 2 (LIVE)].
- **Steps:**
  1. Data preview and preprocessing
  2. 10-bin discretization
  3. Prior probability calculation
  4. Five-fold cross-validation
  5. Naive Bayes Classifier (NBC) and Selective Naive Bayes (SNB) model implementation

### 3. Image Segmentation Dataset

- **Description:**
  - Contains a total of 2310 instances
  - Includes 19 attributes
  - Has 7 classes: [brickface, sky, foliage, cement, window, path, grass]
- **Steps:**
  1. Data overview and preprocessing
  2. 10-bin discretization

## Methodology

### Naive Bayes Classifier (NBC)

- Implemented NBC using all attributes and performed five-fold cross-validation to evaluate the performance.
- Calculated accuracy for each fold and computed the average accuracy.

### Selective Naive Bayes (SNB)

- Incrementally selected attributes to optimize the classification accuracy.
- Evaluated the average accuracy after each step to determine the optimal set of attributes.

## Results

### Glass Identification Dataset

- **NBC Average Accuracy:** 0.5843
- **SNB Best Accuracy:** 0.6401 with attributes ["Al", "Ba", "Ca", "Mg", "Si", "K"]

### Hepatitis Dataset

- **NBC Average Accuracy:** 0.85
- **SNB Best Accuracy:** 0.9625 with attributes ["PROTIME", "ALBUMIN", "SEX", "FATIGUE", "ANOREXIA", "LIVER_BIG"]

## SNB Selection Sheet

The SNB selection process involves incrementally selecting attributes based on their contribution to improving classification accuracy. Here are the selection details for each dataset:

### Glass Identification Dataset SNB Selection

| Step | Selected Attribute | Accuracy |
|------|---------------------|----------|
| 1    | Al                  | 0.5769   |
| 2    | Ba                  | 0.5769   |
| 3    | Ca                  | 0.5769   |
| 4    | Mg                  | 0.6346   |
| 5    | Si                  | 0.6346   |
| 6    | K                   | 0.6401   |

### Hepatitis Dataset SNB Selection

| Step | Selected Attribute | Accuracy |
|------|---------------------|----------|
| 1    | PROTIME             | 0.85     |
| 2    | ALBUMIN             | 0.8875   |
| 3    | SEX                 | 0.9375   |
| 4    | FATIGUE             | 0.95     |
| 5    | ANOREXIA            | 0.95     |
| 6    | LIVER_BIG           | 0.9625   |

## Conclusion

The project demonstrates the implementation and evaluation of Naive Bayes and Selective Naive Bayes classifiers on multiple datasets. The selective approach showed improved performance in terms of accuracy by optimizing the set of attributes used for classification.
