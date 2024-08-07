# Machine Learning Project - Laplace's estimate

## Datasets Used

## 1. Hepatitis Dataset
- 155 instances, reduced to 80 after preprocessing
- 19 attributes
- 2 classes: [1 (DIE), 2 (LIVE)]

## 2. Image Segmentation Dataset
- 2310 instances
- 19 attributes
- 7 classes: [brickface, sky, foliage, cement, window, path, grass]


## Methodology
### Steps
  1. Data overview and preprocessing
  2. 10-bin discretization
  3. Prior probability calculation
  4. Five-fold cross-validation
  5. Naive Bayes Classifier (NBC) and Selective Naive Bayes (SNB) model implementation
     
### Naive Bayes Classifier (NBC)

- Implemented NBC using all attributes and performed five-fold cross-validation to evaluate the performance.
- Calculated accuracy for each fold and computed the average accuracy.

### Selective Naive Bayes (SNB)

- Incrementally selected attributes to optimize the classification accuracy.
- Evaluated the average accuracy after each step to determine the optimal set of attributes.

## Results

## 1. Hepatitis Dataset
### Prior Results:
- Class 1 (DIE): P(Class 1) = 0.371
- Class 2 (LIVE): P(Class 2) = 0.629

### Five-fold cross-validation:
- | Fold | Count|
  |------|------|
  | 1    | 16   |
  | 2    | 16   |
  | 3    | 16   |
  | 4    | 16   |
  | 5    | 16   |
  
### NBC Results:
- NBC Average Accuracy: 0.85
- |ACC 1 |ACC 2 |ACC 3 |ACC 4 |ACC 5 |AVG ACC|
  |------|------|------|------|------|-------|
  |0.9375|0.875 |1.0   |0.625 |0.8125|0.85   |


### SNB Selection:
- SNB Best Accuracy: 0.9625
- | Step | Selected Attribute    | Accuracy | Step | Selected Attribute  | Accuracy |
  |------|-----------------------|----------|------|---------------------|----------| 
  | 1    | PROTIME               | 0.9      | 10   | AGE/MALAISE         | 0.925    |
  | 2    | ALBUMIN               | 0.9125   | 11   | ANTIVIRALS          | 0.9125   |
  | 3    | SEX                   | 0.9375   | 12   | SPLEEN_PALPABLE     | 0.9      |
  | 4    | FATIGUE               | 0.9625   | 13   | ALK_PHOSPHATE/SGOT/HISTOLOGY| 0.8875 |
  | 5    | ANOREXIA              | 0.9625   | 14   | BILIRUBIN/MALAISE/ASCITES   | 0.8625 |
  | 6    | LIVER_BIG             | 0.9625   | 15   | MALAISE             | 0.875    |
  | 7    | LIVER_FIRM            | 0.95     | 16   | HISTOLOGY           | 0.8375   |
  | 8    | STEROID               | 0.95     | 17   | SGOT/ASCITES        | 0.8375   |
  | 9    | VARICES               | 0.9375   | 18   | ASCITES             | 0.8375   |

## 2. Image Segmentation Dataset

### Prior Results:
The number of data points for each class in the test set is the same, the prior is also the same.
- Class 1(BRICKFACE): P(Class 1) = 0.1429
- Class 2(CEMENT): P(Class 2) = 0.1429
- Class 3(FOLIAGE): P(Class 3) = 0.1429
- Class 4(GRASS): P(Class 4) = 0.1429
- Class 5(PATH): P(Class 5) = 0.1429
- Class 6(SKY): P(Class 6) = 0.1429
- Class 7(WINDOW): P(Class 7) = 0.1429


### Five-fold cross-validation:
- |Fold | Count|
  |-----|------|
  | 1   | 42   |
  | 2   | 42   |
  | 3   | 42   |
  | 4   | 42   |
  | 5   | 42   |

### NBC Results:
- NBC Average Accuracy: 0.8048
- |ACC 1 |ACC 2 |ACC 3 |ACC 4 |ACC 5 |AVG ACC|
  |------|------|------|------|------|-------|
  |0.8333|0.8095|0.8571|0.8095|0.7143|0.8048 |


### SNB Selection:
- SNB Best Accuracy: 0.8571
- | Step | Selected Attribute    | Accuracy| Step | Selected Attribute       | Accuracy|
  |------|-----------------------|---------|------|--------------------------|---------| 
  | 1    | exgreen-mean          | 0.5190  |10    | short-line-density-5     | 0.8476  |
  | 2    | exred-mean            | 0.6667  |11    | vedge-sd                 | 0.8429  |
  | 3    | region-centroid-row   | 0.7810  |12    | rawred-mean              | 0.8286  |
  | 4    | intensity-mean        | 0.8333  |13    | hue-mean                 | 0.8571  |
  | 5    | short-line-density-2  | 0.8429  |14    | rawgreen-mean            | 0.8381  |
  | 6    | region-centroid-col   | 0.8524  |15    | rawblue-mean/value-mean  | 0.8238  |
  | 7    | region-pixel-count    | 0.8524  |16    | exblue-mean              | 0.8286  |
  | 8    | hedge-sd              | 0.8476  |17    | saturatoin-mean          | 0.8238  |
  | 9    | vedge-mean            | 0.8476  |18    | value-mean               | 0.8143  |

## Conclusion
The project demonstrates the implementation and evaluation of Naive Bayes and Selective Naive Bayes classifiers on multiple datasets. The selective approach showed improved performance in terms of accuracy by optimizing the set of attributes used for classification.
