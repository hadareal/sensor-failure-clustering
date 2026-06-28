# Machine Failure Classification from Sensor Data

This repository contains my solution for the ML6 Medior ML Engineer challenge use case on machine failure classification from sensor data.

## Problem

The dataset contains **1,600 machine failure events**, each represented by **20 sensor features**.

- **40 failures** have a known failure label
- **1,560 failures** are unlabeled

The goal of this project is to use the small labeled subset together with the large unlabeled subset to assign likely failure labels to the unlabeled machine breakdowns, and to identify **high-confidence predictions** that could support expert review.

---

## Repository Structure

- `notebook/01_sensor_clustering.ipynb`  
  Main notebook containing the full analysis, experiments, evaluation, and final prediction pipeline.

- `outputs/`  
  Generated figures and output files, including:
  - PCA visualization
  - sensor ranking plot
  - model comparison plot
  - final prediction confidence plot
  - final CSV with predicted labels and confidence scores

- `presentation/`  
  Final presentation for the use case.

---

## Approach Summary

The work was structured in several steps:

1. **Initial data exploration**
   - basic dataset inspection
   - comparison between labeled and unlabeled portions
   - PCA visualization of the labeled failures

2. **Unsupervised clustering baselines**
   - KMeans
   - Agglomerative Clustering
   - Gaussian Mixture Models

   These methods showed weak separation in the raw 20-sensor space.

3. **Label-guided approaches**
   I then used the 40 labeled failures to evaluate simple supervised / semi-supervised baselines:
   - k-Nearest Neighbors
   - centroid-based classification
   - weighted kNN
   - Label Spreading

4. **Feature selection**
   Since the full 20-sensor space appeared noisy, I ranked sensors by how much their mean values differed across the known failure labels and selected the **top 5 most discriminative sensors**.

5. **Model comparison**
   I compared the main models both on:
   - all 20 sensors
   - the top 5 selected sensors

   I also tested fully supervised baselines (Logistic Regression, Random Forest) to verify whether
   the semi-supervised approach was actually necessary. The best-performing model was
   **Random Forest on the top 5 sensors** (LOO accuracy 0.725, balanced accuracy 0.717).

6. **Final prediction**
   I trained the final Random Forest model on all 40 labeled failures and used it to assign predicted labels to all **1,560 unlabeled failures**, together with a confidence score for each prediction.

---

## Final Result

The final selected model was:

- **Model:** Random Forest
- **Feature set:** Top 5 discriminative sensors
- **Evaluation:** Leave-one-out validation on the 40 labeled failures
- **Accuracy:** **0.725** (balanced accuracy **0.717**, macro F1 **0.723**)

The final model produced predictions for all **1,560 unlabeled machine failures**.

Using a confidence threshold of **0.7** (LOO-validated: 0.857 accuracy on 35% of the labeled set):

- **524 high-confidence predictions**

Using a stricter threshold of **0.8** (LOO-validated: 1.000 accuracy on 25% of the labeled set):

- **237 high-confidence predictions**

> **Note:** Confidence estimates are based on leave-one-out evaluation on 40 labeled samples
> and should be treated as indicative until validated on additional labeled data.

The final predictions are saved in:

```text
outputs/final_rf_predictions.csv
```
