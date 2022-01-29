# X Education Customer Lead Score Analysis

This repository is about my analysis for X Education Customer using lead scoring.

* [Dataset](https://github.com/dhykac/X_Education_Lead_Score_Customer_Analysis/blob/main/lead_scoring.csv)
* [Data Description](https://github.com/dhykac/X_Education_Lead_Score_Customer_Analysis/blob/main/Leads%20Data%20Dictionary.xlsx)
* [Notebook](https://github.com/dhykac/X_Education_Lead_Score_Customer_Analysis/blob/main/X_Education_Customer_Lead_Score_Analysis.ipynb)

### Packages
```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score, recall_score, precision_score, accuracy_score
```

### Objectives
This code is to predict the customers converted or not using their lead score with help of Random Forest Classifier. The target columns for this objective is `['Converted']`

### Results Overview
![roc curve](https://user-images.githubusercontent.com/92696555/151674862-66070ed8-af76-4cb6-a514-ee25e1a1af1a.png)
* The model succeed to predict with rate 85.06% accuracy.
* The model predicted converted and the customer actually converted is 528.
* The model predicted not converted and the customer actually converted is 85.
* The model predicted converted and the customer actually not converted is 53.
* The model predicted not converted and the customer actually not converted is 258.
* AUC : The model succeed to distinct between True Positive and True Negative with chance 91.9%
* F1 Score : the harmonic mean between precission and recall is 78.9% which is important for us to consider False Positive and False Negative.
* Precission : The rate of model predict results are False Positive (which is 83%)
* Recall : The rate of model predict results are False Negative (which is 75.2%)
