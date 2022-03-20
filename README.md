# X Education Customer Lead Score Analysis

This repository is about my analysis for X Education Customer using lead scoring.

* [Dataset](https://github.com/dhykac/X_Education_Lead_Score_Customer_Analysis/blob/main/Leads_X_Education.csv)
* [Data Description](https://github.com/dhykac/X_Education_Lead_Score_Customer_Analysis/blob/main/Leads%20Data%20Dictionary.xlsx)
* [Notebook](https://github.com/dhykac/X_Education_Lead_Score_Customer_Analysis/blob/main/X_Online_Education_Lead_Scoring_Analysis.ipynb)

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
* The model succeed to predict with rate 86.68% accuracy.
* The model predicted converted and the customer actually converted is 267.
* The model predicted not converted and the customer actually converted is 76.
* The model predicted converted and the customer actually not converted is 47.
* The model predicted not converted and the customer actually not converted is 534.
* AUC : The model succeed to distinct between True Positive and True Negative with chance 92.99%
* F1 Score : the harmonic mean between precission and recall is 81.28% which is important for us to consider False Positive and False Negative.
* Precission : The rate of model predict results are False Positive (which is 83.44%)
* Recall : The rate of model predict results are False Negative (which is 77.84%)
![SHAP](https://user-images.githubusercontent.com/92696555/159148786-d96369a5-daad-4471-b40e-4f13a5d3d058.png)
* From the plot above, the three 3 most important feature from the model is `Total Time Spent on Website` , `Lead Quality` , and `Last Notable Activity` which is make sense because these three could be our basis to determine if the customers converted or not.

### Simulation
By using 924 customers data, with assumed 
* revenue per customers 487 dollars ( Rp 7.000.000 )
* cost per customers 417 dollars ( Rp 6.000.000 ) 
* campaign success rate 100%`

The ML model succeed to 
* boost convertion rate by 49.57% 
* raise the profit by 32.060 dollars ( Rp 460.698.376 )
