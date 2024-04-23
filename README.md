# Body-Pose-detection-Custom-data

## Project Overview
This project uses machine learning techniques to detect human poses in real-time. It leverages OpenCV and MediaPipe's Holistic model to process video frames and identify various body landmarks.

## Dependencies
- OpenCV
- MediaPipe
- Scikit-learn
- Numpy

## Machine Learning Pipelines
The project uses the following machine learning pipelines for pose classification:

1. **Logistic Regression (lr)**: This pipeline standardizes the features using `StandardScaler` and applies `LogisticRegression` with a maximum iteration of 200.

2. **Ridge Classifier (rc)**: This pipeline standardizes the features using `StandardScaler` and applies `RidgeClassifier`.

3. **Random Forest Classifier (rf)**: This pipeline standardizes the features using `StandardScaler` and applies `RandomForestClassifier`.

4. **Gradient Boosting Classifier (gb)**: This pipeline standardizes the features using `StandardScaler` and applies `GradientBoostingClassifier`.

## How to run the above code
1] First git clone above files.

2] Second run the notebook Body Pose Detection.

There first custom data will be asked and then at the end you can have predictions.

## Note
If you want to use of pretrained model then use the saved version of model which is also available.



