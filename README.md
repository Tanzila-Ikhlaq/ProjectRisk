## Project: Risk Assessment

### Overview

The project aims to develop a predictive model to assess the risk associated with project execution. By analyzing various project-related factors such as completion status, sentiment analysis, customer feedback scores (CSAT, NPS, CES), lead status, and industry, the model categorizes projects into different risk levels: No Risk, Medium Risk, and High Risk. This helps project stakeholders make informed decisions and mitigate potential risks.

### Components

1. **Data Preprocessing:** The dataset is cleaned by handling missing values and dropping irrelevant columns. Categorical variables are mapped to numerical values for compatibility with machine learning algorithms.

2. **Risk Factor Calculation:** A function is defined to calculate the risk factor based on project attributes. Depending on factors such as completion status, sentiment, customer feedback scores, and lead status, projects are categorized into different risk levels.

3. **Model Development:** A Random Forest Classifier is trained using a balanced dataset obtained through Synthetic Minority Over-sampling Technique (SMOTE). This model learns from input features to predict project risk levels accurately.

4. **Model Evaluation:** The trained model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score to assess its effectiveness in predicting project risk.

5. **Visualization:** The distribution of risk factors and feature importance are visualized using matplotlib and seaborn to gain insights into the data and model.

6. **Prediction Endpoint:** A Flask web application is created to provide an endpoint for making predictions based on user input. The input data is preprocessed, and the trained model predicts the risk level of the project, returning the predicted risk label.

### Usage

1. **Data Preparation:** Ensure the dataset contains relevant project attributes such as completion status, sentiment, customer feedback scores, lead status, and industry.

2. **Model Training:** Execute the Python script to preprocess the data, train the Random Forest Classifier, and evaluate its performance.

3. **Web Application Deployment:** Run the Flask application to create an API endpoint for making predictions. The endpoint `/predict` accepts POST requests with JSON data containing project attributes and returns the predicted risk label.

### File Structure

- `project_risk.py`: Python script containing data preprocessing, model training, evaluation.
- `app.py`: Flask web application.
- `trained_model.joblib`: Pre-trained Random Forest Classifier model.

### Requirements

- Python 3.11.5
- Libraries: pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn, Flask



