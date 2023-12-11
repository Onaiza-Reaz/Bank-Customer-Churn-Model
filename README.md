**Bank Customer Churn Prediction Model**

This project builds a Random Forest model to predict bank customer churn using data from the Churn_Modelling.csv dataset. It achieves a best accuracy of 82.5% compared to other models like Logistic Regression, Support Vector Machine, K-Nearest Neighbors, Decision Tree, and Gradient Boosting.

**Key Features:**

**Data Cleaning:** Handles missing values, duplicate values, and irrelevant columns.

**Feature Engineering:** Encodes categorical data using one-hot encoding.

**Imbalanced Class Handling:** Addresses class imbalance by applying SMOTE.

**Model Training and Evaluation:** Trains and evaluates various machine learning models using accuracy, precision, recall, and F1-score.

**Model Saving and Deployment:** Saves the best model (Random Forest) for future use.

**Software and Libraries:**

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Imblearn

**Project Structure:**

Bank Customer Churn.ipynb: Jupyter notebook containing the complete analysis and modeling code.
Churn_Prediction_Model: Saved Random Forest model.

**Getting Started:**

Install the required libraries.

Open Bank Customer Churn.ipynb in a Jupyter notebook environment.

Run the code cells to reproduce the analysis and results.

Load the saved model using joblib.

Use the model to predict customer churn on your own data.

**Potential Applications:**

Identify customers at risk of churn

Develop targeted marketing campaigns

Improve customer retention strategies

**Future Work:**

Explore other machine learning models such as Deep Learning algorithms.

Fine-tune hyperparameters for further accuracy improvement.

Implement churn prediction API for real-time application.

By exploring this project, you can gain valuable insights into bank customer churn prediction and machine learning techniques.

**Additional Notes:**

This project is intended for educational purposes only.

The model might not be generalizable to other datasets without further fine-tuning.

Feel free to explore and improve the project further.
