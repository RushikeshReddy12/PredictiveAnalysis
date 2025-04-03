# PredictiveAnalysis

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: Kodakandla Rushikesh Reddy

*INTERN ID*: CTO4WP231

*DOMAIN*: DATA ANALYTICS

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

*I developed a regression model that predicted Purchase_Amount for this task using the provided dataset. The dataset contained a categorical feature called city in addition to numerical features like age and salary. I started by looking at the structure of the dataset to determine which columns would be most useful for training the model. City needed to be correctly encoded before being added to a machine learning model because it is a categorical variable.
I used ColumnTransformer to implement a preprocessing step in order to prepare the data. OneHotEncoder was used to transform the categorical feature (City), and StandardScaler was used to standardize the numerical features (Age and Salary). This made sure that every feature was formatted and scaled correctly prior to training.
After that, I investigated the Random Forest Regressor and Linear Regression models. To assess the performance of the model, I divided the dataset into training and testing sets (80% training, 20% testing). Each model was trained independently, and then predictions were made using the test set. Each model's performance was assessed using the Mean Squared Error (MSE), Mean Absolute Error (MAE), and R2 score.After evaluating the two models to determine which one performed best, I selected the one with the highest R2 score. The evaluation metrics were displayed for the selected final model.*

*OUTPUT*
