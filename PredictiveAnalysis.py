import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

fp = "C:/VSCode - ALL/CodTech-Internship/Task-1/large_dataset.csv"
df = pd.read_csv(fp)

X = df[['Age', 'Salary', 'City']]
y = df['Purchase_Amount']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Age', 'Salary']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['City'])
])

models = {
    'LinearRegression': Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())]),
    'RandomForest': Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model, best_score = None, float('-inf')
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    if r2 > best_score:
        best_model, best_score = pipeline, r2

print("Best Model:", best_model.steps[-1][1].__class__.__name__)
y_pred = best_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))