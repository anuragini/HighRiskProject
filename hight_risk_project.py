import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("cancer-death-rates-by-age.csv")

# Select the relevant age group columns
age_columns = [
    'Deaths - Neoplasms - Sex: Both - Age: Under 5 (Rate)',
    'Deaths - Neoplasms - Sex: Both - Age: 5-14 years (Rate)',
    'Deaths - Neoplasms - Sex: Both - Age: 15-49 years (Rate)',
    'Deaths - Neoplasms - Sex: Both - Age: 50-69 years (Rate)',
    'Deaths - Neoplasms - Sex: Both - Age: 70+ years (Rate)'
]

# Melt the dataframe into long format
df_melted = df.melt(
    id_vars=['Entity', 'Year'],
    value_vars=age_columns,
    var_name='Age Group',
    value_name='Cancer Death Rate'
)

# Clean the age group labels
df_melted['Age Group'] = df_melted['Age Group'].str.extract(r'Age: (.*) \(Rate\)')
print(df_melted)

# Drop missing values
df_clean = df_melted.dropna()

# Features and target
X = df_clean[['Age Group', 'Year', 'Entity']]
y = df_clean['Cancer Death Rate']

# Categorical and numeric features
categorical_features = ['Age Group', 'Entity']
numeric_features = ['Year']


# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE (with Entity included): {rmse:.2f}")


# Define custom order for age groups
age_order = ['Under 5', '5-14 years', '15-49 years', '50-69 years', '70+ years']
df_clean['Age Group'] = pd.Categorical(df_clean['Age Group'], categories=age_order, ordered=True)

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_clean,
    x='Age Group',
    y='Cancer Death Rate',
    estimator='mean',
    ci=None,
    palette='viridis'
)

plt.title('Average Cancer Death Rate by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Cancer Death Rate (per 100,000)', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()


# Plot predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Cancer Death Rate")
plt.ylabel("Predicted Cancer Death Rate")
plt.title("Actual vs Predicted Cancer Death Rates (with Country)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()

