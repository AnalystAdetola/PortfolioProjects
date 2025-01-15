import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Load the file with a different encoding
file_path = r"C:\Users\israe\Documents\WorldHapiness\World_Data.csv"
data = pd.read_csv(file_path, encoding='latin1')

# Preview the data
print(data.head())


# Check for missing values in the dataset
missing_values = data.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)

# Impute missing values
# Numerical columns: Fill with median
numerical_cols = ['Log GDP per capita', 'Healthy life expectancy at birth', 'Social support', 'Happiness Scores']
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

# Check again for nulls
print("\nAfter Imputation, Missing Values:")
print(data.isnull().sum())


# Correlation matrix to examine relationships between key factors
correlation_matrix = data[['Happiness Scores', 'Log GDP per capita', 'Healthy life expectancy at birth', 'Social support']].corr()


# Heatmap for visualization
plt.figure(figsize=(6, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Happiness and Key Factors')
plt.show()


# Calculate average happiness by country
avg_happiness = data.groupby('Country name')['Happiness Scores'].mean().sort_values(ascending=False)

# Bar Charts for Top and Bottom Countries
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Plot top 10 happiest countries
avg_happiness.head(10).plot(
    kind='bar', 
    color='seagreen', 
    ax=axes[0]
)
axes[0].set_title('Top 10 Happiest Countries', fontsize=14)
axes[0].set_xlabel('Country', fontsize=12)
axes[0].set_ylabel('Average Happiness Score', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
for idx, value in enumerate(avg_happiness.head(10)):
    axes[0].text(idx, value + 0.1, f'{value:.2f}', ha='center', fontsize=10)

# Plot bottom 10 least happy countries
avg_happiness.tail(10).plot(
    kind='bar', 
    color='tomato', 
    ax=axes[1]
)
axes[1].set_title('Bottom 10 Least Happy Countries', fontsize=14)
axes[1].set_xlabel('Country', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
for idx, value in enumerate(avg_happiness.tail(10)):
    axes[1].text(idx, value + 0.1, f'{value:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()


# Global Average Happiness Over Time
yearly_avg = data.groupby('year')['Happiness Scores'].mean()
plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_avg, marker='o', linewidth=2, color='steelblue')
plt.title('Global Average Happiness Score Over Time', fontsize=16, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Happiness Score', fontsize=12)
plt.grid(visible=True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
for year, value in yearly_avg.items():
    plt.text(year, value + 0.05, f'{value:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# Create time series for key metrics
metrics = ['Happiness Scores', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth']
yearly_metrics = data.groupby('year')[metrics].mean()

# Normalized Trends of Key Happiness Factors
normalized_metrics = (yearly_metrics - yearly_metrics.mean()) / yearly_metrics.std()
plt.figure(figsize=(12, 6))
for column in normalized_metrics.columns:
    plt.plot(
        normalized_metrics.index, 
        normalized_metrics[column], 
        marker='o', 
        label=column
    )
plt.title('Normalized Trends of Key Happiness Factors Over Time', fontsize=16, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Normalized Score', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Factors")
plt.grid(visible=True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# Select features and target variable
features = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth']
target = 'Happiness Scores'

# Drop rows with missing values in selected columns
regression_data = data[features + [target]].dropna()

# Split data into features (X) and target (y)
X = regression_data[features]
y = regression_data[target]

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)


# GDP-only regression
gdp_model = LinearRegression()
gdp_model.fit(X_train[:, 0].reshape(-1, 1), y_train)  # GDP-only (column 0 of X)

# Predictions and evaluation
y_pred_gdp = gdp_model.predict(X_test[:, 0].reshape(-1, 1))
r2_gdp = r2_score(y_test, y_pred_gdp)
mse_gdp = mean_squared_error(y_test, y_pred_gdp)

print("Simple Linear Regression (GDP only):")
print(f"R² score: {r2_gdp:.2f}")
print(f"Mean Squared Error: {mse_gdp:.2f}")


# Multiple regression model
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_multi = multi_model.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)
mse_multi = mean_squared_error(y_test, y_pred_multi)

print("Multiple Linear Regression (All Factors):")
print(f"R² score: {r2_multi:.2f}")
print(f"Mean Squared Error: {mse_multi:.2f}")

# Coefficients from the model
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': multi_model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("Feature Importance:")
print(coefficients)


# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_multi, alpha=0.7, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Happiness Scores', fontsize=16)
plt.xlabel('Actual Happiness Score', fontsize=12)
plt.ylabel('Predicted Happiness Score', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals plot
residuals = y_test - y_pred_multi
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple', bins=30)
plt.title('Residuals Distribution', fontsize=16)
plt.xlabel('Residual', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
