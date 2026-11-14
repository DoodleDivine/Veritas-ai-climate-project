# Importing Modules/Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# I'll make use of Binary classification

# Importing the CSV file
df = pd.read_csv("MonthlyWeatherDataTokyo_2010_2020.csv", skiprows=11)

print("Printing CSV file...")

print(df)

# Cleanup DataFrame to exclude ANN
df = df.drop(columns=['ANN'], errors='ignore')

# Melting and Restructuring the DataFrame in a way that the parameters are vertical and the Years are horizontal
Month_Cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# Month columns into rows
df_melted = pd.melt(
    df,
    id_vars=['PARAMETER', 'YEAR'],
    value_vars=Month_Cols,
    var_name='MONTH',
    value_name='VALUE'
)

# Parameters sit side by side as independent columns
df_clean = df_melted.pivot_table(
    index=['YEAR', 'MONTH'],
    columns='PARAMETER',
    values='VALUE'
).reset_index()

# Cleaned up index multi-naming
df_clean.columns.name = None

print("\n")

print("Printing cleaned CSV file..")

print(df_clean)

# From the project idea, we will use boundaries for Temperature, Relative Humidity and Wind Speed
# This would be our binary target for the classification

# Human comfort conditions: 
# Relative Humidity below than 85%
# Temperature between 10°C to 24°C
# Wind Speed below 5 m/s

df_clean['GoodMonth'] = (
        (df_clean['RH2M'] < 85) &
        (df_clean['T2M'] >= 10) &
        (df_clean['T2M'] <= 24) &
        (df_clean['WS10M'] < 5.0)  
).astype(int)

# Looking at how many fits the boundary and how many don't
print("\n")
print("Distribution of comfort classes")
counts = df_clean['GoodMonth'].value_counts().to_dict()
print(f"Uncomfortable (0): {counts.get(0,0)} months")
print(f"Comfortable (1): {counts.get(1,0)} months")

# Defining features X with target Y
Features = ['RH2M', 'T2M', 'WS10M']
X = df_clean[Features]
Y = df_clean['GoodMonth']

# Splitting 80% study, and 20% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("\n")
print("----Baseline----")

# Now working on the Baseline Model
# Using Dummy classifier to predict the most frequest class to establish a minimum accuracy bar
Dummy = DummyClassifier(strategy="most_frequent")
Dummy.fit(X_train, Y_train)
Baseline_Accuracy = Dummy.score(X_test, Y_test)

print(f"Baseline Accuracy: {Baseline_Accuracy * 100:.2f}%")

# Moving on to the actual model (Random Forest)
Model = RandomForestClassifier(random_state=42)
Model.fit(X_train, Y_train)

print("\n")
print("----Model----")

# Evaluating results
y_prediction = Model.predict(X_test)
Model_Accuracy = accuracy_score(Y_test, y_prediction)

print(f"Random Forest Model Accuracy: {Model_Accuracy * 100:.2f}%")

print("\n")
print("----Classification Report Below----")
print("\n")
print(classification_report(Y_test, y_prediction))
print("\n")

# This is how the recommendation engine would look like
print("----TRAVELCAST RECOMMENDATION SCAN (2010-2020)----")

df_clean['PredictedGoodMonth'] = Model.predict(X)

Recommendations = df_clean[df_clean['PredictedGoodMonth'] == 1]

print(Recommendations[['YEAR', 'MONTH', 'RH2M', 'T2M', 'WS10M']].to_string(index=False))

# This small engine is used to prove that my AI's decision predictions are accurate and fit
# the boundary data of 'GoodMonth' (Temperature, Relative Humidity, Wind Speed)