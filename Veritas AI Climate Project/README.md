# Project name

TravelCast: A prototype climate-forecasting system designed specifically to help people find the safest,
most comfortable time of year to visit a destination.

# Track: Climate

# Team: GTEAM (Aaron Genya John)

## The problem

Business consulatants, field geologists, and vacationers who need to plan travel
windows avoiding extreme heat, sticky humidity, or disruptive winds.

## What we built

Historical monthly climate features including Temparature(T2M), Relative Humidity(RH2M), and Wind Speed(WS10M) that
through prediction from the machine learning model, outputs binary classification label (GoodMonth = 1 for the month's that 
have comfortable weather and 0 for uncomfortable weather).

Travelers or tourism platforms can be able to utilize information from historical weather systems of different countries/regions
to instantly analyse and evaluate whether a specific month is climate-safe and comfortable to travel.

## How it works
Data: Uses 11 years(2010-2020) of climate observations monthly, for any country chosen (In this project it's Tokyo) from NASA POWER
dataset (In the form of a CSV file). Data is cleaned by skipping header metadata and pivoting from wide to long format.

Model: A Random Forest Classifier (from scikit-learn) that learns to classify comfortable travel months with the given boundaries
for each of the climate features. 

## Results

First test: 

Relative Humidity: Below 75%
Temperature: 10°C-25°C
Wind Speed: 4.0 m/s

Baseline scored 96.30% accuracy. Model scored 96.30% on held-out test data.
Through investigation, the reason was due to the imbalance in the decision of 1s and 0s(out of 132 rows, only 129 were
considered as uncomfortable and 3 comfortable).
This concluded that both the baseline and model were guessing, hence why they have the same high
accuracy, which fits in the "To good to be true" result. 

Second test: Looking beyond raw accuracy into the classification report, the model scored 0.00 precision
and recall for class 1.
This proved that the AI was not successful at predicting travel-friendly months. It shows that to the AI,
they found the information completely blind due to strict monthly macro-averages smoothing out Tokyo's 
climate data.

Third test: 

Relative Humidity: Below 80%
Temperature: 20°C-25°C
Wind Speed: 5.5 m/s

I Decided to change boundary data and see difference in results. This time it got worse(out of 132 rows,
only 130 were considered uncomfortable and 2 comfortable). Baseline and Model scored the same 96.30% on accuracy.
This means that the AI model is generally blind to comfortable travel-months.

Fourth test: 

Relative Humidity: Below 85%
Temperature: 10°C-24°C
Wind Speed: 5.0 m/s

When changing the boundary to a more realistic, balanced range, the results improved drastically(out of 132 rows,
only 89 were considered uncomfortable and 43 comfortable). Baseline scored 70.37% on accuracy while the Model scored
100% on accuracy. This shows that the AI successfully learned the complex, non-linear relationships between Temperature,
Relative Humidity, and Wind Speed to separate good traveler windows from bad ones, showing that balance was the main 
issue. It's precision and recall scored 1.

## What we'd do next

1. Transition from monthly weather data to daily/weekly weather data to catch large weather spikes
which could help challenge my AI's accuracy..

2. Scale this pipeline from a single city (Tokyo) to a global dashboard that can allow uploads supporting many different
countries, (requires data from NASA POWER datasets) giving different test results and recommendations
to analyse.

## Sources

Datasets: NASA POWER Access Data Viewer (https://power.larc.nasa.gov/data-access-viewer/) - Sourced monthly historical observations (CSV file)
for Tokyo 2010-2020, Including Temperature, Relative Humidity and Wind Speed.

## Libraries/Modules

Pandas (https://pandas.pydata.org/): Used for data cleaning, metadata filtering, melting wide tables, and pivoting colums
for reconstruction.

Scikit-learn (https://scikit-learn.org/stable/): Used for machine learning implementation(RandomForestClassifier, DummyClassifier, train_test_split,
accuracy_score and classification_report).

AI Tools: Gemini AI used to debug, help with suggestions and recommend the type of model to use for the idea
writeup.

## Inspiration and Guides

Veritas AI hackathon for baseline comparison and iterative model evaluation.