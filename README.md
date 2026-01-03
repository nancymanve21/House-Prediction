# House-Prediction
Housing Price Prediction using supervised machine learning. Implemented Simple and Multiple Linear Regression with EDA, data visualization, model evaluation using MSE and R², and price prediction based on housing features using Python and scikit-learn.
🏠 Project Overview
This project focuses on predicting housing prices using Supervised Machine Learning (Linear Regression). Both Simple Linear Regression and Multiple Linear Regression models are implemented to analyze how different factors influence house prices and to compare their performance.
The project includes data exploration, visualization, model training, evaluation, and prediction.

📂 Dataset
The dataset contains housing-related features such as:

Median Income (MedInc)
House Age (HouseAge)
Average Rooms (AveRooms)
Latitude
Longitude
Target variable: Price

🔍 Exploratory Data Analysis (EDA)

To understand the data and relationships between variables, the following steps were performed:
Dataset structure and summary statistics (info(), describe())
Pair plots to visualize feature relationships
Correlation heatmap to identify strongly related variables
Histograms for feature distributions
Box plots to detect outliers

📈 Simple Linear Regression
Predictor: Median Income (MedInc)
Target: House Price (Price)

Data split into training and testing sets (80/20)
Model trained using LinearRegression

Evaluation metrics:
Mean Squared Error (MSE)
R² Score

Visualization:
Regression line plotted against actual data
User input feature added to predict house price for a given MedInc value

📊 Multiple Linear Regression

Predictors used:
MedInc
HouseAge
AveRooms
Latitude
Longitude
Model trained on multiple features to improve prediction accuracy

Evaluation metrics:
Mean Squared Error (MSE)
R² Score
Model coefficients displayed to understand feature impact on house prices

🛠️ Technologies Used

Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

📌 Key Learnings

Understanding relationships between housing features and prices
Implementing and comparing Simple vs Multiple Linear Regression
Evaluating regression models using MSE and R²
Visualizing data and model results effectively
Interpreting regression coefficients

🚀 Conclusion

Multiple Linear Regression performed better than Simple Linear Regression as it considers multiple influencing factors, leading to improved prediction accuracy. This project demonstrates a complete end-to-end regression workflow, from data analysis to prediction.
