
Dataset Link : https://www.kaggle.com/datasets/gregorut/videogamesales
### Video Game Sales Prediction Using Linear Regression & Decision Tree

This project involves predicting video game sales using Linear Regression and Decision Tree models. The dataset contains information about video game sales across different platforms, regions, and genres. 

#### Project Structure
- **Data**: `vgsales.csv` - Contains sales data for various video games.
- **Notebook**: The main code is provided in a Jupyter notebook or script format.

#### Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

#### Steps
1. **Data Loading**: Import data from `vgsales.csv` and inspect it.
2. **Data Cleaning**: Drop rows with missing values.
3. **Exploratory Data Analysis (EDA)**:
   - Visualize the top game genres using a pie chart.
   - Display correlation heatmap for numerical features.
4. **Feature Selection**:
   - Select features for prediction (`Rank`, `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`).
   - Set the target variable (`Global_Sales`).
5. **Model Training**:
   - Split data into training and test sets.
   - Train a Decision Tree Regressor and make predictions.
6. **Evaluation**:
   - Compare the predictions against actual values.

#### Example Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load data
data = pd.read_csv('vgsales.csv')

# Drop rows with missing values
data = data.dropna()

# Exploratory Data Analysis
game = data.groupby("Genre")["Global_Sales"].count().head(10)
plt.figure(figsize=(7,7))
plt.pie(game, labels=game.index, autopct='%1.1f%%')
plt.title("Top 10 Categories of Games Sold")
plt.show()

# Correlation Heatmap
numerical_data = data.select_dtypes(include=['number'])
sns.heatmap(numerical_data.corr(), cmap="YlOrBr", annot=True)
plt.show()

# Prepare data for training
x = data[["Rank", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
y = data["Global_Sales"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)
print(predictions)
```

#### Results
- The Decision Tree Regressor provides sales predictions for video games, which can be compared to actual sales figures.

#### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to modify and adapt this project as needed.
