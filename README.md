# E-Commerce Sales Forecasting Using Machine Learning

Predicting monthly e-commerce revenue using time-series regression, built on the Superstore transaction dataset (2014–2017).

## Overview

This project forecasts next-month sales revenue for an e-commerce business using historical transaction data. Rather than treating each sale as an independent data point, the data was aggregated into monthly totals and enriched with lag and rolling-average features, allowing the model to learn from recent sales momentum — similar to how real forecasting systems work in practice.

## Problem Statement

Retail businesses need a reliable read on next month's revenue to plan inventory, staffing, and budgets. Without forecasting, decisions become reactive instead of proactive. This project builds a regression model that estimates monthly revenue using patterns learned from four years of transaction history.

## Dataset

- **Source:** Superstore e-commerce transaction dataset
- **Size:** 9,994 transactions, cleaned from 21 raw columns down to 11 relevant ones
- **Span:** 2014–2017 (4 years)
- **Duplicates:** 0
- **Customer segments:** Consumer (52%), Corporate (30%), Home Office (18%)

## Approach

### 1. Exploratory Data Analysis
- Identified a clear year-over-year revenue growth trend (2014 → 2017)
- Found strong monthly seasonality — November, December, and September consistently outperform other months
- Sales distribution is right-skewed, with a long tail of high-value orders
- Discounting beyond ~30% is associated with negative average profit

### 2. Feature Engineering
- Aggregated daily transactions into 48 monthly sales totals
- Built lag features: sales from 1, 2, and 3 months prior
- Built rolling averages: 3-month and 6-month trailing windows
- Dropped early rows with NaN values from the rolling window, leaving 42 usable months

### 3. Preprocessing
- Chronological train/test split (not random) — order matters in time series
- ~33 months train / ~9 months test
- StandardScaler applied to lag and rolling features
- Month one-hot encoded to let the model learn seasonality directly

### 4. Modeling
Three regression models were trained and compared:

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| **Linear Regression** | 12,054.68 | 14,026.52 | 18.62% | **0.684** |
| Ridge | 15,820.27 | 20,300.11 | 21.28% | 0.338 |
| **Lasso** | 12,054.03 | 14,029.06 | 18.62% | **0.684** |

Linear Regression and Lasso performed best and nearly identically. Ridge underperformed, likely due to inconsistent feature scaling — the `years` feature wasn't scaled while lag/rolling features were, which distorts Ridge's coefficient penalty more than it affects Linear or Lasso.

## Key Insights

- Revenue and profit follow a clear, repeatable seasonal pattern, peaking in Q4 (especially Nov–Dec)
- Technology and high-ticket sub-categories (Phones, Chairs) drive a disproportionate share of revenue
- Central region underperforms on profit relative to its revenue rank — a possible discount or cost-efficiency gap
- Seasonality (month/quarter) is the strongest predictive signal available in this dataset

## Limitations

- Test set covers only 9 months, so results carry meaningful variance
- Model underestimates extreme monthly spikes
- No external variables (promotions, holidays, marketing spend) are included

## Next Steps

- Incorporate promotions, holidays, and marketing spend as additional features
- Compare against models built for seasonality directly (SARIMA, Prophet)
- Expand the test window as more data becomes available

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`

## Project Structure

├── Dataset/
        └── E-commerce_Sales_Data(2014-17).csv
├── Notebook/ 
        └── 01_EDA.ipynb            
                 └── Exploratory data analysis
        └── 02_Model_Building.ipynb 
                 └── Feature engineering, modeling, evaluation
├── Presentation/             
          └── Project presentation.pdf   
├── License 
├── README.md

## Author

**Samarth Singh**
[Portfolio](https://samarthsinghportfolio.netlify.app/) · [GitHub](https://github.com/OfficialSamarthsingh) · [LinkedIn](https://www.linkedin.com/in/samarth-singh-062371215)
