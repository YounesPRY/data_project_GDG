# Visitor Forecasting Pipeline

A machine learning project that predicts tourist visitor counts across different cities using historical data, hotel prices, exchange rates, and event indicators.

## What does this do?

This pipeline takes tourism data (dates, cities, hotel prices, visitor counts, etc.) and builds a Random Forest model to forecast future visitor numbers. It's designed to help understand patterns in tourism and make data-driven predictions.

## The Dataset

We're working with `prevision_visiteurs.csv` which includes:
- **Date**: When the data was recorded
- **City**: Which city (Tokyo, Paris, Marrakech)
- **Prix_Moyen_Hotel**: Average hotel price
- **Taux_Change**: Exchange rate
- **Visiteurs**: Number of visitors (this is what we're trying to predict)
- **Indicateur_Evenement**: Whether there was a special event

## How to run it

1. Make sure you have Python 3.x installed
2. Install the required packages:
```bash
pip install pandas numpy matplotlib scikit-learn
```
3. Put your CSV file in the same folder as the script
4. Run it:
```bash
python visitor_forecasting.py
```

## What happens when you run it?

The pipeline goes through 5 main steps:

1. **Load & Validate** - Reads the CSV and checks if all required columns are there
2. **Exploratory Data Analysis** - Prints statistics about the data (distributions, correlations, missing values)
3. **Preprocessing** - Cleans the data, creates new features like "is it a weekend?", handles missing values properly
4. **Train & Evaluate** - Builds a Random Forest model and tests how well it performs
5. **Visualize** - Shows 6 plots including actual vs predicted visitors, error distributions, and trends

## The Model

We use a **Random Forest Regressor** with:
- 300 trees for stability
- Max depth of 15 to prevent overfitting
- Time series cross-validation (5 splits) to properly evaluate performance

The model uses features like:
- Year, month, day
- Whether it's a weekend
- Previous day's visitor count (lag feature)
- Hotel prices and exchange rates
- Which city it is

## Performance Metrics

After running, you'll see:
- **MAE (Mean Absolute Error)**: Average prediction error in visitor count
- **MSE (Mean Squared Error)**: Penalizes larger errors more heavily
- **R² Score**: How much variance the model explains (closer to 1 is better)
- **Cross-validation scores**: How the model performs across different time periods

## Project Structure
```
VisitorForecastingPipeline (class)
├── load_data()           # Reads CSV and validates columns
├── perform_eda()         # Exploratory analysis and statistics
├── preprocess_data()     # Feature engineering and cleaning
├── train_and_evaluate()  # Model training and metrics
├── visualize_predictions() # Creates 6 plots
└── run()                 # Executes the full pipeline
```

## Notes

- The code handles common data issues like typos in city names (e.g., "Pari" → "Paris")
- Missing values are filled using training data averages only (no data leakage)
- Outliers are clipped using the IQR method
- The 80/20 train-test split respects time ordering (older data for training, newer for testing)

## What we learned

This project taught us about:
- Proper time series forecasting techniques
- Avoiding data leakage in train/test splits
- Feature engineering for temporal data
- Model evaluation and cross-validation
- Clean code structure for ML pipelines

---

Made by Yassine Mouadi & Younes Bouali 
