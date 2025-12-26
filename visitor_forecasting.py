import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class VisitorForecastingPipeline:
    """
    This class manages the full  workflow:
    - We first load and validate data
    - Then, we clean and prepare features
    - And, we train a model
    - Finally, we valuate performance and visualize results
    """

    def __init__(self, csv_path: str):
        
        self.csv_path = csv_path

        self.original_dataset = None
        self.cleaned_dataset = None
        self.model = None

        self.cross_validation_mean = None
        self.cross_validation_std = None
        self.test_predictions = None
        self.test_targets = None

    def load_data(self):
        """We fist load the dataset and verify whether the columns were succesfully loaded"""
        try:
            self.original_dataset = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The file '{self.csv_path}' could not be found."
            )

        expected_columns = {
            "Date",
            "Prix_Moyen_Hotel",
            "Taux_Change",
            "Visiteurs",
            "City",
            "Indicateur_Evenement",
        }

        missing = [col for col in expected_columns if col not in self.original_dataset.columns]

        if missing:
            raise ValueError(
                f"The following columns are missing: {', '.join(missing)}"
            )

        # we fix basic data types and typos here so the EDA is accurate
        dataset = self.original_dataset.copy()

        # we convert the date column into a proper datetime format
        dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
        dataset = dataset.dropna(subset=["Date"])
        dataset = dataset.sort_values("Date").reset_index(drop=True)  # we sort the date values

        # Create temporal features for EDA visibility
        dataset["month"] = dataset["Date"].dt.month
        dataset["is_weekend"] = (dataset["Date"].dt.weekday.isin([5, 6]).astype(int))

        # we standardize city names in the case of errors:
        city_normalization = {
            "yotoko": "Tokyo", "tokyo": "Tokyo", "Kyotoo": "Tokyo",
            "Kyoto": "Tokyo", "Kioto": "Tokyo", "Pari": "Paris",
            "paris": "Paris", "Pariss": "Paris", "Parisss": "Paris",
            "Marakkech": "Marrakech", "Marakesh": "Marrakech",
            "Marakkesh": "Marrakech", "Marrakesh": "Marrakech",
            "marrakech": "Marrakech", "Marakech": "Marrakech",
        }
        dataset["City"] = dataset["City"].replace(city_normalization)

        # we save this version for the EDA to analyze
        self.original_dataset = dataset

    def preprocess_data(self):
        """ Here, we: engineer features, handle outliers, and encode categories"""
        dataset = self.original_dataset.copy()

        # we turn the date into year,month,day
        dataset["year"] = dataset["Date"].dt.year
        dataset["month"] = dataset["Date"].dt.month
        dataset["day"] = dataset["Date"].dt.day

        # we add a new column which determines if the day is a weekend or not
        dataset["is_weekend"] = (dataset["Date"].dt.weekday.isin([5, 6]).astype(int))

        # we add a 'lag' feature to give the model memory of the previous day
        dataset["visiteurs_lag_1"] = dataset.groupby('City')['Visiteurs'].shift(1)
        dataset = dataset.dropna(subset=["visiteurs_lag_1"])

        dataset = dataset.drop(columns=["Date"])
        # finally, we drop the date column after seperating the day,month and year

        # we use the IQR method to clip it (now justified by the EDA histograms)
        lower_quartile = dataset["Visiteurs"].quantile(0.25)
        upper_quartile = dataset["Visiteurs"].quantile(0.75)
        interquartile = upper_quartile - lower_quartile
        dataset["Visiteurs"] = dataset["Visiteurs"].clip(lower_quartile - 1.5 * interquartile,
                                                         upper_quartile + 1.5 * interquartile, )

        # we convert city names into binary columns
        dataset = pd.get_dummies(dataset, columns=["City"], prefix="City")

        # we finally remove duplicate rows and drop the NaN created by the lag
        dataset = dataset.drop_duplicates().dropna().reset_index(drop=True)

        self.cleaned_dataset = dataset

    def perform_eda(self):
        """
        Perform a natural, step-by-step EDA like a real data science project.
        We show: dataset head, info, missing values, descriptive stats,
        distributions, correlations, and simple group analyses.
        """

        dataset = self.original_dataset.copy()

        # we fill missing numeric values using column averages for EDA only
        numeric_columns = ["Prix_Moyen_Hotel", "Taux_Change", "Visiteurs"]
        dataset[numeric_columns] = dataset[numeric_columns].fillna(dataset[numeric_columns].mean())

        print("1) the first 5 rows of the dataset")
        print(dataset.head(), "\n")

        print("2) the dataset's info  ")
        print(dataset.info(), "\n")

        print("3) we check for missing values per column")
        print(dataset.isna().sum(), "\n")

        print("4) the descriptive statistics for numeric columns")
        print(dataset.describe(), "\n")

        print("5) the visitor counts distribution by city")
        print(dataset.groupby("City")["Visiteurs"].describe(), "\n")

        print("6) the visitor counts distribution by month")
        print(dataset.groupby("month")["Visiteurs"].describe(), "\n")

        print("7) the average visitors on weekends vs weekdays")
        print(
            dataset.groupby("is_weekend")["Visiteurs"].describe()
            .rename(index={0: "Weekday", 1: "Weekend"})
        )
        print("\n")

        print("8) the correlations between numeric features and visitors")
        print(dataset[["Visiteurs", "Prix_Moyen_Hotel", "Taux_Change"]].corr(), "\n")

        print("9) finally, we check for duplicates")
        print(f"Total rows after cleaning and deduplication: {len(dataset)}\n")


    def train_and_evaluate(self):
        """This is where we train the model"""
        features = self.cleaned_dataset.drop("Visiteurs", axis=1)
        target = self.cleaned_dataset["Visiteurs"]

        # we split it based off: training on 80% of the features and testing on the remaining 20%
        split_index = int(len(features) * 0.8)
        training_features = features.iloc[:split_index]
        testing_features = features.iloc[split_index:]

        # similarly, but for the target
        training_target = target.iloc[:split_index]
        testing_target = target.iloc[split_index:]

        # we fill missing numeric values using only training data averages
        train_means = training_features.mean()
        training_features = training_features.fillna(train_means)
        testing_features = testing_features.fillna(train_means)

        self.model = RandomForestRegressor(
            n_estimators=300,          # a large forest for stability
            max_depth=15,              #we choose a maximum for the trees' depths to prevent overfitting
            random_state=15,           #we fix the random seed to be 15
            n_jobs=-1                  #we use all cpus to run our model
        )

        #we split the time series into 5 windows to train itself on 5 different segments
        time_splitter = TimeSeriesSplit(n_splits=5)

        cross_validation_scores = cross_val_score(
            self.model,
            training_features,
            training_target,
            cv=time_splitter,
            scoring="neg_mean_absolute_error", # we make the error negative so that 'higher' ( i.e closer to zero) is better.
        )

        self.cross_validation_mean = np.mean(-cross_validation_scores)
        self.cross_validation_std = np.std(-cross_validation_scores)

        # we fit  the model on full training data
        self.model.fit(training_features, training_target)

        # we predict on the unseen test data
        self.test_predictions = self.model.predict(testing_features)
        self.test_targets = testing_target.values

        # for the final evaluation metrics:
        mae = mean_absolute_error(self.test_targets, self.test_predictions)
        mse = mean_squared_error(self.test_targets, self.test_predictions)
        r2 = r2_score(self.test_targets, self.test_predictions)

        print("Cross-validation MAE mean:", round(self.cross_validation_mean, 2))
        print("Cross-validation MAE std :", round(self.cross_validation_std, 2))
        print("Test MAE:", round(mae, 2))
        print("Test MSE:", round(mse, 2))
        print("Test R2 :", round(r2, 3))

    def visualize_predictions(self):
        """
        We show the visuals.
        """

        dataset = self.cleaned_dataset.copy()

        plt.figure(figsize=(16, 18))

        # 1) Visitor distribution
        plt.subplot(3, 2, 1)
        plt.hist(dataset["Visiteurs"], bins=30)
        plt.title("How visitor counts are distributed")
        plt.xlabel("Number of visitors")
        plt.ylabel("Frequency")

        # 2) Visitors by month
        plt.subplot(3, 2, 2)
        monthly_average = dataset.groupby("month")["Visiteurs"].mean()
        plt.plot(monthly_average.index, monthly_average.values, marker="o")
        plt.title("Average visitors per month")
        plt.xlabel("Month")
        plt.ylabel("Average visitors")

        # 3) Visitors by city
        plt.subplot(3, 2, 3)
        # We reconstruct city averages from the One-Hot encoded columns
        city_cols = [col for col in dataset.columns if col.startswith("City_")]
        city_names = [col.replace("City_", "") for col in city_cols]
        city_vals = [dataset[dataset[col] == 1]["Visiteurs"].mean() for col in city_cols]

        plt.bar(city_names, city_vals)
        plt.title("Average visitors per city")
        plt.xlabel("City")
        plt.ylabel("Average visitors")

        # 4) Hotel price vs visitors
        plt.subplot(3, 2, 4)
        plt.scatter(
            dataset["Prix_Moyen_Hotel"],
            dataset["Visiteurs"],
            alpha=0.6
        )
        plt.title("Relationship between hotel price and visitors")
        plt.xlabel("Average hotel price")
        plt.ylabel("Visitors")

        # 5) Actual vs predicted visitors over time
        plt.subplot(3, 2, 5)
        plt.plot(self.test_targets, label="Actual visitors", linewidth=2)
        plt.plot(
            self.test_predictions,
            label="Predicted visitors",
            linestyle="--"
        )
        plt.title("Actual vs predicted visitors (test period)")
        plt.xlabel("Time index")
        plt.ylabel("Visitors")
        plt.legend()

        # 6) Prediction errors
        plt.subplot(3, 2, 6)
        prediction_errors = self.test_targets - self.test_predictions
        plt.hist(prediction_errors, bins=30)
        plt.title("Distribution of prediction errors")
        plt.xlabel("Actual - Predicted")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()
    def run(self):
        """finally, we execute the full pipeline:"""
        self.load_data()
        self.perform_eda()
        self.preprocess_data()
        self.train_and_evaluate()
        self.visualize_predictions()


if __name__ == "__main__":
    pipeline = VisitorForecastingPipeline("prevision_visiteurs.csv")
    pipeline.run()
