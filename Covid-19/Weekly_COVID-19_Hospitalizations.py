#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 07:55:38 2025

@author: Anani Assoutovi
"""

import warnings
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')

class DataDownloader:
    """
    Downloads CSV data from provided URLs.
    """
    @staticmethod
    def download_csv(url: str, output_file: str):
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            with open(output_file, 'wb') as f:
                f.write(resp.content)
            print(f"Downloaded {output_file}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

class COVIDData:
    """
    Loads and preprocesses COVID-19 hospitalization and ICU data.
    """
    def __init__(self, hosp_file: str, vacc_file: str=None):
        self.hosp_file = hosp_file
        self.vacc_file = vacc_file
        self.hosp_df = None
        self.vacc_df = None

    def load_data(self):
        # Load hospitalization/ICU data
        self.hosp_df = pd.read_csv(self.hosp_file)
        self.hosp_df['Date_reported'] = pd.to_datetime(self.hosp_df['Date_reported'])
        
        # Optionally load vaccination data
        if self.vacc_file:
            self.vacc_df = pd.read_csv(self.vacc_file)
            self.vacc_df['Date_reported'] = pd.to_datetime(self.vacc_df['Date_reported'])

    def preprocess(self):
        # Forward/backward fill and smoothing
        self.hosp_df = self.hosp_df.sort_values('Date_reported')
        self.hosp_df.fillna(method='ffill', inplace=True)
        self.hosp_df.fillna(method='bfill', inplace=True)
        self.hosp_df['hospitalisations_7d_avg'] = (
            self.hosp_df['Covid_new_hospitalizations_last_7days']
            .rolling(window=3, min_periods=1)
            .mean()
        )
        self.hosp_df['icu_7d_avg'] = (
            self.hosp_df['Covid_new_ICU_admissions_last_7days']
            .rolling(window=3, min_periods=1)
            .mean()
        )

class Analyzer:
    """
    Performs various analyses on the COVID-19 data.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def trend_analysis(self):
        # Global trend of hospitalizations
        trend = (
            self.df
            .groupby('Date_reported')['hospitalisations_7d_avg']
            .sum()
        )
        plt.figure(figsize=(10, 4))
        trend.plot(title='Weekly Hospitalizations Trend')
        plt.xlabel('Date')
        plt.ylabel('7-day Avg Hospitalizations')
        plt.tight_layout()
        plt.show()

    def segmented_analysis(self, group_col: str):
        # Country/region segmented trends
        seg = (
            self.df
            .groupby(['Date_reported', group_col])['hospitalisations_7d_avg']
            .sum()
            .unstack(group_col)
        )
        seg.plot(figsize=(12, 6), title=f'Hospitalizations by {group_col}')
        plt.xlabel('Date')
        plt.ylabel('7-day Avg Hospitalizations')
        plt.tight_layout()
        plt.show()

    def detect_spikes(self, column: str, threshold: float=2.0):
        # Z-score spike detection
        data = self.df.set_index('Date_reported')[column]
        mean = data.mean()
        std = data.std()
        spikes = data[(data - mean).abs() > threshold * std]
        print(f"Detected {len(spikes)} spikes in {column}.")
        return spikes

    def ratio_metrics(self):
        # ICU-to-hospitalization ratio
        ratio = (
            self.df['icu_7d_avg'] / self.df['hospitalisations_7d_avg']
        ).fillna(0)
        plt.figure(figsize=(10, 4))
        ratio.plot(title='ICU-to-Hospitalization Ratio')
        plt.xlabel('Date')
        plt.ylabel('Ratio')
        plt.tight_layout()
        plt.show()
        return ratio

class Forecaster:
    """
    Forecasts future ICU admissions using ML models.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model_rf = None
        self.model_xgb = None

    def prepare_features(self):
        df = self.df.copy()
        df['weekofyear'] = df['Date_reported'].dt.isocalendar().week
        df['month'] = df['Date_reported'].dt.month
        df['year'] = df['Date_reported'].dt.year
        df = df.fillna(0)
        X = df[['weekofyear', 'month', 'year']]
        y = df['icu_7d_avg']
        return X, y

    def train_random_forest(self):
        X, y = self.prepare_features()
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        self.model_rf = rf
        print('Random Forest trained.')

    def train_xgboost(self):
        X, y = self.prepare_features()
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X, y)
        self.model_xgb = model
        print('XGBoost trained.')

    def forecast(self, future_X: pd.DataFrame):
        if self.model_rf:
            pred_rf = self.model_rf.predict(future_X)
            print('RF forecast available.')
        if self.model_xgb:
            pred_xgb = self.model_xgb.predict(future_X)
            print('XGB forecast available.')
        return pred_rf if self.model_rf else None, pred_xgb if self.model_xgb else None


def main():
    # URLs and file names
    hosp_url = (
        "https://srhdpeuwpubsa.blob.core.windows.net/"
        "whdh/COVID/WHO-COVID-19-global-hosp-icu-data.csv"
    )
    hosp_file = "WHO_COVID19_Hospital_ICU_Data.csv"
    vacc_url = (
        "https://srhdpeuwpubsa.blob.core.windows.net/"
        "whdh/COVID/vaccination-data.csv"
    )
    vacc_file = "vaccination-data.csv"

    # Download data
    downloader = DataDownloader()
    downloader.download_csv(hosp_url, hosp_file)
    downloader.download_csv(vacc_url, vacc_file)

    # Load and preprocess
    data = COVIDData(hosp_file, vacc_file)
    data.load_data()
    data.preprocess()

    # Analysis
    analyzer = Analyzer(data.hosp_df)
    analyzer.trend_analysis()
    analyzer.segmented_analysis(group_col='WHO_region')
    spikes = analyzer.detect_spikes(column='hospitalisations_7d_avg')
    analyzer.ratio_metrics()

    # Forecasting
    forecaster = Forecaster(data.hosp_df)
    forecaster.train_random_forest()
    forecaster.train_xgboost()
    # Prepare future periods
    future = pd.DataFrame({
        'weekofyear': [1,2,3], 'month': [1,1,1], 'year': [2025,2025,2025]
    })
    forecasts = forecaster.forecast(future)
    print('Forecasts:', forecasts)

if __name__ == "__main__":
    main()
