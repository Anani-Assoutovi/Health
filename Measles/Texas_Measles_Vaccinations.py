#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 15:40:14 2025

@author: Anani A. Assoutovi
"""
import pandas as pd
import numpy as np
import json
import folium
from folium.features import GeoJsonTooltip
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px

class DataLoader:
    """
    Load vaccination and exemption data from files.
    """
    def __init__(self, coverage_paths: dict, exemptions_path: str, geojson_path: str):
        self.coverage_paths = coverage_paths
        self.exemptions_path = exemptions_path
        self.geojson_path = geojson_path
        self.coverage_df = None
        self.exemptions_df = None
        self.geojson = None

    def load_coverage(self):
        dfs = []
        for year, path in self.coverage_paths.items():
            df = pd.read_excel(path, header=1)
            df['school_year'] = year
            # identify county column
            county_col = [c for c in df.columns if 'county' in c.lower()][0]
            df = df.rename(columns={county_col: 'county'})
            dfs.append(df)
        self.coverage_df = pd.concat(dfs, ignore_index=True)
        return self.coverage_df

    def load_exemptions(self):
        ex = pd.read_excel(self.exemptions_path, header=2)
        county_col = [c for c in ex.columns if 'county' in c.lower()][0]
        year_cols = [c for c in ex.columns if isinstance(c, str) and '-' in c]
        ex_long = ex.melt(id_vars=[county_col], value_vars=year_cols,
                          var_name='school_year', value_name='exemption_rate')
        ex_long = ex_long.rename(columns={county_col: 'county'})
        ex_long['school_year'] = ex_long['school_year'].str.strip()
        self.exemptions_df = ex_long
        return self.exemptions_df

    def load_geojson(self):
        with open(self.geojson_path) as f:
            self.geojson = json.load(f)
        return self.geojson

class Preprocessor:
    """
    Clean and merge coverage and exemption data.
    """
    def __init__(self, coverage_df: pd.DataFrame, exemptions_df: pd.DataFrame):
        self.coverage_df = coverage_df
        self.exemptions_df = exemptions_df
        self.df = None

    def normalize_columns(self):
        self.coverage_df.columns = (
            self.coverage_df.columns.str.strip()
                                       .str.lower()
                                       .str.replace(' ', '_')
                                       .str.replace('/', '_')
        )
    def merge(self):
        cov = self.coverage_df.copy()
        cov['county'] = cov['county'].str.strip().str.title()
        ex = self.exemptions_df.copy()
        ex['county'] = ex['county'].str.strip().str.title()
        self.df = pd.merge(cov, ex, on=['county','school_year'], how='inner')
        return self.df

class Analyzer:
    """
    Perform various analyses: temporal trends, correlation, clustering, anomalies, thresholds.
    """
    def __init__(self, df: pd.DataFrame, geojson: dict=None):
        self.df = df
        self.geojson = geojson

    def temporal_trends(self):
        cols = [c for c in self.df.select_dtypes(include=[np.number])
                if self.df[c].between(0,1).all()]
        yearly = self.df.groupby('school_year')[cols].mean().reset_index()
        long = yearly.melt(id_vars='school_year', var_name='vaccine', value_name='avg')
        fig = px.line(long, x='school_year', y='avg', color='vaccine', markers=True,
                      title='Temporal Trends')
        fig.show()

    def correlation_analysis(self):
        cols = [c for c in self.df.select_dtypes(include=[np.number])
                if self.df[c].between(0,1).all()]
        corr = self.df[cols].corr()
        fig = px.imshow(corr, x=cols, y=cols, color_continuous_scale='RdBu', zmin=-1, zmax=1,
                        title='Coverage Correlation')
        fig.show()

    def clustering(self, k=4):
        cols = [c for c in self.df.select_dtypes(include=[np.number])
                if self.df[c].between(0,1).all()]
        grp = self.df.groupby('county')[cols].mean().dropna()
        X = StandardScaler().fit_transform(grp)
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        grp['cluster'] = km.labels_
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X)
        coord = pd.DataFrame(pcs, columns=['PC1','PC2'], index=grp.index)
        coord['cluster'] = grp['cluster']
        coord['county'] = coord.index
        fig = px.scatter(coord, x='PC1', y='PC2', color='cluster', hover_data=['county'],
                         title='PCA Clustering')
        fig.show()
        return grp.reset_index()

    def anomaly_detection(self):
        cols = [c for c in self.df.select_dtypes(include=[np.number])
                if self.df[c].between(0,1).all()]
        grp = self.df.groupby(['county','school_year'])[cols].mean().reset_index()
        anomalies = []
        for vac in cols:
            pivot = grp.pivot(index='county', columns='school_year', values=vac)
            diffs = pivot.diff(axis=1).stack(dropna=True)
            Q1,Q3 = diffs.quantile([.25,.75])
            IQR = Q3-Q1
            out = diffs[(diffs< Q1-1.5*IQR)|(diffs>Q3+1.5*IQR)]
            for (county,year),change in out.items():
                anomalies.append(dict(county=county,year=year,vaccine=vac,change=change))
        return pd.DataFrame(anomalies)

    def threshold_monitoring(self, thresholds: dict):
        flags=[]
        for vac,th in thresholds.items():
            if vac in self.df:
                low=self.df[self.df[vac]<th]
                for _,r in low.iterrows():
                    flags.append(dict(
                        county=r['county'], school_year=r['school_year'],
                        vaccine=vac, coverage=r[vac], threshold=th
                    ))
        return pd.DataFrame(flags)

    def spatial_clusters(self, county_stats: pd.DataFrame):
        m = folium.Map(location=[31,-100], zoom_start=6)
        folium.Choropleth(
            geo_data=self.geojson, data=county_stats,
            columns=['geo_county','cluster'], key_on='feature.properties.COUNTY',
            fill_color='Set3', fill_opacity=0.7, line_opacity=0.5,
            legend_name='Cluster'
        ).add_to(m)
        folium.GeoJson(self.geojson,
            style_function=lambda f: {'fillOpacity':0,'weight':0.5,'color':'gray'},
            tooltip=GeoJsonTooltip(fields=['COUNTY'],aliases=['County:'])
        ).add_to(m)
        folium.LayerControl().add_to(m)
        return m

class Forecaster:
    """
    Forecast future coverage using ARIMA.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def arima_forecast(self):
        cols=[c for c in self.df.select_dtypes(include=[np.number])
              if self.df[c].between(0,1).all()]
        yearly=self.df.groupby('school_year')[cols].mean().reset_index()
        last=yearly['school_year'].iloc[-1]
        next_label=f"{last.split('-')[1]}-{int(last.split('-')[1])+1}"
        for vac in cols:
            series=yearly[vac].astype(float)
            m=ARIMA(series,order=(1,1,1)).fit()
            fc=m.get_forecast(1).summary_frame()
            y,low,up=fc['mean'].iloc[0],fc['mean_ci_lower'].iloc[0],fc['mean_ci_upper'].iloc[0]
            print(f"{vac}: {next_label} -> {y:.3f} ({low:.3f}-{up:.3f})")

# Example main workflow
if __name__ == '__main__':
    coverage_paths = {
    "2019-2020": "2019-2020-School-Vaccination-Coverage-Levels-by-District-Private-School-and-County---Seventh-Grade.xls",
    "2020-2021": "2020-2021-School-Vaccination-Coverage-Levels-by-District-Private-School-and-County---Seventh-Grade.xlsx",
    "2021-2022": "2021-2022-School-Vaccination-Coverage-by-District-and-County-Seventh-Grade.xls",
    "2022-2023": "2022-2023-School-Vaccination-Coverage-by-District-and-County-Seventh-Grade.xlsx",
    "2023-2024": "2023-2024_School_Vaccination_Coverage_Levels_Seventh_Grade.xlsx",
}
    loader = DataLoader(coverage_paths, 'tx_counties.geojson')
    cov_df = loader.load_coverage()
    ex_df = loader.load_exemptions()
    geo = loader.load_geojson()

    prep = Preprocessor(cov_df, ex_df)
    prep.normalize_columns()
    df = prep.merge()

    analyzer = Analyzer(df, geojson=geo)
    analyzer.temporal_trends()
    analyzer.correlation_analysis()
    clusters = analyzer.clustering(k=4)
    anomalies = analyzer.anomaly_detection()
    flags = analyzer.threshold_monitoring({
        "mmr": 0.95,
        "tdap/td": 0.90,
        "meningococcal": 0.90,
        "hepatitis_a": 0.85,
        "hepatitis_b": 0.85,
        "polio": 0.95,
        "varicella": 0.95,
    })
    map_obj = analyzer.spatial_clusters(clusters)
    map_obj.save('clusters_map.html')

    fc = Forecaster(df)
    fc.arima_forecast()

