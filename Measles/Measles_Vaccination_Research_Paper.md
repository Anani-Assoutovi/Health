# Trends and Determinants of Measles Vaccination Coverage in Texas (2019–2024)

**Author:** Anani A. Assoutovi  
**Affiliation:** Independent Researcher / Data Scientist  
**Email:** booking.helices4s@icloud.com  
**Kaggle:** https://www.kaggle.com/ananiassoutovi  
**LinkedIn:** www.linkedin.com/in/ananiassoutovi  
**Date:** May 17, 2025

**Abstract**  
This study analyzes statewide trends and county‐level determinants of measles vaccination coverage in Texas between 2019 and 2024. Using public health data from the Texas Department of State Health Services, we assess temporal patterns, identify regions with suboptimal uptake, examine associations with socioeconomic and exemption rates, and forecast future coverage levels. Our findings reveal persistent geographic disparities, a modest upward trend overall, and the critical impact of exemption policies on coverage.

[Data Source](https://www.dshs.texas.gov/immunizations/data/school/coverage)  
[tx_counties.geojson](https://github.com/Cincome/tx.geojson/blob/master/counties/tx_counties.geojson)  
---

## 1. Introduction  
Measles is a highly contagious viral disease preventable by vaccination. Despite a proven safe and effective vaccine, coverage gaps can lead to outbreaks. Texas, with its diverse demographics, presents unique challenges in achieving uniformly high uptake. This paper investigates trends, spatial patterns, and drivers of measles vaccination coverage to inform targeted interventions.

## 2. Data Sources  
- **Vaccination Coverage**: Texas Measles Vaccination dataset (2019–2024) from the Texas Department of State Health Services.  
- **Exemption Rates**: County‐level conscientious and medical exemption records.  
- **Demographics**: County-level socioeconomic indicators (median income, education levels) from the U.S. Census Bureau.

## 3. Methodology  
1. **Data Cleaning**: Standardized county names, handled missing values via interpolation.  
2. **Temporal Analysis**: Computed annual coverage means and year‐over‐year changes.  
3. **Spatial Mapping**: Created county‐level choropleths to highlight geographic disparities.  
4. **Correlation Analysis**: Calculated Pearson correlations between coverage, exemption rates, and demographic covariates.  
5. **Clustering**: Applied K-Means to segment counties by multi‐vaccine profiles.  
6. **Forecasting**: Fitted ARIMA(1,1,1) models to project coverage for 2025.

## 4. Results  
- **Temporal Trends**: Coverage rose from ~92% in 2019 to 96% in 2024, with occasional dips linked to policy changes.  
- **Spatial Disparities**: Rural counties in West Texas and the Panhandle show coverage below 90%, while urban centers exceed 98%.  
- **Exemption Impact**: Conscientious exemption rates negatively correlate with coverage (r = –0.45).  
- **Clustering**: Four clusters emerged, distinguishing high‐uptake urban counties from lower‐uptake rural/exemption‐prone counties.  
- **Forecast**: ARIMA projects statewide coverage reaching 96.5% (95% CI: 96.2–96.8%) in 2025.

## 5. Discussion  
The upward trend is encouraging but masks persistent pockets of vulnerability. Exemption policies and socioeconomic factors are key drivers. Clustering reveals target groups for tailored outreach. Forecast uncertainty underscores the need for continued monitoring.

## 6. Conclusion  
Achieving measles elimination in Texas requires focused strategies in underperforming counties, policy reinforcement to limit non‐medical exemptions, and community engagement. Our integrated analytical framework offers a template for ongoing surveillance and intervention planning.

## References  
1. Texas Department of State Health Services. Measles Vaccination Coverage Data.  
2. U.S. Census Bureau. American Community Survey 5‐Year Estimates.  
3. Hyndman, R.J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*.  
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.