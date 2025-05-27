
# CPI Anomaly Detection Across Countries

## Goal
Detect anomalous consumer price index (CPI) behavior across multiple countries using deep learning models (LSTM, Transformer, etc.) on time series data — and validate anomalies via real-world news.

---

## Methodology Overview

### 1. Data Acquisition

#### A. Primary Source (CPI Data)
- [World Bank](https://www.worldbank.org/en/research/brief/inflation-database)
  - Monthly CPI data for 200+ countries.
  - https://doi.org/10.1016/j.jimonfin.2023.102896

#### B. Data Preprocessing

- Load the CPI data from the World Bank.

- Convert the data into a wide format matrix with countries as columns and dates as rows.

- Handle missing values appropriately (e.g., forward fill, interpolation).

- Ensure the date index is in datetime format.

- Normalize the CPI values.

#### C. Output Format
Target format (wide format matrix):
```
Date       | ABW | AFG | AUT      | ...
-----------|-----|-----|----------|-----
1970-01-01 | NaN | NaN | 22.48557 |
1970-02-01 | NaN | NaN | 22.46568 |
...
```

If in some case you want to avoid NaN values, you can use the following:

```python
hcpi_m.loc["2010-01":"2023-12"].dropna(axis=1)
```
 
### 2. Models

#### A. LSTM Model per country
  - Use LSTM for sequential data modeling.
  - Train on the wide format matrix of CPI data.
  - Detect anomalies based on reconstruction error or prediction error.

#### B. Model per cluster
  - Cluster countries based on CPI trends.
  - Train a single LSTM model per cluster.
  - Detect anomalies within each cluster.

#### C. ...