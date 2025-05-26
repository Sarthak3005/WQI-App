# 💧 Water Quality Index (WQI) Prediction & Forecast App

## 📌 Project Overview

This project focuses on analyzing, predicting, and forecasting the **Water Quality Index (WQI)** using real-world environmental data. The app is designed to assess water quality over time using key parameters and predict future trends using machine learning and time-series forecasting.

The solution supports year-wise WQI analysis, category distribution, and future simulations. It includes multiple regression models and allows users to experiment interactively through a “What-If” simulator.

---

## 🧠 Methodology

* **Dataset:** Water quality dataset with annual records for:
  * pH
  * Dissolved Oxygen (DO)
  * Biological Oxygen Demand (BOD)
  * Temperature
  * Fecal Coliform
  * Total Coliform
  * Conductivity
  * Nitrate (NO3⁻)
  * Year

* **Feature Engineering:**
  * Converted `year_num` to datetime and extracted numeric year.
  * Applied ideal and standard values to compute WQI using weighted quality rating formula.

* **WQI Calculation:**
  * Used weighted aggregation based on Indian standards.
  * Qi = (Parameter value - Ideal) / (Standard - Ideal) × 100 (clipped between 0–100)
  * WQI = Σ(Qi × Wi) / Σ(Wi)

* **Models Used:**
  * Linear Regression
  * Random Forest Regressor
  * Support Vector Regressor (SVR)
  * Decision Tree Regressor

* **Forecasting Technique:**
  * Time-series forecasting with **Prophet** per parameter
  * Applied smoothing via **Savitzky-Golay filter**
  * Clipped predictions to historical bounds for realism

* **Evaluation Metrics:**
  * R² Score
  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)
  * Accuracy (% Based on MAE/WQI Mean)

---

## 📊 Results Summary

After model training and testing on uploaded data:

* **Random Forest** and **SVR** consistently showed strong performance across most metrics.
* WQI trend visualizations and error scatter plots confirmed that the models capture general trends well.
* Combined bar chart displayed **R², MAE, and RMSE** for all models.
* The “Recommended Model” section highlights the top-performing algorithm.

---

## ⚙️ Application Features

* **📁 CSV Upload:** Upload your water quality dataset for analysis.
* **📊 Model Comparison:** Evaluate performance using real-time plots and metrics.
* **📅 WQI Statistics:**
  * Per-year average WQI
  * Category-wise distribution: Excellent, Good, Medium, Poor, Very Poor
* **🔮 Future Forecast:**
  * Predicts parameter-wise trends using Prophet (2012–2027)
  * Computes future WQI and visualizes it
* **🧪 What-If Simulator:**
  * Adjust parameter sliders to simulate WQI
  * View simulated WQI on a real-time gauge
  * Get category badge (Excellent–Very Poor)
  * Year selector for future simulation

---

## ⚠️ Limitations

1. **Parameter Assumptions:**
   * Fixed ideal and standard values assumed per CPCB guidelines.
   * Deviations in regulatory standards across regions not accounted for.

2. **Data Dependency:**
   * Incomplete or sparse data can distort forecasting and model accuracy.

3. **Forecasting Scope:**
   * Prophet is used independently for each parameter (no inter-parameter dependency considered).
   * More advanced multivariate forecasting (e.g. VAR, LSTM) not used.

4. **No Geo-Aware Modeling:**
   * Currently works for single-region datasets only.
   * Geographic water dynamics are not modeled.

---

## ✅ Future Improvements

* Add **multi-location support** with map-based filtering.
* Include **geospatial forecasting** and **climate influence** integration.
* Enable **real-time monitoring** using IoT sensor integrations or APIs.
* Use **deep learning (LSTM, GRU)** for multi-variate sequence forecasting.
* Include **auto-anomaly detection** in water quality data.

---

## 📂 Example CSV Format

| year_num | pH  | DO  | BOD | Temp | FecalColiform | TotalColiform | Conductivity | NO3- |
|----------|-----|-----|-----|------|----------------|----------------|---------------|------|
| 2010     | 7.2 | 6.5 | 3.1 | 23.0 | 40             | 100            | 120           | 15.2 |

---

## 🧰 Tech Stack

* Streamlit  
* Prophet (Facebook)  
* scikit-learn  
* NumPy / Pandas  
* Matplotlib / Plotly  
* Savitzky-Golay Filter

---

## 📬 Contact

**Developer:** Sarthak  
📧 Email: sarthak.wqi@example.com *(replace with your email)*  
🔗 GitHub: [github.com/Sarthak3005](https://github.com/Sarthak3005)

---

## 📄 License

This project is open-sourced under the MIT License.  
See the [LICENSE](LICENSE) file for full details.

