import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.signal import savgol_filter

st.set_page_config(layout="wide")
st.title("💧 Water Quality Index (WQI) Model Comparison")

# WQI formula
def calculate_wqi(df):
    weighted_q = []
    for param in parameters:
        df[param] = pd.to_numeric(df[param], errors='coerce')
        Qi = ((df[param] - ideals[param]) / (standards[param] - ideals[param])) * 100
        Qi = Qi.clip(lower=0, upper=100)
        Wi = weights[param]
        weighted_q.append(Qi * Wi)
    df['WQI'] = sum(weighted_q) / sum(weights.values())
    return df

# Prophet forecasting
def prophet_forecast_parameters(df, parameters, future_years):
    future_df = pd.DataFrame({'year_num': future_years})
    for param in parameters:
        sub = df[['year_num', param]].dropna()
        if len(sub) < 5:
            continue
        yearly_avg = sub.groupby('year_num')[param].mean().reset_index()
        yearly_avg.columns = ['ds', 'y']
        yearly_avg['ds'] = pd.to_datetime(yearly_avg['ds'], format='%Y')

        try:
            m = Prophet(yearly_seasonality=True, daily_seasonality=False)
            m.fit(yearly_avg)
            future = pd.DataFrame({'ds': pd.date_range(start=f"{future_years[0]}", periods=len(future_years), freq='Y')})
            forecast = m.predict(future)
            predicted = forecast['yhat'].values
            min_val, max_val = sub[param].min(), sub[param].max()
            predicted = np.clip(predicted, min_val, max_val)
            future_df[param] = predicted

            fig, ax = plt.subplots()
            ax.plot(yearly_avg['ds'].dt.year, yearly_avg['y'], label='Actual', color='blue')
            ax.plot(future['ds'].dt.year, predicted, label='Forecast', color='red')
            ax.set_title(f"Prophet Forecast for {param}")
            ax.set_xlabel("Year")
            ax.set_ylabel(param)
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"⚠️ Prophet failed for {param}. Reason: {e}")
            continue
    return future_df

# File upload
uploaded_file = st.file_uploader("📁 Upload your Water Quality CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
    df['year'] = pd.to_datetime(df['year_num'], errors='coerce', format='%Y')
    df['year_num'] = df['year'].dt.year

    parameters = ['pH', 'DO', 'BOD', 'Temp', 'FecalColiform', 'TotalColiform', 'Conductivity', 'NO3-']
    weights = {'pH': 0.11, 'DO': 0.17, 'BOD': 0.11, 'Temp': 0.05,
               'FecalColiform': 0.10, 'TotalColiform': 0.10,
               'Conductivity': 0.10, 'NO3-': 0.03}
    standards = {'pH': 8.5, 'DO': 14, 'BOD': 6, 'Temp': 25,
                 'FecalColiform': 100, 'TotalColiform': 100,
                 'Conductivity': 300, 'NO3-': 45}
    ideals = {'pH': 7.0, 'DO': 7.0, 'BOD': 0, 'Temp': 4,
              'FecalColiform': 0, 'TotalColiform': 0,
              'Conductivity': 0, 'NO3-': 0}

    df = calculate_wqi(df).dropna(subset=parameters + ['WQI', 'year_num'])

    X = df[parameters]
    y = df['WQI']
    years = df['year_num']
    X_train, X_test, y_train, y_test, years_train, years_test = train_test_split(
        X, y, years, test_size=0.2, random_state=42
    )

    def plot_wqi_trend(model_name, wqi_values, years_array):
        fig, ax = plt.subplots()
        ax.scatter(years_array, wqi_values, color='blue', label="Actual WQI")
        ax.set_title(f'WQI vs Year - {model_name}')
        ax.set_xlabel("Year")
        ax.set_ylabel("WQI")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    def plot_error_analysis(y_true, y_pred, model_name):
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.6)
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
        ax.set_title(f"Error Analysis - {model_name}")
        ax.set_xlabel("Actual WQI")
        ax.set_ylabel("Predicted WQI")
        ax.grid(True)
        st.pyplot(fig)

    # Model training
    model_results = []
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'SVR': SVR(C=100, epsilon=0.1),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }

    predictions = {}

    for name, model in models.items():
        if name == 'SVR':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        predictions[name] = preds
        plot_wqi_trend(name, preds, years_test)
        plot_error_analysis(y_test, preds, name)
        model_results.append({
            'Model': name,
            'R2': r2_score(y_test, preds),
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'WQI Mean': preds.mean()
        })

    df_results = pd.DataFrame(model_results)
    df_results['Accuracy (%)'] = (1 - df_results['MAE'] / y_test.mean()) * 100
    st.markdown("### 📊 Model Comparison Table:")
    st.dataframe(df_results.style.highlight_max(axis=0, color='lightgreen'))

    # WQI per year
    st.markdown("### 📅 WQI Per Year")
    per_year_df = pd.DataFrame({'year': years_test})
    for name, preds in predictions.items():
        per_year_df[name] = preds
    st.dataframe(per_year_df.groupby('year').mean().reset_index().style.format("{:.2f}"))

    # Recommendation
    st.markdown("### 🤖 Recommended Model")
    best_model = df_results.sort_values(by="R2", ascending=False).iloc[0]
    best_model_name = best_model["Model"]
    avg_wqi = best_model["WQI Mean"]
    if avg_wqi <= 25:
        category = "Excellent"
    elif avg_wqi <= 50:
        category = "Good"
    elif avg_wqi <= 75:
        category = "Medium"
    elif avg_wqi <= 100:
        category = "Poor"
    else:
        category = "Very Poor"
    st.success(f"✅ Best Model: {best_model_name}, Avg WQI: {avg_wqi:.2f} → {category}")

    # 🔮 Future Prediction (Prophet Forecast)
    st.markdown("### 🔮 Future WQI Prediction (2012–2027 using Prophet)")
    future_years = np.arange(2012, 2028)
    future_df = prophet_forecast_parameters(df, parameters, future_years)
    future_df = calculate_wqi(future_df)

    # Future WQI Prediction
    if best_model_name == 'SVR':
        future_wqi = models['SVR'].predict(scaler.transform(future_df[parameters]))
    else:
        future_wqi = models[best_model_name].predict(future_df[parameters])

    # Clip & smooth
    wqi_min, wqi_max = np.percentile(df['WQI'], 5), np.percentile(df['WQI'], 95)
    future_wqi = np.clip(future_wqi, wqi_min, wqi_max)
    if len(future_wqi) >= 5:
        future_wqi = savgol_filter(future_wqi, window_length=5, polyorder=2)

    fig, ax = plt.subplots()
    ax.plot(future_df['year_num'], future_wqi, marker='o', color='purple')
    ax.set_title(f"Future WQI Forecast - {best_model_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("WQI")
    ax.grid(True)
    st.pyplot(fig)

else:
    st.warning("⚠️ Please upload a CSV file to begin.")
