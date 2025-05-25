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
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("💧 Water Quality Index (WQI) Prediction & Forecast App")

# Sidebar: File Upload
st.sidebar.header("📁 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your Water Quality CSV file", type=["csv"])

# Parameter Setup
parameters = ['pH', 'DO', 'BOD', 'Temp', 'FecalColiform', 'TotalColiform', 'Conductivity', 'NO3-']
weights = {'pH': 0.11, 'DO': 0.17, 'BOD': 0.11, 'Temp': 0.05,
           'FecalColiform': 0.10, 'TotalColiform': 0.10, 'Conductivity': 0.10, 'NO3-': 0.03}
standards = {'pH': 8.5, 'DO': 14, 'BOD': 6, 'Temp': 25,
             'FecalColiform': 100, 'TotalColiform': 100, 'Conductivity': 300, 'NO3-': 45}
ideals = {'pH': 7.0, 'DO': 7.0, 'BOD': 0, 'Temp': 4,
          'FecalColiform': 0, 'TotalColiform': 0, 'Conductivity': 0, 'NO3-': 0}

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

# Prophet Forecasting
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
            m = Prophet(yearly_seasonality=True)
            m.fit(yearly_avg)
            future = pd.DataFrame({'ds': pd.date_range(start=f"{future_years[0]}", periods=len(future_years), freq='Y')})
            forecast = m.predict(future)
            predicted = forecast['yhat'].values
            min_val, max_val = sub[param].min(), sub[param].max()
            predicted = np.clip(predicted, min_val, max_val)
            future_df[param] = predicted
            with st.expander(f"📈 Forecast Plot for {param}", expanded=False):
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
            st.warning(f"⚠️ Prophet failed for {param}: {e}")
    return future_df

# --- MAIN LOGIC ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
    df['year'] = pd.to_datetime(df['year_num'], errors='coerce', format='%Y')
    df['year_num'] = df['year'].dt.year
    df = calculate_wqi(df).dropna(subset=parameters + ['WQI', 'year_num'])

    X = df[parameters]
    y = df['WQI']
    years = df['year_num']
    X_train, X_test, y_train, y_test, years_train, years_test = train_test_split(X, y, years, test_size=0.2, random_state=42)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Training", "📅 WQI Stats", "🔮 Forecast", "🧪 Try Your Own Inputs"])

    with tab1:
        st.subheader("📊 Model Evaluation & Error Analysis")
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
            with st.expander(f"{name} - Error Analysis", expanded=False):
                fig, ax = plt.subplots()
                ax.scatter(y_test, preds, alpha=0.6)
                ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
                ax.set_title(f"{name}: Actual vs Predicted WQI")
                st.pyplot(fig)
            model_results.append({
                'Model': name,
                'R2': r2_score(y_test, preds),
                'MAE': mean_absolute_error(y_test, preds),
                'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
                'WQI Mean': preds.mean()
            })

        df_results = pd.DataFrame(model_results)
        df_results['Accuracy (%)'] = (1 - df_results['MAE'] / y_test.mean()) * 100
        st.dataframe(df_results.style.highlight_max(axis=0, color='lightgreen'))

        # 📊 Combined Comparison Chart
        st.markdown("### 📊 Combined Model Comparison (R², MAE, RMSE)")
        melted = df_results.melt(id_vars='Model', value_vars=['R2', 'MAE', 'RMSE'], var_name='Metric', value_name='Score')
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = ['R2', 'MAE', 'RMSE']
        bar_width = 0.2
        positions = np.arange(len(df_results))
        for i, metric in enumerate(metrics):
            values = melted[melted['Metric'] == metric]['Score'].values
            ax.bar(positions + i * bar_width, values, width=bar_width, label=metric)
        ax.set_xticks(positions + bar_width)
        ax.set_xticklabels(df_results['Model'])
        ax.set_title("Model Comparison - R², MAE, RMSE")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    with tab2:
        st.subheader("📅 WQI Per Year + Category Distribution")
        per_year_df = pd.DataFrame({'Year': years_test})
        for name, preds in predictions.items():
            per_year_df[name] = preds
        avg_df = per_year_df.groupby('Year').mean().reset_index()
        st.dataframe(avg_df.style.format("{:.2f}"))

        def wqi_category(val):
            if val <= 25: return 'Excellent'
            elif val <= 50: return 'Good'
            elif val <= 75: return 'Medium'
            elif val <= 100: return 'Poor'
            else: return 'Very Poor'
        df['Category'] = df['WQI'].apply(wqi_category)
        cat_dist = df.groupby(['year_num', 'Category']).size().unstack(fill_value=0)
        cat_dist.plot(kind='bar', stacked=True, figsize=(10, 5))
        plt.title("WQI Category Distribution Over Years")
        plt.xlabel("Year")
        plt.ylabel("Number of Records")
        st.pyplot(plt.gcf())

    with tab3:
        st.subheader("🔮 Future WQI Prediction (2012–2027)")
        future_years = np.arange(2012, 2028)
        future_df = prophet_forecast_parameters(df, parameters, future_years)
        future_df = calculate_wqi(future_df)

        best_model = df_results.sort_values(by="R2", ascending=False).iloc[0]
        best_model_name = best_model["Model"]
        if best_model_name == 'SVR':
            future_wqi = models['SVR'].predict(scaler.transform(future_df[parameters]))
        else:
            future_wqi = models[best_model_name].predict(future_df[parameters])

        future_wqi = np.clip(future_wqi, np.percentile(df['WQI'], 5), np.percentile(df['WQI'], 95))
        if len(future_wqi) >= 5:
            future_wqi = savgol_filter(future_wqi, 5, 2)

        fig, ax = plt.subplots()
        ax.plot(future_df['year_num'], future_wqi, marker='o', color='purple')
        ax.set_title(f"Future WQI Forecast - {best_model_name}")
        ax.set_xlabel("Year")
        ax.set_ylabel("WQI")
        ax.grid(True)
        st.pyplot(fig)

    with tab4:
        st.subheader("🧪 Try Your Own Parameters (What-If Simulator)")

        sim_year = st.slider("📅 Simulate WQI for Year", 2025, 2050, 2030)

        user_input = {}
        col_row1 = st.columns(4)
        for i, param in enumerate(parameters[:4]):
            with col_row1[i]:
                min_val = float(df[param].min())
                max_val = float(df[param].max())
                mean_val = float(df[param].mean())
                user_input[param] = st.slider(f"{param}", min_val, max_val, mean_val)

        col_row2 = st.columns(4)
        for i, param in enumerate(parameters[4:]):
            with col_row2[i]:
                min_val = float(df[param].min())
                max_val = float(df[param].max())
                mean_val = float(df[param].mean())
                user_input[param] = st.slider(f"{param}", min_val, max_val, mean_val)

        user_df = pd.DataFrame([user_input])
        user_df = calculate_wqi(user_df)
        wqi_value = float(user_df['WQI'].iloc[0])

        def get_category(wqi):
            if wqi <= 25: return "🟢 Excellent"
            elif wqi <= 50: return "🟢 Good"
            elif wqi <= 75: return "🟠 Medium"
            elif wqi <= 100: return "🔴 Poor"
            else: return "⚫ Very Poor"

        category = get_category(wqi_value)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=wqi_value,
                title={'text': "Simulated WQI"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 25], 'color': "green"},
                        {'range': [25, 50], 'color': "lime"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"},
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': wqi_value}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### WQI Category")
            st.markdown(f"<h2 style='text-align:center'>{category}</h2>", unsafe_allow_html=True)

else:
    st.info("👈 Upload your CSV from the sidebar to get started.")
