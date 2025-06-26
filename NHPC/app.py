import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.seasonal import STL
from prophet import Prophet
from scipy.stats import iqr
import plotly.graph_objects as go
import streamlit as st

# Title
st.title("⚡ Electricity Generation Analysis and Forecast Dashboard")

# Load data
df = pd.read_excel("GenerationData.xlsx")
df.columns = ['Date', 'Type'] + [f'Time_{i}' for i in range(1, len(df.columns) - 1)]
df = df[df['Type'] == 'Scheduled Generation (MW)'].drop(columns=['Type'])

# Melt into long format
df_melted = df.melt(id_vars=['Date'], var_name='TimeBlock', value_name='Generation')
time_blocks = pd.date_range(start="00:00", periods=96, freq="15min").time
time_block_mapping = dict(zip([f'Time_{i}' for i in range(1, 97)], time_blocks))
df_melted['Time'] = df_melted['TimeBlock'].map(time_block_mapping)
df_melted['Datetime'] = pd.to_datetime(df_melted['Date'].astype(str)) + pd.to_timedelta(df_melted['Time'].astype(str))
df_timeseries = df_melted[['Datetime', 'Generation']].sort_values('Datetime').reset_index(drop=True)
daily_df = df_timeseries.resample('D', on='Datetime').mean().reset_index()

# Add time parts
daily_df['Month'] = daily_df['Datetime'].dt.strftime('%m')
daily_df['Year'] = daily_df['Datetime'].dt.strftime('%Y')
daily_df['MonthYear'] = daily_df['Datetime'].dt.strftime('%Y-%m')

st.success("✅ Data loaded and preprocessed.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 STL Trend by Month", "📦 Trend Boxplot by Month", "📉 Forecast + Maintenance", "📅 Full-Year STL Trend by Year"])

# --- TAB 1: STL Decomposition Trend Lines by Selected Month Across Years ---
with tab1:
    st.markdown("### 📈 STL Trend Lines by Month Across Years")
    available_months = sorted(daily_df['Month'].unique())
    selected_month = st.selectbox("Select Month", options=available_months, index=5)
    
    all_years = sorted(daily_df['Year'].unique())
    max_years_to_show = st.slider("Number of Most Recent Years to Show", min_value=1, max_value=len(all_years), value=4)
    selected_years = all_years[-max_years_to_show:]

    trace_labels = []
    trend_store = {}
    fig_trend = go.Figure()

    for year in selected_years:
        my = f"{year}-{selected_month}"
        month_df = daily_df[daily_df['MonthYear'] == my]
        if len(month_df) < 14:
            continue
        ts = month_df.set_index('Datetime')['Generation']
        res = STL(ts, period=7).fit()
        trend_values = res.trend.values
        x_vals = list(range(1, len(trend_values) + 1))
        trend_store.setdefault(selected_month, []).append(trend_values)
        fig_trend.add_trace(go.Scatter(x=x_vals, y=trend_values, mode='lines', name=f"{selected_month}-{year}"))
        trace_labels.append((selected_month, year))

    # IQR bands
    trends = trend_store.get(selected_month, [])
    if trends:
        max_len = max(len(t) for t in trends)
        padded = np.array([np.pad(t, (0, max_len - len(t)), constant_values=np.nan) for t in trends])
        q1 = np.nanpercentile(padded, 25, axis=0)
        q3 = np.nanpercentile(padded, 75, axis=0)
        x_vals = list(range(1, max_len + 1))
        fig_trend.add_trace(go.Scatter(x=x_vals, y=q1, fill=None, mode='lines',
                                       line=dict(color='rgba(0,100,80,0.2)'), showlegend=False))
        fig_trend.add_trace(go.Scatter(x=x_vals, y=q3, fill='tonexty', mode='lines',
                                       line=dict(color='rgba(0,100,80,0.2)'), showlegend=False))

    fig_trend.update_layout(title=f"STL Trend for Month: {selected_month}", height=600,
                            xaxis_title="Day of Month", yaxis_title="Trend Value")
    st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 2: Boxplot of Monthly Trends ---
with tab2:
    st.markdown("#### 📦 Distribution of Trend Values per Month (Boxplot)")
    box_data = []
    for my in sorted(daily_df['MonthYear'].unique()):
        month_df = daily_df[daily_df['MonthYear'] == my]
        if len(month_df) < 14:
            continue
        ts = month_df.set_index('Datetime')['Generation']
        res = STL(ts, period=7).fit()
        for val in res.trend:
            box_data.append({'Month': my[-2:], 'Trend': val})

    box_df = pd.DataFrame(box_data)
    box_df['Month'] = pd.Categorical(box_df['Month'], categories=sorted(box_df['Month'].unique()), ordered=True)

    box_fig = go.Figure()
    for m in box_df['Month'].cat.categories:
        box_fig.add_trace(go.Box(y=box_df[box_df['Month'] == m]['Trend'], name=m))
    box_fig.update_layout(title="Boxplot of Trend Values per Month", yaxis_title="Trend Value")
    st.plotly_chart(box_fig, use_container_width=True)

# --- TAB 3: Forecast + Rolling Loss ---
with tab3:
    st.markdown("#### 📉 Prophet Forecast & Rolling Loss")

    prophet_df = daily_df[['Datetime', 'Generation']].rename(columns={'Datetime': 'ds', 'Generation': 'y'}).dropna()
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=600)
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Datetime', 'yhat': 'Forecast'})

    # IQR filter
    q1 = forecast_df['Forecast'].quantile(0.25)
    q3 = forecast_df['Forecast'].quantile(0.75)
    lower = q1 - 1.5 * iqr(forecast_df['Forecast'])
    upper = q3 + 1.5 * iqr(forecast_df['Forecast'])
    clean_forecast = forecast_df[(forecast_df['Forecast'] >= lower) & (forecast_df['Forecast'] <= upper)]

    # Forecast Plot
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=forecast_df['Datetime'], y=forecast_df['Forecast'], name='Forecast', line=dict(color='blue')))
    forecast_fig.add_trace(go.Scatter(x=clean_forecast['Datetime'], y=clean_forecast['Forecast'], name='Filtered (IQR)', line=dict(color='green')))
    forecast_fig.update_layout(title="Forecast vs IQR-filtered Forecast", xaxis_title="Date", yaxis_title="MW")
    st.plotly_chart(forecast_fig, use_container_width=True)

    # Maintenance optimization
    maintenance_days = st.number_input("Enter number of days for maintenance", min_value=1, max_value=60, value=7)
    month_input = st.text_input("Enter month(s) or range (e.g., '12' or '11-2')", value='12')

    def parse_month_input(month_str):
        if '-' in month_str:
            start, end = map(int, month_str.split('-'))
            return list(range(start, 13)) + list(range(1, end + 1)) if start > end else list(range(start, end + 1))
        else:
            return [int(month_str)]

    target_months = parse_month_input(month_input)
    cutoff = pd.to_datetime("2024-04-01")
    filtered = clean_forecast[(clean_forecast['Datetime'] >= cutoff) &
                              (clean_forecast['Datetime'].dt.month.isin(target_months))]
    filtered['RollingLoss'] = filtered['Forecast'].rolling(window=maintenance_days).sum()
    best_windows = filtered.dropna().nsmallest(3, 'RollingLoss')

    # Loss Plot
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=filtered['Datetime'], y=filtered['RollingLoss'], name='Rolling Loss', line=dict(color='orange')))
    for i, row in best_windows.iterrows():
        loss_fig.add_vrect(
            x0=row['Datetime'] - timedelta(days=maintenance_days - 1),
            x1=row['Datetime'],
            fillcolor="red", opacity=0.2,
            annotation_text="Suggested", annotation_position="top left"
        )
    loss_fig.update_layout(title="Rolling Loss Over Time", xaxis_title="Date", yaxis_title="Total Loss in MW")
    st.plotly_chart(loss_fig, use_container_width=True)

    if not best_windows.empty:
        st.success("✅ Top Maintenance Windows:")
        for i, row in best_windows.iterrows():
            end_date = row['Datetime'].date()
            start_date = (row['Datetime'] - timedelta(days=maintenance_days - 1)).date()
            avg_loss = round(row['RollingLoss'] / maintenance_days, 2)
            total_loss = round(row['RollingLoss'], 2)
            st.markdown(f"🗓️ **{start_date} to {end_date}** → Avg Loss: `{avg_loss}` MW/day | Total: `{total_loss}` MW")
    else:
        st.warning("No valid maintenance windows found. Try adjusting your input.")

# --- TAB 4: Full-Year STL Trend Line Comparison ---
# --- TAB 4: Full-Year STL Trend Line Comparison ---
# --- TAB 4: Full-Year STL Trend Line Comparison ---
# --- TAB 4: Full-Year STL Trend Line Comparison ---
with tab4:
    st.markdown("### 📅 Full-Year STL Trend Comparison (Overlapped by Month)")

    all_years = sorted(daily_df['Year'].unique())
    max_years_tab4 = st.slider("Select number of most recent years to compare", min_value=1, max_value=len(all_years), value=4)
    selected_tab4_years = all_years[-max_years_tab4:]

    fig_full_year = go.Figure()

    for year in selected_tab4_years:
        year_df = daily_df[daily_df['Year'] == year].copy()
        if len(year_df) < 100:
            continue

        # Normalize to year 2000 for X-axis alignment
        year_df['NormalizedDate'] = year_df['Datetime'].apply(lambda d: d.replace(year=2000))
        ts = year_df.set_index('NormalizedDate')['Generation']
        res = STL(ts, period=7).fit()

        fig_full_year.add_trace(go.Scatter(
            x=ts.index,  # Normalized to 2000
            y=res.trend.values,
            mode='lines',
            name=f"{year}"
        ))

    fig_full_year.update_layout(
        title="Overlapping STL Trends (Jan–Dec) for Selected Years",
        height=650,
        xaxis_title="Month",
        yaxis_title="Trend Value",
        legend_title="Year",
        xaxis=dict(
            tickformat="%b",  # Jan, Feb, etc.
            dtick="M1",
            tickangle=0,
            tickfont=dict(size=10)
        )
    )
    st.plotly_chart(fig_full_year, use_container_width=True)
