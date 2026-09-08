import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Executive Sales & Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #475569;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Function: Sample Data Generator ---
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
    categories = ["Electronics", "Home & Kitchen", "Apparel", "Office Supplies", "Health & Fitness"]
    regions = ["North", "South", "East", "West", "Central"]
    
    n_rows = 3000
    chosen_dates = np.random.choice(dates, n_rows)
    
    # Generate realistic seasonal trend
    base_sales = np.random.exponential(scale=200, size=n_rows) + 50
    seasonal_boost = [(d.month / 12.0) * 100 for d in pd.to_datetime(chosen_dates)]
    
    df = pd.DataFrame({
        "Date": chosen_dates,
        "Category": np.random.choice(categories, n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        "Region": np.random.choice(regions, n_rows),
        "Sales": np.round(base_sales + seasonal_boost, 2),
        "Quantity": np.random.randint(1, 12, size=n_rows),
    })
    df["Profit"] = np.round(df["Sales"] * np.random.uniform(0.12, 0.38, size=n_rows), 2)
    df["Order_ID"] = [f"ORD-{2000 + i}" for i in range(n_rows)]
    return df.sort_values("Date").reset_index(drop=True)


# --- Sidebar: Data Ingestion & Mapping ---
with st.sidebar:
    st.title("⚙️ Controls & Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Sales CSV",
        type=["csv"],
        help="Upload a CSV with sales and date fields."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            st.stop()
    else:
        st.info("💡 Using demo dataset (2-year sample).")
        df = generate_sample_data()

    st.markdown("---")
    st.subheader("🔍 Column Mapping")
    
    def find_match(candidates, columns):
        for col in columns:
            if any(cand in col.lower() for cand in candidates):
                return col
        return columns[0] if len(columns) > 0 else None

    cols = list(df.columns)
    
    date_col = st.selectbox("Date Column", cols, index=cols.index(find_match(["date", "time", "day", "order_date"], cols)))
    sales_col = st.selectbox("Sales Column", cols, index=cols.index(find_match(["sales", "revenue", "amount", "total"], cols)))
    profit_col = st.selectbox("Profit Column (Optional)", ["None"] + cols, index=(cols.index(find_match(["profit", "margin"], cols)) + 1) if find_match(["profit", "margin"], cols) else 0)
    category_col = st.selectbox("Category Column", cols, index=cols.index(find_match(["category", "product", "item", "segment"], cols)))
    region_col = st.selectbox("Region Column", cols, index=cols.index(find_match(["region", "location", "city", "country", "state"], cols)))

# --- Data Cleaning ---
try:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)
    if profit_col != "None":
        df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0)
except Exception as e:
    st.error(f"Data conversion error: {e}")
    st.stop()

# --- Sidebar Filters ---
with st.sidebar:
    st.markdown("---")
    st.subheader("🎯 Filters")
    
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    
    date_range = st.date_input("Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)
    
    all_categories = sorted(df[category_col].dropna().astype(str).unique().tolist())
    selected_categories = st.multiselect("Category", all_categories, default=all_categories)
    
    all_regions = sorted(df[region_col].dropna().astype(str).unique().tolist())
    selected_regions = st.multiselect("Region", all_regions, default=all_regions)

# Filter dataset
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = df[
        (df[date_col].dt.date >= start_d) &
        (df[date_col].dt.date <= end_d) &
        (df[category_col].astype(str).isin(selected_categories)) &
        (df[region_col].astype(str).isin(selected_regions))
    ]
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("⚠️ No records match the selected filters. Please expand your filter range.")
    st.stop()

# --- Top Header ---
st.title("📈 Executive Sales & Predictive Dashboard")

# --- Tabs Navigation ---
tab1, tab2, tab3 = st.tabs(["📊 Performance Overview", "🔮 Predictive Forecasting", "📋 Raw Data & Export"])

# ==============================================================================
# TAB 1: PERFORMANCE OVERVIEW
# ==============================================================================
with tab1:
    total_revenue = filtered_df[sales_col].sum()
    total_orders = len(filtered_df)
    aov = total_revenue / total_orders if total_orders > 0 else 0
    total_profit = filtered_df[profit_col].sum() if profit_col != "None" else None
    profit_margin = (total_profit / total_revenue * 100) if (profit_col != "None" and total_revenue > 0) else None

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Revenue</div><div class="metric-value">${total_revenue:,.2f}</div></div>""", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Orders</div><div class="metric-value">{total_orders:,}</div></div>""", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Order Value</div><div class="metric-value">${aov:,.2f}</div></div>""", unsafe_allow_html=True)
    with kpi4:
        if total_profit is not None:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Profit (Margin)</div><div class="metric-value">${total_profit:,.2f} <span style="font-size: 15px; color: #4ade80;">({profit_margin:.1f}%)</span></div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Active Segments</div><div class="metric-value">{filtered_df[category_col].nunique()} Categories</div></div>""", unsafe_allow_html=True)

    st.write("")
    
    c1, c2 = st.columns([3, 2])
    with c1:
        trend_freq = st.radio("Aggregation Interval:", ["Day", "Week", "Month"], horizontal=True, index=2)
        freq_map = {"Day": "D", "Week": "W-MON", "Month": "MS"}
        
        timeline_df = filtered_df.set_index(date_col).resample(freq_map[trend_freq])[sales_col].sum().reset_index()
        
        fig_timeline = px.area(
            timeline_df, x=date_col, y=sales_col,
            title=f"Historical Sales Trend ({trend_freq}ly)",
            labels={date_col: "Date", sales_col: "Revenue ($)"},
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8"]
        )
        fig_timeline.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        st.plotly_chart(fig_timeline, use_container_width=True)

    with c2:
        cat_df = filtered_df.groupby(category_col)[sales_col].sum().reset_index()
        fig_donut = px.pie(
            cat_df, names=category_col, values=sales_col, hole=0.55,
            title="Revenue Distribution by Category",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_donut.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_donut, use_container_width=True)


# ==============================================================================
# TAB 2: PREDICTIVE FORECASTING (PROPHET + MOVING AVERAGE)
# ==============================================================================
with tab2:
    st.subheader("🔮 Sales Demand & Revenue Forecasting")
    st.markdown("Generate forward-looking projections with confidence boundaries and trend decomposition.")

    # Aggregate daily sales for modeling
    daily_sales = (
        filtered_df.groupby(pd.Grouper(key=date_col, freq="D"))[sales_col]
        .sum()
        .reset_index()
        .rename(columns={date_col: "ds", sales_col: "y"})
    )
    daily_sales["y"] = daily_sales["y"].fillna(0)

    # User Forecast Parameters
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        forecast_horizon = st.slider("Forecast Horizon (Days)", min_value=14, max_value=180, value=60, step=7)
    with f_col2:
        model_type = st.selectbox("Forecasting Engine", ["Prophet (AI Seasonality & Trend)", "Moving Averages (Baseline)"])
    with f_col3:
        include_ma = st.checkbox("Overlay 7-Day & 30-Day Moving Averages", value=True)

    # Compute Moving Averages
    daily_sales["MA_7"] = daily_sales["y"].rolling(window=7, min_periods=1).mean()
    daily_sales["MA_30"] = daily_sales["y"].rolling(window=30, min_periods=1).mean()

    if len(daily_sales) < 14:
        st.error("Insufficient historical records to generate a meaningful forecast. Please provide at least 14 days of data.")
    else:
        if model_type.startswith("Prophet"):
            try:
                from prophet import Prophet
                
                with st.spinner("Fitting Prophet model and generating confidence intervals..."):
                    m = Prophet(
                        yearly_seasonality="auto",
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        interval_width=0.80  # 80% confidence interval
                    )
                    m.fit(daily_sales[["ds", "y"]])
                    
                    future = m.make_future_dataframe(periods=forecast_horizon)
                    forecast = m.predict(future)

                    # Bound lower predictions at 0 (sales cannot be negative)
                    forecast["yhat"] = forecast["yhat"].clip(lower=0)
                    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
                    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

                    # Forecast summary metrics
                    future_predictions = forecast[forecast["ds"] > daily_sales["ds"].max()]
                    projected_total = future_predictions["yhat"].sum()
                    recent_window_actual = daily_sales.tail(forecast_horizon)["y"].sum()
                    growth_delta = ((projected_total - recent_window_actual) / recent_window_actual * 100) if recent_window_actual > 0 else 0

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""<div class="metric-card"><div class="metric-label">Projected Revenue ({forecast_horizon} Days)</div><div class="metric-value">${projected_total:,.2f}</div></div>""", unsafe_allow_html=True)
                    with m2:
                        color = "#4ade80" if growth_delta >= 0 else "#f87171"
                        st.markdown(f"""<div class="metric-card"><div class="metric-label">Growth vs Prior {forecast_horizon} Days</div><div class="metric-value" style="color: {color};">{growth_delta:+.2f}%</div></div>""", unsafe_allow_html=True)
                    with m3:
                        peak_date = future_predictions.loc[future_predictions["yhat"].idxmax(), "ds"].strftime("%b %d, %Y")
                        st.markdown(f"""<div class="metric-card"><div class="metric-label">Projected Peak Sales Day</div><div class="metric-value" style="font-size:22px;">{peak_date}</div></div>""", unsafe_allow_html=True)

                    st.write("")

                    # --- Interactive Forecast Visualization ---
                    fig_fc = go.Figure()

                    # Historical Actuals
                    fig_fc.add_trace(go.Scatter(
                        x=daily_sales["ds"], y=daily_sales["y"],
                        mode="lines", name="Actual Daily Sales",
                        line=dict(color="#64748b", width=1.2),
                        opacity=0.7
                    ))

                    # 7-Day & 30-Day Rolling Averages
                    if include_ma:
                        fig_fc.add_trace(go.Scatter(
                            x=daily_sales["ds"], y=daily_sales["MA_7"],
                            mode="lines", name="7-Day Rolling Avg",
                            line=dict(color="#f59e0b", width=1.5, dash="dot")
                        ))
                        fig_fc.add_trace(go.Scatter(
                            x=daily_sales["ds"], y=daily_sales["MA_30"],
                            mode="lines", name="30-Day Rolling Avg",
                            line=dict(color="#10b981", width=1.8)
                        ))

                    # Confidence Intervals (Upper & Lower Band)
                    fig_fc.add_trace(go.Scatter(
                        x=future_predictions["ds"], y=future_predictions["yhat_upper"],
                        mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip"
                    ))
                    fig_fc.add_trace(go.Scatter(
                        x=future_predictions["ds"], y=future_predictions["yhat_lower"],
                        mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(56, 189, 248, 0.15)",
                        name="80% Confidence Interval"
                    ))

                    # Forecast Prediction Line
                    fig_fc.add_trace(go.Scatter(
                        x=future_predictions["ds"], y=future_predictions["yhat"],
                        mode="lines", name="Prophet Forecast",
                        line=dict(color="#38bdf8", width=3)
                    ))

                    fig_fc.update_layout(
                        title=f"Prophet Demand Forecast (Next {forecast_horizon} Days)",
                        xaxis_title="Date",
                        yaxis_title="Revenue ($)",
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        hovermode="x unified",
                        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)

                    # Seasonal Components Breakdown
                    with st.expander("🔎 View Model Seasonality Breakdown (Weekly & Trend)"):
                        st.markdown("Shows underlying weekly purchasing cycles extracted by Prophet.")
                        forecast["day_of_week"] = forecast["ds"].dt.day_name()
                        weekly_pattern = forecast.groupby("day_of_week")["weekly"].mean().reindex(
                            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                        ).reset_index()

                        fig_weekly = px.bar(
                            weekly_pattern, x="day_of_week", y="weekly",
                            title="Weekly Seasonality Effect",
                            labels={"day_of_week": "Day", "weekly": "Impact on Sales ($)"},
                            template="plotly_dark",
                            color="weekly",
                            color_continuous_scale="Viridis"
                        )
                        fig_weekly.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_weekly, use_container_width=True)

            except ImportError:
                st.error("The `prophet` package is not installed. Please run `pip install prophet` or select 'Moving Averages'.")

        else:
            # Moving Average Projection
            window = 30
            last_ma = daily_sales["y"].tail(window).mean()
            last_date = daily_sales["ds"].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
            
            projected_df = pd.DataFrame({
                "ds": future_dates,
                "yhat": [last_ma] * forecast_horizon
            })

            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=daily_sales["ds"], y=daily_sales["y"], name="Actual Daily Sales", line=dict(color="#64748b", width=1)))
            fig_ma.add_trace(go.Scatter(x=daily_sales["ds"], y=daily_sales["MA_7"], name="7-Day MA", line=dict(color="#f59e0b", width=1.5)))
            fig_ma.add_trace(go.Scatter(x=daily_sales["ds"], y=daily_sales["MA_30"], name="30-Day MA", line=dict(color="#10b981", width=1.8)))
            fig_ma.add_trace(go.Scatter(x=projected_df["ds"], y=projected_df["yhat"], name=f"{window}-Day MA Baseline Forecast", line=dict(color="#38bdf8", width=2.5, dash="dash")))

            fig_ma.update_layout(
                title=f"Moving Average Forecast Baseline (Next {forecast_horizon} Days)",
                xaxis_title="Date",
                yaxis_title="Revenue ($)",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_ma, use_container_width=True)


# ==============================================================================
# TAB 3: RAW DATA & EXPORT
# ==============================================================================
with tab3:
    st.subheader("Data Export & Raw Inspection")
    st.dataframe(filtered_df.sort_values(by=date_col, ascending=False), use_container_width=True)
    
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Sliced Data as CSV",
        data=csv_data,
        file_name="sales_filtered_export.csv",
        mime="text/csv"
    )