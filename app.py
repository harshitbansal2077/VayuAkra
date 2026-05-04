import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import KDTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Project VayuAkra",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0a0e1a; }
    [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a 0%, #0a0e1a 100%); border-right: 1px solid #1e3a5f; }
    .hero-title { font-size: 2.8rem; font-weight: 900; background: linear-gradient(90deg, #00d4ff, #00ff9d, #ffb800); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; letter-spacing: -1px; margin-bottom: 0; }
    .hero-sub { text-align: center; color: #7ba3c8; font-size: 1rem; font-weight: 300; margin-top: 4px; letter-spacing: 2px; text-transform: uppercase; }
    .metric-card { background: linear-gradient(135deg, #0d1f35, #0a1628); border: 1px solid #1e3a5f; border-radius: 12px; padding: 20px; text-align: center; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 0.75rem; color: #7ba3c8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .section-header { font-size: 1.2rem; font-weight: 700; color: #00d4ff; border-left: 3px solid #00d4ff; padding-left: 12px; margin: 20px 0 12px 0; }
    .badge-urgent { background: #ff3b3b22; color: #ff6b6b; border: 1px solid #ff3b3b55; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
    .badge-hybrid { background: #00d4ff22; color: #00d4ff; border: 1px solid #00d4ff55; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
    .badge-solar { background: #ffb80022; color: #ffb800; border: 1px solid #ffb80055; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
    .badge-wind { background: #00ff9d22; color: #00ff9d; border: 1px solid #00ff9d55; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
    .plant-card { background: linear-gradient(135deg, #0d1f35, #0a1628); border: 1px solid #1e3a5f; border-radius: 16px; padding: 24px; margin-top: 12px; }
    [data-testid="stMetricValue"] { color: #00d4ff !important; }
    .stSlider > div > div { background: #1e3a5f; }
    div[data-testid="stSelectbox"] > div { background: #0d1f35; border-color: #1e3a5f; }
    .stDataFrame { background: #0d1f35; border-radius: 8px; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    .chat-msg-user { background: linear-gradient(135deg, #1a3a5c, #1e4a7a); border: 1px solid #2a5a8c; border-radius: 16px 16px 4px 16px; padding: 14px 18px; margin: 8px 0 8px 60px; color: #e8f4fd; font-size: 0.92rem; line-height: 1.6; }
    .chat-msg-ai { background: linear-gradient(135deg, #0d1f35, #0f2840); border: 1px solid #1e3a5f; border-radius: 16px 16px 16px 4px; padding: 14px 18px; margin: 8px 60px 8px 0; color: #c8dff0; font-size: 0.92rem; line-height: 1.6; }
    .chat-avatar-ai { font-size: 1.4rem; margin-bottom: 4px; }
    .chat-container { max-height: 520px; overflow-y: auto; padding: 12px 0; }
    .suggestion-pill { display: inline-block; background: #0d1f35; border: 1px solid #2a5a8c; border-radius: 20px; padding: 6px 14px; margin: 4px; cursor: pointer; color: #7ba3c8; font-size: 0.8rem; transition: all 0.2s; }
    .suggestion-pill:hover { background: #1e3a5f; color: #00d4ff; border-color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

STATE_CENTROIDS = {
    "Andhra Pradesh": (15.9, 79.7),
    "Arunachal Pradesh": (28.2, 94.7),
    "Assam": (26.2, 92.9),
    "Bihar": (25.1, 85.3),
    "Chhattisgarh": (21.3, 81.9),
    "Goa": (15.3, 74.0),
    "Gujarat": (22.3, 71.2),
    "Haryana": (29.1, 76.1),
    "Himachal Pradesh": (31.1, 77.2),
    "Jharkhand": (23.6, 85.3),
    "Karnataka": (15.3, 75.7),
    "Kerala": (10.9, 76.3),
    "Madhya Pradesh": (22.9, 78.7),
    "Maharashtra": (19.7, 75.7),
    "Manipur": (24.7, 93.9),
    "Meghalaya": (25.5, 91.4),
    "Mizoram": (23.2, 92.9),
    "Nagaland": (26.2, 94.6),
    "Odisha": (20.9, 85.1),
    "Punjab": (31.1, 75.3),
    "Rajasthan": (27.0, 74.2),
    "Sikkim": (27.5, 88.5),
    "Tamil Nadu": (11.1, 78.7),
    "Telangana": (17.4, 79.1),
    "Tripura": (23.9, 91.9),
    "Uttar Pradesh": (27.1, 80.9),
    "Uttarakhand": (30.1, 79.3),
    "West Bengal": (22.9, 87.9),
    "Delhi": (28.7, 77.1),
    "Jammu and Kashmir": (33.7, 76.9),
    "Ladakh": (34.2, 77.6),
}

FOSSIL_FUELS = ["Coal", "Gas", "Oil"]
CURRENT_YEAR = 2024

@st.cache_data
def load_plants():
    df = pd.read_csv("india_plants_renewable_full.csv")
    df = df.dropna(subset=["latitude", "longitude", "capacity_mw"])
    return df

@st.cache_data
def load_monthly():
    df = pd.read_csv("india_monthly_full_release_long_format.csv")
    return df

@st.cache_data
def compute_fossil_ratio(monthly_df):
    cap = monthly_df[
        (monthly_df["Category"] == "Capacity") &
        (monthly_df["State type"] == "state") &
        (monthly_df["Variable"].isin(["Fossil", "Clean"]))
    ]
    latest = cap.sort_values("Date").groupby(["State", "Variable"])["Value"].last().reset_index()
    pivot = latest.pivot(index="State", columns="Variable", values="Value").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"State": "state_name"})
    for col in ["Fossil", "Clean"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot["Fossil"] = pivot["Fossil"].fillna(0)
    pivot["Clean"] = pivot["Clean"].fillna(0)
    pivot["total_cap"] = pivot["Fossil"] + pivot["Clean"]
    pivot["fossil_ratio"] = np.where(
        pivot["total_cap"] > 0,
        pivot["Fossil"] / pivot["total_cap"],
        0.0
    )
    return pivot[["state_name", "Fossil", "Clean", "total_cap", "fossil_ratio"]]

def assign_states(plants_df):
    state_names = list(STATE_CENTROIDS.keys())
    centroids = np.array([STATE_CENTROIDS[s] for s in state_names])
    coords = plants_df[["latitude", "longitude"]].values
    tree = KDTree(centroids)
    _, idx = tree.query(coords)
    plants_df = plants_df.copy()
    plants_df["state_name"] = [state_names[i] for i in idx]
    return plants_df

def compute_age_score(row):
    fuel = row["primary_fuel"]
    cy = row["commissioning_year"]
    if pd.isna(cy):
        assumed_age = 30 if fuel == "Coal" else 20
    else:
        assumed_age = CURRENT_YEAR - int(cy)
    assumed_age = max(0, assumed_age)
    return min(100.0, (assumed_age / 60.0) * 100.0)

def compute_renewable_score(row):
    solar_cols = [f"solar_m{str(i).zfill(2)}" for i in range(1, 13)]
    solar_vals = [row[c] for c in solar_cols if pd.notna(row.get(c))]
    solar_mean = np.mean(solar_vals) if solar_vals else 0.0
    wind = row.get("wind_speed_100m", 0.0)
    wind = wind if pd.notna(wind) else 0.0
    return solar_mean, wind

@st.cache_data
def build_scored_dataset(plants_df, fossil_ratio_df):
    df = assign_states(plants_df)
    df = df.merge(fossil_ratio_df[["state_name", "fossil_ratio"]], on="state_name", how="left")
    df["fossil_ratio"] = df["fossil_ratio"].fillna(0.0)

    solar_cols = [f"solar_m{str(i).zfill(2)}" for i in range(1, 13)]
    df["solar_mean"] = df[solar_cols].mean(axis=1)

    solar_p75 = df["solar_mean"].quantile(0.75)
    wind_p75 = df["wind_speed_100m"].quantile(0.75)

    fossil_mask = df["primary_fuel"].isin(FOSSIL_FUELS)

    age_scores = df.apply(compute_age_score, axis=1)
    solar_norm = (df["solar_mean"] / df["solar_mean"].max().clip(min=1)) * 100
    wind_norm = (df["wind_speed_100m"] / df["wind_speed_100m"].max().clip(min=1)) * 100
    renewable_scores = (solar_norm * 0.5 + wind_norm * 0.5)
    grid_scores = df["fossil_ratio"] * 100

    transition_score = (
        age_scores * 0.40 +
        renewable_scores * 0.40 +
        grid_scores * 0.20
    )
    transition_score = transition_score.clip(0, 100)

    df["age_score"] = age_scores
    df["renewable_score"] = renewable_scores
    df["grid_score"] = grid_scores
    df["transition_score"] = np.where(fossil_mask, transition_score, np.nan)

    def classify(row):
        if not row["primary_fuel"] in FOSSIL_FUELS:
            return "Already Clean"
        s = row["transition_score"]
        if s >= 70:
            return "Urgent Replacement"
        elif s >= 40:
            return "Moderate Priority"
        else:
            return "Low Priority"

    df["classification"] = df.apply(classify, axis=1)

    def renewable_mix(row):
        if row["primary_fuel"] not in FOSSIL_FUELS:
            return "N/A"
        high_solar = row["solar_mean"] >= solar_p75
        high_wind = row["wind_speed_100m"] >= wind_p75 if pd.notna(row["wind_speed_100m"]) else False
        if high_solar and high_wind:
            return "Hybrid Candidate"
        elif high_solar:
            return "Solar"
        elif high_wind:
            return "Wind"
        else:
            return "Solar"

    df["recommended_mix"] = df.apply(renewable_mix, axis=1)
    return df

def score_color(score):
    if pd.isna(score):
        return "#00cc88"
    r = int(255 * (score / 100))
    g = int(255 * (1 - score / 100))
    return f"rgb({r},{g},50)"

def main():
    plants_raw = load_plants()
    monthly_raw = load_monthly()
    fossil_ratio_df = compute_fossil_ratio(monthly_raw)
    df = build_scored_dataset(plants_raw, fossil_ratio_df)

    st.markdown('<h1 class="hero-title">⚡ Project VayuAkra</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">India\'s AI-Driven Hybrid Green Transition Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown("---")

    fossil_df = df[df["primary_fuel"].isin(FOSSIL_FUELS)].copy()
    urgent_df = fossil_df[fossil_df["classification"] == "Urgent Replacement"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Plants</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(fossil_df):,}</div><div class="metric-label">Fossil Plants</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(urgent_df):,}</div><div class="metric-label">Urgent Replacements</div></div>""", unsafe_allow_html=True)
    with col4:
        total_retirable = urgent_df["capacity_mw"].sum()
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_retirable:,.0f}</div><div class="metric-label">Retirable MW</div></div>""", unsafe_allow_html=True)
    with col5:
        hybrid_count = fossil_df[fossil_df["recommended_mix"] == "Hybrid Candidate"].shape[0]
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{hybrid_count:,}</div><div class="metric-label">Hybrid Candidates</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🎛️ Mission Control")
        st.markdown("---")

        min_score = st.slider(
            "🔢 Minimum Transition Score",
            min_value=0, max_value=100, value=0, step=5,
            help="Hide fossil plants below this feasibility score"
        )

        fuel_filter = st.multiselect(
            "⛽ Fuel Types",
            options=["Coal", "Gas", "Oil", "Solar", "Wind", "Hydro", "Biomass", "Nuclear", "Other"],
            default=["Coal", "Gas", "Oil"],
        )

        class_filter = st.multiselect(
            "🚦 Classification",
            options=["Urgent Replacement", "Moderate Priority", "Low Priority", "Already Clean"],
            default=["Urgent Replacement", "Moderate Priority", "Low Priority", "Already Clean"],
        )

        st.markdown("---")
        st.markdown("### 🔎 Plant Inspector")
        fossil_names = fossil_df.sort_values("transition_score", ascending=False)["name"].tolist()
        selected_plant = st.selectbox("Select a Fossil Plant", options=["— None —"] + fossil_names)

        st.markdown("---")
        st.markdown("""<div style="color:#3a6080;font-size:0.72rem;text-align:center;">
        VayuAkra v1.0 · Scoring: Age 40% · Resource 40% · Grid 20%
        </div>""", unsafe_allow_html=True)

    display_df = df.copy()
    if fuel_filter:
        display_df = display_df[display_df["primary_fuel"].isin(fuel_filter)]
    if class_filter:
        display_df = display_df[display_df["classification"].isin(class_filter)]

    fossil_display = display_df[display_df["primary_fuel"].isin(FOSSIL_FUELS)]
    fossil_display = fossil_display[
        fossil_display["transition_score"].isna() | (fossil_display["transition_score"] >= min_score)
    ]
    clean_display = display_df[~display_df["primary_fuel"].isin(FOSSIL_FUELS)]
    display_df = pd.concat([fossil_display, clean_display])

    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Geospatial Intelligence Map", "🏆 State Leaderboard", "☀️💨 Surya-Vayu Analysis", "🤖 VayuAkra AI"])

    with tab1:
        st.markdown('<div class="section-header">Transition Score Geospatial Overview</div>', unsafe_allow_html=True)

        map_df = display_df.dropna(subset=["latitude", "longitude"]).copy()
        map_df["score_display"] = map_df["transition_score"].fillna(0)
        map_df["marker_size"] = (map_df["capacity_mw"].clip(upper=3000) / 3000) * 28 + 5
        map_df["hover_text"] = (
            "<b>" + map_df["name"] + "</b><br>" +
            "Fuel: " + map_df["primary_fuel"] + "<br>" +
            "Capacity: " + map_df["capacity_mw"].round(0).astype(str) + " MW<br>" +
            "State: " + map_df["state_name"] + "<br>" +
            "Score: " + map_df["score_display"].round(1).astype(str) + "<br>" +
            "Status: " + map_df["classification"] + "<br>" +
            "Mix: " + map_df["recommended_mix"]
        )

        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="score_display",
            size="marker_size",
            hover_name="name",
            custom_data=["hover_text"],
            color_continuous_scale=[
                [0.0, "#00cc88"],
                [0.3, "#4de877"],
                [0.5, "#ffb800"],
                [0.7, "#ff6633"],
                [1.0, "#ff1a1a"],
            ],
            range_color=[0, 100],
            zoom=4.3,
            center={"lat": 20.5, "lon": 79.0},
            mapbox_style="carto-darkmatter",
            height=600,
        )
        fig_map.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>",
            marker=dict(opacity=0.85),
        )
        fig_map.update_layout(
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0a0e1a",
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(
                title="Transition<br>Score",
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0<br>Clean", "25", "50", "75", "100<br>Urgent"],
                title_font_color="#7ba3c8",
                tickfont_color="#7ba3c8",
                bgcolor="#0d1f35",
                bordercolor="#1e3a5f",
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fuel_cap = display_df.groupby("primary_fuel")["capacity_mw"].sum().reset_index()
            fuel_cap.columns = ["Fuel", "Capacity_MW"]
            fuel_cap = fuel_cap.sort_values("Capacity_MW", ascending=False)
            colors_fuel = ["#ff3b3b" if f in FOSSIL_FUELS else "#00cc88" for f in fuel_cap["Fuel"]]
            fig_fuel = go.Figure(go.Bar(
                x=fuel_cap["Fuel"],
                y=fuel_cap["Capacity_MW"],
                marker_color=colors_fuel,
                text=fuel_cap["Capacity_MW"].round(0).astype(int),
                textposition="outside",
                textfont=dict(color="#7ba3c8", size=10),
            ))
            fig_fuel.update_layout(
                title=dict(text="Installed Capacity by Fuel Type (MW)", font=dict(color="#00d4ff", size=14)),
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#0d1f35",
                xaxis=dict(color="#7ba3c8", gridcolor="#1e3a5f"),
                yaxis=dict(color="#7ba3c8", gridcolor="#1e3a5f"),
                margin=dict(l=0, r=0, t=40, b=0),
                height=320,
            )
            st.plotly_chart(fig_fuel, use_container_width=True)

        with col_b:
            class_dist = fossil_df["classification"].value_counts().reset_index()
            class_dist.columns = ["Classification", "Count"]
            color_map = {
                "Urgent Replacement": "#ff3b3b",
                "Moderate Priority": "#ffb800",
                "Low Priority": "#00d4ff",
                "Already Clean": "#00cc88",
            }
            fig_pie = px.pie(
                class_dist,
                names="Classification",
                values="Count",
                color="Classification",
                color_discrete_map=color_map,
                hole=0.6,
            )
            fig_pie.update_layout(
                title=dict(text="Fossil Plant Classification Breakdown", font=dict(color="#00d4ff", size=14)),
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#0a0e1a",
                legend=dict(font=dict(color="#7ba3c8"), bgcolor="#0d1f35"),
                margin=dict(l=0, r=0, t=40, b=0),
                height=320,
            )
            fig_pie.update_traces(textfont_color="white")
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">State Leaderboard — Retirable Coal Capacity</div>', unsafe_allow_html=True)

        coal_df = df[df["primary_fuel"] == "Coal"].copy()
        state_coal = coal_df.groupby("state_name").agg(
            total_coal_mw=("capacity_mw", "sum"),
            plant_count=("name", "count"),
            avg_score=("transition_score", "mean"),
            urgent_count=("classification", lambda x: (x == "Urgent Replacement").sum()),
        ).reset_index().sort_values("total_coal_mw", ascending=False)

        state_coal = state_coal.merge(
            fossil_ratio_df[["state_name", "fossil_ratio"]],
            on="state_name", how="left"
        )
        state_coal["fossil_pct"] = (state_coal["fossil_ratio"] * 100).round(1)
        state_coal["avg_score"] = state_coal["avg_score"].round(1)

        col_lb1, col_lb2 = st.columns([3, 2])

        with col_lb1:
            fig_lb = go.Figure()
            top15 = state_coal.head(15)
            bar_colors = [
                "#ff3b3b" if r >= 0.7 else "#ffb800" if r >= 0.4 else "#00cc88"
                for r in top15["fossil_ratio"].fillna(0)
            ]
            fig_lb.add_trace(go.Bar(
                y=top15["state_name"],
                x=top15["total_coal_mw"],
                orientation="h",
                marker_color=bar_colors,
                text=top15["total_coal_mw"].round(0).astype(int).astype(str) + " MW",
                textposition="outside",
                textfont=dict(color="#7ba3c8", size=10),
                customdata=top15[["plant_count", "avg_score", "fossil_pct"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>Coal Capacity: %{x:.0f} MW<br>"
                    "Plants: %{customdata[0]}<br>Avg Score: %{customdata[1]}<br>"
                    "Fossil Grid %: %{customdata[2]}%<extra></extra>"
                ),
            ))
            fig_lb.update_layout(
                title=dict(text="Top 15 States by Retirable Coal Capacity", font=dict(color="#00d4ff", size=14)),
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#0d1f35",
                xaxis=dict(color="#7ba3c8", gridcolor="#1e3a5f", title="Total Coal Capacity (MW)"),
                yaxis=dict(color="#7ba3c8", autorange="reversed"),
                margin=dict(l=0, r=60, t=40, b=0),
                height=500,
            )
            st.plotly_chart(fig_lb, use_container_width=True)

        with col_lb2:
            st.markdown('<div class="section-header">Fossil-to-Clean Grid Ratio by State</div>', unsafe_allow_html=True)
            ratio_df = fossil_ratio_df.sort_values("fossil_ratio", ascending=False).head(15).copy()
            ratio_df["clean_ratio"] = 1 - ratio_df["fossil_ratio"]

            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Bar(
                name="Fossil",
                y=ratio_df["state_name"],
                x=ratio_df["fossil_ratio"] * 100,
                orientation="h",
                marker_color="#ff4444",
            ))
            fig_ratio.add_trace(go.Bar(
                name="Clean",
                y=ratio_df["state_name"],
                x=ratio_df["clean_ratio"] * 100,
                orientation="h",
                marker_color="#00cc88",
            ))
            fig_ratio.update_layout(
                barmode="stack",
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#0d1f35",
                xaxis=dict(color="#7ba3c8", gridcolor="#1e3a5f", title="Grid Composition %"),
                yaxis=dict(color="#7ba3c8", autorange="reversed"),
                legend=dict(font=dict(color="#7ba3c8"), bgcolor="#0d1f35", orientation="h", y=1.08),
                margin=dict(l=0, r=0, t=30, b=0),
                height=500,
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

        st.markdown('<div class="section-header">Full State Rankings Table</div>', unsafe_allow_html=True)
        display_table = state_coal[["state_name", "total_coal_mw", "plant_count", "avg_score", "urgent_count", "fossil_pct"]].copy()
        display_table.columns = ["State", "Coal Capacity (MW)", "# Plants", "Avg Transition Score", "Urgent Plants", "Fossil Grid %"]
        display_table["Coal Capacity (MW)"] = display_table["Coal Capacity (MW)"].round(0).astype(int)
        display_table = display_table.reset_index(drop=True)
        display_table.index += 1
        st.dataframe(
            display_table,
            use_container_width=True,
            height=350,
        )

    with tab3:
        st.markdown('<div class="section-header">☀️ Surya-Vayu Plant Deep Dive</div>', unsafe_allow_html=True)

        if selected_plant == "— None —":
            st.info("👈 Select a fossil plant from the sidebar to begin the Surya-Vayu analysis.")
        else:
            plant_row = fossil_df[fossil_df["name"] == selected_plant].iloc[0]

            solar_cols = [f"solar_m{str(i).zfill(2)}" for i in range(1, 13)]
            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            solar_vals = [plant_row.get(c, 0) for c in solar_cols]
            solar_vals_clean = [v if pd.notna(v) else 0 for v in solar_vals]
            wind_speed = plant_row.get("wind_speed_100m", 0)
            wind_speed = wind_speed if pd.notna(wind_speed) else 0
            score = plant_row["transition_score"]
            mix = plant_row["recommended_mix"]
            cap_mw = plant_row["capacity_mw"]
            state = plant_row["state_name"]
            cy = plant_row.get("commissioning_year")
            age = CURRENT_YEAR - int(cy) if pd.notna(cy) else (30 if plant_row["primary_fuel"] == "Coal" else 20)

            badge_map = {
                "Hybrid Candidate": "badge-hybrid",
                "Solar": "badge-solar",
                "Wind": "badge-wind",
            }
            badge_class = badge_map.get(mix, "badge-solar")

            col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
            with col_h1:
                st.metric("Transition Score", f"{score:.1f}/100")
            with col_h2:
                st.metric("Capacity", f"{cap_mw:.0f} MW")
            with col_h3:
                st.metric("Plant Age", f"~{age} yrs")
            with col_h4:
                st.metric("Avg Solar", f"{np.mean(solar_vals_clean):.2f} kWh/m²/d")
            with col_h5:
                st.metric("Wind @ 100m", f"{wind_speed:.1f} m/s")

            st.markdown(f"""
            <div class="plant-card">
                <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
                    <div style="font-size:1.5rem;font-weight:700;color:#fff;">{selected_plant}</div>
                    <span class="{badge_class}">🎯 {mix}</span>
                    <span style="color:#7ba3c8;font-size:0.85rem;">{plant_row['primary_fuel']} Plant · {state}</span>
                </div>
                <div style="color:#a0b8cc;font-size:0.9rem;line-height:1.8;">
                    {'🔴 <b>URGENT REPLACEMENT CANDIDATE</b>' if score >= 70 else '🟡 <b>MODERATE PRIORITY</b>' if score >= 40 else '🟢 <b>LOW PRIORITY</b>'}
                    <br>
                    Score Breakdown: Age Component = <b>{plant_row['age_score']:.1f}</b> · 
                    Resource Component = <b>{plant_row['renewable_score']:.1f}</b> · 
                    Grid Priority = <b>{plant_row['grid_score']:.1f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_sv1, col_sv2 = st.columns([3, 2])

            with col_sv1:
                fig_solar = go.Figure()
                fig_solar.add_trace(go.Bar(
                    x=month_labels,
                    y=solar_vals_clean,
                    name="Solar Irradiance",
                    marker=dict(
                        color=solar_vals_clean,
                        colorscale=[[0, "#ff8c00"], [0.5, "#ffb800"], [1, "#ffe066"]],
                        showscale=False,
                    ),
                    text=[f"{v:.2f}" for v in solar_vals_clean],
                    textposition="outside",
                    textfont=dict(color="#ffb800", size=9),
                ))
                solar_mean_val = np.mean(solar_vals_clean)
                fig_solar.add_hline(
                    y=solar_mean_val,
                    line_dash="dash",
                    line_color="#00d4ff",
                    annotation_text=f"Annual Mean: {solar_mean_val:.2f}",
                    annotation_font_color="#00d4ff",
                )
                fig_solar.update_layout(
                    title=dict(text="12-Month Solar Irradiance Profile (kWh/m²/day)", font=dict(color="#00d4ff", size=14)),
                    paper_bgcolor="#0a0e1a",
                    plot_bgcolor="#0d1f35",
                    xaxis=dict(color="#7ba3c8", gridcolor="#1e3a5f"),
                    yaxis=dict(color="#7ba3c8", gridcolor="#1e3a5f", title="kWh/m²/day"),
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=360,
                    showlegend=False,
                )
                st.plotly_chart(fig_solar, use_container_width=True)

            with col_sv2:
                wind_pct = min(wind_speed / 15 * 100, 100)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=wind_speed,
                    number=dict(suffix=" m/s", font=dict(color="#00ff9d", size=36)),
                    title=dict(text="Wind Speed @ 100m", font=dict(color="#00d4ff", size=14)),
                    gauge=dict(
                        axis=dict(range=[0, 15], tickcolor="#7ba3c8", tickfont=dict(color="#7ba3c8")),
                        bar=dict(color="#00ff9d", thickness=0.25),
                        bgcolor="#0d1f35",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 4], color="#0d1f35"),
                            dict(range=[4, 7], color="#1e3a5f"),
                            dict(range=[7, 10], color="#0a3a5f"),
                            dict(range=[10, 15], color="#003366"),
                        ],
                        threshold=dict(line=dict(color="#ffb800", width=3), thickness=0.75, value=7.0),
                    ),
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="#0a0e1a",
                    font=dict(color="#7ba3c8"),
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=0),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                mix_details = {
                    "Hybrid Candidate": {
                        "icon": "⚡",
                        "color": "#00d4ff",
                        "title": "Hybrid Solar + Wind",
                        "desc": "Both solar and wind resources are above the 75th percentile for this region. Deploy a co-located Solar-Wind hybrid system to maximize capacity factor and minimize grid intermittency.",
                        "solar_share": 55,
                        "wind_share": 45,
                    },
                    "Solar": {
                        "icon": "☀️",
                        "color": "#ffb800",
                        "title": "Utility-Scale Solar PV",
                        "desc": "Strong solar irradiance makes this site ideal for a large-scale photovoltaic installation. Consider pairing with BESS (Battery Energy Storage System) for 24-hour reliability.",
                        "solar_share": 100,
                        "wind_share": 0,
                    },
                    "Wind": {
                        "icon": "💨",
                        "color": "#00ff9d",
                        "title": "Wind Farm",
                        "desc": "High wind speeds at 100m hub height indicate strong wind resource. A wind farm installation would effectively replace this plant's capacity with clean energy.",
                        "solar_share": 0,
                        "wind_share": 100,
                    },
                }
                m = mix_details.get(mix, mix_details["Solar"])
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#0d1f35,#0a1628);border:1px solid {m['color']}55;border-radius:12px;padding:18px;margin-top:8px;">
                    <div style="font-size:1.1rem;font-weight:700;color:{m['color']};margin-bottom:10px;">{m['icon']} Recommended: {m['title']}</div>
                    <div style="color:#a0b8cc;font-size:0.82rem;line-height:1.7;">{m['desc']}</div>
                    <div style="margin-top:14px;">
                        <div style="color:#7ba3c8;font-size:0.75rem;margin-bottom:6px;">REPLACEMENT CAPACITY TARGET</div>
                        <div style="font-size:1.4rem;font-weight:700;color:#fff;">{cap_mw:.0f} MW</div>
                        {'<div style="color:#a0b8cc;font-size:0.78rem;margin-top:4px;">☀️ ' + str(m['solar_share']) + '% Solar · 💨 ' + str(m['wind_share']) + '% Wind</div>' if m['solar_share'] and m['wind_share'] else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Nearby Plant Comparison</div>', unsafe_allow_html=True)
            nearby = fossil_df[
                (fossil_df["state_name"] == state) &
                (fossil_df["name"] != selected_plant)
            ].sort_values("transition_score", ascending=False).head(5)

            if not nearby.empty:
                comp_data = nearby[["name", "primary_fuel", "capacity_mw", "transition_score", "classification", "recommended_mix"]].copy()
                comp_data.columns = ["Plant Name", "Fuel", "Capacity (MW)", "Transition Score", "Classification", "Recommended Mix"]
                comp_data["Capacity (MW)"] = comp_data["Capacity (MW)"].round(0).astype(int)
                comp_data["Transition Score"] = comp_data["Transition Score"].round(1)
                st.dataframe(comp_data, use_container_width=True, hide_index=True)
            else:
                st.info("No other fossil plants found in the same state.")

@st.cache_resource
def load_qa_engine():
    qa_df = pd.read_csv("qa_data.csv")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(qa_df["question"].str.lower())
    return qa_df, vectorizer, tfidf_matrix

def find_best_answer(query, qa_df, vectorizer, tfidf_matrix, threshold=0.12):
    q_vec = vectorizer.transform([query.lower()])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    best_idx = scores.argmax()
    best_score = scores[best_idx]
    if best_score < threshold:
        return None, best_score
    return qa_df.iloc[best_idx]["answer"], best_score

def get_live_plant_answer(df, query):
    fossil_df = df[df["primary_fuel"].isin(FOSSIL_FUELS)].copy()
    query_words = query.lower().split()
    best_match = None
    best_len = 0
    for _, row in fossil_df.iterrows():
        plant_words = row["name"].lower().split()
        common = sum(1 for w in query_words if any(w in pw or pw in w for pw in plant_words))
        if common > best_len and common >= 2:
            best_len = common
            best_match = row
    if best_match is not None:
        row = best_match
        solar_cols = [f"solar_m{str(i).zfill(2)}" for i in range(1, 13)]
        solar_vals = [row.get(c, 0) for c in solar_cols]
        solar_mean = float(np.mean([v for v in solar_vals if pd.notna(v)]))
        age_note = ""
        cy = row.get("commissioning_year")
        if pd.notna(cy):
            age = CURRENT_YEAR - int(cy)
            age_note = f"commissioned in {int(cy)} ({age} years old)"
        else:
            assumed = 30 if row["primary_fuel"] == "Coal" else 20
            age_note = f"commissioning year unknown — assumed {assumed} years old"
        mix_desc = {
            "Hybrid Candidate": "a Hybrid Candidate — both solar and wind are strong here, ideal for co-located generation",
            "Solar": "Solar-primary — excellent solar irradiance makes utility-scale PV the best replacement",
            "Wind": "Wind-primary — strong winds at 100m make a wind farm the optimal replacement",
        }.get(row["recommended_mix"], row["recommended_mix"])
        urgency = {
            "Urgent Replacement": "🔴 URGENT — top priority for immediate retirement planning",
            "Moderate Priority": "🟡 MODERATE — significant retirement case, not immediately critical",
            "Low Priority": "🟢 LOW PRIORITY — lower urgency based on current scoring",
        }.get(row["classification"], row["classification"])
        return (
            f"\U0001f4ca **{row['name']}**\n\n"
            f"\u2022 **Fuel:** {row['primary_fuel']} | **Capacity:** {row['capacity_mw']:.0f} MW\n"
            f"\u2022 **State:** {row['state_name']} | **Age:** {age_note}\n"
            f"\u2022 **Transition Score:** {row['transition_score']:.1f}/100 \u2014 {urgency}\n"
            f"\u2022 **Score Breakdown:** Age {row['age_score']:.1f} + Resource {row['renewable_score']:.1f} + Grid {row['grid_score']:.1f}\n"
            f"\u2022 **Recommended Mix:** {mix_desc}\n"
            f"\u2022 **Solar Potential:** {solar_mean:.2f} kWh/m\u00b2/day annual mean\n"
            f"\u2022 **Wind Speed @ 100m:** {row.get('wind_speed_100m', 'N/A')} m/s"
        )
    return None

def get_live_state_answer(df, fossil_ratio_df, query):
    query_lower = query.lower()
    state_keywords = {
        "chhattisgarh": "Chhattisgarh", "jharkhand": "Jharkhand", "bihar": "Bihar",
        "madhya pradesh": "Madhya Pradesh", "mp": "Madhya Pradesh", "uttar pradesh": "Uttar Pradesh",
        "up": "Uttar Pradesh", "rajasthan": "Rajasthan", "gujarat": "Gujarat",
        "maharashtra": "Maharashtra", "karnataka": "Karnataka", "tamil nadu": "Tamil Nadu",
        "andhra pradesh": "Andhra Pradesh", "telangana": "Telangana", "west bengal": "West Bengal",
        "odisha": "Odisha", "punjab": "Punjab", "haryana": "Haryana", "kerala": "Kerala",
        "assam": "Assam", "himachal": "Himachal Pradesh",
    }
    matched_state = None
    for keyword, state_name in state_keywords.items():
        if keyword in query_lower:
            matched_state = state_name
            break
    if matched_state:
        fossil_df = df[df["primary_fuel"].isin(FOSSIL_FUELS)].copy()
        state_plants = fossil_df[fossil_df["state_name"] == matched_state]
        state_row = fossil_ratio_df[fossil_ratio_df["state_name"] == matched_state]
        if not state_row.empty:
            fr = state_row.iloc[0]
            fossil_pct = fr["fossil_ratio"] * 100
            top3 = state_plants.nlargest(3, "transition_score")[["name", "transition_score", "recommended_mix"]]
            top3_str = "\n".join([f"  {i+1}. {r['name']} \u2014 Score {r['transition_score']:.1f} ({r['recommended_mix']})" for i, (_, r) in enumerate(top3.iterrows())])
            urgent = (state_plants["transition_score"] >= 70).sum()
            return (
                f"\U0001f4cd **{matched_state} \u2014 State Energy Profile**\n\n"
                f"\u2022 **Fossil Grid Share:** {fossil_pct:.1f}% fossil / {100-fossil_pct:.1f}% clean\n"
                f"\u2022 **Total Fossil Capacity:** {fr['Fossil']:,.0f} MW | **Clean Capacity:** {fr['Clean']:,.0f} MW\n"
                f"\u2022 **Coal Plants in State:** {len(state_plants)} | **Urgent Replacements:** {urgent}\n\n"
                f"\U0001f3ed **Top 3 Priority Plants to Retire:**\n{top3_str}"
            )
    return None

def build_system_prompt(df, fossil_ratio_df):
    fossil_df = df[df["primary_fuel"].isin(FOSSIL_FUELS)]
    top_urgent = fossil_df.nlargest(5, "transition_score")[["name", "primary_fuel", "capacity_mw", "transition_score", "state_name", "recommended_mix"]].to_string(index=False)
    state_summary = fossil_ratio_df.sort_values("fossil_ratio", ascending=False).head(8)[["state_name", "fossil_ratio", "Fossil", "Clean"]].to_string(index=False)
    total_fossil_mw = fossil_df["capacity_mw"].sum()
    urgent_count = (fossil_df["transition_score"] >= 70).sum()
    hybrid_count = (fossil_df["recommended_mix"] == "Hybrid Candidate").sum()
    solar_count = (fossil_df["recommended_mix"] == "Solar").sum()
    wind_count = (fossil_df["recommended_mix"] == "Wind").sum()

    return f"""You are VayuAkra AI, an expert energy transition analyst embedded inside the VayuAkra dashboard — India's AI-driven fossil-to-renewable transition intelligence platform.

PLATFORM CONTEXT:
- VayuAkra analyses {len(df):,} power plants across India, of which {len(fossil_df)} are fossil-fuel plants (Coal, Gas, Oil)
- Total fossil capacity under analysis: {total_fossil_mw:,.0f} MW
- Urgent replacement candidates (score ≥ 70): {urgent_count} plants
- Hybrid Candidates (high solar + high wind): {hybrid_count} plants
- Solar-primary recommendations: {solar_count} plants
- Wind-primary recommendations: {wind_count} plants

SCORING SYSTEM (0-100, higher = more urgent to replace):
- Age Component (40%): Older plants score higher. Missing commissioning year → assumes 30 yrs for Coal, 20 yrs for Gas
- Renewable Resource (40%): Mean of 12-month solar irradiance bands + wind speed at 100m, both normalised nationally
- Regional Grid Priority (20%): States with high fossil dependency get elevated scores
- Classifications: Urgent ≥70 | Moderate 40-69 | Low Priority <40 | Already Clean (renewables)
- Hybrid Candidate: Plant site exceeds 75th percentile for BOTH solar AND wind

TOP 5 MOST URGENT PLANTS:
{top_urgent}

STATE FOSSIL DEPENDENCY (Top 8 by fossil ratio):
{state_summary}

YOUR BEHAVIOUR:
- Answer questions about any plant, state, score, renewable mix, or transition strategy in the dataset
- Be concise but insightful — bullet points for comparisons, prose for explanations
- When asked about a specific plant, give its score breakdown, recommended mix, and strategic context
- When asked about a state, explain its fossil ratio, top plants to retire, and renewable potential
- You can explain what Solar, Wind, and Hybrid Candidate classifications mean for site planning
- Use emojis sparingly for readability (☀️ 💨 ⚡ 🏭 📊)
- If asked something outside the energy/India context, politely redirect to VayuAkra topics
- Never make up plant names or scores not in the dataset; say you don't have that data if unsure
"""

def get_plant_context(df, query):
    fossil_df = df[df["primary_fuel"].isin(FOSSIL_FUELS)].copy()
    query_lower = query.lower()
    matched = fossil_df[fossil_df["name"].str.lower().str.contains(query_lower, na=False)]
    if not matched.empty:
        row = matched.iloc[0]
        solar_cols = [f"solar_m{str(i).zfill(2)}" for i in range(1, 13)]
        solar_vals = [row.get(c, 0) for c in solar_cols]
        solar_mean = float(np.mean([v for v in solar_vals if pd.notna(v)]))
        return f"""\n\nDETAILED PLANT DATA for {row['name']}:
- Fuel: {row['primary_fuel']} | Capacity: {row['capacity_mw']:.0f} MW
- State: {row['state_name']} | Transition Score: {row['transition_score']:.1f}/100
- Classification: {row['classification']} | Recommended Mix: {row['recommended_mix']}
- Age Score: {row['age_score']:.1f} | Resource Score: {row['renewable_score']:.1f} | Grid Score: {row['grid_score']:.1f}
- Solar Mean: {solar_mean:.2f} kWh/m²/day | Wind @ 100m: {row.get('wind_speed_100m', 'N/A')} m/s
- Commissioning Year: {row.get('commissioning_year', 'Unknown')}
- Location: {row['latitude']:.2f}°N, {row['longitude']:.2f}°E"""
    return ""

def render_chatbot_tab(df, fossil_ratio_df):
    st.markdown('<div class="section-header">⚡ VayuAkra AI — Your Energy Transition Analyst</div>', unsafe_allow_html=True)

    qa_df, vectorizer, tfidf_matrix = load_qa_engine()

    col_intro, col_stats = st.columns([3, 1])
    with col_intro:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0d1f35,#0a1628);border:1px solid #1e3a5f;border-radius:12px;padding:16px 20px;margin-bottom:16px;">
            <div style="color:#00d4ff;font-weight:700;font-size:1rem;margin-bottom:6px;">🤖 Ask me anything about India's energy transition</div>
            <div style="color:#7ba3c8;font-size:0.83rem;line-height:1.7;">
                I know every plant score renewable potential and replacement strategy. Ask about specific plants states scoring logic or renewable mixes. Type a plant or state name for live data!
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        fossil_df_c = df[df["primary_fuel"].isin(FOSSIL_FUELS)]
        urgent = (fossil_df_c["transition_score"] >= 70).sum()
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a0a0a,#2a0a0a);border:1px solid #ff3b3b44;border-radius:12px;padding:14px;text-align:center;">
            <div style="font-size:1.8rem;font-weight:700;color:#ff6b6b;">{urgent}</div>
            <div style="font-size:0.72rem;color:#ff9999;text-transform:uppercase;letter-spacing:1px;">Urgent Plants</div>
        </div>
        """, unsafe_allow_html=True)

    if "vayuakra_messages" not in st.session_state:
        st.session_state.vayuakra_messages = []

    suggestions = [
        "🏭 Which plant needs replacement most urgently?",
        "☀️ What is a Hybrid Candidate?",
        "📊 Which state has the worst fossil dependency?",
        "💨 How is the Transition Score calculated?",
        "⚡ What does Urgent Replacement mean?",
        "🔴 Which region has the best hybrid potential?",
    ]

    if not st.session_state.vayuakra_messages:
        st.markdown("**Quick questions — click to ask:**")
        cols = st.columns(3)
        for i, sug in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.vayuakra_messages.append({"role": "user", "content": sug})
                    st.rerun()

    chat_html = '<div class="chat-container">'
    for msg in st.session_state.vayuakra_messages:
        if msg["role"] == "user":
            chat_html += f'''<div class="chat-msg-user"><b>You</b><br>{msg["content"]}</div>'''
        else:
            formatted = msg["content"].replace("\n", "<br>").replace("**", "<b>", 1)
            i = 0
            result = ""
            bold_open = False
            for ch in msg["content"]:
                pass
            display_text = msg["content"].replace("\n", "<br>")
            chat_html += f'''<div class="chat-msg-ai"><span class="chat-avatar-ai">⚡</span> <b>VayuAkra AI</b><br>{display_text}</div>'''
    chat_html += "</div>"

    if st.session_state.vayuakra_messages:
        st.markdown(chat_html, unsafe_allow_html=True)

    col_input, col_btn, col_clear = st.columns([7, 1, 1])
    with col_input:
        user_input = st.text_input(
            "Ask VayuAkra AI...",
            placeholder="e.g. Tell me about Korba Power Plant  |  Which state is worst?  |  What is a hybrid candidate?",
            label_visibility="collapsed",
            key="chat_input"
        )
    with col_btn:
        send = st.button("Send ➤", use_container_width=True, type="primary")
    with col_clear:
        if st.button("Clear 🗑", use_container_width=True):
            st.session_state.vayuakra_messages = []
            st.rerun()

    if send and user_input.strip():
        query = user_input.strip()
        st.session_state.vayuakra_messages.append({"role": "user", "content": query})

        plant_answer = get_live_plant_answer(df, query)
        if plant_answer:
            reply = plant_answer
        else:
            state_answer = get_live_state_answer(df, fossil_ratio_df, query)
            if state_answer:
                reply = state_answer
            else:
                answer, score = find_best_answer(query, qa_df, vectorizer, tfidf_matrix)
                if answer:
                    reply = f"💡 {answer}"
                else:
                    fossil_df_r = df[df["primary_fuel"].isin(FOSSIL_FUELS)]
                    top_plant = fossil_df_r.nlargest(1, "transition_score").iloc[0]
                    reply = (
                        f"I didn't quite catch that — try asking about a specific plant name state or topic like scoring hybrid candidates or renewable mixes!\n\n"
                        f"💡 Quick fact: The most urgent plant right now is **{top_plant['name']}** in {top_plant['state_name']} "
                        f"with a Transition Score of {top_plant['transition_score']:.1f}/100."
                    )

        st.session_state.vayuakra_messages.append({"role": "assistant", "content": reply})
        st.rerun()

main()
