"""
CatTracker – Streamlit Dashboard
"""
import os
import json
import requests
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
LAT_HOME = float(os.getenv("LAT_HOME", "48.8566"))
LON_HOME = float(os.getenv("LON_HOME", "2.3522"))

st.set_page_config(
    page_title="🐱 CatTracker",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #e94560;
    text-align: center;
}
.stButton>button {
    background: linear-gradient(90deg, #e94560, #0f3460);
    color: white; border: none; border-radius: 8px;
    font-family: 'Space Mono', monospace;
}
.stButton>button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def get_cats():
    try:
        r = requests.get(f"{BACKEND}/cats/", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def get_positions(chat_id: int, limit: int = 1000):
    try:
        r = requests.get(f"{BACKEND}/positions/{chat_id}?limit={limit}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def get_home_range(chat_id: int):
    try:
        r = requests.get(f"{BACKEND}/positions/{chat_id}/home-range", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_prediction(chat_id: int):
    try:
        r = requests.get(f"{BACKEND}/predict/{chat_id}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def upload_csv(chat_id: int, file_bytes, filename: str):
    try:
        r = requests.post(
            f"{BACKEND}/upload/{chat_id}",
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("🐱 CatTracker – Suivi GPS Félin")
st.caption("Monitoring GPS temps-réel • Domaine Vital • Prédiction LSTM")
st.divider()

# Sidebar
with st.sidebar:
    st.header("🐾 Configuration")

    cats = get_cats()
    if not cats:
        st.error("Backend inaccessible. Démarrez docker-compose.")
        st.stop()

    cat_options = {c["nom"]: c for c in cats}
    selected_name = st.selectbox("Sélectionner un chat", list(cat_options.keys()))
    cat = cat_options[selected_name]
    chat_id = cat["id"]

    st.markdown(f"""
    **Race :** {cat.get('race', '—')}
    **Couleur :** {cat.get('couleur', '—')}
    **Poids :** {cat.get('poids_kg', '—')} kg
    **Maison :** {cat['lat_home']:.4f}, {cat['lon_home']:.4f}
    """)

    st.divider()
    st.subheader("📤 Charger un CSV")
    uploaded = st.file_uploader("Fichier GPS (.csv)", type=["csv"])
    if uploaded and st.button("Importer & Entraîner"):
        with st.spinner("Insertion + entraînement LSTM..."):
            result = upload_csv(chat_id, uploaded.getvalue(), uploaded.name)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(
                f"✅ {result['inserted']} points insérés, "
                f"{result['skipped']} ignorés. "
                f"Réentraînement LSTM en cours..."
            )
        st.cache_data.clear()

    st.divider()
    nb_points = st.slider("Points affichés", 100, 5000, 1000, 100)
    show_heatmap   = st.toggle("Heatmap", value=True)
    show_homerange = st.toggle("Domaine vital", value=True)
    show_pred      = st.toggle("Prédiction LSTM", value=True)


# ── Fetch data ────────────────────────────────────────────────────────────────

positions_raw = get_positions(chat_id, nb_points)
if not positions_raw:
    st.warning("Aucune position disponible. Importez un CSV via le panneau latéral.")
    st.stop()

df = pd.DataFrame(positions_raw)
df["ts"] = pd.to_datetime(df["ts"])
df.sort_values("ts", inplace=True)
df.reset_index(drop=True, inplace=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("📍 Points", len(df))
with c2:
    avg_speed = df["vitesse_ms"].dropna().mean()
    st.metric("⚡ Vitesse moy.", f"{avg_speed:.2f} m/s" if not np.isnan(avg_speed) else "—")
with c3:
    max_dist = df["distance_home_m"].dropna().max()
    st.metric("🏠 Distance max", f"{max_dist:.0f} m" if not np.isnan(max_dist) else "—")
with c4:
    st.metric("🕐 Première pos.", df["ts"].min().strftime("%d/%m %H:%M"))
with c5:
    st.metric("🕐 Dernière pos.", df["ts"].max().strftime("%d/%m %H:%M"))

st.divider()

# ── Map + charts ──────────────────────────────────────────────────────────────

col_map, col_charts = st.columns([3, 2], gap="large")

with col_map:
    st.subheader("🗺️ Carte interactive")

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB dark_matter")

    # Trajectory
    coords = list(zip(df["latitude"].tolist(), df["longitude"].tolist()))
    folium.PolyLine(coords, color="#e94560", weight=2, opacity=0.7, tooltip="Trajectoire").add_to(m)

    # Start / End markers
    folium.CircleMarker(coords[0],  radius=6, color="#2ecc71", fill=True, tooltip="Départ").add_to(m)
    folium.CircleMarker(coords[-1], radius=6, color="#e94560", fill=True, tooltip="Arrivée").add_to(m)

    # Home marker
    folium.Marker(
        [cat["lat_home"], cat["lon_home"]],
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        tooltip="🏠 Maison",
    ).add_to(m)

    # Heatmap
    if show_heatmap:
        HeatMap(
            [[r["latitude"], r["longitude"]] for _, r in df.iterrows()],
            radius=12, blur=10, min_opacity=0.3, max_zoom=18,
        ).add_to(m)

    # Home range polygon
    if show_homerange:
        hr = get_home_range(chat_id)
        if hr:
            folium.GeoJson(
                hr["polygon_geojson"],
                style_function=lambda _: {
                    "fillColor": "#f39c12", "color": "#f39c12",
                    "weight": 2, "fillOpacity": 0.15,
                },
                tooltip=f"Domaine vital : {hr['area_km2']} km²",
            ).add_to(m)

    # LSTM prediction
    if show_pred:
        pred = get_prediction(chat_id)
        if pred:
            folium.Marker(
                [pred["predicted_latitude"], pred["predicted_longitude"]],
                icon=folium.Icon(color="purple", icon="question", prefix="fa"),
                tooltip=f"🔮 Prédiction LSTM\n({pred['predicted_latitude']:.5f}, {pred['predicted_longitude']:.5f})",
            ).add_to(m)

    st_folium(m, height=480, use_container_width=True)

    if show_homerange and hr:
        st.info(f"📐 **Domaine vital (MCP) :** {hr['area_km2']} km²  •  {hr['n_points']} points")

    if show_pred and pred:
        st.info(f"🔮 **Prédiction LSTM :** lat={pred['predicted_latitude']:.5f}, lon={pred['predicted_longitude']:.5f}")


with col_charts:
    st.subheader("📊 Analyses temporelles")

    # Speed over time
    df_speed = df.dropna(subset=["vitesse_ms"])
    if not df_speed.empty:
        fig_speed = px.line(
            df_speed, x="ts", y="vitesse_ms",
            labels={"ts": "Horodatage", "vitesse_ms": "Vitesse (m/s)"},
            title="Vitesse au fil du temps",
            color_discrete_sequence=["#e94560"],
        )
        fig_speed.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=220, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_speed, use_container_width=True)

    # Distance to home over time
    df_dist = df.dropna(subset=["distance_home_m"])
    if not df_dist.empty:
        fig_dist = px.area(
            df_dist, x="ts", y="distance_home_m",
            labels={"ts": "Horodatage", "distance_home_m": "Distance (m)"},
            title="Distance à la maison",
            color_discrete_sequence=["#0f3460"],
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=220, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Activity by hour
    df["hour"] = df["ts"].dt.hour
    hourly = df.groupby("hour")["vitesse_ms"].mean().reset_index()
    fig_hour = px.bar(
        hourly, x="hour", y="vitesse_ms",
        labels={"hour": "Heure", "vitesse_ms": "Vitesse moy. (m/s)"},
        title="Activité par heure",
        color_discrete_sequence=["#e94560"],
    )
    fig_hour.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=220, margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_hour, use_container_width=True)

# ── Raw data expander ─────────────────────────────────────────────────────────

with st.expander("🗃️ Données brutes"):
    st.dataframe(
        df[["ts", "latitude", "longitude", "vitesse_ms", "distance_home_m"]].tail(200),
        use_container_width=True,
    )
