import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import re

st.set_page_config(page_title="Growth Fit Pro", layout="wide")
st.title("📈 Growth Fit Pro")

# ── Models ─────────────────────────────────────────────────────────────────────

def weibull_growth(x, bot, top, lag, scale, shape):
    z = np.clip(x - lag, 0, None)
    return bot + (top - bot) * (1 - np.exp(-(z / scale) ** shape))

def modified_gompertz(x, bot, top, lag, mu_max):
    t = np.asarray(x)
    A = top - bot
    if A <= 0: A = 1e-9 
    z = np.clip(t - lag, 0, None)
    return bot + A * np.exp(-np.exp((mu_max * np.e / A) * (lag - t) + 1))

def baranyi(x, bot, top, lag, mu_max):
    t = np.asarray(x, dtype=float)
    h0 = mu_max * lag
    with np.errstate(over='ignore', invalid='ignore'):
        A_t = t + (1 / mu_max) * np.log(np.exp(-mu_max * t) + np.exp(-h0) - np.exp(-mu_max * t - h0))
        y = bot + mu_max * A_t - np.log(1 + (np.exp(mu_max * A_t) - 1) / np.exp(top - bot))
    return y

def parse_values(text):
    if not text.strip(): return np.array([])
    nums = re.split(r'[\s,\t]+', text.strip())
    return np.array([float(v) for v in nums if v])

def calc_advanced_metrics(y, y_pred, k):
    n = len(y)
    rss = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - rss / ss_tot if ss_tot != 0 else 0.0
    rmse = np.sqrt(rss / n)
    safe_rss = max(rss, 1e-15)
    aic = 2 * k + n * np.log(safe_rss / n)
    bic = k * np.log(n) + n * np.log(safe_rss / n)
    return r2, rmse, aic, bic

MODEL_DEFS = {
    "Modified Gompertz": {
        "func": modified_gompertz, "n_params": 4,
        "p0_fn": lambda x, y: [y.min(), y.max(), x[len(x)//4], 0.3],
        "bounds_low": [-np.inf, -np.inf, -np.inf, 0.001],
        "bounds_high": [np.inf, np.inf, np.inf, 20],
        "param_names": ["bot", "top", "lag", "mu_max"],
    },
    "Baranyi": {
        "func": baranyi, "n_params": 4,
        "p0_fn": lambda x, y: [y.min(), y.max(), x[len(x)//4], (y.max()-y.min())/(x.max()-x.min())],
        "bounds_low": [-np.inf, -np.inf, 0, 0.001],
        "bounds_high": [np.inf, np.inf, np.inf, 20],
        "param_names": ["bot", "top", "lag", "mu_max"],
    },
    "Weibull": {
        "func": weibull_growth, "n_params": 5,
        "p0_fn": lambda x, y: [y.min(), y.max(), x[len(x)//2], (x.max()-x.min())/6, 3],
        "bounds_low": [-np.inf, -np.inf, -np.inf, 0.001, 0.1],
        "bounds_high": [np.inf, np.inf, np.inf, np.inf, 20],
        "param_names": ["bot", "top", "lag", "scale", "shape"],
    },
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📥 ข้อมูลและโมเดล")
    title_input = st.text_input("ชื่อกราฟ", value="Growth Fit")
    model_choice = st.selectbox("โมเดล", options=list(MODEL_DEFS.keys()), index=0)
    x_input = st.text_area("Time [h]", height=100, placeholder="0 0.5 1 1.5 2 ...")
    y_input = st.text_area("OD / Signal", height=100, placeholder="0.097 0.100 0.150 ...")

    st.divider()
    
    # ── การตั้งค่าแกน (Axis Settings) ──
    st.header("📏 การตั้งค่าแกน")
    scale_mode = st.radio("รูปแบบสเกลแกน", ["Auto (อัตโนมัติ)", "Manual (กำหนดเอง)"])
    
    # ดึงค่า x, y มาคำนวณเบื้องต้นสำหรับ Default Manual
    x_raw = parse_values(x_input)
    y_raw = parse_values(y_input)
    
    if scale_mode == "Manual (กำหนดเอง)" and len(x_raw) > 0 and len(y_raw) > 0:
        col1, col2 = st.columns(2)
        with col1:
            x_min_val = st.number_input("X Min", value=float(min(x_raw)))
            y_min_val = st.number_input("Y Min", value=float(min(y_raw)))
        with col2:
            x_max_val = st.number_input("X Max", value=float(max(x_raw) * 1.1))
            y_max_val = st.number_input("Y Max", value=float(max(y_raw) * 1.2))
        fixed_x_range = [x_min_val, x_max_val]
        fixed_y_range = [y_min_val, y_max_val]
    else:
        # ค่า Auto เดิม
        if len(x_raw) > 0 and len(y_raw) > 0:
            x_pad = (max(x_raw) - min(x_raw)) * 0.05
            y_pad = (max(y_raw) - min(y_raw)) * 0.20
            fixed_x_range = [min(x_raw) - x_pad, max(x_raw) + x_pad]
            fixed_y_range = [min(y_raw) - y_pad, max(y_raw) + y_pad]
        else:
            fixed_x_range = None
            fixed_y_range = None

    with st.expander("🧮 ดูสมการ (Equations)"):
        st.markdown("**1. Modified Gompertz**")
        st.latex(r"y(t) = bot + A \cdot \exp\left(-\exp\left[\frac{\mu_{max} \cdot e}{A}(\lambda - t) + 1\right]\right)")
        st.divider()
        st.markdown("**2. Baranyi-Roberts**")
        st.latex(r"y(t) = bot + \mu_{max} A(t) - \ln \left( 1 + \frac{e^{\mu_{max} A(t)} - 1}{e^{top - bot}} \right)")

# ── Main ───────────────────────────────────────────────────────────────────────
if len(x_raw) >= 4 and len(x_raw) == len(y_raw):
    mdef = MODEL_DEFS[model_choice]
    try:
        popt, _ = curve_fit(mdef["func"], x_raw, y_raw, p0=mdef["p0_fn"](x_raw, y_raw), bounds=(mdef["bounds_low"], mdef["bounds_high"]), maxfev=500000)
        y_pred = mdef["func"](x_raw, *popt)
        r2, rmse, aic, bic = calc_advanced_metrics(y_raw, y_pred, mdef["n_params"])
        
        # UI Metrics
        st.subheader(f"ผลลัพธ์: **{model_choice}**")
        c = st.columns(5)
        c[0].metric("R²", f"{r2:.4f}")
        c[1].metric("RMSE", f"{rmse:.4f}")
        c[2].metric("AIC", f"{aic:.2f}")
        c[3].metric("BIC", f"{bic:.2f}")

        # กราฟและการคำนวณเส้นสัมผัส
        xd = np.linspace(x_raw.min(), x_raw.max(), 5000)
        yd = mdef["func"](xd, *popt)
        dy = np.gradient(yd, xd)
        idx = np.argmax(dy)
        x_ms, slope_val, y_ms = xd[idx], dy[idx], yd[idx]
        y_tan = slope_val * (np.linspace(x_ms-2, x_ms+2, 100) - x_ms) + y_ms
        c[4].metric("Max Slope", f"{slope_val:.4f}/h")

        st.write("**Parameters:**")
        st.table({n: [f"{v:.6f}"] for n, v in zip(mdef["param_names"], popt)})

        # ── Chart Plotly ──
        fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.04, subplot_titles=(title_input, "Growth Rate"))
        fig.add_trace(go.Scatter(x=x_raw, y=y_raw, mode='markers', name='Raw Data'), row=1, col=1)
        fig.add_trace(go.Scatter(x=xd, y=yd, mode='lines', name='Fit Line', line=dict(color='tomato')), row=1, col=1)
        fig.add_trace(go.Scatter(x=xd, y=dy, mode='lines', name='Rate', line=dict(color='darkorange')), row=2, col=1)

        # เพิ่มเส้น Asymptotes และ Tangent
        fig.add_hline(y=popt[1], line=dict(dash='dash', color='green'), row=1, col=1)
        fig.add_hline(y=popt[0], line=dict(dash='dash', color='purple'), row=1, col=1)

        fig.update_layout(height=800, template="plotly_white")
        fig.update_xaxes(range=fixed_x_range)
        fig.update_yaxes(range=fixed_y_range, row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Fit Error: {e}")
else:
    st.info("👈 กรุณาใส่ข้อมูลให้ครบถ้วนใน Sidebar")
