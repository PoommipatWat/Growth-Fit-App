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
    title_input = st.text_input("ชื่อกราฟ", value="Growth Curve Analysis")
    model_choice = st.selectbox("เลือกโมเดล (Default: Gompertz)", options=list(MODEL_DEFS.keys()), index=0)
    x_input = st.text_area("Time [h]", height=100, placeholder="0 1 2 3...")
    y_input = st.text_area("OD / Signal", height=100, placeholder="0.1 0.2 0.5...")

    st.divider()
    
    # ── 📏 ตั้งค่าขนาดภาพและสัดส่วน (Aspect Ratio Settings) ──
    st.header("📏 ตั้งค่าขนาดภาพ (Export)")
    export_mode = st.radio("รูปแบบการแสดงผล", ["Responsive (Auto)", "Fixed Size (Manual)"], help="Fixed Size จะช่วยให้เซฟภาพออกมาขนาดเท่ากันทุกเครื่อง")
    
    if export_mode == "Fixed Size (Manual)":
        export_w = st.slider("ความกว้างภาพ (Width px)", 600, 1920, 1000)
        export_h = st.slider("ความสูงภาพ (Height px)", 400, 1440, 700)
        use_container_width = False
    else:
        export_w = None
        export_h = 800 # Default height for auto mode
        use_container_width = True

    st.divider()
    
    with st.expander("🧮 ดูสมการที่ใช้ (Equations)"):
        st.markdown("**1. Modified Gompertz**")
        st.latex(r"y(t) = bot + A \cdot e^{-e^{\frac{\mu_{max} \cdot e}{A}(\lambda - t) + 1}}")
        st.markdown("**2. Baranyi-Roberts**")
        st.latex(r"y(t) = bot + \mu_{max} A(t) - \ln \left( 1 + \frac{e^{\mu_{max} A(t)} - 1}{e^{top - bot}} \right)")
        st.markdown("**3. Weibull Growth**")
        st.latex(r"y(t) = bot + (top - bot) (1 - e^{-(\frac{t - \lambda}{scale})^{shape}})")

# ── Main ───────────────────────────────────────────────────────────────────────
x_raw = parse_values(x_input)
y_raw = parse_values(y_input)

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

        # การคำนวณเส้นสัมผัสและพารามิเตอร์
        xd = np.linspace(x_raw.min(), x_raw.max(), 5000)
        yd = mdef["func"](xd, *popt)
        dy = np.gradient(yd, xd)
        idx = np.argmax(dy)
        x_ms, slope_val, y_ms = xd[idx], dy[idx], yd[idx]
        c[4].metric("Max Rate", f"{slope_val:.4f}/h")

        st.write("**Parameters Table:**")
        st.table({n: [f"{v:.6f}"] for n, v in zip(mdef["param_names"], popt)})

        # ── Chart Plotly ──
        fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(title_input, "Growth Rate (d/dt)"))
        
        fig.add_trace(go.Scatter(x=x_raw, y=y_raw, mode='markers', name='Raw Data', marker=dict(size=8, color='steelblue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=xd, y=yd, mode='lines', name='Fit Line', line=dict(color='tomato', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=xd, y=dy, mode='lines', name='Rate', line=dict(color='darkorange')), row=2, col=1)

        # เส้นกำกับและสเกล
        fig.add_hline(y=popt[1], line=dict(dash='dash', color='green', width=1), annotation_text="Top", row=1, col=1)
        fig.add_hline(y=popt[0], line=dict(dash='dash', color='purple', width=1), annotation_text="Bot", row=1, col=1)

        # ล็อกสเกลตามโหมดที่เลือก
        fig.update_layout(
            height=export_h, 
            width=export_w, 
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # คงสเกลแกนให้อัตโนมัติเวลาเปลี่ยนโมเดล
        x_pad = (x_raw.max() - x_raw.min()) * 0.05
        y_pad = (y_raw.max() - y_raw.min()) * 0.15
        fig.update_xaxes(range=[x_raw.min() - x_pad, x_raw.max() + x_pad])
        fig.update_yaxes(range=[y_raw.min() - y_pad, y_raw.max() + y_pad], row=1, col=1)

        st.plotly_chart(fig, use_container_width=use_container_width)

    except Exception as e:
        st.error(f"Fit Error: {e}")
else:
    st.info("👈 กรุณาใส่ข้อมูล Time และ OD อย่างน้อย 4 จุดเพื่อเริ่มการวิเคราะห์")
