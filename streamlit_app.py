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

# ── จัดการ Session State ──
if 'export_mode' not in st.session_state: st.session_state['export_mode'] = "Responsive (Auto)"
if 'export_w' not in st.session_state: st.session_state['export_w'] = 1200
if 'export_h' not in st.session_state: st.session_state['export_h'] = 800
if 'scale_mode' not in st.session_state: st.session_state['scale_mode'] = "Auto (อัตโนมัติ)"

def sync_w_from_slider(): st.session_state['export_w'] = st.session_state['w_slider']
def sync_w_from_num(): st.session_state['export_w'] = st.session_state['w_num']
def sync_h_from_slider(): st.session_state['export_h'] = st.session_state['h_slider']
def sync_h_from_num(): st.session_state['export_h'] = st.session_state['h_num']

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📥 ข้อมูลและโมเดล")
    title_input = st.text_input("ชื่อกราฟ", value="Growth Curve Analysis")
    model_choice = st.selectbox("เลือกโมเดล (Default: Gompertz)", options=list(MODEL_DEFS.keys()), index=0)
    x_input = st.text_area("Time [h]", height=100, placeholder="0 1 2 3...")
    y_input = st.text_area("OD / Signal", height=100, placeholder="0.1 0.2 0.5...")

    x_raw = parse_values(x_input)
    y_raw = parse_values(y_input)

    st.divider()
    
    st.header("📏 ตั้งค่าขนาดภาพ (Export)")
    export_mode = st.radio("รูปแบบการแสดงผล", ["Responsive (Auto)", "Fixed Size (Manual)"], index=0 if st.session_state['export_mode'] == "Responsive (Auto)" else 1, key="radio_export_mode")
    st.session_state['export_mode'] = export_mode
    
    if export_mode == "Fixed Size (Manual)":
        st.markdown("**ความกว้างภาพ (Width px)**")
        w_col1, w_col2 = st.columns([3, 1])
        w_col1.slider("W_Slider", 600, 2500, value=st.session_state['export_w'], key="w_slider", on_change=sync_w_from_slider, label_visibility="collapsed")
        w_col2.number_input("W_Num", 600, 2500, value=st.session_state['export_w'], key="w_num", step=10, on_change=sync_w_from_num, label_visibility="collapsed")

        st.markdown("**ความสูงภาพ (Height px)**")
        h_col1, h_col2 = st.columns([3, 1])
        h_col1.slider("H_Slider", 400, 1500, value=st.session_state['export_h'], key="h_slider", on_change=sync_h_from_slider, label_visibility="collapsed")
        h_col2.number_input("H_Num", 400, 1500, value=st.session_state['export_h'], key="h_num", step=10, on_change=sync_h_from_num, label_visibility="collapsed")
        
        export_w, export_h, use_container_width = st.session_state['export_w'], st.session_state['export_h'], False
    else:
        export_w, export_h, use_container_width = None, 800, True

    st.divider()
    
    st.header("🎯 การตั้งค่าแกนกราฟ")
    scale_mode = st.radio("รูปแบบสเกลแกน", ["Auto (อัตโนมัติ)", "Manual (กำหนดเอง)"], index=0 if st.session_state['scale_mode'] == "Auto (อัตโนมัติ)" else 1, key="radio_scale_mode")
    st.session_state['scale_mode'] = scale_mode
    
    if scale_mode == "Manual (กำหนดเอง)" and len(x_raw) > 0 and len(y_raw) > 0:
        if 'x_min_val' not in st.session_state: st.session_state['x_min_val'] = float(min(x_raw))
        if 'x_max_val' not in st.session_state: st.session_state['x_max_val'] = float(max(x_raw) * 1.1)
        if 'y_min_val' not in st.session_state: st.session_state['y_min_val'] = float(min(y_raw))
        if 'y_max_val' not in st.session_state: st.session_state['y_max_val'] = float(max(y_raw) * 1.2)

        col1, col2 = st.columns(2)
        with col1:
            x_min_val = st.number_input("X Min", value=st.session_state['x_min_val'], key="x_min_val_input")
            y_min_val = st.number_input("Y Min", value=st.session_state['y_min_val'], key="y_min_val_input")
        with col2:
            x_max_val = st.number_input("X Max", value=st.session_state['x_max_val'], key="x_max_val_input")
            y_max_val = st.number_input("Y Max", value=st.session_state['y_max_val'], key="y_max_val_input")
            
        st.session_state['x_min_val'], st.session_state['x_max_val'] = x_min_val, x_max_val
        st.session_state['y_min_val'], st.session_state['y_max_val'] = y_min_val, y_max_val
        fixed_x_range, fixed_y_range = [x_min_val, x_max_val], [y_min_val, y_max_val]
    else:
        if len(x_raw) > 0 and len(y_raw) > 0:
            x_pad, y_pad = (max(x_raw) - min(x_raw)) * 0.05, (max(y_raw) - min(y_raw)) * 0.20
            fixed_x_range, fixed_y_range = [min(x_raw) - x_pad, max(x_raw) + x_pad], [min(y_raw) - y_pad, max(y_raw) + y_pad]
        else:
            fixed_x_range, fixed_y_range = None, None

    with st.expander("🧮 ดูสมการที่ใช้ (Equations)"):
        st.markdown("**1. Modified Gompertz**")
        st.latex(r"y(t) = bot + A \cdot e^{-e^{\frac{\mu_{max} \cdot e}{A}(\lambda - t) + 1}}")
        st.markdown("**2. Baranyi-Roberts**")
        st.latex(r"y(t) = bot + \mu_{max} A(t) - \ln \left( 1 + \frac{e^{\mu_{max} A(t)} - 1}{e^{top - bot}} \right)")
        st.markdown("**3. Weibull Growth**")
        st.latex(r"y(t) = bot + (top - bot) (1 - e^{-(\frac{t - \lambda}{scale})^{shape}})")

# ── Main ───────────────────────────────────────────────────────────────────────
if len(x_raw) >= 4 and len(x_raw) == len(y_raw):
    mdef = MODEL_DEFS[model_choice]
    try:
        popt, _ = curve_fit(mdef["func"], x_raw, y_raw, p0=mdef["p0_fn"](x_raw, y_raw), bounds=(mdef["bounds_low"], mdef["bounds_high"]), maxfev=500000)
        y_pred = mdef["func"](x_raw, *popt)
        r2, rmse, aic, bic = calc_advanced_metrics(y_raw, y_pred, mdef["n_params"])
        
        # ── UI Metrics ──
        st.subheader(f"ผลลัพธ์: **{model_choice}**")
        c = st.columns(5)
        c[0].metric("R²", f"{r2:.6f}")
        c[1].metric("RMSE", f"{rmse:.6f}")
        c[2].metric("AIC", f"{aic:.2f}")
        c[3].metric("BIC", f"{bic:.2f}")

        xd = np.linspace(x_raw.min(), x_raw.max(), 5000)
        yd = mdef["func"](xd, *popt)
        dy = np.gradient(yd, xd)
        idx = np.argmax(dy)
        x_ms, slope_val, y_ms = xd[idx], dy[idx], yd[idx]
        
        span = (x_raw.max() - x_raw.min()) * 0.25
        x_tan = np.linspace(x_ms - span, x_ms + span, 300)
        y_tan = slope_val * (x_tan - x_ms) + y_ms
        bot_disp, top_disp = popt[0], popt[1]
        x_bot_intersect = x_ms - (y_ms - bot_disp) / slope_val if slope_val != 0 else x_ms
        
        c[4].metric("Max Rate", f"{slope_val:.4f}/h")

        # ── ตาราง Parameters ──
        st.write("**Parameters Table:**")
        st.table({n: [f"{v:.6f}"] for n, v in zip(mdef["param_names"], popt)})

        # ── 📌 ส่วนสร้างสมการพร้อมใช้งาน (Fitted Equations) ──
        st.markdown("### 📌 สมการพร้อมนำไปใช้งาน (Fitted Equation)")
        st.markdown("คัดลอกสมการด้านล่างที่แทนค่าพารามิเตอร์ของข้อมูลคุณเรียบร้อยแล้วไปใช้ได้เลย")
        
        eq_bot, eq_top = popt[0], popt[1]
        eq_A = eq_top - eq_bot
        
        with st.container(border=True):
            if model_choice == "Modified Gompertz":
                eq_lag, eq_mu = popt[2], popt[3]
                coef = (eq_mu * np.e) / eq_A
                st.latex(rf"y(t) = {eq_bot:.4f} + {eq_A:.4f} \cdot \exp\left(-\exp\left[{coef:.4f} \cdot ({eq_lag:.4f} - t) + 1\right]\right)")
                st.caption("Copy-paste Python code:")
                st.code(f"y_t = {eq_bot:.4f} + {eq_A:.4f} * np.exp(-np.exp({coef:.4f} * ({eq_lag:.4f} - t) + 1))", language="python")

            elif model_choice == "Baranyi":
                eq_lag, eq_mu = popt[2], popt[3]
                eq_h0 = eq_mu * eq_lag
                st.latex(rf"A(t) = t + \frac{{1}}{{{eq_mu:.4f}}} \ln\left( e^{{-{eq_mu:.4f} t}} + e^{{-{eq_h0:.4f}}} - e^{{-{eq_mu:.4f}(t + {eq_lag:.4f})}} \right)")
                st.latex(rf"y(t) = {eq_bot:.4f} + {eq_mu:.4f} \cdot A(t) - \ln\left( 1 + \frac{{e^{{{eq_mu:.4f} \cdot A(t)}} - 1}}{{e^{{{eq_A:.4f}}}}} \right)")
                st.caption("Copy-paste Python code:")
                st.code(
                    f"A_t = t + (1 / {eq_mu:.4f}) * np.log(np.exp(-{eq_mu:.4f} * t) + np.exp(-{eq_h0:.4f}) - np.exp(-{eq_mu:.4f} * (t + {eq_lag:.4f})))\n"
                    f"y_t = {eq_bot:.4f} + {eq_mu:.4f} * A_t - np.log(1 + (np.exp({eq_mu:.4f} * A_t) - 1) / np.exp({eq_A:.4f}))", 
                    language="python"
                )

            elif model_choice == "Weibull":
                eq_lag, eq_scale, eq_shape = popt[2], popt[3], popt[4]
                st.latex(rf"y(t) = {eq_bot:.4f} + {eq_A:.4f} \left( 1 - \exp\left[-\left(\frac{{t - {eq_lag:.4f}}}{{{eq_scale:.4f}}}\right)^{{{eq_shape:.4f}}}\right] \right)")
                st.caption("Copy-paste Python code:")
                st.code(f"y_t = {eq_bot:.4f} + {eq_A:.4f} * (1 - np.exp(-((t - {eq_lag:.4f}) / {eq_scale:.4f})**{eq_shape:.4f}))", language="python")

        st.divider()

        # ── Chart Plotly ──
        fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(title_input, "Growth Rate (d/dt)"))
        
        fig.add_trace(go.Scatter(x=x_raw, y=y_raw, mode='markers', name='Raw Data', marker=dict(size=8, color='steelblue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=xd, y=yd, mode='lines', name=f'{model_choice} Fit', line=dict(color='tomato', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_tan, y=y_tan, mode='lines', name='Max slope', line=dict(color='darkorange', width=2, dash='dash'), hovertemplate='t = %{x:.3f} h<br>OD = %{y:.6f}<extra>Tangent</extra>'), row=1, col=1)

        fig.add_hline(y=top_disp, line=dict(dash='dash', color='green', width=1), annotation_text=f"Top={top_disp:.4f}", annotation_position="right", row=1, col=1)
        fig.add_hline(y=bot_disp, line=dict(dash='dash', color='purple', width=1), annotation_text=f"Bot={bot_disp:.4f}", annotation_position="right", row=1, col=1)

        fig.add_vline(x=x_bot_intersect, line=dict(color='royalblue', dash='dash', width=1.5), annotation_text=f"Lag = {x_bot_intersect:.2f} h", annotation_position="top left", row=1, col=1)
        fig.add_vline(x=x_ms, line=dict(color='crimson', dash='dash', width=1.5), row=1, col=1)

        fig.add_trace(go.Scatter(x=xd, y=dy, mode='lines', name='Rate', line=dict(color='darkorange')), row=2, col=1)
        fig.add_trace(go.Scatter(x=[x_ms], y=[slope_val], mode='markers', name='Peak Rate', marker=dict(color='red', size=10, symbol='circle'), hovertemplate=f't = {x_ms:.3f} h<br>Peak = {slope_val:.5f} /h<extra>Max rate</extra>'), row=2, col=1)
        fig.add_vline(x=x_bot_intersect, line=dict(color='royalblue', dash='dash', width=1), row=2, col=1)
        fig.add_vline(x=x_ms, line=dict(color='crimson', dash='dash', width=1), row=2, col=1)

        fig.update_layout(
            height=st.session_state['export_h'] if not use_container_width else 800, 
            width=st.session_state['export_w'] if not use_container_width else None, 
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(range=fixed_x_range)
        fig.update_yaxes(range=fixed_y_range, row=1, col=1)

        st.plotly_chart(fig, use_container_width=use_container_width)

    except Exception as e:
        st.error(f"Fit Error: {e}")
else:
    st.info("👈 กรุณาใส่ข้อมูล Time และ OD อย่างน้อย 4 จุดเพื่อเริ่มการวิเคราะห์")
