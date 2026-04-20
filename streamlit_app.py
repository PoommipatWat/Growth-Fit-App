import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import re

st.set_page_config(page_title="Growth Fit", layout="wide")
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

# สลับ Modified Gompertz ขึ้นมาเป็นตัวแรก เพื่อให้เป็น Default
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

# ── Sidebar: input ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📥 ใส่ข้อมูล")
    title_input = st.text_input("ชื่อกราฟ", value="Growth Fit")
    model_choice = st.selectbox("โมเดล", options=list(MODEL_DEFS.keys()), index=0)
    x_input = st.text_area("Time [h]  (คั่นด้วย , หรือ space หรือ Enter)", height=100, placeholder="0 0.5 1 1.5 2 ...")
    y_input = st.text_area("OD / Signal", height=100, placeholder="0.097 0.100 0.150 ...")

# ── Main ───────────────────────────────────────────────────────────────────────
x = parse_values(x_input)
y = parse_values(y_input)

if len(x) == 0 or len(y) == 0:
    st.info("👈 โปรดใส่ข้อมูล Time และ OD ในแถบด้านซ้ายเพื่อแสดงกราฟ")
    st.stop()
elif len(x) != len(y):
    st.warning(f"⚠️ จำนวนข้อมูลไม่เท่ากัน: x มี {len(x)} ตัว, y มี {len(y)} ตัว")
    st.stop()
elif len(x) < 4:
    st.warning("⚠️ โปรดใส่ข้อมูลอย่างน้อย 4 คู่ เพื่อให้เพียงพอต่อการทำ Curve Fitting")
    st.stop()

mdef = MODEL_DEFS[model_choice]
func = mdef["func"]

p0 = mdef["p0_fn"](x, y)
bounds = (mdef["bounds_low"], mdef["bounds_high"])

try:
    popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=500000)
except Exception as e:
    st.error(f"❌ Fit ไม่สำเร็จ: ข้อมูลอาจแกว่งเกินไป หรือไม่เหมาะกับโมเดล (Error: {e})")
    st.stop()

y_pred_fit = func(x, *popt)
r2, rmse, aic, bic = calc_advanced_metrics(y, y_pred_fit, mdef["n_params"])

# ── Dense curve (คำนวณเส้นสมูท) ────────────────────────────────────────────────
xd = np.linspace(x.min(), x.max(), 5000)
yd_fit = func(xd, *popt)

dy_fit = np.gradient(yd_fit, xd)
idx = np.argmax(dy_fit)
x_ms, slope_val, y_ms = xd[idx], dy_fit[idx], yd_fit[idx]

span = (x.max() - x.min()) * 0.25
x_tan = np.linspace(x_ms - span, x_ms + span, 300)

bot_disp = popt[0]
top_disp = popt[1]
y_tan = slope_val * (x_tan - x_ms) + y_ms

x_bot_intersect = x_ms - (y_ms - bot_disp) / slope_val if slope_val != 0 else x_ms

# ── คำนวณ Fixed Scale เพื่อไม่ให้กราฟขยับเวลาเปลี่ยนโมเดล ──
x_pad = (x.max() - x.min()) * 0.05
y_pad = (y.max() - y.min()) * 0.20
fixed_x_range = [x.min() - x_pad, x.max() + x_pad]
fixed_y_range = [y.min() - y_pad, y.max() + y_pad]

# ── UI Metrics ────────────────────────────────────────────────────────────
st.subheader(f"ผลลัพธ์: **{model_choice}**")
cols = st.columns(5)
cols[0].metric("R²", f"{r2:.6f}")
cols[1].metric("RMSE", f"{rmse:.6f}")
cols[2].metric("AIC", f"{aic:.2f}")
cols[3].metric("BIC", f"{bic:.2f}")
cols[4].metric("Max Slope", f"{slope_val:.5f} /h", f"at t = {x_ms:.4f} h")

st.write("**Parameters:**")
params_df = {pn: [f"{pv:.6f}"] for pn, pv in zip(mdef["param_names"], popt)}
st.table(params_df)

st.divider()

# ── Chart Plotly ────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.75, 0.25],
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=(title_input, "Growth Rate (d/dt)"),
)

# Raw data
fig.add_trace(go.Scatter(
    x=x, y=y, mode='markers', name='Raw data',
    marker=dict(color='steelblue', size=8, opacity=0.8),
    hovertemplate='t = %{x:.3f} h<br>OD = %{y:.6f}<extra>Raw</extra>',
), row=1, col=1)

# Fit curve 
fig.add_trace(go.Scatter(
    x=xd, y=yd_fit, mode='lines',
    name=f'{model_choice} fit',
    line=dict(color='tomato', width=2.5),
    hovertemplate='t = %{x:.3f} h<br>OD = %{y:.6f}<extra>Fit</extra>',
), row=1, col=1)

# Tangent
fig.add_trace(go.Scatter(
    x=x_tan, y=y_tan, mode='lines',
    name=f'Max slope',
    line=dict(color='darkorange', width=2, dash='dash'),
    hovertemplate='t = %{x:.3f} h<br>OD = %{y:.6f}<extra>Tangent</extra>',
), row=1, col=1)

# Asymptotes
fig.add_hline(y=top_disp, line=dict(color='green', dash='dash', width=1),
              annotation_text=f"Top={top_disp:.4f}",
              annotation_position="right", row=1, col=1)
fig.add_hline(y=bot_disp, line=dict(color='purple', dash='dash', width=1),
              annotation_text=f"Bot={bot_disp:.4f}",
              annotation_position="right", row=1, col=1)

# Vertical lines
fig.add_vline(x=x_bot_intersect, line=dict(color='royalblue', dash='dash', width=1.5),
              annotation_text=f"Lag = {x_bot_intersect:.2f} h", annotation_position="top left", row=1, col=1)
fig.add_vline(x=x_ms, line=dict(color='crimson', dash='dash', width=1.5), row=1, col=1)

# Derivative subplot
fig.add_trace(go.Scatter(
    x=xd, y=dy_fit, mode='lines', name='d/dt',
    line=dict(color='darkorange', width=2),
    hovertemplate='t = %{x:.3f} h<br>Rate = %{y:.5f}<extra>d/dt</extra>',
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=[x_ms], y=[slope_val], mode='markers',
    name=f'Peak Rate',
    marker=dict(color='red', size=10, symbol='circle'),
    hovertemplate=f't = {x_ms:.3f} h<br>Peak = {slope_val:.5f} /h<extra>Max rate</extra>',
), row=2, col=1)

fig.add_vline(x=x_bot_intersect, line=dict(color='royalblue', dash='dash', width=1), row=2, col=1)
fig.add_vline(x=x_ms, line=dict(color='crimson', dash='dash', width=1), row=2, col=1)

fig.update_layout(
    height=800,
    autosize=True,
    hovermode='closest',
    legend=dict(orientation='h', yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor='#fafafa',
    paper_bgcolor='white',
    margin=dict(l=40, r=40, t=80, b=40),
)

fig.update_xaxes(showgrid=True, gridcolor='lightgray', range=fixed_x_range)
fig.update_yaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(title_text="OD / Signal", range=fixed_y_range, row=1, col=1)
fig.update_xaxes(title_text="Time [h]", row=2, col=1)
fig.update_yaxes(title_text="d/dt", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)
