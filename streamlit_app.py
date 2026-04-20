import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

st.set_page_config(page_title="Growth Fit", layout="wide")
st.title("Growth Fit")

# ── Models ─────────────────────────────────────────────────────────────────────

def weibull_growth(x, bot, top, lag, scale, shape):
    """Weibull growth: bot + (top-bot)*(1 - exp(-((x-lag)/scale)^shape))"""
    z = np.clip(x - lag, 0, None)
    return bot + (top - bot) * (1 - np.exp(-(z / scale) ** shape))


def modified_gompertz(x, bot, top, lag, mu_max):
    """Modified Gompertz: bot + (top-bot)*exp(-b*exp(-c*(t-lag)))"""
    t = np.asarray(x)
    b = 6.0
    c = mu_max * np.e / b
    z = np.clip(t - lag, 0, None)
    return bot + (top - bot) * np.exp(-b * np.exp(-c * z))


def baranyi(x, log_bot, log_top, lag, mu_max, hill):
    """Baranyi–Roberts model — all in log10 space.
    y input and output are log10(signal)."""
    t = np.asarray(x, dtype=float)

    def safe_exp(v):
        return np.exp(np.clip(v, -700, 700))

    alpha  = safe_exp(-mu_max * (t - lag))
    alpha0 = safe_exp(-mu_max * lag)

    log_frac = np.log1p(alpha) - np.log1p(alpha0)

    A = np.where(
        t <= lag,
        (1.0 / mu_max) * log_frac,
        t + (1.0 / mu_max) * log_frac,
    )

    denom = (log_top - log_bot) * safe_exp(-mu_max * A)
    y_log = log_top + np.log10((1.0 + safe_exp(-mu_max * A)) / (1.0 + denom))
    return y_log


def parse_values(text):
    import re
    nums = re.split(r'[\s,\t]+', text.strip())
    return np.array([float(v) for v in nums if v])


def calc_metrics(y, y_pred):
    """R² and RMSE in the same space as y (linear or log10)."""
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    return r2, rmse


MODEL_DEFS = {
    "Weibull": {
        "func":       weibull_growth,
        "n_params":   5,
        "log_space":   False,
        "p0_fn": lambda x, y: [y.min(), y.max(), x[len(x)//2],
                                (x.max()-x.min())/6, 3],
        "bounds_low":  [-np.inf, -np.inf, -np.inf, 0.001, 0.1],
        "bounds_high": [ np.inf,  np.inf,  np.inf, np.inf, 20],
        "param_names": ["bot", "top", "lag", "scale", "shape"],
    },
    "Modified Gompertz": {
        "func":       modified_gompertz,
        "n_params":   4,
        "log_space":   False,
        "p0_fn": lambda x, y: [y.min(), y.max(), x[len(x)//4], 0.3],
        "bounds_low":  [-np.inf, -np.inf, -np.inf, 0.001],
        "bounds_high": [ np.inf,  np.inf,  np.inf, 20],
        "param_names": ["bot", "top", "lag", "mu_max"],
    },
    "Baranyi": {
        "func":       baranyi,
        "n_params":   5,
        "log_space":   True,   # y must be converted to log10 before fit
        "p0_fn": lambda x, y_log: [y_log.min(), y_log.max(),
                                    x[len(x)//4], 0.3, 1.0],
        "bounds_low":  [-10, -10, -np.inf, 0.001, 0.1],
        "bounds_high": [ 10,  10,  np.inf, 20, 20],
        "param_names": ["log_bot", "log_top", "lag", "mu_max", "hill"],
    },
}


# ── Sidebar: input ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📥 ใส่ข้อมูล")

    title_input = st.text_input("ชื่อกราฟ", value="Growth Fit")

    model_choice = st.selectbox(
        "โมเดล",
        options=list(MODEL_DEFS.keys()),
        index=0,
    )

    if model_choice == "Baranyi":
        st.caption("ℹ️ Baranyi ใช้ log10(y) — ค่า y ควรเป็น log10 หรือ OD ที่มีค่า > 0")

    x_input = st.text_area(
        "Time [h]  (คั่นด้วย , หรือ space หรือ Enter)",
        height=100,
        placeholder="0 0.5 1 1.5 2 ..."
    )
    y_input = st.text_area(
        "OD / Signal",
        height=100,
        placeholder="0.097 0.100 0.150 ..."
    )

    calc_btn = st.button("⚙️ Calculate", type="primary", use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────────────
if calc_btn:
    try:
        x = parse_values(x_input)
        y = parse_values(y_input)
    except Exception:
        st.error("❌ ตรวจสอบข้อมูล x, y — ต้องเป็นตัวเลขเท่านั้น")
        st.stop()

    if len(x) != len(y):
        st.error(f"❌ จำนวน x ({len(x)}) ≠ จำนวน y ({len(y)})")
        st.stop()

    if len(x) < 4:
        st.error("❌ ต้องมีข้อมูลอย่างน้อย 4 จุด")
        st.stop()

    if np.any(y <= 0):
        st.error("❌ ค่า y ต้องมากกว่า 0 สำหรับทุกโมเดล")
        st.stop()

    mdef  = MODEL_DEFS[model_choice]
    func  = mdef["func"]
    is_log = mdef["log_space"]

    # Transform y to log10 if model requires it
    if is_log:
        y_fit = np.log10(y)
    else:
        y_fit = y.copy()

    p0     = mdef["p0_fn"](x, y_fit)
    bounds = (mdef["bounds_low"], mdef["bounds_high"])

    try:
        popt, _ = curve_fit(func, x, y_fit, p0=p0,
                            bounds=bounds, maxfev=500000)
    except Exception as e:
        st.error(f"❌ Fit ไม่สำเร็จ ({model_choice}): {e}")
        st.stop()

    y_pred_fit = func(x, *popt)      # in fit-space (log10 for Baranyi)
    r2, rmse   = calc_metrics(y_fit, y_pred_fit)

    # ── Dense curve ─────────────────────────────────────────────────────────
    xd        = np.linspace(x.min(), x.max(), 5000)
    yd_fit    = func(xd, *popt)

    # Derivative always in fit-space
    dy_fit    = np.gradient(yd_fit, xd)
    idx       = np.argmax(dy_fit)
    x_ms, slope_val, y_ms = xd[idx], dy_fit[idx], yd_fit[idx]

    span  = (x.max() - x.min()) * 0.25
    x_tan = np.linspace(x_ms - span, x_ms + span, 300)
    y_tan = slope_val * (x_tan - x_ms) + y_ms

    bot_fit = popt[0]
    top_fit = popt[1]

    # Convert key values back to display space
    if is_log:
        bot_disp = 10 ** bot_fit
        top_disp = 10 ** top_fit
        y_ms_disp = 10 ** y_ms
    else:
        bot_disp  = bot_fit
        top_disp  = top_fit
        y_ms_disp = y_ms

    x_bot_intersect = x_ms - (y_ms_disp - bot_disp) / slope_val if slope_val != 0 else x_ms

    # ── Metrics ────────────────────────────────────────────────────────────
    st.subheader(f"ผลลัพธ์: **{model_choice}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",      f"{r2:.6f}")
    c2.metric("RMSE",    f"{rmse:.6f}")
    c3.metric("R²+RMSE", f"{r2 + rmse:.6f}")
    c4.metric("Max Slope", f"{slope_val:.5f} /h", f"at t = {x_ms:.4f} h")

    # Parameters
    params_df = {pn: [f"{pv:.6f}"] for pn, pv in zip(mdef["param_names"], popt)}
    st.write("**Parameters:**")
    st.table(params_df)

    st.divider()

    # ── Chart ──────────────────────────────────────────────────────────────
    yaxis_title = "log10(Signal)" if is_log else "Signal"

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(title_input, "d/dt"),
    )

    # Raw data — plot in fit-space
    fig.add_trace(go.Scatter(
        x=x, y=y_fit, mode='markers', name='Raw data',
        marker=dict(color='steelblue', size=6, opacity=0.6),
        hovertemplate='t = %{x:.3f} h<br>y = %{y:.6f}<extra>Raw data</extra>',
    ), row=1, col=1)

    # Fit curve
    fig.add_trace(go.Scatter(
        x=xd, y=yd_fit, mode='lines',
        name=f'{model_choice} fit  (R²={r2:.4f}  RMSE={rmse:.5f})',
        line=dict(color='tomato', width=2.5),
        hovertemplate='t = %{x:.3f} h<br>y = %{y:.6f}<extra>fit</extra>',
    ), row=1, col=1)

    # Tangent
    fig.add_trace(go.Scatter(
        x=x_tan, y=y_tan, mode='lines',
        name=f'Max slope = {slope_val:.4f} /h',
        line=dict(color='darkorange', width=2, dash='dash'),
        hovertemplate='t = %{x:.3f} h<br>y = %{y:.6f}<extra>Tangent</extra>',
    ), row=1, col=1)

    # Asymptotes
    fig.add_hline(y=top_fit, line=dict(color='green', dash='dash', width=1),
                  annotation_text=f"Top={top_fit:.4f}",
                  annotation_position="right", row=1, col=1)
    fig.add_hline(y=bot_fit, line=dict(color='purple', dash='dash', width=1),
                  annotation_text=f"Bot={bot_fit:.4f}",
                  annotation_position="right", row=1, col=1)

    # Vertical lines
    fig.add_vline(x=x_bot_intersect,
                  line=dict(color='royalblue', dash='dash', width=1.5),
                  annotation_text=f"Tangent∩Bot<br>t={x_bot_intersect:.2f} h",
                  annotation_position="top left")
    fig.add_vline(x=x_ms,
                  line=dict(color='crimson', dash='dash', width=1.5),
                  annotation_text=f"Max slope<br>t={x_ms:.2f} h",
                  annotation_position="top right")

    # Derivative subplot
    fig.add_trace(go.Scatter(
        x=xd, y=dy_fit, mode='lines', name='d/dt',
        line=dict(color='darkorange', width=2),
        hovertemplate='t = %{x:.3f} h<br>d/dt = %{y:.5f}<extra>d/dt</extra>',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[x_ms], y=[slope_val], mode='markers',
        name=f'Peak={slope_val:.4f}/h',
        marker=dict(color='red', size=10, symbol='circle'),
        hovertemplate=f't = {x_ms:.3f} h<br>Peak = {slope_val:.5f} /h<extra>Max slope</extra>',
    ), row=2, col=1)
    fig.add_vline(x=x_bot_intersect,
                  line=dict(color='royalblue', dash='dash', width=1))
    fig.add_vline(x=x_ms,
                  line=dict(color='crimson', dash='dash', width=1))

    fig.update_layout(
        height=1200,
        autosize=True,
        hovermode='closest',
        hoverdistance=30,
        spikedistance=-1,
        legend=dict(orientation='h', yanchor="bottom", y=1.02,
                     xanchor="right", x=1, bgcolor='rgba(255,255,255,0)'),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=80, b=40),
    )

    spike = dict(showspikes=True, spikemode='across', spikesnap='cursor',
                 spikecolor='gray', spikethickness=1, spikedash='dot')
    fig.update_xaxes(**spike)
    fig.update_yaxes(**spike)
    fig.update_xaxes(title_text="Time [h]", row=2, col=1, showticklabels=True)
    fig.update_yaxes(title_text=yaxis_title, row=1, col=1)
    fig.update_yaxes(title_text="d/dt",     row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 ใส่ข้อมูลในแถบซ้าย แล้วกด Calculate")
