import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

st.set_page_config(page_title="Growth Fit", layout="wide")
st.title("Growth Fit")

# ── Model ──────────────────────────────────────────────────────────────────────
def weibull_growth(x, bot, top, lag, scale, shape):
    z = np.clip(x - lag, 0, None)
    return bot + (top - bot) * (1 - np.exp(-(z / scale) ** shape))

def parse_values(text):
    import re
    nums = re.split(r'[\s,\t]+', text.strip())
    return np.array([float(v) for v in nums if v])

# ── Sidebar: input ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📥 ใส่ข้อมูล")

    title_input = st.text_input("ชื่อกราฟ", value="Growth Fit")

    x_input = st.text_area(
        "Time [h]  (คั่นด้วย , หรือ space หรือ Enter)",
        height=150,
        placeholder="0 0.5 1 1.5 2 ..."
    )
    y_input = st.text_area(
        "OD / Signal  (คั่นด้วย , หรือ space หรือ Enter)",
        height=150,
        placeholder="0.097 0.097 0.098 ..."
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

    if len(x) < 6:
        st.error("❌ ต้องมีข้อมูลอย่างน้อย 6 จุด")
        st.stop()

    try:
        p0 = [y.min(), y.max(), x[len(x)//2], (x.max()-x.min())/6, 3]
        bounds = ([-np.inf,-np.inf,-np.inf,0.001,0.1],
                  [ np.inf, np.inf, np.inf, np.inf, 20])
        popt, pcov = curve_fit(weibull_growth, x, y, p0=p0,
                               bounds=bounds, maxfev=200000)
    except Exception as e:
        st.error(f"❌ Fit ไม่สำเร็จ: {e}")
        st.stop()

    bot, top, lag, scale, shape = popt

    y_pred = weibull_growth(x, *popt)
    rmse   = np.sqrt(np.mean((y - y_pred)**2))

    xd  = np.linspace(x.min(), x.max(), 5000)
    yd  = weibull_growth(xd, *popt)
    dy  = np.gradient(yd, xd)
    idx = np.argmax(dy)
    x_ms, slope_val, y_ms = xd[idx], dy[idx], yd[idx]

    span  = (x.max() - x.min()) * 0.25
    x_tan = np.linspace(x_ms - span, x_ms + span, 300)
    y_tan = slope_val * (x_tan - x_ms) + y_ms

    x_bot_intersect = x_ms - (y_ms - bot) / slope_val

    # ── Metrics ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.6f}")
    col2.metric("Max Slope", f"{slope_val:.5f} /h", f"at t = {x_ms:.4f} h")
    col3.metric("Tangent ∩ Bot  (t)", f"{x_bot_intersect:.4f} h",
                f"y = {bot:.6f}")

    st.divider()

    # ── Plotly interactive chart ───────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(title_input, "d(Signal)/dt")
    )

    # Raw data
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers', name='Raw data',
        marker=dict(color='steelblue', size=6, opacity=0.6),
        hovertemplate='t = %{x:.3f} h<br>y = %{y:.6f}<extra>Raw data</extra>'
    ), row=1, col=1)

    # Fit curve
    fig.add_trace(go.Scatter(
        x=xd, y=yd, mode='lines',
        name=f'Weibull fit (RMSE={rmse:.5f})',
        line=dict(color='tomato', width=2.5),
        hovertemplate='t = %{x:.3f} h<br>y = %{y:.6f}<extra>Weibull fit</extra>'
    ), row=1, col=1)

    # Tangent line
    fig.add_trace(go.Scatter(
        x=x_tan, y=y_tan, mode='lines',
        name=f'Max slope = {slope_val:.4f} /h',
        line=dict(color='darkorange', width=2, dash='dash'),
        hovertemplate='t = %{x:.3f} h<br>y = %{y:.6f}<extra>Tangent</extra>'
    ), row=1, col=1)

    # Top / Bot horizontal lines
    fig.add_hline(y=top, line=dict(color='green', dash='dash', width=1),
                  annotation_text=f"Top={top:.4f}",
                  annotation_position="right", row=1, col=1)
    fig.add_hline(y=bot, line=dict(color='purple', dash='dash', width=1),
                  annotation_text=f"Bot={bot:.4f}",
                  annotation_position="right", row=1, col=1)

    # Vertical lines — บน ax1
    fig.add_vline(x=x_bot_intersect,
                  line=dict(color='royalblue', dash='dash', width=1.5),
                  annotation_text=f"Tangent∩Bot<br>t={x_bot_intersect:.2f} h",
                  annotation_position="top left")
    fig.add_vline(x=x_ms,
                  line=dict(color='crimson', dash='dash', width=1.5),
                  annotation_text=f"Max slope<br>t={x_ms:.2f} h",
                  annotation_position="top right")

    # Derivative
    fig.add_trace(go.Scatter(
        x=xd, y=dy, mode='lines', name='d/dt',
        line=dict(color='darkorange', width=2),
        hovertemplate='t = %{x:.3f} h<br>d/dt = %{y:.5f}<extra>d/dt</extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[x_ms], y=[slope_val], mode='markers',
        name=f'Peak={slope_val:.4f}/h',
        marker=dict(color='red', size=10, symbol='circle'),
        hovertemplate=f't = {x_ms:.3f} h<br>Peak = {slope_val:.5f} /h<extra>Max slope</extra>'
    ), row=2, col=1)

    fig.add_vline(x=x_bot_intersect,
                  line=dict(color='royalblue', dash='dash', width=1))
    fig.add_vline(x=x_ms,
                  line=dict(color='crimson', dash='dash', width=1))

    fig.update_layout(
        height=1200,
        hovermode='closest',        # ← snap ไปหาจุดที่ใกล้สุด
        hoverdistance=30,            # ← รัศมีดูด pixel
        spikedistance=-1,            # ← crosshair ลากตลอด
        legend=dict(
            orientation='v',
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray', borderwidth=1
        ),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        margin=dict(l=60, r=80, t=60, b=40),
    )

    # Crosshair spike lines ทั้ง 2 axes
    spike_style = dict(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='gray',
        spikethickness=1,
        spikedash='dot',
    )
    fig.update_xaxes(**spike_style)
    fig.update_yaxes(**spike_style)

    fig.update_xaxes(title_text="Time [h]", row=2, col=1, showticklabels=True)
    fig.update_yaxes(title_text="Signal",   row=1, col=1)
    fig.update_yaxes(title_text="d/dt",     row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 ใส่ข้อมูลในแถบซ้าย แล้วกด Calculate")
