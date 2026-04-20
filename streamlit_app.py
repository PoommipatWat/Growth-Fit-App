import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# 1. ฟังก์ชันสมการการเติบโต (Growth Models)
# ==========================================

def weibull(t, A, k, n):
    """Weibull Model"""
    return A * (1 - np.exp(-(k * t)**n))

def modified_gompertz(t, A, mu_max, lam):
    """Modified Gompertz Model (ตามภาพแนบ)"""
    # y(t) = A * exp(-exp((mu_max * e / A) * (lam - t) + 1))
    return A * np.exp(-np.exp((mu_max * np.e / A) * (lam - t) + 1))

def baranyi(t, y0, ymax, mu_max, lam):
    """Baranyi-Roberts Model (Explicit Approximation)"""
    h0 = mu_max * lam
    # ป้องกัน Overflow จาก np.exp
    with np.errstate(over='ignore', invalid='ignore'):
        A_t = t + (1 / mu_max) * np.log(np.exp(-mu_max * t) + np.exp(-h0) - np.exp(-mu_max * t - h0))
        y = y0 + mu_max * A_t - np.log(1 + (np.exp(mu_max * A_t) - 1) / np.exp(ymax - y0))
    return y

# ==========================================
# 2. ส่วนหน้าแอป (Streamlit UI)
# ==========================================

st.set_page_config(page_title="Growth Model Fitting App", layout="wide")
st.title("📈 Growth Model Fitting App")
st.markdown("แอปพลิเคชันสำหรับทำ Curve Fitting ด้วยโมเดลการเติบโตต่างๆ (Weibull, Modified Gompertz, Baranyi)")

# ----- Sidebar: ตัวเลือก -----
st.sidebar.header("ตั้งค่าโมเดลและข้อมูล")

# 1. Bar ให้เลือกโมเดล (ตามความต้องการ)
model_choice = st.sidebar.selectbox(
    "เลือกโมเดล (Model Selection):",
    ("Weibull", "Modified Gompertz", "Baranyi")
)

# จำลองข้อมูลตัวอย่าง (หรือจะให้ผู้ใช้อัปโหลด CSV ก็ได้)
st.sidebar.markdown("---")
use_dummy_data = st.sidebar.checkbox("ใช้ข้อมูลตัวอย่าง (Dummy Data)", value=True)

if use_dummy_data:
    # สร้างข้อมูลจำลองที่มี Noise เล็กน้อยให้ดูเหมือนการเติบโตจริง
    t_data = np.linspace(0, 20, 20)
    # สมมติฐานข้อมูลให้คล้ายรูปตัว S (Sigmoidal)
    y_data = 10 * np.exp(-np.exp((1.2 * np.e / 10) * (5 - t_data) + 1)) + np.random.normal(0, 0.2, len(t_data))
    y_data = np.abs(y_data) # ป้องกันค่าติดลบ
    df = pd.DataFrame({'Time (t)': t_data, 'Growth (y)': y_data})
else:
    uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV (คอลัมน์: t, y)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        t_col, y_col = df.columns[0], df.columns[1]
        t_data = df[t_col].values
        y_data = df[y_col].values
    else:
        st.warning("โปรดอัปโหลดไฟล์ CSV หรือเลือก 'ใช้ข้อมูลตัวอย่าง'")
        st.stop()

st.sidebar.dataframe(df.head())

# ==========================================
# 3. การประมวลผล (Curve Fitting)
# ==========================================

try:
    # ตั้งค่า Initial Guess (p0) เพื่อช่วยให้ Optimize ทำงานง่ายขึ้น
    y_max = np.max(y_data)
    y_min = np.min(y_data)
    
    if model_choice == "Weibull":
        p0 = [y_max, 0.1, 1]
        bounds = (0, np.inf)
        popt, pcov = curve_fit(weibull, t_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
        y_pred = weibull(t_data, *popt)
        
    elif model_choice == "Modified Gompertz":
        p0 = [y_max, 1.0, 5.0] # [A, mu_max, lambda]
        popt, pcov = curve_fit(modified_gompertz, t_data, y_data, p0=p0, maxfev=5000)
        y_pred = modified_gompertz(t_data, *popt)
        
    elif model_choice == "Baranyi":
        p0 = [y_min, y_max, 1.0, 5.0] # [y0, ymax, mu_max, lambda]
        popt, pcov = curve_fit(baranyi, t_data, y_data, p0=p0, maxfev=5000)
        y_pred = baranyi(t_data, *popt)

    # ==========================================
    # 4. คำนวณตัวชี้วัด (Metrics) ตามที่ระบุ
    # ==========================================
    r2 = r2_score(y_data, y_pred)
    rmse = np.sqrt(mean_squared_error(y_data, y_pred))
    
    # คำนวณ R** + RSME 
    combined_metric = r2 + rmse

    # ==========================================
    # 5. แสดงผลลัพธ์ (UI Dashboard)
    # ==========================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"กราฟแสดงผลลัพธ์ (Fitting Result) - {model_choice}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(t_data, y_data, label='Actual Data', color='red')
        
        # วาดเส้น Fit ที่มีความละเอียดสูงขึ้น
        t_smooth = np.linspace(min(t_data), max(t_data), 100)
        if model_choice == "Weibull":
            y_smooth = weibull(t_smooth, *popt)
        elif model_choice == "Modified Gompertz":
            y_smooth = modified_gompertz(t_smooth, *popt)
        elif model_choice == "Baranyi":
            y_smooth = baranyi(t_smooth, *popt)
            
        ax.plot(t_smooth, y_smooth, label=f'Fitted {model_choice}', color='blue')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Growth (y)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

    with col2:
        st.subheader("พารามิเตอร์ที่คำนวณได้")
        if model_choice == "Weibull":
            st.code(f"A = {popt[0]:.4f}\nk = {popt[1]:.4f}\nn = {popt[2]:.4f}")
        elif model_choice == "Modified Gompertz":
            st.code(f"A (Asymptote) = {popt[0]:.4f}\nμ_max (Max rate) = {popt[1]:.4f}\nλ (Lag time) = {popt[2]:.4f}")
        elif model_choice == "Baranyi":
            st.code(f"y0 = {popt[0]:.4f}\nymax = {popt[1]:.4f}\nμ_max = {popt[2]:.4f}\nλ = {popt[3]:.4f}")

        # เพิ่มตัวยืนยันข้อมูลตาม Request
        st.subheader("ตัวยืนยันข้อมูล (Metrics)")
        st.metric(label="R² Score", value=f"{r2:.4f}")
        st.metric(label="RMSE", value=f"{rmse:.4f}")
        st.metric(label="R² + RMSE", value=f"{combined_metric:.4f}", 
                  help="คะแนนรวมระหว่าง R² และ RMSE ตามที่ระบุ")

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการประมวลผล (Curve Fitting Failed): {e}")
    st.info("คำแนะนำ: ข้อมูลอาจไม่เหมาะสมกับโมเดลที่เลือก หรืออาจต้องปรับค่า Initial Guess ในส่วนโค้ดของการทำ curve_fit")
