import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import differential_evolution

# --- STYLING ---
st.set_page_config(page_title="Pro Pyrolysis Optimizer", layout="wide")

# --- DATA & MODEL ENGINE ---
FEATURE_COLS = ['M', 'Ash', 'VM', 'FC', 'C', 'H', 'O', 'N', 'PS', 'FT', 'HR', 'FR']
TARGET_COLS = ['Solid phase', 'Liquid phase', 'Gas phase']

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists('pyrolysis.csv'):
        return None, None
    
    try:
        df = pd.read_csv('pyrolysis.csv')
        # 🔥 FIX: Remove extra spaces from column names
        df.columns = df.columns.str.strip()
        
        # Numeric conversion and dropping NaNs
        for col in FEATURE_COLS + TARGET_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
        
        X, y = df[FEATURE_COLS], df[TARGET_COLS]
        scaler = StandardScaler().fit(X)
        model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
        model.fit(scaler.transform(X), y)
        return model, scaler
    except Exception as e:
        st.error(f"Data loading mein error: {e}")
        return None, None

model, scaler = load_model_and_scaler()

# --- UI DESIGN ---
st.title("🧪 Smart Biomass Pyrolysis Optimizer")

if model is None:
    st.error("❌ 'pyrolysis.csv' file missing ya corrupt hai. Bhai, check karo folder!")
else:
    # INPUT SECTION
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🍃 Biomass Details")
        m = st.number_input("Moisture (%)", 10.0)
        ash = st.number_input("Ash (%)", 5.0)
        vm = st.number_input("Volatile Matter (%)", 70.0)
        fc = st.number_input("Fixed Carbon (%)", 15.0)
        c_val = st.number_input("Carbon (C)", 50.0)
        h_val = st.number_input("Hydrogen (H)", 6.0)
        o_val = st.number_input("Oxygen (O)", 40.0)
        n_val = st.number_input("Nitrogen (N)", 0.5)

    with col2:
        st.subheader("⚙️ Process Parameters")
        ps = st.number_input("Particle Size (mm)", 1.0)
        ft = st.number_input("Final Temp (°C)", 500)
        hr = st.number_input("Heating Rate", 10)
        fr = st.number_input("Flow Rate", 50)
        
        st.divider()
        goal = st.selectbox("🎯 Maximize:", TARGET_COLS)
        # Unique key added to button to avoid state issues
        optimize_click = st.button("🚀 Start Powerful Optimization", key="opt_btn")

    # --- CALCULATION LOGIC ---
    if optimize_click:
        try:
            with st.spinner("Finding the **optimal** solution..."):
                # Prepare data
                biomass_inputs = [m, ash, vm, fc, c_val, h_val, o_val, n_val]
                current_process = [ps, ft, hr, fr]
                
                def predict_fn(proc_params):
                    combined = np.array([biomass_inputs + list(proc_params)])
                    scaled = scaler.transform(combined)
                    return model.predict(scaled)[0]

                # 1. Current Yield
                curr_y = predict_fn(current_process)
                
                # 2. Optimize (Differential Evolution)
                goal_idx = TARGET_COLS.index(goal)
                
                # Boundary conditions
                bounds = [(0.1, 5.0), (300, 900), (1, 100), (10, 200)]
                
                # Optimization function (Minimize negative to maximize)
                result = differential_evolution(lambda p: -predict_fn(p)[goal_idx], bounds, seed=42)
                
                opt_params = result.x
                opt_yields = predict_fn(opt_params)

                # 3. Display Results
                st.divider()
                st.balloons()
                st.success(f"Successfully Optimized for **{goal}**!")
                
                res_df = pd.DataFrame({
                    "Parameter / Yield": ["Particle Size", "Temp", "Heating Rate", "Flow Rate"] + TARGET_COLS,
                    "Current": current_process + [f"{v:.2f}%" for v in curr_y],
                    "Optimized": [round(v, 2) for v in opt_params] + [f"{v:.2f}%" for v in opt_yields]
                })
                
                st.table(res_df)
                
                improvement = ((opt_yields[goal_idx] - curr_y[goal_idx]) / curr_y[goal_idx]) * 100
                st.metric("Yield Increase", f"{opt_yields[goal_idx]:.2f}%", f"{improvement:+.2f}%")

        except Exception as e:
            st.error(f"Optimization fail ho gayi: {e}")