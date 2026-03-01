import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Literal, List, Tuple
from fastapi.middleware.cors import CORSMiddleware
import re
import warnings

FEATURE_COLS = ['M', 'Ash', 'VM', 'FC', 'C', 'H', 'O', 'N', 'PS', 'FT', 'HR', 'FR']
OPTIMIZABLE_PARAMS = ['PS', 'FT', 'HR', 'FR']
TARGET_COLS = ['Solid phase', 'Liquid phase', 'Gas phase']
models, scaler = {}, None


def clean_and_convert(value):
    if isinstance(value, (int, float)): return float(value)
    s_value = str(value).strip()
    range_match = re.match(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", s_value)
    if range_match:
        try:
            low, high = float(range_match.group(1)), float(range_match.group(2))
            return (low + high) / 2.0
        except (ValueError, IndexError): return np.nan
    num_match = re.search(r"(\d+\.?\d*)", s_value)
    if num_match:
        try: return float(num_match.group(1))
        except (ValueError, IndexError): return np.nan
    return np.nan

def train_and_load_model():
    global models, scaler
    print("INFO: Starting model training process...")
    df = pd.read_csv('pyrolysis.csv')
    df.columns = [col.strip() for col in df.columns]
    for col in FEATURE_COLS + TARGET_COLS:
        if col not in df.columns: raise ValueError(f"Required column '{col}' not found.")
        df[col] = df[col].apply(clean_and_convert)
    df.dropna(subset=FEATURE_COLS + TARGET_COLS, inplace=True)
    
    X, Y = df[FEATURE_COLS], df[TARGET_COLS]
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    for target in TARGET_COLS:
        print(f"INFO: Training model for {target}...")
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train_scaled, Y_train[target])
        models[target] = model
    print("INFO: All models trained and ready.")


def get_predictions(biomass_data: dict, process_params: dict) -> dict:
    full_params = {**biomass_data, **process_params}
    input_df = pd.DataFrame([full_params], columns=FEATURE_COLS)
    input_scaled = scaler.transform(input_df)
    raw_predictions = {target: max(0, model.predict(input_scaled)[0]) for target, model in models.items()}
    total = sum(raw_predictions.values())
    return {k: (v / total) * 100 for k, v in raw_predictions.items()} if total > 0 else raw_predictions

def run_optimization(biomass_data: dict, goal: str, user_bounds: List[Tuple[float, float]]) -> dict:

    def objective_function(params):
        ps, ft, hr, fr = params
        current_process_params = {'PS': ps, 'FT': ft, 'HR': hr, 'FR': fr}
        predictions = get_predictions(biomass_data, current_process_params)
        
        return -predictions[goal]

    result = differential_evolution(func=objective_function, bounds=user_bounds, seed=42, polish=True)
    if not result.success: warnings.warn("Optimization may not have fully converged.")
    return dict(zip(OPTIMIZABLE_PARAMS, result.x))


app = FastAPI(title="Industrial Pyrolysis Optimizer", version="Yield-Focused", on_startup=[train_and_load_model])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class BiomassProperties(BaseModel): M: float; Ash: float; VM: float; FC: float; C: float; H: float; O: float; N: float
class ProcessParams(BaseModel): PS: float; FT: float; HR: float; FR: float
class OptimizationConstraints(BaseModel): PS: Tuple[float, float]; FT: Tuple[float, float]; HR: Tuple[float, float]; FR: Tuple[float, float]

class OptimizationRequest(BaseModel):
    biomass_properties: BiomassProperties
    current_process_params: ProcessParams
    optimization_goal: Literal['Solid phase', 'Liquid phase', 'Gas phase']
    constraints: OptimizationConstraints

class OptimizationResult(BaseModel):
    optimization_goal: str; current_yields: Dict[str, float]; optimized_yields: Dict[str, float]
    optimized_params: ProcessParams; yield_improvement: Dict[str, str]

@app.post("/api/optimize", response_model=OptimizationResult)
async def optimize_pyrolysis_process(request: OptimizationRequest):
    try:
        biomass_data = request.biomass_properties.model_dump()
        current_params = request.current_process_params.model_dump()
        goal = request.optimization_goal
        user_bounds = [request.constraints.PS, request.constraints.FT, request.constraints.HR, request.constraints.FR]
        
        current_pred = get_predictions(biomass_data, current_params)
       
        optimized_params_dict = run_optimization(biomass_data, goal, user_bounds)
        optimized_pred = get_predictions(biomass_data, optimized_params_dict)
        
        improvement = {phase: f"{((optimized_pred[phase] - current_pred[phase]) / current_pred[phase]) * 100:+.2f}%" for phase in TARGET_COLS if current_pred.get(phase, 0) > 0}

        return OptimizationResult(
            optimization_goal=goal,
            current_yields={k: round(v, 2) for k, v in current_pred.items()},
            optimized_yields={k: round(v, 2) for k, v in optimized_pred.items()},
            optimized_params=ProcessParams(**optimized_params_dict),
            yield_improvement=improvement
        )
    except Exception as e:
        print(f"ERROR in optimization endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))