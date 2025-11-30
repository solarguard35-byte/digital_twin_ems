# -*- coding: utf-8 -*-
import sys
# Force UTF-8 encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import webbrowser 
import os 
from datetime import datetime, timedelta
import warnings
import random
import asyncio 
import websockets 
import json 

# Conditional import for clipboard support
try:
    import pyperclip
except ImportError:
    pyperclip = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 0. GLOBAL CONFIGURATION ---
WS_PORT = 8765
WS_HOST = "localhost" 
CONNECTED_CLIENTS = set() 
NUM_HOURS_SIMULATION = 8760

# --- 1. SYNTHETIC DATA GENERATION ---

def generate_synthetic_data(num_hours=NUM_HOURS_SIMULATION):
    """Generates realistic energy consumption data."""
    print(f"Generating {num_hours} hours of synthetic energy data with 10 equipment items...")
    timestamps = pd.to_datetime(pd.Series(range(num_hours)).apply(lambda x: datetime(2025, 1, 1) + timedelta(hours=x)))
    
    df = pd.DataFrame({'timestamp': timestamps})
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Simulate temperature and base load
    sine_wave = np.sin((df['month'] / 12) * 2 * np.pi) 
    cosine_wave = np.cos((df['hour'] / 24) * 2 * np.pi)
    df['temperature_celsius'] = 15 + 10 * sine_wave + 5 * cosine_wave + np.random.normal(0, 2, num_hours)
    base_load = np.random.normal(20, 2, num_hours)
    
    production_hours_mask = (df['hour'].between(6, 22)) & (~df['is_weekend'])
    df['production_units'] = 0.5 * production_hours_mask + np.random.normal(0, 0.1, num_hours)
    
    # --- 10 EQUIPMENT DATA GENERATION ---
    equipment_params = {
        'production_line': {'base': 40, 'rand': 5, 'degr': 1.10, 'mask': production_hours_mask},
        'hvac': {'base': 10, 'temp_factor': 0.5, 'degr': 1.05, 'mask': True},
        'compressors': {'base': 25, 'rand': 2, 'degr': 1.20, 'mask': production_hours_mask},
        'lighting': {'base': 10, 'degr': 1.02, 'mask': True}, 
        'conveyor': {'base': 15, 'rand': 3, 'degr': 1.05, 'mask': production_hours_mask},
        'heavy_machinery': {'base': 50, 'rand': 8, 'degr': 1.15, 'mask': production_hours_mask},
        'server_cooling': {'base': 12, 'rand': 1, 'degr': 1.08, 'mask': True},
        'hvac_2': {'base': 15, 'temp_factor': 0.8, 'degr': 1.18, 'mask': True},
        'pump_motors': {'base': 30, 'rand': 5, 'degr': 1.16, 'mask': production_hours_mask},
        'exterior_lighting': {'base': 5, 'rand': 1, 'degr': 1.25, 'mask': (df['hour'] < 6) | (df['hour'] > 18)}
    }

    for eq, p in equipment_params.items():
        num_hours_float = float(num_hours)
        degradation_factor = np.linspace(1, p['degr'], num_hours)
        
        if 'temp_factor' in p:
            load = p['temp_factor'] * df['temperature_celsius'] + p['base'] + np.random.normal(0, 3, num_hours)
        elif eq == 'lighting':
             lighting_kwh_int = 10 * (np.exp(-(df['hour'] - 12)**2 / 50) + 0.1)
             load = lighting_kwh_int
        else:
            load = p['mask'] * (p['base'] + p['rand'] * np.random.rand(num_hours))

        load *= degradation_factor
        df[f'{eq}_kwh'] = np.maximum(0, load)

    # Combine and add noise
    df['base_load_kwh'] = np.maximum(5, base_load)
    
    consumption_cols = [col for col in df.columns if col.endswith('_kwh')]
    df['total_consumption_kwh'] = df[consumption_cols].sum(axis=1)
    
    # Add a couple of sharp anomalies for the detector test
    spike_indices = np.random.randint(0, num_hours, 5)
    df.loc[spike_indices, 'total_consumption_kwh'] *= 2.5
    
    print("✓ Synthetic data generated.")
    return df

# --- 2. NEXTGEN ENERGY MANAGER CLASS ---

class NextGenEnergyManager:
    """Advanced energy management features, primarily for getting static health predictions."""
    
    def __init__(self, df):
        self.df = df
        
    def predict_equipment_health(self):
        """Simulates predictive maintenance for 10 equipment items."""
        equipment_list = [
            'production_line', 'hvac', 'compressors', 'lighting', 
            'conveyor', 'heavy_machinery', 'server_cooling', 
            'hvac_2', 'pump_motors', 'exterior_lighting'
        ]
        health_predictions = []
        window = 168 
        
        for equipment in equipment_list:
            consumption = self.df[f'{equipment}_kwh'].values
            
            if len(consumption) < window * 2 or consumption.sum() == 0:
                health_predictions.append({'equipment': equipment.replace('_', ' ').title(), 'risk_level': "LOW"})
                continue
                
            baseline = consumption[:window].mean()
            recent = consumption[-window:].mean()
            degradation_rate = ((recent - baseline) / baseline) * 100 if baseline != 0 else 0
            health_score = max(0, min(100, 100 - abs(degradation_rate * 2)))
            
            if health_score < 70:
                risk_level = "HIGH"
            elif health_score < 85:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            health_predictions.append({
                'equipment': equipment.replace('_', ' ').title(),
                'health_score': round(health_score, 1),
                'risk_level': risk_level
            })
        
        return health_predictions


# --- 3. DIGITAL TWIN AND MANAGER (Real-time Simulation) ---

class DigitalTwin:
    """Represents a Digital Twin with state and simulated consumption."""
    def __init__(self, equipment_id, initial_state="ON", initial_kwh=50):
        self.equipment_id = equipment_id
        self.state = initial_state
        self.consumption_kwh = initial_kwh + random.uniform(-5, 5) 

    def get_real_time_data(self):
        """Simulates sensor reading adjusted by twin state."""
        
        if self.state == "ON":
            base_load = 50
        elif self.state == "OPTIMIZED":
            base_load = 35 
        elif self.state == "HIGH_CONSUMPTION":
            base_load = 75 
        elif self.state == "LOW_CONSUMPTION":
            base_load = 10
        else: # OFF
            base_load = 0
            
        self.consumption_kwh = base_load + random.uniform(-2, 2)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "equipment_id": self.equipment_id,
            "state": self.state,
            "consumption_kwh": round(max(0, self.consumption_kwh), 2),
            "status": "OPERATIONAL"
        }

    def process_command(self, command):
        """Simulates command execution (e.g., Ditto to device)."""
        action = command.get("action", "").upper()
        
        if action == "SHUTDOWN":
            self.state = "OFF"
        elif action == "OPTIMIZE":
            self.state = "OPTIMIZED"
        elif action == "STARTUP":
            self.state = "ON"
        elif action == "FAIL":
            self.state = "HIGH_CONSUMPTION"
        elif action == "LOW_LOAD":
            self.state = "LOW_CONSUMPTION"
        
        return f"SUCCESS: State changed to {self.state}"

class DigitalTwinManager:
    """Manages the collection of Digital Twins and the real-time sync loop."""
    def __init__(self, equipment_names):
        self.twins = {}
        for name in equipment_names:
            twin_id = name.replace('_kwh', '').replace('_', ' ').title()
            initial_kwh = 40 + 20 * random.random()
            self.twins[twin_id] = DigitalTwin(twin_id, initial_kwh=initial_kwh)

    async def simulate_mqtt_sync(self, nextgen_manager):
        """Infinite loop for real-time simulation and WebSocket broadcast."""
        print("\n--- STARTING REAL-TIME SYNCHRONIZATION (Digital Twins) ---")
        print("--- Press Ctrl+C to stop the server ---")
        
        health_data = nextgen_manager.predict_equipment_health()
        
        cycle = 0
        while True: # INFINITE LOOP FIX
            cycle += 1
            print(f"\n[CYCLE {cycle}] ---------------------------------")
            update_data = {}

            for twin_id, twin in self.twins.items():
                
                # STEP 1: Read Real-time Data
                data = twin.get_real_time_data()
                update_data[twin_id] = data
                
                # STEP 2: Command Logic (Repeating every 15 cycles)
                health_info = next((h for h in health_data if h['equipment'] == twin_id), None)
                
                # Use modulo to create repeating patterns
                logic_step = cycle % 15
                
                # Scenario 1: Predictive Maintenance (Stop HIGH RISK)
                if health_info and health_info['risk_level'] == "HIGH" and twin.state != "OFF" and 'Compressors' in twin_id and logic_step == 4:
                    twin.process_command({"action": "SHUTDOWN"})
                    print(f"  < COMMAND '{twin_id}':  HIGH RISK. Immediate Maintenance Shutdown.")
                
                # Scenario 2: Demand Response (Optimize HVAC)
                elif ('Hvac 2' in twin_id) and logic_step == 6 and twin.state == "ON":
                      twin.process_command({"action": "OPTIMIZE"})
                      print(f"  < COMMAND '{twin_id}': Peak Load Detected. Optimizing (Load Shedding).")
                      
                # Scenario 3: Simulate Failure (Heavy Machinery)
                elif ('Heavy Machinery' in twin_id) and logic_step == 9:
                      twin.process_command({"action": "FAIL"})
                      print(f"  < COMMAND '{twin_id}':  Simulating Failure. HIGH_CONSUMPTION.")
                      
                # Scenario 4: Return to Normal
                elif twin.state in ["OPTIMIZED", "OFF", "HIGH_CONSUMPTION"] and logic_step == 12:
                      twin.process_command({"action": "STARTUP"})
                      print(f"  < COMMAND '{twin_id}': Returning to Normal (STARTUP).")
                
            # STEP 3: Broadcast to 3D Engine
            await broadcast(json.dumps(update_data))
            
            print(f"  > BROADCAST: {len(update_data)} assets updated to 3D engine.")
            
            # SLEEP: Adjusted to 2 seconds so you can see the visualization changes
            await asyncio.sleep(2) 

# --- 4. WEB SOCKET SERVER ---

async def register(websocket):
    """Registers a new WebSocket client."""
    CONNECTED_CLIENTS.add(websocket)

async def unregister(websocket):
    """Unregisters a WebSocket client."""
    CONNECTED_CLIENTS.remove(websocket)

async def broadcast(message):
    """Broadcasts message to all connected clients."""
    if CONNECTED_CLIENTS:
        to_remove = set()
        for client in CONNECTED_CLIENTS:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                to_remove.add(client)
            except Exception:
                to_remove.add(client)
                
        for client in to_remove:
            CONNECTED_CLIENTS.remove(client)
            print("  [WS] Client removed due to connection error.")


# === FIX: Removed 'path' argument for compatibility with newer websockets versions ===
async def ws_server_handler(websocket): 
    """Handles client connection and keeps it open."""
    await register(websocket)
    try:
        print(f"  [WS] New client connected. Total clients: {len(CONNECTED_CLIENTS)}")
        await websocket.wait_closed()
    finally:
        await unregister(websocket)
        print(f"  [WS] Client disconnected. Total clients: {len(CONNECTED_CLIENTS)}")

# --- 5. MACHINE LEARNING & UTILS ---

def load_data(df):
    """Prepares data for ML."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    return df

def train_models(df):
    """Trains Prediction and Anomaly Detection models."""
    print("Training AI models...")
    
    features = ['hour', 'day_of_week', 'month', 'temperature_celsius', 'production_units', 'is_weekend']
    X = df[features]
    y = df['total_consumption_kwh']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Prediction Model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    
    # Anomaly Detector (Isolation Forest)
    anomaly_features = ['total_consumption_kwh', 'hour', 'day_of_week', 'temperature_celsius']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[anomaly_features])
    detector = IsolationForest(contamination=0.01, random_state=42)
    anomaly_labels = detector.fit_predict(X_scaled)
    
    print(f" Prediction Model R²: {r2:.3f}")
    print(f" Anomaly Detector found: {(anomaly_labels == -1).sum()} anomalies")
    
    return model, anomaly_labels


# --- 6. MAIN ASYNCHRONOUS EXECUTION ---
async def run_main():
    """Main async entry point."""
    
    # 1. STATIC ANALYSIS (ML & Data Prep)
    df = generate_synthetic_data(num_hours=NUM_HOURS_SIMULATION)
    df = load_data(df)
    model, anomaly_labels = train_models(df)
    
    equipment_cols = [col for col in df.columns if col.endswith('_kwh') and col != 'total_consumption_kwh']
    manager = NextGenEnergyManager(df)
    dt_manager = DigitalTwinManager(equipment_cols)

    # 2. START WEBSOCKET SERVER
    start_server = websockets.serve(ws_server_handler, WS_HOST, WS_PORT)
    # We await the server start object, but we don't cancel it.
    server = await start_server
    
    print(f"\n\n WebSocket Server listening on ws://{WS_HOST}:{WS_PORT}")
    
    # --- UPDATED CLIENT FILE HANDLING ---
    
    script_path = os.path.abspath(sys.argv[0])
    script_dir = os.path.dirname(script_path)
    
    # List of HTML client files to check and attempt to open
    client_files = [
        'websocket_client.html', 
        'Digital_Twin.html', 
        'Dashboard.html'
    ]
    opened_files = [] # To track successfully opened files

    print("\n--- Attempting to auto-open client HTML files ---")

    # Iterate through the list of files
    for file_name in client_files:
        html_file_path = os.path.join(script_dir, file_name)

        if os.path.exists(html_file_path):
            browser_path = f"file://{html_file_path.replace(os.path.sep, '/')}"
            
            # Open the file in the browser
            webbrowser.open(browser_path, new=2)
            print(f"✅ Auto-opened: **{file_name}**")
            opened_files.append(browser_path)
        else:
            print(f"⚠️ WARNING: Client file **{file_name}** not found at: {html_file_path}")

    # --- Clipboard / Action Required Section (Focus on the first file) ---

    if opened_files:
        # Use the path of the first successfully opened file for clipboard operations
        primary_browser_path = opened_files[0] 
        
        # Determine the name of the primary file for better print feedback
        primary_file_name = client_files[0] if len(client_files) > 0 else "Client"

        if pyperclip:
            pyperclip.copy(primary_browser_path)
            print("\n-------------------------------------------------------------------------")
            print(f" Primary client path (**{primary_file_name}**) COPIED to clipboard.")
            print(" ACTION REQUIRED: Paste (Ctrl+V) the path below into your browser:")
            print(f"   {primary_browser_path}")
            print("-------------------------------------------------------------------------")
        else:
            print("\n-------------------------------------------------------------------------")
            print(" Open your browser and paste this path:")
            print(f"   {primary_browser_path}")
            print("-------------------------------------------------------------------------")
            
        if len(opened_files) > 1:
            print(f"💡 Opened {len(opened_files)} client windows/tabs. Check your browser.")
    else:
        print("\n--- ACTION REQUIRED ---")
        print(" None of the client HTML files were found.")
        print(f" Expected files: {', '.join(client_files)}")
        print(" Please ensure they are in the same folder as the server script.")

    # 3. START REAL-TIME SIMULATION (Infinite Loop)
    # We await this. Since it's infinite, the script will stay alive here forever.
    await dt_manager.simulate_mqtt_sync(manager)
    
    # This part is technically unreachable now, but good for cleanup if loop breaks
    server.close()
    await server.wait_closed()
    print("\n[END] Simulation finished.")
def main():
    """Entry point."""
    try:
        asyncio.run(run_main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        if 'websockets' in str(e):
             print("\n Module 'websockets' missing. Run 'pip install websockets'.")
        
if __name__ == "__main__":
    main()