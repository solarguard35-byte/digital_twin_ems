import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys

# Supprimer les avertissements pour une sortie propre
warnings.filterwarnings('ignore')

def generate_synthetic_data(num_hours=8760):
    """
    Génère des données de consommation d'énergie réalistes pour une installation industrielle
    avec 10 équipements, dont 5 présentent une dégradation élevée pour forcer un risque HIGH.
    """
    print(f"Génération de {num_hours} heures de données synthétiques avec 10 équipements...", file=sys.stderr)
    timestamps = pd.to_datetime(pd.Series(range(num_hours)).apply(lambda x: datetime(2025, 1, 1) + timedelta(hours=x)))
    
    df = pd.DataFrame({'timestamp': timestamps})
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek # Lundi=0, Dimanche=6
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Simulation de la température
    sine_wave = np.sin((df['month'] / 12) * 2 * np.pi) 
    cosine_wave = np.cos((df['hour'] / 24) * 2 * np.pi)
    df['temperature_celsius'] = 15 + 10 * sine_wave + 5 * cosine_wave + np.random.normal(0, 2, num_hours)
    
    base_load = np.random.normal(20, 2, num_hours)
    
    # Masque des heures de production
    production_hours_mask = (df['hour'].between(6, 22)) & (~df['is_weekend'])
    df['production_units'] = 0.5 * production_hours_mask + np.random.normal(0, 0.1, num_hours)
    
    # --- 10 ÉQUIPEMENTS ET LEUR DÉGRADATION FORCÉE ---
    
    # 1. Production Line (Risque MEDIUM: 10% dégradation)
    production_line_kwh = production_hours_mask * (40 + 5 * np.random.rand(num_hours))
    degradation_factor_1 = np.linspace(1, 1.10, num_hours) 
    df['production_line_kwh'] = np.maximum(0, production_line_kwh * degradation_factor_1)
    
    # 2. HVAC 1 (Risque LOW: 5% dégradation)
    hvac_load_1 = 0.5 * df['temperature_celsius'] + 10 + np.random.normal(0, 3, num_hours)
    degradation_factor_2 = np.linspace(1, 1.05, num_hours) 
    df['hvac_kwh'] = np.maximum(0, hvac_load_1 * degradation_factor_2)

    # 3. Compressors (Risque HIGH: 20% dégradation) -> HIGH FORCÉ
    compressors_kwh = production_hours_mask * (25 + 2 * np.random.rand(num_hours))
    degradation_factor_3 = np.linspace(1, 1.20, num_hours) 
    df['compressors_kwh'] = np.maximum(0, compressors_kwh * degradation_factor_3)
    
    # 4. Lighting (Risque LOW: 2% dégradation)
    lighting_kwh_int = 10 * (np.exp(-(df['hour'] - 12)**2 / 50) + 0.1)
    degradation_factor_4 = np.linspace(1, 1.02, num_hours) 
    df['lighting_kwh'] = np.maximum(0, lighting_kwh_int * degradation_factor_4)
    
    # 5. Conveyor System (Risque LOW: 5% dégradation)
    conveyor_kwh = production_hours_mask * (15 + 3 * np.random.rand(num_hours))
    degradation_factor_5 = np.linspace(1, 1.05, num_hours) 
    df['conveyor_kwh'] = np.maximum(0, conveyor_kwh * degradation_factor_5)
    
    # 6. Heavy Machinery (Risque HIGH: 15% dégradation) -> HIGH FORCÉ
    heavy_machinery_kwh = production_hours_mask * (50 + 8 * np.random.rand(num_hours))
    degradation_factor_6 = np.linspace(1, 1.15, num_hours) 
    df['heavy_machinery_kwh'] = np.maximum(0, heavy_machinery_kwh * degradation_factor_6)
    
    # 7. Server Room Cooling (Risque LOW: 8% dégradation)
    server_cooling_kwh = np.random.normal(12, 1, num_hours) 
    degradation_factor_7 = np.linspace(1, 1.08, num_hours) 
    df['server_cooling_kwh'] = np.maximum(0, server_cooling_kwh * degradation_factor_7)

    # 8. HVAC 2 (Risque HIGH: 18% dégradation) -> HIGH FORCÉ
    hvac_load_2 = 0.8 * df['temperature_celsius'] + 15 + np.random.normal(0, 4, num_hours)
    degradation_factor_8 = np.linspace(1, 1.18, num_hours) 
    df['hvac_2_kwh'] = np.maximum(0, hvac_load_2 * degradation_factor_8)
    
    # 9. Pump Motors (Risque HIGH: 16% dégradation) -> HIGH FORCÉ
    pump_motors_kwh = production_hours_mask * (30 + 5 * np.random.rand(num_hours))
    degradation_factor_9 = np.linspace(1, 1.16, num_hours) 
    df['pump_motors_kwh'] = np.maximum(0, pump_motors_kwh * degradation_factor_9)
    
    # 10. Exterior Lighting (Risque HIGH: 25% dégradation) -> HIGH FORCÉ
    exterior_lighting_mask = (df['hour'] < 6) | (df['hour'] > 18)
    exterior_lighting_kwh = exterior_lighting_mask * (5 + 1 * np.random.rand(num_hours))
    degradation_factor_10 = np.linspace(1, 1.25, num_hours) 
    df['exterior_lighting_kwh'] = np.maximum(0, exterior_lighting_kwh * degradation_factor_10)

    # Combinaison et calculs finaux
    df['base_load_kwh'] = np.maximum(5, base_load)
    
    df['total_consumption_kwh'] = (
        df['base_load_kwh'] + df['production_line_kwh'] + df['hvac_kwh'] + 
        df['compressors_kwh'] + df['lighting_kwh'] + df['conveyor_kwh'] +
        df['heavy_machinery_kwh'] + df['server_cooling_kwh'] + df['hvac_2_kwh'] +
        df['pump_motors_kwh'] + df['exterior_lighting_kwh']
    )
    
    # Anomalies
    spike_indices = np.random.randint(0, num_hours, 5)
    df.loc[spike_indices, 'total_consumption_kwh'] *= 2.5
    
    df['cost_usd'] = df['total_consumption_kwh'] * 0.12 # 0.12 $/kWh
    
    return df

# Exécution et affichage des données
if __name__ == "__main__":
    df_new = generate_synthetic_data(num_hours=8760)
    print("✓ Données générées. Affichage du jeu de données (CSV) :", file=sys.stderr)
    print(df_new.to_csv(index=False))