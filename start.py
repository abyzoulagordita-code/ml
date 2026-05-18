#!/usr/bin/env python3
"""
Punto de entrada para Railway.
Entrena el modelo si no existe, luego arranca la API Flask.
"""
import os
import sys

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demand_model.pkl')

if not os.path.exists(MODEL_PATH):
    print("Modelo no encontrado — entrenando...")
    import train
    # train.py ejecuta todo en __main__, lo importamos como módulo
    from train import load_data, engineer_features, create_demand_labels, train, save_model, plot_insights
    df_raw, df_svc, df_emp = load_data()
    df = engineer_features(df_raw, df_svc)
    df, q33, q66 = create_demand_labels(df)
    clf, features, model_name = train(df)
    save_model(clf, features, {'bajo': float(q33), 'medio': float(q66)}, model_name)
    print("Entrenamiento completado.")
else:
    print(f"Modelo existente encontrado: {MODEL_PATH}")

# Arrancar API
from api import app
port = int(os.getenv('PORT', 5000))
print(f"Iniciando API en puerto {port}")
app.run(host='0.0.0.0', port=port, debug=False)
