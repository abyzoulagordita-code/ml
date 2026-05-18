#!/usr/bin/env python3
import os, sys

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demand_model.pkl')

if not os.path.exists(MODEL_PATH):
    print("Modelo no encontrado — entrenando...")
    from train import load_data, engineer_features, create_demand_labels, train, save_model, plot_insights, business_stats
    df_raw, df_svc, svc_names = load_data()
    df    = engineer_features(df_raw, df_svc)
    stats = business_stats(df, df_svc, svc_names)
    df, q33, q66 = create_demand_labels(df)
    clf, features, model_name = train(df)
    save_model(clf, features, {'bajo': q33, 'medio': q66}, model_name, stats)
    plot_insights(df, stats)
    print("Entrenamiento completado.")
else:
    print(f"Modelo existente encontrado: {MODEL_PATH}")

from api import app
port = int(os.getenv('PORT', 5000))
print(f"Iniciando API en puerto {port}")
app.run(host='0.0.0.0', port=port, debug=False)
