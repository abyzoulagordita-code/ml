#!/usr/bin/env python3
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demand_model.pkl')

if not os.path.exists(MODEL_PATH):
    print("Modelo no encontrado — entrenando...")
    from train import (load_data, engineer_features, create_demand_labels,
                       train, save_model, plot_insights, business_stats,
                       build_services_for_optimizer)

    df_raw, df_svc, svc_names, svc_catalog, svc_by_hour = load_data()
    df    = engineer_features(df_raw, df_svc)
    stats = business_stats(df, df_svc, svc_names)
    df, q33, q66 = create_demand_labels(df)
    clf, features, model_name = train(df)

    result = build_services_for_optimizer(svc_catalog, df_svc, svc_by_hour)
    services, pop_by_hour = result if result else ([], {})

    save_model(clf, features, {'bajo': q33, 'medio': q66},
               model_name, stats, services, pop_by_hour)
    plot_insights(df, stats)
    print("Entrenamiento completado.")
else:
    print(f"Modelo existente: {MODEL_PATH}")

from api import app
port = int(os.getenv('PORT', 5000))
print(f"Iniciando API en puerto {port}")
app.run(host='0.0.0.0', port=port, debug=False)
