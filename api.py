#!/usr/bin/env python3
"""
WashApp PRO — API Flask de Predicción
Sirve el modelo entrenado (demand_model.pkl) como endpoint HTTP.
Mismo contrato JSON que el PredictionController de Spring Boot.

Uso:
    python api.py
    # corre en http://localhost:5000

Endpoints:
    GET  /health          → estado del servicio
    POST /api/predecir    → predicción de demanda
"""

import os
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demand_model.pkl')

app = Flask(__name__)
CORS(app)

# Cargar modelo al iniciar
try:
    artifact = joblib.load(MODEL_PATH)
    clf      = artifact['model']
    features = artifact['features']
    print(f"✅ Modelo '{artifact['model_name']}' cargado ({artifact['trained_at'][:10]})")
    print(f"   Clases: {artifact['classes']}")
except FileNotFoundError:
    print("❌ No se encontró demand_model.pkl — ejecuta train.py primero")
    raise

DAY_MAP = {
    'Lunes': 0, 'Martes': 1, 'Miercoles': 2, 'Jueves': 3,
    'Viernes': 4, 'Sabado': 5, 'Domingo': 6,
}

def make_prediction(hora, dia_semana, num_services=1):
    dow        = DAY_MAP.get(dia_semana, 0)
    month      = datetime.now().month
    is_weekend = 1 if dow >= 5 else 0

    X     = pd.DataFrame([[hora, dow, month, is_weekend, num_services]], columns=features)
    pred  = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    conf  = f"{max(proba) * 100:.1f}%"
    return pred, conf

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':     'ok',
        'model':      artifact['model_name'],
        'trained_at': artifact['trained_at'][:10],
        'classes':    artifact['classes'],
    })

@app.route('/api/predecir', methods=['POST'])
def predecir():
    """
    Body JSON esperado (mismo que el formulario del frontend):
    {
        "hora":              9,
        "diaSemana":         "Lunes",
        "clima":             "Soleado",       (ignorado — no está en el modelo)
        "tipoServicio":      "Basico",        (ignorado por ahora)
        "idCliente":         1,               (ignorado por ahora)
        "historialVisitas":  5,               (ignorado por ahora)
        "temperatura":       24,              (ignorado — no está en el modelo)
        "promocionesActivas":"No"             (ignorado — no está en el modelo)
    }

    Respuesta:
    {
        "prediccion": "Alto",
        "confianza":  "87.3%",
        "id":         "python-model"
    }
    """
    try:
        data        = request.get_json(force=True) or {}
        hora        = int(data.get('hora', datetime.now().hour))
        dia_semana  = str(data.get('diaSemana', 'Lunes'))
        num_svc     = int(data.get('historialVisitas', 1)) or 1

        if dia_semana not in DAY_MAP:
            return jsonify({'error': f"diaSemana inválido: {dia_semana}. "
                                     f"Valores válidos: {list(DAY_MAP.keys())}"}), 400
        if not (0 <= hora <= 23):
            return jsonify({'error': 'hora debe estar entre 0 y 23'}), 400

        pred, conf = make_prediction(hora, dia_semana, num_svc)

        return jsonify({
            'prediccion': pred,
            'confianza':  conf,
            'id':         'python-model',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predecir/batch', methods=['POST'])
def predecir_batch():
    """Predice la demanda para todas las horas del día dado un día de la semana."""
    try:
        data       = request.get_json(force=True) or {}
        dia_semana = str(data.get('diaSemana', 'Lunes'))

        results = []
        for h in range(7, 19):
            pred, conf = make_prediction(h, dia_semana)
            results.append({'hora': h, 'prediccion': pred, 'confianza': conf})

        return jsonify({'dia': dia_semana, 'predicciones': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"\n🚀 API corriendo en http://localhost:{port}")
    print(f"   GET  /health         → estado")
    print(f"   POST /api/predecir   → predicción simple")
    print(f"   POST /api/predecir/batch → predicción todo el día\n")
    app.run(host='0.0.0.0', port=port, debug=False)
