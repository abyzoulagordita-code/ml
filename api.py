#!/usr/bin/env python3
"""
WashApp PRO — API de Predicción
Sirve predicciones basadas en datos reales del negocio.
"""

import os, joblib, pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demand_model.pkl')

app = Flask(__name__)
CORS(app)

try:
    artifact    = joblib.load(MODEL_PATH)
    clf         = artifact['model']
    features    = artifact['features']
    stats       = artifact.get('stats', {})
    services    = artifact.get('services', [])
    pop_by_hour = artifact.get('pop_by_hour', {})
    print(f"✅ Modelo '{artifact['model_name']}' cargado ({artifact['trained_at'][:10]})")
    print(f"   Clases: {artifact['classes']}")
    print(f"   Servicios para optimizador: {[s['name'] for s in services]}")
except FileNotFoundError:
    print("❌ No se encontró demand_model.pkl — ejecuta train.py primero")
    raise

DAY_MAP = {'Lunes':0,'Martes':1,'Miercoles':2,'Jueves':3,'Viernes':4,'Sabado':5,'Domingo':6}

DEMAND_COLORS = {'Alto': '#43a047', 'Medio': '#fb8c00', 'Bajo': '#e53935'}


def predict_hour(hora, dia_semana, num_services=1):
    dow        = DAY_MAP.get(dia_semana, 0)
    is_weekend = 1 if dow >= 5 else 0
    month      = datetime.now().month
    X    = pd.DataFrame([[hora, dow, month, is_weekend, num_services]], columns=features)
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0]
    conf = f"{max(prob)*100:.1f}%"
    return pred, conf


def staff_recommendation(demand_level, washes_by_hour, hora):
    base = washes_by_hour.get(hora, 0)
    avg  = stats.get('avg_washes_per_day', 8) / 11  # horas activas
    if demand_level == 'Alto' or base > avg * 1.5:
        return 3
    if demand_level == 'Medio' or base > avg * 0.8:
        return 2
    return 1


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
    """Predicción para una hora específica (compatibilidad con Java backend)."""
    try:
        data       = request.get_json(force=True) or {}
        hora       = int(data.get('hora', datetime.now().hour))
        dia_semana = str(data.get('diaSemana', 'Lunes'))
        num_svc    = int(data.get('historialVisitas', 1) or 1)

        if dia_semana not in DAY_MAP:
            return jsonify({'error': f"diaSemana inválido: {dia_semana}"}), 400

        pred, conf = predict_hour(hora, dia_semana, num_svc)

        return jsonify({'prediccion': pred, 'confianza': conf, 'id': 'python-model'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dia/<dia_semana>', methods=['GET'])
def plan_dia(dia_semana):
    """
    Devuelve el plan completo del día: predicción hora por hora + resumen del negocio.
    Esto es lo que usa el frontend para mostrar el 'Plan del Día'.
    """
    try:
        if dia_semana not in DAY_MAP:
            return jsonify({'error': f"Día inválido: {dia_semana}"}), 400

        washes_by_hour = stats.get('washes_by_hour', {})

        horas = []
        peak_demand = 'Bajo'
        peak_hour   = None
        staff_max   = 1

        for h in range(7, 19):
            pred, conf = predict_hour(h, dia_semana)
            staff = staff_recommendation(pred, washes_by_hour, h)
            historical = washes_by_hour.get(h, 0)
            horas.append({
                'hora':       h,
                'prediccion': pred,
                'confianza':  conf,
                'color':      DEMAND_COLORS.get(pred, '#6e8efb'),
                'empleados':  staff,
                'historico':  historical,
            })
            if pred == 'Alto' and peak_demand != 'Alto':
                peak_demand = 'Alto'
                peak_hour   = h
            elif pred == 'Medio' and peak_demand == 'Bajo':
                peak_demand = 'Medio'
                peak_hour   = h
            if staff > staff_max:
                staff_max = staff

        # Resumen del día
        alto_count  = sum(1 for h in horas if h['prediccion'] == 'Alto')
        medio_count = sum(1 for h in horas if h['prediccion'] == 'Medio')

        avg = stats.get('avg_washes_per_day', 8)
        dow = DAY_MAP[dia_semana]
        day_historical = stats.get('washes_by_day', {}).get(dia_semana, 0)
        total_days = max(1, sum(stats.get('washes_by_day', {dia_semana: 1}).values()) // 7)
        est_washes = round(day_historical / max(total_days, 1)) if day_historical else round(avg)

        top_service = stats.get('top_service', 'Basico')
        top_services = stats.get('top_services', {})

        resumen = {
            'dia':              dia_semana,
            'demanda_general':  peak_demand,
            'hora_pico':        peak_hour,
            'empleados_max':    staff_max,
            'franjas_altas':    alto_count,
            'franjas_medias':   medio_count,
            'lavados_estimados': est_washes,
            'ingreso_estimado': round(est_washes * stats.get('avg_revenue_per_wash', 0)),
            'servicio_top':     top_service,
            'top_services':     top_services,
            'dia_pico_negocio': stats.get('peak_day', ''),
            'hora_pico_negocio': stats.get('peak_hour', 10),
        }

        return jsonify({'resumen': resumen, 'horas': horas})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/optimizar/<dia_semana>', methods=['GET'])
def optimizar_dia(dia_semana):
    """
    Maximización de ingresos usando Programación Lineal.
    Retorna el mix óptimo de servicios por hora y el ingreso máximo posible.
    """
    try:
        from optimizer import optimize_revenue_day

        if dia_semana not in DAY_MAP:
            return jsonify({'error': f"Día inválido: {dia_semana}"}), 400

        if not services:
            return jsonify({'error': 'No hay servicios registrados en la BD'}), 404

        # Demanda y empleados por hora (desde el modelo ML)
        demand_by_hour   = {}
        employees_by_hour = {}
        washes_by_hour   = stats.get('washes_by_hour', {})

        for h in range(7, 19):
            pred, _ = predict_hour(h, dia_semana)
            # Convertir predicción a número esperado de autos
            base = washes_by_hour.get(h, 1)
            avg  = stats.get('avg_washes_per_day', 8) / 11
            if pred == 'Alto':
                demand_by_hour[h]    = max(base, avg * 1.4)
                employees_by_hour[h] = 3
            elif pred == 'Medio':
                demand_by_hour[h]    = max(base, avg * 1.0)
                employees_by_hour[h] = 2
            else:
                demand_by_hour[h]    = max(base, avg * 0.6)
                employees_by_hour[h] = 1

        # Ajustar popularidad por hora si está disponible
        svc_list = []
        for s in services:
            h_pop = pop_by_hour.get(str(h), {})
            svc_list.append({**s, 'popularity': h_pop.get(s['id'], s['popularity'])})

        resultado = optimize_revenue_day(services, demand_by_hour, employees_by_hour)
        resultado['dia']      = dia_semana
        resultado['servicios'] = [{'name': s['name'], 'price': s['price'],
                                   'popularity': round(s['popularity']*100, 1)}
                                  for s in services]

        return jsonify(resultado)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/negocio', methods=['GET'])
def negocio():
    """Estadísticas generales del negocio aprendidas del historial real."""
    return jsonify(stats)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"\n🚀 API en http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
