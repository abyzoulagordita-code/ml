#!/usr/bin/env python3
"""
WashApp PRO — Optimizador de Ingresos
Programación Lineal (LP) para maximizar ingresos dado capacidad de empleados.

Variables de decisión:  x[s] = cuántos lavados del servicio s hacer en una hora
Objetivo:               Maximizar  Σ precio[s] * x[s]
Restricciones:
  - Σ duracion[s] * x[s]  ≤  empleados × 60 min   (capacidad de tiempo)
  - x[s]                  ≤  demanda × popularidad[s]  (límite por demanda)
  - x[s]                  ≥  0
"""

import numpy as np
from scipy.optimize import linprog


def parse_duration(duration_str):
    """Convierte string de duración a minutos."""
    if not duration_str:
        return 30
    s = str(duration_str).lower().strip()
    try:
        return float(s)
    except ValueError:
        pass
    if 'hora' in s or 'hour' in s:
        try:
            return float(s.replace('horas','').replace('hora','').replace('hours','').replace('hour','').strip()) * 60
        except Exception:
            return 60
    if 'min' in s:
        try:
            return float(s.replace('minutos','').replace('minuto','').replace('min','').strip())
        except Exception:
            return 30
    return 30


def optimize_revenue_day(services, demand_by_hour, employees_by_hour):
    """
    Calcula el mix óptimo de servicios hora por hora para maximizar ingresos.

    Parámetros:
        services:          lista de dicts {name, price, duration_min, popularity}
        demand_by_hour:    {hora(int): autos_esperados(float)}
        employees_by_hour: {hora(int): empleados(int)}

    Retorna: lista de dicts por hora con mix óptimo e ingresos.
    """
    if not services:
        return []

    n = len(services)
    prices    = np.array([s['price']        for s in services], dtype=float)
    durations = np.array([s['duration_min'] for s in services], dtype=float)
    pop       = np.array([s['popularity']   for s in services], dtype=float)

    if pop.sum() > 0:
        pop = pop / pop.sum()
    else:
        pop = np.ones(n) / n

    total_opt_revenue = 0
    total_avg_revenue = 0
    results = []

    for hora in range(7, 19):
        n_emp    = max(1, employees_by_hour.get(hora, 1))
        demand   = max(0.5, float(demand_by_hour.get(hora, 1)))
        emp_mins = n_emp * 60          # minutos disponibles por hora

        # ── LP: minimizar el negativo de los ingresos ──────────────────────────
        c     = -prices                                       # objetivo
        A_ub  = [durations.tolist()]                          # restricción de tiempo
        b_ub  = [float(emp_mins)]
        bounds = [(0.0, max(0.05, demand * float(p))) for p in pop]   # demanda por servicio

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            x_opt      = res.x
            rev_opt    = float(-res.fun)
        else:
            x_opt   = np.array([demand * float(p) for p in pop])
            rev_opt = float(prices @ x_opt)

        # Ingreso promedio sin optimizar (asignación proporcional simple)
        x_avg   = np.array([demand * float(p) for p in pop])
        rev_avg = float(prices @ x_avg)

        mix = {
            services[i]['name']: round(float(x_opt[i]), 1)
            for i in range(n) if x_opt[i] >= 0.05
        }

        cap_usada = min(100, round(float(durations @ x_opt) / emp_mins * 100))

        total_opt_revenue += rev_opt
        total_avg_revenue += rev_avg

        results.append({
            'hora':             hora,
            'mix_optimo':       mix,
            'ingreso_optimo':   round(rev_opt),
            'ingreso_promedio': round(rev_avg),
            'capacidad_usada':  cap_usada,
            'autos_optimos':    round(float(x_opt.sum()), 1),
            'empleados':        n_emp,
        })

    return {
        'horas':                results,
        'total_optimo':         round(total_opt_revenue),
        'total_promedio':       round(total_avg_revenue),
        'ganancia_adicional':   round(total_opt_revenue - total_avg_revenue),
        'pct_mejora':           round((total_opt_revenue / max(1, total_avg_revenue) - 1) * 100, 1),
    }
