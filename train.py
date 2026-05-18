#!/usr/bin/env python3
"""
WashApp PRO — Entrenamiento ML
Lee datos reales de la BD, aprende los patrones del negocio y guarda el modelo.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

DB_HOST     = os.getenv('DB_HOST',     'localhost')
DB_PORT     = int(os.getenv('DB_PORT', '3306'))
DB_NAME     = os.getenv('DB_NAME',     'DB_SanFelipe')
DB_USER     = os.getenv('DB_USER',     'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUT_DIR, 'demand_model.pkl')
PLOT_DIR   = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

DAY_MAP = {0:'Lunes',1:'Martes',2:'Miercoles',3:'Jueves',4:'Viernes',5:'Sabado',6:'Domingo'}


# ── 1. Carga desde BD ──────────────────────────────────────────────────────────
def load_data():
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, connect_timeout=10
        )
        print(f"✅ Conectado a {DB_NAME}@{DB_HOST}:{DB_PORT}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        sys.exit(1)

    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT id, date, total, employee, client FROM wash_record ORDER BY date")
    df = pd.DataFrame(cur.fetchall())

    try:
        cur.execute("SELECT wash_record_id, service_offered FROM wash_record_service_offered")
        df_svc = pd.DataFrame(cur.fetchall())
    except Exception:
        df_svc = pd.DataFrame(columns=['wash_record_id','service_offered'])

    conn.close()
    print(f"   Lavados cargados: {len(df)}")
    return df, df_svc


# ── 2. Feature engineering ────────────────────────────────────────────────────
def engineer_features(df, df_svc):
    df = df.copy()
    df['date']        = pd.to_datetime(df['date'])
    df['hour']        = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek   # 0=Lun … 6=Dom
    df['day_name']    = df['day_of_week'].map(DAY_MAP)
    df['month']       = df['date'].dt.month
    df['is_weekend']  = df['day_of_week'].isin([5,6]).astype(int)
    df['date_only']   = df['date'].dt.date

    # Servicios por lavado
    if not df_svc.empty:
        cnt = df_svc.groupby('wash_record_id').size().rename('num_services')
        df  = df.merge(cnt, left_on='id', right_index=True, how='left')
    else:
        df['num_services'] = 1
    df['num_services'] = df['num_services'].fillna(1).astype(int)

    return df


# ── 3. Estadísticas del negocio ────────────────────────────────────────────────
def business_stats(df, df_svc):
    stats = {}

    # Lavados por día de semana
    by_day = df.groupby('day_name').size()
    order  = ['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo']
    stats['washes_by_day'] = {d: int(by_day.get(d,0)) for d in order}

    # Lavados por hora (7-18)
    by_hour = df[df['hour'].between(7,18)].groupby('hour').size()
    stats['washes_by_hour'] = {int(h): int(v) for h,v in by_hour.items()}

    # Hora pico global
    stats['peak_hour'] = int(by_hour.idxmax()) if not by_hour.empty else 10

    # Día pico global
    busiest_day_idx = df.groupby('day_of_week').size().idxmax()
    stats['peak_day'] = DAY_MAP[busiest_day_idx]

    # Promedio lavados por día
    daily_count = df.groupby('date_only').size()
    stats['avg_washes_per_day'] = round(float(daily_count.mean()), 1)
    stats['max_washes_per_day'] = int(daily_count.max())

    # Ingreso promedio por lavado
    stats['avg_revenue_per_wash'] = round(float(df['total'].mean()), 0)

    # Servicio más pedido
    if not df_svc.empty:
        top_svc = df_svc['service_offered'].value_counts()
        stats['top_services'] = {str(k): int(v) for k,v in top_svc.items()}
        stats['top_service']  = str(top_svc.idxmax())
    else:
        stats['top_services'] = {}
        stats['top_service']  = 'Basico'

    # Empleado más activo
    if 'employee' in df.columns:
        emp = df['employee'].value_counts()
        stats['top_employee'] = str(emp.idxmax()) if not emp.empty else ''

    # Días activos en la semana (tienen datos reales)
    stats['active_days'] = [d for d in order if stats['washes_by_day'][d] > 0]

    print(f"\n📊 Estadísticas del negocio:")
    print(f"   Día pico     : {stats['peak_day']}")
    print(f"   Hora pico    : {stats['peak_hour']}:00")
    print(f"   Promedio/día : {stats['avg_washes_per_day']} lavados")
    print(f"   Servicio top : {stats['top_service']}")

    return stats


# ── 4. Etiquetas de demanda (por franja horaria) ───────────────────────────────
def create_demand_labels(df):
    # Agrupamos por día + franja de 2 horas para tener granularidad real
    df = df.copy()
    df['slot'] = df['date_only'].astype(str) + '_' + df['hour'].astype(str)

    slot_count = df.groupby(['date_only','day_of_week','hour','is_weekend']).size().reset_index(name='count')

    q33 = slot_count['count'].quantile(0.33)
    q66 = slot_count['count'].quantile(0.66)

    def label(n):
        if n <= q33: return 'Bajo'
        if n <= q66: return 'Medio'
        return 'Alto'

    slot_count['demand'] = slot_count['count'].apply(label)

    # Merge de vuelta al df principal
    df = df.merge(slot_count[['date_only','hour','demand']],
                  on=['date_only','hour'], how='left')
    df['demand'] = df['demand'].fillna('Bajo')

    print(f"\n📊 Umbrales franja: Bajo≤{q33:.0f} | Medio≤{q66:.0f} | Alto>{q66:.0f} lavados/franja")
    print(slot_count['demand'].value_counts().rename('franjas').to_string())

    return df, float(q33), float(q66)


# ── 5. Entrenamiento ───────────────────────────────────────────────────────────
def train(df):
    FEATURES = ['hour','day_of_week','month','is_weekend','num_services']

    clean = df[FEATURES + ['demand']].dropna()
    X = clean[FEATURES]
    y = clean['demand']

    if y.nunique() < 2:
        print("⚠️  Pocos datos para entrenar — usando modelo simple")
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        return clf, FEATURES, 'RandomForest-simple'

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42,
            class_weight='balanced', n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }

    print("\n🔬 Evaluando modelos (CV 5-fold):")
    best_clf, best_score, best_name = None, 0.0, ''
    kf = StratifiedKFold(n_splits=min(5, y.value_counts().min()), shuffle=True, random_state=42)
    for name, clf in candidates.items():
        cv = cross_val_score(clf, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
        print(f"   {name:20s}: {cv.mean():.2%} ± {cv.std():.2%}")
        if cv.mean() > best_score:
            best_score, best_clf, best_name = cv.mean(), clf, name

    print(f"\n🏆 Ganador: {best_name}  (CV {best_score:.2%})")
    best_clf.fit(X_tr, y_tr)
    return best_clf, FEATURES, best_name


# ── 6. Guardar modelo + estadísticas ──────────────────────────────────────────
def save_model(clf, features, thresholds, model_name, stats):
    artifact = {
        'model':      clf,
        'features':   features,
        'thresholds': thresholds,
        'model_name': model_name,
        'trained_at': datetime.now().isoformat(),
        'classes':    list(clf.classes_),
        'stats':      stats,       # <── estadísticas reales del negocio
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n💾 Modelo guardado: {MODEL_PATH}")


# ── 7. Gráficas ────────────────────────────────────────────────────────────────
def plot_insights(df, stats):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('WashApp PRO — Análisis del Negocio', fontsize=16, fontweight='bold')

    order = ['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo']

    # (a) Lavados por hora
    hc = pd.Series(stats['washes_by_hour'])
    peak = stats['peak_hour']
    axes[0,0].bar(hc.index, hc.values,
                  color=['#fb8c00' if h == peak else '#6e8efb' for h in hc.index])
    axes[0,0].set_title('Lavados por Hora')
    axes[0,0].set_xlabel('Hora'); axes[0,0].set_ylabel('Total lavados')

    # (b) Lavados por día
    dc = pd.Series({d: stats['washes_by_day'][d] for d in order})
    max_d = max(stats['washes_by_day'], key=stats['washes_by_day'].get)
    axes[0,1].bar(dc.index, dc.values,
                  color=['#fb8c00' if d == max_d else '#6e8efb' for d in order])
    axes[0,1].set_title('Lavados por Día de Semana')
    axes[0,1].tick_params(axis='x', rotation=30)

    # (c) Ingresos por mes
    mc = df.groupby('month')['total'].sum()
    axes[1,0].plot(mc.index, mc.values, marker='o', color='#fb8c00', lw=2)
    axes[1,0].fill_between(mc.index, mc.values, alpha=0.15, color='#fb8c00')
    axes[1,0].set_title('Ingresos por Mes')
    axes[1,0].set_xlabel('Mes'); axes[1,0].set_ylabel('Ingresos')

    # (d) Servicios más pedidos
    if stats.get('top_services'):
        svc = pd.Series(stats['top_services']).sort_values(ascending=True).tail(5)
        axes[1,1].barh(svc.index, svc.values, color='#6e8efb')
        axes[1,1].set_title('Servicios Más Pedidos')
    else:
        axes[1,1].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'analisis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"📈 Gráficas: {path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  WashApp PRO — Entrenamiento ML")
    print("=" * 55)

    df_raw, df_svc = load_data()
    df = engineer_features(df_raw, df_svc)
    stats = business_stats(df, df_svc)
    df, q33, q66 = create_demand_labels(df)
    clf, features, model_name = train(df)
    save_model(clf, features, {'bajo': q33, 'medio': q66}, model_name, stats)
    plot_insights(df, stats)

    print("\n✅ Entrenamiento completo.")
