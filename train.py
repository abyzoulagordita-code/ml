#!/usr/bin/env python3
"""
WashApp PRO — Script de Entrenamiento ML
Conecta a MySQL, entrena un clasificador de demanda (Bajo/Medio/Alto)
y guarda el modelo listo para servir via api.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

# ── Configuración ──────────────────────────────────────────────────────────────
DB_HOST     = os.getenv('DB_HOST',     'localhost')
DB_PORT     = int(os.getenv('DB_PORT', '3306'))
DB_NAME     = os.getenv('DB_NAME',     'DB_SanFelipe')
DB_USER     = os.getenv('DB_USER',     'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# Para Railway usa:
# DB_HOST=mysql.railway.internal  DB_NAME=railway
# DB_USER=root  DB_PASSWORD=BtBIwDvSZMEMLztCZSSjpglrbvuSLgDj  DB_PORT=3306

OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUT_DIR, 'demand_model.pkl')
PLOT_DIR   = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

DAY_MAP = {0:'Lunes', 1:'Martes', 2:'Miercoles', 3:'Jueves',
           4:'Viernes', 5:'Sabado', 6:'Domingo'}

# ── 1. Carga ───────────────────────────────────────────────────────────────────
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

    cur.execute("SELECT id, date, total, employee, client, car FROM wash_record ORDER BY date")
    df = pd.DataFrame(cur.fetchall())

    try:
        cur.execute("SELECT wash_record_id, service_offered FROM wash_record_service_offered")
        df_svc = pd.DataFrame(cur.fetchall())
    except Exception:
        df_svc = pd.DataFrame(columns=['wash_record_id', 'service_offered'])

    try:
        cur.execute("SELECT id, name, last_name FROM employee")
        df_emp = pd.DataFrame(cur.fetchall())
        df_emp['full_name'] = df_emp['name'] + ' ' + df_emp['last_name']
    except Exception:
        df_emp = pd.DataFrame()

    conn.close()
    print(f"   Lavados : {len(df)} registros")
    return df, df_svc, df_emp

# ── 2. Feature engineering ─────────────────────────────────────────────────────
def engineer_features(df, df_svc):
    df = df.copy()
    df['date']        = pd.to_datetime(df['date'])
    df['hour']        = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek        # 0=Lunes … 6=Domingo
    df['day_name']    = df['day_of_week'].map(DAY_MAP)
    df['month']       = df['date'].dt.month
    df['week']        = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
    df['date_only']   = df['date'].dt.date

    if not df_svc.empty:
        cnt = df_svc.groupby('wash_record_id').size().rename('num_services')
        df  = df.merge(cnt, left_on='id', right_index=True, how='left')
    else:
        df['num_services'] = 1
    df['num_services'] = df['num_services'].fillna(1).astype(int)

    return df

# ── 3. Etiqueta de demanda diaria ──────────────────────────────────────────────
def create_demand_labels(df):
    daily = df.groupby('date_only').agg(
        wash_count=('id', 'count'),
        revenue=('total', 'sum'),
    ).reset_index()

    q33 = daily['wash_count'].quantile(0.33)
    q66 = daily['wash_count'].quantile(0.66)

    def label(n):
        if n <= q33:  return 'Bajo'
        if n <= q66:  return 'Medio'
        return 'Alto'

    daily['demand_label'] = daily['wash_count'].apply(label)
    df = df.merge(daily[['date_only', 'demand_label', 'wash_count', 'revenue']],
                  on='date_only', how='left')

    print(f"\n📊 Umbrales: Bajo ≤ {q33:.0f}  |  Medio ≤ {q66:.0f}  |  Alto > {q66:.0f}  lavados/día")
    print(daily['demand_label'].value_counts().rename('días').to_string())
    return df, q33, q66

# ── 4. Entrenamiento ───────────────────────────────────────────────────────────
def train(df):
    FEATURES = ['hour', 'day_of_week', 'month', 'is_weekend', 'num_services']

    clean = df[FEATURES + ['demand_label']].dropna()
    X = clean[FEATURES]
    y = clean['demand_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=10,
            random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.1, random_state=42
        ),
    }

    print("\n🔬 Evaluando modelos (CV 5-fold):")
    best_clf, best_score, best_name = None, 0.0, ''
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, clf in candidates.items():
        cv = cross_val_score(clf, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
        print(f"   {name:20s}: {cv.mean():.2%} ± {cv.std():.2%}")
        if cv.mean() > best_score:
            best_score, best_clf, best_name = cv.mean(), clf, name

    print(f"\n🏆 Ganador: {best_name}  (CV accuracy {best_score:.2%})")
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    print(f"\n📋 Reporte en test set:\n{classification_report(y_test, y_pred)}")

    # Importancia de features
    imp = pd.Series(best_clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("📌 Importancia de features:")
    for feat, val in imp.items():
        bar = '█' * int(val * 40)
        print(f"   {feat:20s} {bar} {val:.3f}")

    return best_clf, FEATURES, best_name

# ── 5. Guardar ─────────────────────────────────────────────────────────────────
def save_model(clf, features, thresholds, model_name):
    artifact = {
        'model':      clf,
        'features':   features,
        'thresholds': thresholds,
        'model_name': model_name,
        'trained_at': datetime.now().isoformat(),
        'classes':    list(clf.classes_),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n💾 Modelo guardado en: {MODEL_PATH}")

# ── 6. Visualizaciones ─────────────────────────────────────────────────────────
def plot_insights(df, df_emp):
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('WashApp PRO — Dashboard de Análisis', fontsize=16, fontweight='bold', y=1.01)

    # (a) Lavados por hora
    hc = df.groupby('hour').size()
    axes[0,0].bar(hc.index, hc.values, color=['#fb8c00' if v == hc.max() else '#6e8efb' for v in hc.values])
    axes[0,0].set_title('Lavados por Hora del Día')
    axes[0,0].set_xlabel('Hora'); axes[0,0].set_ylabel('Lavados')

    # (b) Lavados por día de semana
    order = ['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo']
    dc = df.groupby('day_name').size().reindex(order, fill_value=0)
    clr = ['#3949ab' if d in ['Sabado','Domingo'] else '#6e8efb' for d in order]
    axes[0,1].bar(dc.index, dc.values, color=clr)
    axes[0,1].set_title('Lavados por Día de la Semana')
    axes[0,1].tick_params(axis='x', rotation=30)

    # (c) Ingresos por mes
    mc = df.groupby('month')['total'].sum()
    axes[1,0].plot(mc.index, mc.values, marker='o', color='#fb8c00', lw=2)
    axes[1,0].fill_between(mc.index, mc.values, alpha=0.15, color='#fb8c00')
    axes[1,0].set_title('Ingresos por Mes')
    axes[1,0].set_xlabel('Mes'); axes[1,0].set_ylabel('Ingresos ($COP)')

    # (d) Distribución de demanda
    lc = df['demand_label'].value_counts().reindex(['Alto','Medio','Bajo'], fill_value=0)
    axes[1,1].pie(lc, labels=lc.index, autopct='%1.1f%%',
                  colors=['#43a047','#fb8c00','#e53935'], startangle=90)
    axes[1,1].set_title('Distribución de Demanda Diaria')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'analisis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"📈 Gráficas guardadas: {path}")
    plt.close()

# ── 7. Prueba rápida ───────────────────────────────────────────────────────────
def quick_test():
    art = joblib.load(MODEL_PATH)
    clf = art['model']
    day_rev = {v: k for k, v in DAY_MAP.items()}
    print("\n🧪 Prueba rápida de predicción:")
    for hora, dia in [(9,'Lunes'), (14,'Viernes'), (18,'Domingo'), (11,'Sabado')]:
        dow = day_rev[dia]
        X = pd.DataFrame([[hora, dow, datetime.now().month, int(dow>=5), 1]],
                         columns=art['features'])
        pred  = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        conf  = f"{max(proba)*100:.1f}%"
        print(f"   {dia:10s} {hora:02d}:00  →  {pred:6s}  (confianza {conf})")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  WashApp PRO — Entrenamiento ML")
    print("=" * 55)

    df_raw, df_svc, df_emp = load_data()
    df = engineer_features(df_raw, df_svc)
    df, q33, q66 = create_demand_labels(df)
    clf, features, model_name = train(df)
    save_model(clf, features, {'bajo': float(q33), 'medio': float(q66)}, model_name)
    plot_insights(df, df_emp)
    quick_test()

    print("\n✅ Listo! Ejecuta  python api.py  para servir predicciones por HTTP.")
