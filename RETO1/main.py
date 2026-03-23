import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(42)

# ── Cargar datos ──────────────────────────────────────────────────────────────
df_original = sns.load_dataset('penguins')
df = df_original.dropna().reset_index(drop=True)

X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ── Funciones ────────────────────────────────────────────────────────────────

def clasificador_humano(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    if flipper_length_mm >= 203: # Gentoo tiene aletas largas
        if body_mass_g > 4000 and bill_depth_mm < 18: # Confirmar que si pesa mucho y tiene pico poco profundo, es Gentoo
         return "Gentoo"
        elif bill_depth_mm < 18: # Adelie también puede tener aleta larga pero su pico es menos profundo
         return "Adelie"
        else: # Cualquier otro caso es Chinstrap
         return "Chinstrap"
    elif body_mass_g < 4000 and bill_length_mm < 45: # Si pesa poco, no tiene una aleta tan larga y el pico no es tan largo, es un Adelie
         return "Adelie"
    elif bill_length_mm < 45 and bill_depth_mm < 20: # Separar más Adelie
         return "Adelie"
    elif bill_length_mm > 45: # Separar los chinstrap esperando que no queden muchos Adelie con pico largo
         return "Chinstrap"
    else: # Por descarte quedan Adelie y pocos Chinstrap
        if bill_depth_mm >= 19: # Los Chinstrap que quedan tienen el pico menos profundo
            return "Chinstrap"
        else:
            return "Adelie"


def arbol_decision():
    modelo_ml = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, random_state=42)
    modelo_ml.fit(X_train, y_train)
    return modelo_ml


def clasificador_humano_v2(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """
    Versión mejorada del clasificador humano.
    
    DOCUMENTA TUS CAMBIOS:
    1. Identifiqué que si divido los pingüinos restantes por una longitud del pico menor a 42mm, puedo clasificar muchos Adelie sin riesgo de equivocarme.
         Por lo tanto, bill_length_mm < 42 es "Adelie"
    2. Posterior a esto, también identifico gracias al gráfico de las medidas del pico, que los Chinstrap tienen un pico con una mayor longitud, por esto:
         bill_length_mm >= 47 es "Chinstrap"
    3. Continuando con las observaciones, clasifico casos especiales de Adelie que identifico apartados de Chinstrap y puedo obtener sin riesgo de equivocarme.
         bill_length_mm < 45 y bill_depth_mm > 18 es "Adelie"
    4. Como otro filtro, los pingüinos Adelie suelen ser más pesados, por esto decido agregar una condición extra por cualquiera de esta especie que se me pasara.
         body_mass_g > 4000 es "Adelie"
    5. Finalmente, por descarte solo deben quedarme puros "Chinstrap".

    """
    # CLASIFICADOR BRISA:
    if flipper_length_mm >= 203: # Gentoo tiene aletas largas
        if body_mass_g > 4000 and bill_depth_mm < 18: # Confirmar que si pesa mucho y tiene pico poco profundo, es Gentoo
         return "Gentoo"
        elif bill_depth_mm < 18: # Adelie también puede tener aleta larga pero su pico es menos profundo
         return "Adelie"
        else: # Cualquier otro caso es Chinstrap
         return "Chinstrap"
    elif bill_length_mm < 42: # Separo Adelies de los que estoy segura PASO 1
          return "Adelie"
    elif bill_length_mm >= 47: # Separar Chinstrap que conozco PASO 2
          return "Chinstrap"
    elif bill_length_mm < 45 and bill_depth_mm > 18: # Separar casos especiales de Adelie PASO 3
          return "Adelie"
    elif body_mass_g > 4000: # Si pesa mucho es un Adelie
         return "Adelie"
    else: # Por descarte quedan Chinstrap
        return "Chinstrap"


def resultado_final(modelo_ml):
    print("\n\n")

    pred_humano = []
    pred_maquina = modelo_ml.predict(X_test)
    pred_humano_v2 = []

    for idx, row in X_test.iterrows():
        pred_humano.append(clasificador_humano(
            row['bill_length_mm'], row['bill_depth_mm'],
            row['flipper_length_mm'], row['body_mass_g']))
        pred_humano_v2.append(clasificador_humano_v2(
            row['bill_length_mm'], row['bill_depth_mm'],
            row['flipper_length_mm'], row['body_mass_g']))

    # Construir el dataframe
    resultados = X_test.copy()
    resultados.columns = ['Pico L', 'Pico D', 'Aleta', 'Masa']
    resultados['Real']      = y_test.values
    resultados['Humano']    = pred_humano
    resultados['Maquina']   = pred_maquina
    resultados['Humano v2'] = pred_humano_v2

    # Resumen de accuracy
    print("="*80)
    print("\t\t\tAccuracy por clasificador")
    print("="*80)

    for col in ['Humano', 'Maquina', 'Humano v2']:
        acc = accuracy_score(resultados['Real'], resultados[col])
        print(f"  {col:<12}: {acc:.2%}")

    return resultados


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    modelo_ml = arbol_decision()

    resultados = resultado_final(modelo_ml)
    print("\n")
    print("=" * 80)
    print("\t\t\tCONTENIDO DEL CSV COMPARATIVO")
    print("=" * 80)
    print(resultados.to_string(index=False))

    resultados.to_csv('resultados_clasificadores.csv', index=False)
    print("\nDataframe guardado en: resultados_clasificadores.csv\n")