import numpy as np
import pandas as pd

# ============================================================
# FUNCIONES
# ============================================================

def f_1d(x):
    """f(x) = (x - 3)^2 + 5"""
    return (x - 3)**2 + 5

def df_1d(x):
    """Derivada de f: f'(x) = 2(x - 3)"""
    return 2 * (x - 3)

def f_2d(x, y):
    """f(x, y) = x^2 + y^2 - 4x - 2y + 5"""
    return x**2 + y**2 - 4*x - 2*y + 5

def grad_2d(x, y):
    """Gradiente de f: [2x - 4, 2y - 2]"""
    return np.array([2*x - 4, 2*y - 2])

# ============================================================
# GRADIENTE DESCENDENTE 1D
# ============================================================

def gradiente_descendente_1d(x_inicial, learning_rate, max_iter=1000, tolerancia=1e-6):
    # 1. Inicializa las variables
    x_actual = x_inicial
    historial_x = [x_inicial]
    historial_f = [f_1d(x_inicial)]
    convergido = False

    # 2. Loop principal
    for i in range(max_iter):
        #    a. Calcula el gradiente
        g = df_1d(x_actual)
        #    b. Actualiza x
        x_nuevo = x_actual - learning_rate * g

        # Detener si diverge
        if not np.isfinite(x_nuevo) or abs(x_nuevo) > 1e10:
            break

        #    c. Guarda en historial
        historial_x.append(x_nuevo)
        historial_f.append(f_1d(x_nuevo))

        #    d. Verifica convergencia
        if abs(x_nuevo - x_actual) < tolerancia:
            convergido = True
            break

        #    e. Actualiza x_actual
        x_actual = x_nuevo

    # 3. Retorna el diccionario con resultados
    return {
        'x_final':     x_actual,
        'f_final':     f_1d(x_actual),
        'iteraciones': len(historial_x) - 1,
        'convergido':  convergido,
        'historial_x': historial_x,
        'historial_f': historial_f,
    }

# ============================================================
# GRADIENTE DESCENDENTE 2D
# ============================================================

def gradiente_descendente_2d(x_inicial, y_inicial, learning_rate, max_iter=1000, tolerancia=1e-6):
    # 1. Inicializa
    x_actual, y_actual = x_inicial, y_inicial
    historial_x = [x_inicial]
    historial_y = [y_inicial]
    historial_f = [f_2d(x_inicial, y_inicial)]
    convergido = False

    # 2. Loop principal
    for i in range(max_iter):
        #    a. Calcula gradiente
        g = grad_2d(x_actual, y_actual)

        #    b. Actualiza ambos parametros
        x_nuevo = x_actual - learning_rate * g[0]
        y_nuevo = y_actual - learning_rate * g[1]

        # Detener si diverge
        if not np.isfinite(x_nuevo) or not np.isfinite(y_nuevo) or abs(x_nuevo) > 1e10:
            break

        #    c. Guarda en historiales
        historial_x.append(x_nuevo)
        historial_y.append(y_nuevo)
        historial_f.append(f_2d(x_nuevo, y_nuevo))

        #    d. Verifica convergencia
        if np.linalg.norm(g) < tolerancia:
            convergido = True
            break

        #    e. Actualiza x_actual, y_actual
        x_actual, y_actual = x_nuevo, y_nuevo

    # 3. Retorna diccionario con resultados
    return {
        'x_final':     x_actual,
        'y_final':     y_actual,
        'f_final':     f_2d(x_actual, y_actual),
        'iteraciones': len(historial_x) - 1,
        'convergido':  convergido,
        'historial_x': historial_x,
        'historial_y': historial_y,
        'historial_f': historial_f,
    }

# ============================================================
# GENERACION DEL CSV
# ============================================================

learning_rates_1d = [0.001, 0.01, 0.1, 0.5, 0.9, 1.0, 1.5]
learning_rates_2d = [0.001, 0.01, 0.1, 0.5]
puntos_iniciales  = [(-1.0, 4.0), (5.0, -1.0), (0.0, 0.0)]

filas = []

# --- Experimentos 1D ---
for lr in learning_rates_1d:
    res = gradiente_descendente_1d(x_inicial=-2.0, learning_rate=lr, max_iter=200)
    filas.append({
        'learning_rate': lr,
        'dimension':     '1D',
        'x_inicial':     -2.0,
        'y_inicial':     np.nan,
        'x_final':       res['x_final'],
        'y_final':       np.nan,
        'valor_minimo':  res['f_final'],
        'iteraciones':   res['iteraciones'],
        'convergido':    res['convergido'],
    })

# --- Experimentos 2D ---
for lr in learning_rates_2d:
    for (xi, yi) in puntos_iniciales:
        res = gradiente_descendente_2d(x_inicial=xi, y_inicial=yi, learning_rate=lr, max_iter=200)
        filas.append({
            'learning_rate': lr,
            'dimension':     '2D',
            'x_inicial':     xi,
            'y_inicial':     yi,
            'x_final':       res['x_final'],
            'y_final':       res['y_final'],
            'valor_minimo':  res['f_final'],
            'iteraciones':   res['iteraciones'],
            'convergido':    res['convergido'],
        })

# Crea el DataFrame y guarda CSV
df_experimentos = pd.DataFrame(filas).sort_values('iteraciones').reset_index(drop=True)
df_experimentos.to_csv('experimentos_gd.csv', index=False)

# Muestra resultados
print("CONTENIDO DEL CSV DE EXPERIMENTOS")
print("=" * 80)
print(df_experimentos.to_string(index=False))
print(f"\nTotal de experimentos: {len(df_experimentos)}")
print("Archivo guardado como: experimentos_gd.csv")
