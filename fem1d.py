"""MEF P1 para -(k u')' = f en un intervalo; material de la sesión 2.

Requiere Python 3 y NumPy. Ejecutar: python3 fem1d.py
Comprobaciones: python3 fem1d.py --verificar
Exportación: python3 fem1d.py --csv resultados_sesion2.csv

Las matrices densas hacen visible el ensamblaje; para mallas grandes se deben
usar matrices dispersas y un método de resolución apropiado.
"""
from pathlib import Path
import argparse
import csv
import numpy as np


def _valores(funcion, puntos, nombre):
    """Admite funciones vectorizadas y funciones constantes, como lambda x: 1."""
    valores = np.asarray(funcion(puntos), dtype=float)
    valores = np.broadcast_to(valores, puntos.shape)
    if not np.all(np.isfinite(valores)):
        raise ValueError(f"{nombre} debe devolver valores finitos")
    return valores


def ensamblar(nodos, f, k=lambda x: 1.0):
    """Devuelve nodos, matriz completa y cargas antes de imponer el contorno.

    Usa dos puntos de Gauss por elemento. La rigidez es exacta si k tiene
    grado <= 3; las cargas son exactas si f tiene grado <= 2 por elemento.
    Para coeficientes o cargas con discontinuidades, alinear la malla con
    ellas o integrar cada tramo por separado.
    """
    x = np.asarray(nodos, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("Se necesitan al menos dos nodos en un vector")
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("Los nodos deben ser finitos y estrictamente crecientes")
    A = np.zeros((len(x), len(x)))
    F = np.zeros(len(x))
    xi = np.array([(1 - 1 / np.sqrt(3)) / 2,
                   (1 + 1 / np.sqrt(3)) / 2])
    pesos = np.array([0.5, 0.5])
    formas = np.column_stack((1 - xi, xi))
    patron = np.array([[1.0, -1.0], [-1.0, 1.0]])
    for e in range(len(x) - 1):
        ids = np.array([e, e + 1])
        h = x[e + 1] - x[e]
        xq = x[e] + h * xi
        kq = _valores(k, xq, "k")
        if np.any(kq <= 0):
            raise ValueError("k debe ser positivo en los puntos de cuadratura")
        fq = _valores(f, xq, "f")
        Ae = (pesos @ kq) / h * patron
        Fe = h * (formas.T @ (pesos * fq))
        A[np.ix_(ids, ids)] += Ae
        F[ids] += Fe
    return x, A, F


def resolver(nodos, f, k=lambda x: 1.0, u_izq=0.0,
             u_der=0.0, flujo_der=0.0):
    """Resuelve con Dirichlet izquierdo y Dirichlet o Neumann derecho.

    Si u_der es un número, impone ese valor de Dirichlet y flujo_der debe
    ser cero. Si u_der=None, impone k(b)*u'(b)=flujo_der.
    Este dato es la derivada con coeficiente, no el flujo físico -k*u'.
    Se supone k uniformemente positivo en el intervalo; el código sólo
    comprueba la positividad en los puntos de Gauss.
    """
    x, A, F = ensamblar(nodos, f, k)
    if not np.isfinite(u_izq):
        raise ValueError("El dato izquierdo debe ser finito")
    U = np.zeros(len(x))
    U[0] = u_izq
    if u_der is None:
        if not np.isfinite(flujo_der):
            raise ValueError("El dato de Neumann debe ser finito")
        D = np.array([0])
        F[-1] += flujo_der
    else:
        if not np.isfinite(u_der) or flujo_der != 0:
            raise ValueError("Elegir Dirichlet derecho o Neumann derecho")
        U[-1] = u_der
        D = np.array([0, len(x) - 1])
    I = np.setdiff1d(np.arange(len(x)), D)
    # Corregir las cargas ANTES de eliminar las columnas conocidas.
    b = F[I] - A[np.ix_(I, D)] @ U[D]
    if len(I):
        U[I] = np.linalg.solve(A[np.ix_(I, I)], b)
    return x, U


def errores(nodos, U, exacta, derivada_exacta):
    """Integra errores de las funciones, no sólo diferencias en los nodos.

    Tres puntos de Gauss por elemento. Devuelve L2, seminorma H1 y norma H1.
    Para coeficiente k=1, la seminorma H1 coincide con la norma de energía.
    """
    x = np.asarray(nodos, dtype=float)
    U = np.asarray(U, dtype=float)
    if x.ndim != 1 or U.shape != x.shape or len(x) < 2:
        raise ValueError("Nodos y U deben ser vectores de igual tamaño")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(U)):
        raise ValueError("Los nodos y la solución deben ser finitos")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Los nodos deben ser estrictamente crecientes")
    xi = np.array([(1 - np.sqrt(3 / 5)) / 2, 0.5,
                   (1 + np.sqrt(3 / 5)) / 2])
    pesos = np.array([5 / 18, 4 / 9, 5 / 18])
    e0 = e1 = 0.0
    for e in range(len(x) - 1):
        h = x[e + 1] - x[e]
        xq = x[e] + h * xi
        uh = (1 - xi) * U[e] + xi * U[e + 1]
        duh = (U[e + 1] - U[e]) / h
        e0 += h * np.sum(pesos * (_valores(exacta, xq, "u") - uh) ** 2)
        e1 += h * np.sum(pesos *
                         (_valores(derivada_exacta, xq, "u'") - duh) ** 2)
    return np.sqrt(e0), np.sqrt(e1), np.sqrt(e0 + e1)


def verificar():
    """Comprueba casos con solución conocida y propiedades del ensamblaje."""
    _, Ak, Fk = ensamblar([0, 1], lambda x: x)
    np.testing.assert_allclose(Fk, [1/6, 1/3], atol=1e-13)
    _, Ak, _ = ensamblar([0, 0.5], lambda x: 0.0, k=lambda x: 1+x)
    np.testing.assert_allclose(Ak, 2.5*np.array([[1, -1], [-1, 1]]), atol=1e-13)
    x = np.array([0.0, 0.1, 0.4, 0.7, 1.0])
    _, A, F = ensamblar(x, lambda x: 1.0)
    np.testing.assert_allclose(A, A.T, atol=1e-13)
    np.testing.assert_allclose(A @ np.ones(len(x)), 0, atol=1e-13)
    np.testing.assert_allclose(np.sum(F), 1, atol=1e-13)
    np.linalg.cholesky(A[1:-1, 1:-1])
    # Para k=1 y carga integrada exactamente, P1 interpola en los nodos.
    _, U = resolver(x, lambda x: 1.0)
    np.testing.assert_allclose(U, x * (1 - x) / 2, atol=1e-13)
    # Levantamiento: detecta la omisión de A_ID U_D.
    _, U = resolver(x, lambda x: 0.0, u_izq=1.0, u_der=2.0)
    np.testing.assert_allclose(U, 1 + x, atol=1e-13)
    # Problema mixto con dato natural no nulo.
    _, U = resolver(x, lambda x: 1.0, u_der=None, flujo_der=2.0)
    np.testing.assert_allclose(U, 3 * x - x**2 / 2, atol=1e-13)
    # Intervalo de longitud distinta de uno.
    y = np.array([2.0, 2.2, 2.9, 3.0])
    _, U = resolver(y, lambda x: 0.0, u_izq=5.0, u_der=7.0)
    np.testing.assert_allclose(U, 5 + 2 * (y - 2), atol=1e-13)
    # Error funcional positivo aunque el error nodal sea cero.
    x2, U2 = resolver([0, 0.5, 1], lambda x: 1.0)
    e0, e1, _ = errores(x2, U2, lambda x: x * (1 - x) / 2,
                        lambda x: 0.5 - x)
    np.testing.assert_allclose([e0**2, e1**2], [1 / 1920, 1 / 48], rtol=1e-12)
    # Solución fabricada con coeficiente variable: k=1+x, u=x(1-x).
    eprev = None
    for N in (16, 32, 64):
        t, W = resolver(np.linspace(0, 1, N + 1), lambda x: 1 + 4*x,
                        k=lambda x: 1 + x)
        ecur = errores(t, W, lambda x: x*(1-x), lambda x: 1-2*x)
        if eprev is not None:
            assert 3.8 < eprev[0] / ecur[0] < 4.2
            assert 1.9 < eprev[1] / ecur[1] < 2.1
        eprev = ecur
    print("Verificación correcta: ensamblaje, contorno, errores y coeficiente variable.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verificar", action="store_true")
    parser.add_argument("--csv", type=Path, help="Guardar la tabla de errores")
    args = parser.parse_args()
    if args.verificar:
        verificar()
    x, U = resolver(np.linspace(0, 1, 5), lambda x: 1.0)
    print("\nEjemplo de cuatro elementos; columnas x y U:")
    print(np.column_stack((x, U)))
    print("\nN     h           error L2       seminorma H1   tasa L2   tasa H1")
    filas = []
    anteriores = None
    for N in (4, 8, 16, 32, 64):
        x, U = resolver(np.linspace(0, 1, N + 1), lambda x: 1.0)
        e0, e1, _ = errores(x, U, lambda x: x*(1-x)/2, lambda x: 0.5-x)
        tasas = (np.nan, np.nan) if anteriores is None else (
            np.log2(anteriores[0]/e0), np.log2(anteriores[1]/e1))
        filas.append((N, 1/N, e0, e1, *tasas))
        texto_tasas = ["  -  " if np.isnan(t) else f"{t:.3f}" for t in tasas]
        print(f"{N:<5} {1/N:.6f}    {e0:.6e}   {e1:.6e}   "
              f"{texto_tasas[0]}     {texto_tasas[1]}")
        anteriores = (e0, e1)
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(["N", "h", "error_L2", "seminorma_H1", "tasa_L2", "tasa_H1"])
            for fila in filas:
                escritor.writerow(["" if isinstance(v, float) and np.isnan(v) else v
                                    for v in fila])
        print(f"\nTabla guardada en {args.csv}")


if __name__ == "__main__":
    main()
