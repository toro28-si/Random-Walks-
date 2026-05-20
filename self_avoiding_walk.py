"""
MODELO DE CAMINATA AUTOEVITANTE (SAW)
======================================
Simulación de polímeros o cadenas que no se cruzan a sí mismas.
Se calcula:
- Distancia promedio <d_n> vs número de pasos
- Distancia cuadrática promedio <d_n²> vs pasos
- Exponente de Flory v (teórico = 0.75 en 2D)

Autor: Rodrigo Estrada
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns 

# ==================================================
# 1. FUNCIONES PRINCIPALES
# ==================================================

def es_caminata_valida(p_x, p_y):
    """Verifica que la caminata no se cruce a sí misma"""
    posiciones = set()
    for i in range(len(p_x)):
        pos = (p_x[i], p_y[i])
        if pos in posiciones:
            return False
        posiciones.add(pos)
    return True

def generar_caminata_autoevitante(n_pasos):
    """
    Genera UNA caminata autoevitante de n_pasos
    Retorna: arrays de x, y, y la distancia final
    """
    x = 0
    y = 0
    p_x = np.zeros(n_pasos)
    p_y = np.zeros(n_pasos)
    
    for i in range(n_pasos):
        # Intentar mover en una dirección aleatoria
        intentos = 0
        movido = False
        
        while not movido and intentos < 100:
            b = random.random()
            if b < 0.25:
                x_nuevo, y_nuevo = x + 1, y
            elif b < 0.5:
                x_nuevo, y_nuevo = x, y + 1
            elif b < 0.75:
                x_nuevo, y_nuevo = x - 1, y
            else:
                x_nuevo, y_nuevo = x, y - 1
            
            # Verificar si la nueva posición ya fue visitada
            if (x_nuevo, y_nuevo) not in set(zip(p_x[:i], p_y[:i])):
                x, y = x_nuevo, y_nuevo
                movido = True
            intentos += 1
        
        if not movido:
            # Si no se pudo mover, la caminata murió
            return p_x[:i], p_y[:i], np.sqrt(x**2 + y**2), i
        
        p_x[i] = x
        p_y[i] = y
    
    return p_x, p_y, np.sqrt(x**2 + y**2), n_pasos

# ==================================================
# 2. SIMULACIÓN DE MÚLTIPLES CAMINATAS
# ==================================================

def simular_caminatas_autoevitantes(n_pasos_max=100, n_caminatas=100):
    """
    Simula n_caminatas caminatas autoevitantes
    Calcula <d_n> y <d_n²> para cada número de pasos
    """
    # Inicializar matrices para almacenar resultados
    distancias_por_paso = [[] for _ in range(n_pasos_max)]
    
    print(f"Simulando {n_caminatas} caminatas autoevitantes...")
    
    for caminata in range(n_caminatas):
        if caminata % 20 == 0:
            print(f"  Progreso: {caminata}/{n_caminatas}")
        
        p_x, p_y, dist_final, pasos_realizados = generar_caminata_autoevitante(n_pasos_max)
        
        # Guardar distancia en cada paso
        for paso in range(len(p_x)):
            dist = np.sqrt(p_x[paso]**2 + p_y[paso]**2)
            distancias_por_paso[paso].append(dist)
    
    # Calcular promedios
    dn_promedio = []
    dn2_promedio = []
    pasos_validos = []
    
    for paso in range(n_pasos_max):
        if len(distancias_por_paso[paso]) > 10:  # Suficientes muestras
            dn_promedio.append(np.mean(distancias_por_paso[paso]))
            dn2_promedio.append(np.mean([d**2 for d in distancias_por_paso[paso]]))
            pasos_validos.append(paso + 1)
    
    return np.array(pasos_validos), np.array(dn_promedio), np.array(dn2_promedio)

# ==================================================
# 3. ANÁLISIS DEL EXPONENTE DE FLORY
# ==================================================

def analizar_exponentes(pasos, dn, dn2):
    """
    Ajusta una ley de potencia d ∝ n^ν
    Usa regresión lineal en escala log-log
    """
    log_pasos = np.log(pasos)
    log_dn = np.log(dn)
    log_dn2 = np.log(dn2)
    
    # Regresión para <d_n>
    slope_dn, intercept_dn, r_dn, p_dn, std_dn = stats.linregress(log_pasos, log_dn)
    nu_dn = slope_dn
    A_dn = np.exp(intercept_dn)
    
    # Regresión para <d_n²>
    slope_dn2, intercept_dn2, r_dn2, p_dn2, std_dn2 = stats.linregress(log_pasos, log_dn2)
    nu_dn2 = slope_dn2 / 2  # ν = exponente de <d_n²>/2
    A_dn2 = np.exp(intercept_dn2)
    
    return {
        'nu_dn': nu_dn,
        'nu_dn2': nu_dn2,
        'r2_dn': r_dn**2,
        'r2_dn2': r_dn2**2,
        'A_dn': A_dn,
        'A_dn2': A_dn2
    }

# ==================================================
# 4. GRÁFICAS PROFESIONALES
# ==================================================

def graficar_resultados(pasos, dn, dn2, resultados_ajuste):
    """Genera gráficas de calidad profesional"""
    
    sns.set_theme(style="whitegrid", palette="viridis")
    
    # Predicciones del ajuste
    dn_ajustado = resultados_ajuste['A_dn'] * pasos ** resultados_ajuste['nu_dn']
    dn2_ajustado = resultados_ajuste['A_dn2'] * pasos ** (2 * resultados_ajuste['nu_dn2'])
    
    # FIGURA 1: <d_n> vs n (escala lineal)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(pasos, dn, 'bo-', markersize=4, linewidth=1.5, label='Simulación')
    plt.plot(pasos, dn_ajustado, 'r--', linewidth=2, 
             label=f'Ajuste: ν = {resultados_ajuste["nu_dn"]:.3f}')
    plt.xlabel('Número de pasos (n)', fontsize=12)
    plt.ylabel('<dₙ>', fontsize=12)
    plt.title('Distancia promedio vs Pasos', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.loglog(pasos, dn, 'bo-', markersize=4, linewidth=1.5, label='Simulación')
    plt.loglog(pasos, dn_ajustado, 'r--', linewidth=2, 
               label=f'ν = {resultados_ajuste["nu_dn"]:.3f}')
    plt.xlabel('Pasos (n) - escala log', fontsize=12)
    plt.ylabel('<dₙ> - escala log', fontsize=12)
    plt.title('Ley de escala (log-log)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('saw_distancia_vs_pasos.png', dpi=150)
    plt.show()
    
    # FIGURA 2: <d_n²> vs n
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(pasos, dn2, 'go-', markersize=4, linewidth=1.5, label='Simulación')
    plt.plot(pasos, dn2_ajustado, 'r--', linewidth=2,
             label=f'Ajuste: ν = {resultados_ajuste["nu_dn2"]:.3f}')
    plt.xlabel('Número de pasos (n)', fontsize=12)
    plt.ylabel('<dₙ²>', fontsize=12)
    plt.title('Distancia cuadrática promedio vs Pasos', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.loglog(pasos, dn2, 'go-', markersize=4, linewidth=1.5, label='Simulación')
    plt.loglog(pasos, dn2_ajustado, 'r--', linewidth=2,
               label=f'2ν = {2*resultados_ajuste["nu_dn2"]:.3f}')
    plt.xlabel('Pasos (n) - escala log', fontsize=12)
    plt.ylabel('<dₙ²> - escala log', fontsize=12)
    plt.title('Ley de escala para <dₙ²>', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('saw_distancia_cuad_vs_pasos.png', dpi=150)
    plt.show()

def graficar_una_caminata_ejemplo(n_pasos=50):
    """Muestra una caminata autoevitante individual"""
    p_x, p_y, dist, _ = generar_caminata_autoevitante(n_pasos)
    
    plt.figure(figsize=(8, 8))
    plt.plot(p_x, p_y, 'b-', linewidth=1.5, alpha=0.7, label='Trayectoria')
    plt.plot(p_x, p_y, 'bo', markersize=4, alpha=0.8)
    plt.plot(0, 0, 'go', markersize=10, label='Inicio (0,0)')
    plt.plot(p_x[-1], p_y[-1], 'ro', markersize=10, 
             label=f'Final ({p_x[-1]:.0f}, {p_y[-1]:.0f})')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(f'Caminata Autoevitante - {len(p_x)} pasos\nDistancia final: {dist:.2f}', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('saw_ejemplo.png', dpi=150)
    plt.show()
    
    return dist

# ==================================================
# 5. MAIN
# ==================================================

def main():
    print("=" * 60)
    print("   CAMINATA AUTOEVITANTE (Self-Avoiding Walk)")
    print("   Modelo de polímeros en 2D")
    print("=" * 60)
    
    # Parámetros
    n_pasos_max = 80
    n_caminatas = 150
    
    print(f"\nParámetros de la simulación:")
    print(f"  • Máximo de pasos por caminata: {n_pasos_max}")
    print(f"  • Número de caminatas: {n_caminatas}")
    print(f"  • Exponente teórico de Flory (2D): ν = 0.75")
    
    # Simular
    print("\n" + "-" * 40)
    pasos, dn, dn2 = simular_caminatas_autoevitantes(n_pasos_max, n_caminatas)
    
    # Analizar exponentes
    resultados = analizar_exponentes(pasos, dn, dn2)
    
    # Mostrar resultados
    print("\n" + "=" * 60)
    print("RESULTADOS DEL ANÁLISIS")
    print("=" * 60)
    print(f"Exponente ν (de <dₙ>):     {resultados['nu_dn']:.4f}")
    print(f"Exponente ν (de <dₙ²>/2):  {resultados['nu_dn2']:.4f}")
    print(f"Teórico (Flory, 2D):       0.7500")
    print(f"\nR² (calidad del ajuste):")
    print(f"  • Para <dₙ>:  {resultados['r2_dn']:.4f}")
    print(f"  • Para <dₙ²>: {resultados['r2_dn2']:.4f}")
    
    # Verificar si cumple con Flory
    error = abs(resultados['nu_dn2'] - 0.75) / 0.75 * 100
    if error < 10:
        print(f"\n✅ Buen ajuste: ν = {resultados['nu_dn2']:.3f} (error {error:.1f}%)")
    else:
        print(f"\n⚠️ El exponente difiere del teórico (error {error:.1f}%)")
        print(" Se necesitan más caminatas o más pasos.") 
    
    # Exportar resultados
    df = pd.DataFrame({
        'pasos': pasos,
        'distancia_promedio': dn,
        'distancia_cuad_promedio': dn2,
        'ajuste_dn': resultados['A_dn'] * pasos ** resultados['nu_dn'],
        'ajuste_dn2': resultados['A_dn2'] * pasos ** (2 * resultados['nu_dn2'])
    })
    df.to_csv('resultados_saw.csv', index=False)
    print(f"\n✅ Datos guardados en 'resultados_saw.csv'")
    
    # Gráficas
    print("\nGenerando gráficas...")
    graficar_resultados(pasos, dn, dn2, resultados)
    
    # Mostrar una caminata ejemplo
    print("\nGenerando ejemplo visual de una caminata autoevitante...")
    graficar_una_caminata_ejemplo(50)
    
    print("\n" + "=" * 60)
    print("¡Simulación completada!")
    print("=" * 60)

if __name__ == "__main__":
    main()