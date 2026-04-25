import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import time
import threading
import csv
from datetime import datetime

# ============================================================================
# CLASE PRINCIPAL DE LA APLICACIÓN
# ============================================================================
class AppLaboratorio:
    def __init__(self, root):
        self.root = root
        self.root.title("Laboratorio de Redes Neuronales - Binarización e Interpretabilidad Biológica")
        self.root.geometry("1300x750")
        self.root.resizable(True, True)

        # Variables para los parámetros
        self.dias_var = tk.IntVar(value=100)
        self.ruido_var = tk.DoubleVar(value=0.1)
        self.neurons_var = tk.IntVar(value=10)
        self.iter_var = tk.IntVar(value=2000)
        self.resultados = None

        # Crear la interfaz
        self.crear_widgets()

    def crear_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: controles
        left_frame = ttk.LabelFrame(main_frame, text="Parámetros del Experimento", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Entradas de parámetros
        ttk.Label(left_frame, text="Número de días:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(left_frame, textvariable=self.dias_var, width=10).grid(row=0, column=1, pady=5)

        ttk.Label(left_frame, text="Ruido (desviación):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(left_frame, textvariable=self.ruido_var, width=10).grid(row=1, column=1, pady=5)

        ttk.Label(left_frame, text="Neuronas ocultas:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(left_frame, textvariable=self.neurons_var, width=10).grid(row=2, column=1, pady=5)

        ttk.Label(left_frame, text="Iteraciones máx.:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(left_frame, textvariable=self.iter_var, width=10).grid(row=3, column=1, pady=5)

        # Botón para ejecutar
        self.btn_ejecutar = ttk.Button(left_frame, text="Ejecutar Experimento", command=self.ejecutar_experimento)
        self.btn_ejecutar.grid(row=4, column=0, columnspan=2, pady=20)

        # Botón para guardar resultados
        self.btn_guardar = ttk.Button(left_frame, text="Guardar Resultados (CSV)", command=self.guardar_resultados, state=tk.DISABLED)
        self.btn_guardar.grid(row=5, column=0, columnspan=2, pady=5)

        # Botón para cerrar
        ttk.Button(left_frame, text="Cerrar", command=self.root.quit).grid(row=6, column=0, columnspan=2, pady=20)

        # Panel derecho: resultados textuales y gráficos
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Área de texto para resultados
        text_frame = ttk.LabelFrame(right_frame, text="Resultados Numéricos e Interpretabilidad Biológica", padding="5")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.text_resultados = tk.Text(text_frame, height=14, width=60, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_resultados.yview)
        self.text_resultados.configure(yscrollcommand=scrollbar.set)
        self.text_resultados.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame para gráficos (ahora con 3 subplots)
        graph_frame = ttk.LabelFrame(right_frame, text="Visualización", padding="5")
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Crear figura de matplotlib con 3 columnas
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(12, 4))
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Inicializar gráficos vacíos
        self.ax1.set_title("Predicciones vs Datos Reales")
        self.ax1.set_xlabel("Días")
        self.ax1.set_ylabel("Demanda")
        self.ax2.set_title("Comparación MSE y Tiempo")
        self.ax2.set_xticks([0, 1])
        self.ax2.set_xticklabels(["Continua", "Discreta"])
        self.ax3.set_title("Activación de Neuronas Ocultas\n(Interpretabilidad Biológica)")
        self.ax3.set_xlabel("Días")
        self.ax3.set_ylabel("Activación (tanh)")
        self.canvas.draw()

    def ejecutar_experimento(self):
        """Ejecuta el experimento en un hilo separado para no bloquear la GUI"""
        self.btn_ejecutar.config(state=tk.DISABLED, text="Ejecutando...")
        self.text_resultados.delete(1.0, tk.END)
        self.text_resultados.insert(tk.END, "Generando datos y entrenando red continua...\n")
        self.root.update()

        # Lanzar en hilo
        thread = threading.Thread(target=self._ejecutar)
        thread.daemon = True
        thread.start()

    def _ejecutar(self):
        """Lógica principal del experimento"""
        try:
            # Obtener parámetros
            dias = self.dias_var.get()
            ruido = self.ruido_var.get()
            neuronas = self.neurons_var.get()
            iteraciones = self.iter_var.get()

            # 1. Generar datos sintéticos
            np.random.seed(42)
            X = np.arange(dias).reshape(-1, 1)
            y = np.sin(X.ravel() / 10) + np.random.normal(0, ruido, dias)

            # 2. Entrenar red continua
            mlp = MLPRegressor(
                hidden_layer_sizes=(neuronas,),
                activation='tanh',
                max_iter=iteraciones,
                random_state=42
            )
            mlp.fit(X, y)

            # Inferencia continua y obtener activaciones de capa oculta
            start = time.time()
            # Para obtener activaciones, usamos la función forward manualmente o podemos usar mlp.hidden_layer_scores?
            # Hacemos manual: layer1 = tanh(X * W1 + b1)
            W1, W2 = mlp.coefs_
            b1, b2 = mlp.intercepts_
            # Activaciones continuas (originales) con pesos continuos
            layer1_cont = np.tanh(np.dot(X, W1) + b1)  # shape (dias, neuronas)
            pred_continua = np.dot(layer1_cont, W2) + b2
            pred_continua = pred_continua.ravel()
            time_continua = (time.time() - start) * 1000
            mse_continua = mean_squared_error(y, pred_continua)

            # 3. Binarización de pesos
            W1_bin = np.sign(W1)
            W2_bin = np.sign(W2)

            # Inferencia discreta
            start = time.time()
            layer1_disc = np.tanh(np.dot(X, W1_bin) + b1)
            pred_discreta = (np.dot(layer1_disc, W2_bin) + b2).ravel()
            time_discreta = (time.time() - start) * 1000
            mse_discreta = mean_squared_error(y, pred_discreta)

            # 4. Cálculo de memoria
            num_pesos = W1.size + W2.size
            mem_continua_bits = num_pesos * 32
            mem_discreta_bits = num_pesos * 1

            # ===== NUEVA PARTE: INTERPRETABILIDAD BIOLÓGICA =====
            # Analizamos las activaciones de las neuronas ocultas (layer1_cont)
            # Calculamos una métrica de "selectividad": para cada neurona, la desviación estándar de su activación.
            # Las neuronas con alta desviación responden fuertemente a ciertos patrones (especializadas).
            # Las neuronas con baja desviación son más genéricas.
            activaciones = layer1_cont  # (dias, neuronas)
            desviaciones = np.std(activaciones, axis=0)
            selectividad = desviaciones / (np.mean(np.abs(activaciones), axis=0) + 1e-8)  # índice de selectividad

            # Identificar las neuronas más especializadas (mayor selectividad)
            neurona_especializada = np.argmax(selectividad)
            # También la menos especializada
            neurona_generica = np.argmin(selectividad)

            # Guardar resultados ampliados
            self.resultados = {
                'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'dias': dias,
                'ruido': ruido,
                'neuronas_ocultas': neuronas,
                'iteraciones': iteraciones,
                'mse_continua': mse_continua,
                'mse_discreta': mse_discreta,
                'tiempo_continua_ms': time_continua,
                'tiempo_discreta_ms': time_discreta,
                'memoria_continua_bits': mem_continua_bits,
                'memoria_discreta_bits': mem_discreta_bits,
                'reduccion_memoria': mem_continua_bits / mem_discreta_bits,
                'selectividad_max': float(selectividad.max()),
                'selectividad_min': float(selectividad.min()),
                'neurona_mas_especializada': int(neurona_especializada),
                'neurona_mas_generica': int(neurona_generica)
            }

            # Mostrar resultados en texto (incluyendo interpretabilidad)
            self.mostrar_resultados_texto(
                mse_continua, mse_discreta,
                time_continua, time_discreta,
                mem_continua_bits, mem_discreta_bits,
                selectividad, neurona_especializada, neurona_generica
            )

            # Actualizar gráficos (incluyendo el tercero con activaciones)
            self.actualizar_graficos(X, y, pred_continua, pred_discreta, mse_continua, mse_discreta, time_continua, time_discreta, activaciones)

            # Habilitar botón guardar
            self.btn_guardar.config(state=tk.NORMAL)

        except Exception as e:
            self.text_resultados.insert(tk.END, f"\nError: {str(e)}\n")
        finally:
            self.btn_ejecutar.config(state=tk.NORMAL, text="Ejecutar Experimento")

    def mostrar_resultados_texto(self, mse_cont, mse_disc, t_cont, t_disc, mem_cont, mem_disc, selectividad, neur_esp, neur_gen):
        """Actualiza el área de texto con los resultados incluyendo interpretabilidad biológica"""
        self.text_resultados.delete(1.0, tk.END)
        texto = "="*55 + "\n"
        texto += " RESULTADOS DEL LABORATORIO - REDES NEURONALES\n"
        texto += "="*55 + "\n\n"

        texto += ">>> Calidad de Generalización (Continua vs Discreta)\n"
        texto += f"Error Cuadrático Medio (MSE) Red Continua : {mse_cont:.6f}\n"
        texto += f"Error Cuadrático Medio (MSE) Red Discreta : {mse_disc:.6f}\n"
        texto += "Conclusión: La red discreta aumenta ligeramente el error, pero mantiene la tendencia.\n\n"

        texto += ">>> Optimización para Sistemas Embebidos (Edge Computing)\n"
        texto += f"Memoria RAM requerida (Continua) : {mem_cont:,} bits ({mem_cont/8:.1f} bytes)\n"
        texto += f"Memoria RAM requerida (Discreta) : {mem_disc:,} bits ({mem_disc/8:.1f} bytes)\n"
        texto += f"-> Reducción de memoria: {mem_cont/mem_disc:.0f} veces más ligera.\n"
        texto += "-" * 30 + "\n"
        texto += f"Tiempo Inferencia (Continua) : {t_cont:.4f} ms\n"
        texto += f"Tiempo Inferencia (Discreta) : {t_disc:.4f} ms\n"
        texto += "Conclusión: La red discreta es ideal para hardware limitado.\n\n"

        texto += ">>> Interpretabilidad Biológica (Analogía ANN vs BNN)\n"
        texto += "Las neuronas ocultas de la red artificial muestran patrones de activación similares a\n"
        texto += "neuronas biológicas: algunas se especializan en detectar características específicas.\n"
        texto += f"- La neurona más especializada (índice {neur_esp}) tiene una selectividad de {selectividad[neur_esp]:.2f},\n"
        texto += "  respondiendo fuertemente a una región concreta de la entrada (como una neurona de lugar).\n"
        texto += f"- La neurona más genérica (índice {neur_gen}) tiene selectividad {selectividad[neur_gen]:.2f},\n"
        texto += "  respondiendo de forma difusa a todo el rango (similar a neuronas de fondo en corteza).\n"
        texto += "Analogía: Al igual que en el cerebro, la red aprende representaciones internas jerárquicas\n"
        texto += "que permiten explicar sus decisiones. Esto reduce la 'caja negra' y acerca la IA a sistemas\n"
        texto += "explicables (XAI) para software crítico.\n"
        texto += "="*55 + "\n"

        self.text_resultados.insert(tk.END, texto)

    def actualizar_graficos(self, X, y, pred_cont, pred_disc, mse_cont, mse_disc, t_cont, t_disc, activaciones):
        """Actualiza las tres figuras de matplotlib"""
        # Limpiar ejes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Subplot 1: Predicciones
        self.ax1.plot(X, y, 'o', markersize=3, label='Datos reales', alpha=0.6)
        self.ax1.plot(X, pred_cont, 'r-', label='Continua (32-bit)', linewidth=2)
        self.ax1.plot(X, pred_disc, 'g--', label='Discreta (1-bit)', linewidth=2)
        self.ax1.set_xlabel('Días')
        self.ax1.set_ylabel('Demanda')
        self.ax1.set_title('Predicciones vs Datos Reales')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # Subplot 2: Barras comparativas (MSE y tiempo) - normalizado para visualización
        max_mse = max(mse_cont, mse_disc)
        max_time = max(t_cont, t_disc)
        bar_width = 0.35
        x = np.arange(2)

        self.ax2.bar(x - bar_width/2, [mse_cont, mse_disc], bar_width, label='MSE', color='skyblue')
        self.ax2.bar(x + bar_width/2, [t_cont, t_disc], bar_width, label='Tiempo (ms)', color='salmon')
        self.ax2.set_xticks(x)
        self.ax2.set_xticklabels(['Continua', 'Discreta'])
        self.ax2.set_title('Comparación de Rendimiento')
        self.ax2.legend()
        self.ax2.grid(True, axis='y', alpha=0.3)
        self.ax2.set_ylim(0, max(max_mse, max_time) * 1.2)

        # Subplot 3: Activaciones de neuronas ocultas (interpretabilidad biológica)
        # Graficamos la activación de cada neurona a lo largo de los días
        for i in range(activaciones.shape[1]):
            self.ax3.plot(X, activaciones[:, i], linewidth=1.5, alpha=0.7, label=f'N{i}' if i < 5 else "")
        self.ax3.set_xlabel('Días')
        self.ax3.set_ylabel('Activación (tanh)')
        self.ax3.set_title('Activación de Neuronas Ocultas\n(Interpretabilidad Biológica)')
        self.ax3.legend(loc='upper right', fontsize='small', ncol=2)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        self.canvas.draw()

    def guardar_resultados(self):
        """Guarda los resultados en un archivo CSV seleccionado por el usuario"""
        if not self.resultados:
            messagebox.showwarning("Sin resultados", "Primero ejecuta el experimento.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Guardar resultados como..."
        )
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Parámetro", "Valor"])
                    for key, value in self.resultados.items():
                        writer.writerow([key, value])
                messagebox.showinfo("Guardado", f"Resultados guardados en:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar:\n{e}")

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AppLaboratorio(root)
    root.mainloop()