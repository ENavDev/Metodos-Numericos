import tkinter as tk
from tkinter import ttk
import math
import matplotlib.pyplot as plt
import sympy as sp
from tkinter import messagebox
import re


def abrir_punto_fijo():
    global entry_funcion, entry_x0, entry_tolerancia, entry_max_iter, tree, label_resultado

    ventana_punto_fijo = tk.Toplevel(root)
    ventana_punto_fijo.title("Método de Punto Fijo")

    tk.Label(ventana_punto_fijo, text="Función g(x):").grid(row=0, column=0)
    entry_funcion = tk.Entry(ventana_punto_fijo)
    entry_funcion.grid(row=0, column=1)
    entry_funcion.insert(0, "0.5*(x + 2/x)")  # Usar la función como ejemplo

    tk.Label(ventana_punto_fijo, text="Valor inicial x0:").grid(row=1, column=0)
    entry_x0 = tk.Entry(ventana_punto_fijo)
    entry_x0.grid(row=1, column=1)
    entry_x0.insert(0, "1")  # Cambia el valor inicial según sea necesario

    tk.Label(ventana_punto_fijo, text="Tolerancia:").grid(row=2, column=0)
    entry_tolerancia = tk.Entry(ventana_punto_fijo)
    entry_tolerancia.grid(row=2, column=1)
    entry_tolerancia.insert(0, "1e-5")

    tk.Label(ventana_punto_fijo, text="Máximo de iteraciones:").grid(row=3, column=0)
    entry_max_iter = tk.Entry(ventana_punto_fijo)
    entry_max_iter.grid(row=3, column=1)
    entry_max_iter.insert(0, "100")

    button_calcular = tk.Button(ventana_punto_fijo, text="Calcular", command=calcular)
    button_calcular.grid(row=4, columnspan=2)

    tree = ttk.Treeview(ventana_punto_fijo, columns=('Iteración', 'x', 'Error Absoluto'), show='headings')
    tree.heading('Iteración', text='Iteración')
    tree.heading('x', text='x')
    tree.heading('Error Absoluto', text='Error Absoluto')
    tree.grid(row=5, columnspan=2)

    label_resultado = tk.Label(ventana_punto_fijo, text="")
    label_resultado.grid(row=6, columnspan=2)

def evaluar_funcion(funcion, x):
    local_scope = {'x': x, 'sin': math.sin, 'cos': math.cos, 
                   'tan': math.tan, 'exp': math.exp, 'log': math.log, 
                   'sqrt': math.sqrt}
    try:
        result = eval(funcion, {"__builtins__": None}, local_scope)
        return result
    except Exception as e:
        raise ValueError(f"Error al evaluar la función: {str(e)}")

def graficar(iteraciones):
    iteraciones_num = [i[0] for i in iteraciones]
    valores_x = [i[1] for i in iteraciones]
    errores = [i[2] for i in iteraciones]
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(iteraciones_num, valores_x, marker='o', label='Valores de x')
    plt.title('Convergencia de x en cada iteración')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de x')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iteraciones_num, errores, marker='o', color='r', label='Error Absoluto')
    plt.title('Error Absoluto en cada iteración')
    plt.xlabel('Iteración')
    plt.ylabel('Error Absoluto')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def calcular():
    try:
        funcion = entry_funcion.get()
        x0 = float(entry_x0.get())
        tolerancia = float(entry_tolerancia.get())
        max_iter = int(entry_max_iter.get())

        g = lambda x: evaluar_funcion(funcion, x)

        resultado, iteraciones = calcular_punto_fijo(g, x0, tolerancia, max_iter)

        for i in tree.get_children():
            tree.delete(i)

        for it in iteraciones:
            tree.insert("", "end", values=it)

        label_resultado.config(text=f"Resultado: {resultado:.6f}")

        graficar(iteraciones)

    except Exception as e:
        label_resultado.config(text=f"Error: {str(e)}")

def calcular_punto_fijo(g, x0, tolerancia, max_iter):
    xrold = x0
    iteraciones = []
    
    for i in range(max_iter):
        xr = g(xrold)
        error = abs(xr - xrold)
        iteraciones.append((i + 1, xr, error))
        
        if error < tolerancia:
            return xr, iteraciones
        
        xrold = xr

    raise ValueError("No se alcanzó la convergencia en el número máximo de iteraciones.")




def abrir_biseccion():
    global entry_funcion_biseccion, entry_a, entry_b, entry_tolerancia_biseccion, tree_biseccion, label_resultado_biseccion

    ventana_biseccion = tk.Toplevel(root)
    ventana_biseccion.title("Método de Bisección")

    # Entrada de la función
    tk.Label(ventana_biseccion, text="Función f(x):").grid(row=0, column=0)
    entry_funcion_biseccion = tk.Entry(ventana_biseccion)
    entry_funcion_biseccion.grid(row=0, column=1)
    entry_funcion_biseccion.insert(0, "x^3 - x - 2")  # Ejemplo de función por defecto

    # Entrada para el valor 'a'
    tk.Label(ventana_biseccion, text="Valor inicial a:").grid(row=1, column=0)
    entry_a = tk.Entry(ventana_biseccion)
    entry_a.grid(row=1, column=1)
    entry_a.insert(0, "1")  # Valor por defecto

    # Entrada para el valor 'b'
    tk.Label(ventana_biseccion, text="Valor inicial b:").grid(row=2, column=0)
    entry_b = tk.Entry(ventana_biseccion)
    entry_b.grid(row=2, column=1)
    entry_b.insert(0, "2")  # Valor por defecto

    # Entrada para la tolerancia
    tk.Label(ventana_biseccion, text="Tolerancia:").grid(row=3, column=0)
    entry_tolerancia_biseccion = tk.Entry(ventana_biseccion)
    entry_tolerancia_biseccion.grid(row=3, column=1)
    entry_tolerancia_biseccion.insert(0, "0.001")  # Tolerancia por defecto

    # Botón para ejecutar el cálculo
    button_calcular_biseccion = tk.Button(ventana_biseccion, text="Calcular", command=calcular_biseccion)
    button_calcular_biseccion.grid(row=4, columnspan=2)

    # Tabla de resultados
    tree_biseccion = ttk.Treeview(ventana_biseccion, columns=('Iteración', 'a', 'b', 'c', 'f(c)'), show='headings')
    tree_biseccion.heading('Iteración', text='Iteración')
    tree_biseccion.heading('a', text='a')
    tree_biseccion.heading('b', text='b')
    tree_biseccion.heading('c', text='c')
    tree_biseccion.heading('f(c)', text='f(c)')
    tree_biseccion.grid(row=5, columnspan=2)

    # Resultado final
    label_resultado_biseccion = tk.Label(ventana_biseccion, text="")
    label_resultado_biseccion.grid(row=6, columnspan=2)

def calcular_biseccion():
    try:
        funcion = entry_funcion_biseccion.get().replace('^', '**')  # Reemplaza ^ con ** para Python
        a = float(entry_a.get())
        b = float(entry_b.get())
        tolerancia = float(entry_tolerancia_biseccion.get())

        iteraciones_biseccion = []

        # Evalúa la función con las operaciones matemáticas correctas
        def f(x):
            return eval(funcion, {"__builtins__": None}, {'x': x, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 
                                                          'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt})

        if f(a) * f(b) >= 0:
            raise ValueError("El método de bisección no se puede aplicar porque f(a) y f(b) tienen el mismo signo.")

        iteracion = 1
        while (b - a) / 2.0 > tolerancia:
            c = (a + b) / 2.0
            fc = f(c)

            iteraciones_biseccion.append((iteracion, a, b, c, fc))
            iteracion += 1

            if fc == 0:
                break  # c es una raíz exacta
            elif f(a) * fc < 0:
                b = c  # La raíz está en el intervalo [a, c]
            else:
                a = c  # La raíz está en el intervalo [c, b]

        # Mostrar los resultados en la tabla
        for i in tree_biseccion.get_children():
            tree_biseccion.delete(i)

        for it in iteraciones_biseccion:
            tree_biseccion.insert("", "end", values=it)

        # Mostrar el resultado final
        c_final = (a + b) / 2.0
        label_resultado_biseccion.config(text=f"Raíz aproximada: {c_final:.6f}")

        # Graficar la convergencia
        graficar_biseccion(iteraciones_biseccion)

    except Exception as e:
        label_resultado_biseccion.config(text=f"Error: {str(e)}")

def graficar_biseccion(iteraciones):
    iteraciones_num = [i[0] for i in iteraciones]
    valores_c = [i[3] for i in iteraciones]

    plt.figure(figsize=(6, 4))
    plt.plot(iteraciones_num, valores_c, marker='o', label='Valor de c')
    plt.title('Convergencia del Método de Bisección')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de c')
    plt.grid(True)
    plt.legend()
    plt.show()


def abrir_newton_raphson():
    global entry_funcion_newton, entry_x0, entry_tolerancia_newton, tree_newton, label_resultado_newton

    ventana_newton = tk.Toplevel(root)
    ventana_newton.title("Método de Newton-Raphson")

    # Entrada de la función
    tk.Label(ventana_newton, text="Función f(x):").grid(row=0, column=0)
    entry_funcion_newton = tk.Entry(ventana_newton)
    entry_funcion_newton.grid(row=0, column=1)
    entry_funcion_newton.insert(0, "x**3 - x - 2")  # Ejemplo de función por defecto

    # Entrada para el valor inicial x0
    tk.Label(ventana_newton, text="Valor inicial x0:").grid(row=1, column=0)
    entry_x0 = tk.Entry(ventana_newton)
    entry_x0.grid(row=1, column=1)
    entry_x0.insert(0, "1.5")  # Valor por defecto

    # Entrada para la tolerancia
    tk.Label(ventana_newton, text="Tolerancia:").grid(row=2, column=0)
    entry_tolerancia_newton = tk.Entry(ventana_newton)
    entry_tolerancia_newton.grid(row=2, column=1)
    entry_tolerancia_newton.insert(0, "0.001")  # Tolerancia por defecto

    # Botón para ejecutar el cálculo
    button_calcular_newton = tk.Button(ventana_newton, text="Calcular", command=calcular_newton_raphson)
    button_calcular_newton.grid(row=3, columnspan=2)

    # Tabla de resultados
    tree_newton = ttk.Treeview(ventana_newton, columns=('Iteración', 'x0', 'f(x0)', 'f\'(x0)', 'x1'), show='headings')
    tree_newton.heading('Iteración', text='Iteración')
    tree_newton.heading('x0', text='x0')
    tree_newton.heading('f(x0)', text='f(x0)')
    tree_newton.heading('f\'(x0)', text="f'(x0)")
    tree_newton.heading('x1', text='x1')
    tree_newton.grid(row=4, columnspan=2)

    # Resultado final
    label_resultado_newton = tk.Label(ventana_newton, text="")
    label_resultado_newton.grid(row=5, columnspan=2)

def calcular_newton_raphson():
    try:
        # Definir la variable simbólica y la función
        x = sp.symbols('x')
        funcion = entry_funcion_newton.get().replace('^', '**')  # Reemplaza ^ con ** para Python
        
        # Convertir la función a expresión de SymPy
        f_expr = sp.sympify(funcion)
        
        # Derivar la función automáticamente
        df_expr = sp.diff(f_expr, x)
        
        # Convertir las expresiones en funciones evaluables
        f = sp.lambdify(x, f_expr, "math")
        df = sp.lambdify(x, df_expr, "math")
        
        # Obtener el valor inicial y la tolerancia
        x0 = float(entry_x0.get())
        tolerancia = float(entry_tolerancia_newton.get())

        iteraciones_newton = []
        iteracion = 1
        
        while True:
            fx0 = f(x0)
            dfx0 = df(x0)

            if dfx0 == 0:
                raise ValueError("La derivada se hizo cero. El método no puede continuar.")

            x1 = x0 - fx0 / dfx0
            iteraciones_newton.append((iteracion, x0, fx0, dfx0, x1))

            if abs(x1 - x0) < tolerancia:
                break

            x0 = x1
            iteracion += 1

        # Mostrar los resultados en la tabla
        for i in tree_newton.get_children():
            tree_newton.delete(i)

        for it in iteraciones_newton:
            tree_newton.insert("", "end", values=it)

        # Mostrar el resultado final
        label_resultado_newton.config(text=f"Raíz aproximada: {x1:.6f}")

        # Graficar la convergencia
        graficar_newton(iteraciones_newton)

    except Exception as e:
        label_resultado_newton.config(text=f"Error: {str(e)}")

def graficar_newton(iteraciones):
    iteraciones_num = [i[0] for i in iteraciones]
    valores_x1 = [i[4] for i in iteraciones]

    plt.figure(figsize=(6, 4))
    plt.plot(iteraciones_num, valores_x1, marker='o', label='Valor de x1')
    plt.title('Convergencia del Método de Newton-Raphson')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de x1')
    plt.grid(True)
    plt.legend()
    plt.show()

def salir():
    root.quit()

def abrir_menu_metodos():
    ventana_menu = tk.Toplevel(root)
    ventana_menu.title("Banco de Métodos")

    tk.Label(ventana_menu, text="Banco de Métodos", font=("Arial", 16)).pack(pady=10)

    button_punto_fijo = tk.Button(ventana_menu, text="Método de Punto Fijo", command=abrir_punto_fijo)
    button_punto_fijo.pack(pady=5)


    button_biseccion = tk.Button(ventana_menu, text="Método de Bisección", command=abrir_biseccion)
    button_biseccion.pack(pady=5)

    button_biseccion = tk.Button(ventana_menu, text="Metodo Newton-Raphson", command=abrir_newton_raphson)
    button_biseccion.pack(pady=5)

    button_salir = tk.Button(ventana_menu, text="Salir", command=salir)
    button_salir.pack(pady=5)


root = tk.Tk()
root.title("Banco de Métodos")

menu_abierto = False

abrir_menu_metodos()

root.mainloop()
