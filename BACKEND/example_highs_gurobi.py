import pulp
import random


# Set seed for reproducibility
random.seed(4200)

# 1. Initialize the Problema
model = pulp.LpProblem("Millions_of_Nodes_Challenge", pulp.LpMinimize)

# 2. Parameters
n_routes = 5000  
demand = 500 

# 3. Decision Variables
routes = [pulp.LpVariable(f"route_{i}", cat='Binary') for i in range(n_routes)]

# 4. Objective Function
model += pulp.lpSum([(10 + random.uniform(0, 0.1)) * routes[i] for i in range(n_routes)])

# 5. Constraints
model += pulp.lpSum([(10 + random.uniform(0, 0.05)) * routes[i] for i in range(n_routes)]) >= demand

for i in range(n_routes - 1):
    model += routes[i] + routes[i+1] <= 1

# 6. Solve with CBC
# We use pulp.PULP_CBC_CMD to call the CBC solver
# msg=True: show logs
# timeLimit: max seconds to run
# threads: parallel processing
solver = pulp.GUROBI(
    msg=True,          # Muestra el progreso del solver en consola (nodos, gap, etc.)
    timeLimit=1200,    # Segundos máximos de ejecución. Si se alcanza, devuelve la mejor solución encontrada
    gapRel=0.01,       # Diferencia % máxima entre solución y cota teórica para parar (0.01 = 1%)
    mip=True,          # True=resuelve entero (binarias/enteras). False=ignora integralidad, resuelve LP continuo

    # --- Terminación ---
    # MIPGapAbs=1e-8,       # Para cuando la diferencia absoluta entre solución y cota es menor a este valor
    # BestObjStop=520.0,    # Para cuando encuentra una solución con objetivo <= este valor
    # SolutionLimit=10,     # Para después de encontrar N soluciones factibles (sin importar calidad)

    # --- Rendimiento MIP ---
    Threads=12,              # Hilos CPU en paralelo. Más no siempre es mejor por overhead de coordinación
    MIPFocus=2,             # Estrategia=0=balanced, 1=encontrar factibles rápido, 2=demostrar optimalidad, 3=mejorar cota
    Presolve=2,             # Simplificación previa=0=off, 1=conservador, 2=agresivo (reduce modelo pero cuesta tiempo)
    PrePasses=-1,           # Pasadas de presolve. -1=sin límite, Gurobi decide cuándo parar
    # PreSparsify=1,        # Reduce densidad de la matriz de restricciones para acelerar álgebra lineal
    # PreDual=-1,           # Dualiza el modelo en presolve. A veces el dual se resuelve más rápido

    # --- Estructura ---
    Symmetry=2,             # Detecta soluciones intercambiables para podar ramas redundantes. 0=off, 1=conservador, 2=agresivo
    Disconnected=2,       # Detecta subproblemas independientes para resolverlos por separado
    IntegralityFocus=1,   # Se esfuerza más en que las enteras sean exactamente enteras (no casi enteras)

    # --- Método solver ---
    Method=1,            # Para LP=-1=auto, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent
    # NodeMethod=1,           # LP en nodos MIP=-1=auto, 0=primal, 1=dual (reutiliza base del padre), 2=barrier
    # ConcurrentMIP=1,      # Lanza N solves MIP con distintas estrategias. El primero que termina gana

    # --- Solution Pool (guarda múltiples soluciones factibles) ---
    # PoolSolutions=10,     # Cuántas soluciones guarda además de la óptima
    # PoolGap=0.1,          # Solo guarda soluciones dentro de este % del óptimo
    # PoolSearchMode=0,     # 0=guarda las que encuentra de paso, 1=busca más activamente, 2=busca las N mejores

)

model.solve(solver)

# 7. Output results
print("-" * 30)
print(f"Status: {pulp.LpStatus[model.status]}")
if model.status == pulp.LpStatusOptimal:
    print(f"Optimal Cost: {pulp.value(model.objective)}")
    print(f"Active Routes: {sum(pulp.value(r) for r in routes)}")