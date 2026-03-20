
import pulp

# ─────────────────────────────────────────────────────────────────
# §1  INDICES & SETS
# ─────────────────────────────────────────────────────────────────

people = ["Ana", "Bob", "Cho", "Dan", "Eva"]
tasks  = ["Sound", "Lights", "Door"]
hours  = [1, 2, 3, 4, 5]
days   = ["Sat", "Sun"]

# ─────────────────────────────────────────────────────────────────
# §2  PARAMETERS  (replace with your real data)
# ─────────────────────────────────────────────────────────────────

# --- Availability  A[p][h][d] ∈ {0,1} ---
A = {
    p: {h: {d: 1 for d in days} for h in hours}
    for p in people
}
# Example tweak: Ana is NOT available hour 5 on Sunday
A["Ana"][5]["Sun"] = 0
A["Dan"][1]["Sat"] = 0
A["Dan"][2]["Sat"] = 0

# --- Demand  D_demand[t][h][d] ≥ 0 ---
D_demand = {
    t: {h: {d: 1 for d in days} for h in hours}
    for t in tasks
}
# Sound needs 2 people at hour 1 on Saturday
D_demand["Sound"][1]["Sat"] = 2

# --- Skills  S_skill[p][t] ∈ {0,1} ---
S_skill = {p: {t: 1 for t in tasks} for p in people}
# Bob can't do Sound
S_skill["Bob"]["Sound"] = 0

# --- Force matrix  F[p][t][h][d] ∈ {0,1} ---
F = {
    p: {t: {h: {d: 0 for d in days} for h in hours} for t in tasks}
    for p in people
}
# Manager forces Ana → Lights at hour 1 on Saturday
F["Ana"]["Lights"][1]["Sat"] = 1

# --- Social affinity  E_social[p][p'] ∈ {1, 0, -1} ---
E_social = {p: {pp: 0 for pp in people} for p in people}
# Ana & Bob should work TOGETHER
E_social["Ana"]["Bob"] =  1
E_social["Bob"]["Ana"] =  1
# Cho & Dan must stay APART
E_social["Cho"]["Dan"] = -1
E_social["Dan"]["Cho"] = -1

# --- Min quota wish  L[p][t] ∈ {0,1} ---
L = {p: {t: 0 for t in tasks} for p in people}
# Eva wants to try Sound at least once
L["Eva"]["Sound"] = 1

# --- Preference cost  B[p][t] ≥ 0  (higher = more dislike) ---
B = {p: {t: 0.0 for t in tasks} for p in people}
B["Dan"]["Door"] = 3.0   # Dan really dislikes Door

# --- Rotation switch  R[t] ∈ {0,1} ---
R = {t: 0 for t in tasks}
R["Door"] = 1  # Door must rotate every hour

# --- Anchor plan  X_prev[p][t][h][d] ∈ {0,1} ---
X_prev = {
    p: {t: {h: {d: 0 for d in days} for h in hours} for t in tasks}
    for p in people
}

# --- Priority weights ---
W_COVERAGE  = 1000
W_STABILITY = 50
W_MANDATE   = 200
W_EQ_DAY    = 10
W_EQ_TOTAL  = 20
W_ROTATION  = 5
W_SOCIAL    = 8
W_GAP       = 15
W_QUOTA     = 12
W_PREF      = 2


# ─────────────────────────────────────────────────────────────────
# §3  MODEL & DECISION VARIABLES
# ─────────────────────────────────────────────────────────────────

model = pulp.LpProblem("Staffing_Optimizer", pulp.LpMinimize)

# --- Main assignment  x[p,t,h,d] ∈ {0,1} ---
x = pulp.LpVariable.dicts(
    "x", (people, tasks, hours, days), cat="Binary"
)

# --- Missing staff  m[t,h,d] ≥ 0 ---
m = pulp.LpVariable.dicts(
    "m", (tasks, hours, days), lowBound=0, cat="Continuous"
)

# --- Mandate penalty  U[p,t,h,d] ≥ 0 ---
U = pulp.LpVariable.dicts(
    "U", (people, tasks, hours, days), lowBound=0, cat="Continuous"
)

# --- Daily equity bounds ---
n_max = pulp.LpVariable.dicts("n_max", days, lowBound=0, cat="Continuous")
n_min = pulp.LpVariable.dicts("n_min", days, lowBound=0, cat="Continuous")

# --- Global equity bounds ---
N_max = pulp.LpVariable("N_max", lowBound=0, cat="Continuous")
N_min = pulp.LpVariable("N_min", lowBound=0, cat="Continuous")

# --- Rotation penalty  C[p,t,h,d] ∈ {0,1}  (only for h < H) ---
C = pulp.LpVariable.dicts(
    "C",
    (people, tasks, [h for h in hours if h < max(hours)], days),
    cat="Binary",
)

# --- Social penalties  (indexed per relevant pair × t × h × d) ---
together_pairs = [
    (p, pp)
    for p in people for pp in people
    if p < pp and E_social[p][pp] == 1
]
separate_pairs = [
    (p, pp)
    for p in people for pp in people
    if p < pp and E_social[p][pp] == -1
]

T_soc = pulp.LpVariable.dicts(
    "Tsoc", (together_pairs, tasks, hours, days), lowBound=0, cat="Continuous"
)
S_soc = pulp.LpVariable.dicts(
    "Ssoc", (separate_pairs, tasks, hours, days), lowBound=0, cat="Continuous"
)

# --- Quota penalty  Q[p,t] ∈ {0,1} ---
Q = pulp.LpVariable.dicts("Q", (people, tasks), cat="Binary")

# --- Stability error  E_stab[p,t,h,d] ∈ {0,1} ---
E_stab = pulp.LpVariable.dicts(
    "Estab", (people, tasks, hours, days), cat="Binary"
)

# --- Workday envelope ---
Started      = pulp.LpVariable.dicts("Started",      (people, hours, days), cat="Binary")
Active_Later = pulp.LpVariable.dicts("ActiveLater",   (people, hours, days), cat="Binary")
Z_gap        = pulp.LpVariable.dicts("Zgap",          (people, hours, days), cat="Binary")


# ─────────────────────────────────────────────────────────────────
# §4  CONSTRAINTS
# ─────────────────────────────────────────────────────────────────

# ── A. Task Coverage ─────────────────────────────────────────────
for t in tasks:
    for h in hours:
        for d in days:
            model += (
                pulp.lpSum(x[p][t][h][d] for p in people) + m[t][h][d]
                == D_demand[t][h][d],
                f"Coverage_{t}_{h}_{d}",
            )

# ── B. Manual Mandates ──────────────────────────────────────────
for p in people:
    for t in tasks:
        for h in hours:
            for d in days:
                model += (
                    F[p][t][h][d] - x[p][t][h][d] <= U[p][t][h][d],
                    f"Mandate_{p}_{t}_{h}_{d}",
                )

# ── C. Physical Availability & Uniqueness (HARD) ────────────────
for p in people:
    for h in hours:
        for d in days:
            model += (
                pulp.lpSum(x[p][t][h][d] for t in tasks) <= A[p][h][d],
                f"Availability_{p}_{h}_{d}",
            )

# ── D. Skill Filter (HARD) ──────────────────────────────────────
for p in people:
    for t in tasks:
        for h in hours:
            for d in days:
                model += (
                    x[p][t][h][d] <= S_skill[p][t],
                    f"Skill_{p}_{t}_{h}_{d}",
                )

# ── E. Equity — Daily Bounds ────────────────────────────────────
for p in people:
    for d in days:
        total_day = pulp.lpSum(x[p][t][h][d] for t in tasks for h in hours)
        model += (total_day <= n_max[d], f"DayMax_{p}_{d}")
        model += (total_day >= n_min[d], f"DayMin_{p}_{d}")

# ── E. Equity — Global Bounds ───────────────────────────────────
for p in people:
    total_all = pulp.lpSum(
        x[p][t][h][d] for t in tasks for h in hours for d in days
    )
    model += (total_all <= N_max, f"GlobalMax_{p}")
    model += (total_all >= N_min, f"GlobalMin_{p}")

# ── F. Selective Rotation ────────────────────────────────────────
for p in people:
    for t in tasks:
        for h in hours:
            if h < max(hours):
                for d in days:
                    model += (
                        x[p][t][h][d] + x[p][t][h + 1][d] - C[p][t][h][d]
                        <= 2 - R[t],
                        f"Rotation_{p}_{t}_{h}_{d}",
                    )

# ── G. Social Pairing — Together ─────────────────────────────────
for (p, pp) in together_pairs:
    for t in tasks:
        for h in hours:
            for d in days:
                model += (
                    x[p][t][h][d] - x[pp][t][h][d] <= T_soc[(p, pp)][t][h][d],
                    f"TogetherA_{p}_{pp}_{t}_{h}_{d}",
                )
                model += (
                    x[pp][t][h][d] - x[p][t][h][d] <= T_soc[(p, pp)][t][h][d],
                    f"TogetherB_{p}_{pp}_{t}_{h}_{d}",
                )

# ── G. Social Pairing — Separate ────────────────────────────────
for (p, pp) in separate_pairs:
    for t in tasks:
        for h in hours:
            for d in days:
                model += (
                    x[p][t][h][d] + x[pp][t][h][d] - S_soc[(p, pp)][t][h][d]
                    <= 1,
                    f"Separate_{p}_{pp}_{t}_{h}_{d}",
                )

# ── H. Experience Quota ──────────────────────────────────────────
for p in people:
    for t in tasks:
        model += (
            pulp.lpSum(x[p][t][h][d] for h in hours for d in days)
            + Q[p][t]
            >= L[p][t],
            f"Quota_{p}_{t}",
        )

# ── J. Stability — Removed assignments ──────────────────────────
for p in people:
    for t in tasks:
        for h in hours:
            for d in days:
                model += (
                    E_stab[p][t][h][d]
                    >= X_prev[p][t][h][d] - x[p][t][h][d],
                    f"StabRem_{p}_{t}_{h}_{d}",
                )

# ── J. Stability — New assignments ──────────────────────────────
for p in people:
    for t in tasks:
        for h in hours:
            for d in days:
                model += (
                    E_stab[p][t][h][d]
                    >= x[p][t][h][d] - X_prev[p][t][h][d],
                    f"StabNew_{p}_{t}_{h}_{d}",
                )

# ── K. Workday Envelope — Start Line ────────────────────────────
for p in people:
    for d in days:
        for h in hours:
            # Started ≥ working now
            model += (
                Started[p][h][d]
                >= pulp.lpSum(x[p][t][h][d] for t in tasks),
                f"StartWork_{p}_{h}_{d}",
            )
            # Started propagates forward
            if h > min(hours):
                model += (
                    Started[p][h][d] >= Started[p][h - 1][d],
                    f"StartProp_{p}_{h}_{d}",
                )

# ── K. Workday Envelope — Finish Line ───────────────────────────
for p in people:
    for d in days:
        for h in hours:
            # Active_Later ≥ working now
            model += (
                Active_Later[p][h][d]
                >= pulp.lpSum(x[p][t][h][d] for t in tasks),
                f"LaterWork_{p}_{h}_{d}",
            )
            # Active_Later propagates backward
            if h < max(hours):
                model += (
                    Active_Later[p][h][d] >= Active_Later[p][h + 1][d],
                    f"LaterProp_{p}_{h}_{d}",
                )

# ── K. Workday Envelope — Gap Sensor ────────────────────────────
for p in people:
    for h in hours:
        for d in days:
            model += (
                Z_gap[p][h][d]
                >= Started[p][h][d]
                + Active_Later[p][h][d]
                - 1
                - pulp.lpSum(x[p][t][h][d] for t in tasks),
                f"Gap_{p}_{h}_{d}",
            )


# ─────────────────────────────────────────────────────────────────
# §5  OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────

obj_coverage = W_COVERAGE * pulp.lpSum(m[t][h][d] for t in tasks for h in hours for d in days)

obj_stability = W_STABILITY * pulp.lpSum(
    E_stab[p][t][h][d]
    for p in people for t in tasks for h in hours for d in days)

obj_mandate = W_MANDATE * pulp.lpSum(
    U[p][t][h][d]
    for p in people for t in tasks for h in hours for d in days
)

obj_eq_day = W_EQ_DAY * pulp.lpSum(
    n_max[d] - n_min[d] for d in days
)

obj_eq_total = W_EQ_TOTAL * (N_max - N_min)

obj_rotation = W_ROTATION * pulp.lpSum(
    C[p][t][h][d]
    for p in people for t in tasks
    for h in hours if h < max(hours)
    for d in days
)

obj_social = W_SOCIAL * (
    pulp.lpSum(
        T_soc[(p, pp)][t][h][d]
        for (p, pp) in together_pairs
        for t in tasks for h in hours for d in days
    )
    + pulp.lpSum(
        S_soc[(p, pp)][t][h][d]
        for (p, pp) in separate_pairs
        for t in tasks for h in hours for d in days
    )
)

obj_gap = W_GAP * pulp.lpSum(
    Z_gap[p][h][d] for p in people for h in hours for d in days
)

obj_quota = W_QUOTA * pulp.lpSum(
    Q[p][t] for p in people for t in tasks
)

obj_pref = W_PREF * pulp.lpSum(
    B[p][t] * x[p][t][h][d]
    for p in people for t in tasks for h in hours for d in days
)

model += (
    obj_coverage
    + obj_stability
    + obj_mandate
    + obj_eq_day
    + obj_eq_total
    + obj_rotation
    + obj_social
    + obj_gap
    + obj_quota
    + obj_pref
)


# ─────────────────────────────────────────────────────────────────
# §6  SOLVE & REPORT
# ─────────────────────────────────────────────────────────────────

solver = pulp.HiGHS(msg=True, timeLimit=12000, threads=12)
model.solve(solver)

status = pulp.LpStatus[model.status]
print(f"Estado: {status}  |  Costo total: {pulp.value(model.objective):.2f}\n")

if status == "Optimal":
    # ── Schedule table ───────────────────────────────────────────
    for d in days:
        print(f"{'═' * 50}")
        print(f"  DÍA: {d}")
        print(f"{'═' * 50}")
        header = f"{'Hora':<6}" + "".join(f"{p:<10}" for p in people)
        print(header)
        print("─" * len(header))
        for h in hours:
            row = f"{h:<6}"
            for p in people:
                assigned = [
                    t for t in tasks
                    if pulp.value(x[p][t][h][d]) is not None
                    and pulp.value(x[p][t][h][d]) > 0.5
                ]
                cell = assigned[0] if assigned else "—"
                row += f"{cell:<10}"
            print(row)
        print()

    # ── Missing staff warnings ───────────────────────────────────
    missing_found = False
    for t in tasks:
        for h in hours:
            for d in days:
                val = pulp.value(m[t][h][d])
                if val is not None and val > 0.01:
                    if not missing_found:
                        print("⚠  PERSONAL FALTANTE:")
                        missing_found = True
                    print(f"   {t} | Hora {h} | {d} → faltan {val:.0f}")

    # ── Hours summary per person ─────────────────────────────────
    print(f"\n{'═' * 40}")
    print("  RESUMEN DE HORAS")
    print(f"{'═' * 40}")
    for p in people:
        total = sum(
            1
            for t in tasks for h in hours for d in days
            if pulp.value(x[p][t][h][d]) is not None
            and pulp.value(x[p][t][h][d]) > 0.5
        )
        per_day = {}
        for d in days:
            per_day[d] = sum(
                1
                for t in tasks for h in hours
                if pulp.value(x[p][t][h][d]) is not None
                and pulp.value(x[p][t][h][d]) > 0.5
            )
        day_str = " | ".join(f"{d}: {per_day[d]}" for d in days)
        print(f"  {p:<8} Total: {total:>2}  ({day_str})")