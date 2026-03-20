# Ultimo cambio que le meti fue que tanto horas como días funcionaran con labels, y no con numeros



from pulp import *

# ============================================================
# 0.  EXAMPLE DATA — Replace with your own DataFrames / Dicts
# ============================================================

# --- Dimensions ---
people = ["Alice", "Bob", "Carol"]                        # P
tasks  = ["Reception", "Backstage", "Security"]           # T
hours  = ["08:00", "09:00", "10:00", "11:00", "12:00",   # H  (now accepts any label)
          "13:00", "14:00", "15:00", "16:00", "17:00"]
days   = ["Mon", "Tue", "Wed"]                            # J  (now accepts any label)

# --- Helper look-ups for "next hour" logic ---
hour_index = {h: i for i, h in enumerate(hours)}
next_hour  = {hours[i]: hours[i + 1] for i in range(len(hours) - 1)}

# --- INPUT Parameters ---
# A[p,h,j] : Availability (1 = present)
availability = {(p,h,j): 1 for p in people for h in hours for j in days}

# D[t,h,j] : Demand (how many people needed)
demand = {(t,h,j): 1 for t in tasks for h in hours for j in days}

# S[p,t] : Skill matrix (1 = qualified)
skills = {(p,t): 1 for p in people for t in tasks}

# F[p,t,h,j] : Force / Mandate matrix (1 = manager requires it)
force = {(p,t,h,j): 0 for p in people for t in tasks for h in hours for j in days}

# E[p,p'] : Social affinity  (1 = friends/together, -1 = enemies/separate, 0 = neutral)
social = {}
for i, p1 in enumerate(people):
    for p2 in people[i + 1:]:
        social[(p1,p2)] = 0

# L[p,t] : Min-quota wish (1 = person wants to try task at least once)
min_quota_wish = {(p,t): 0 for p in people for t in tasks}

# B[p,t] : Preference cost (higher = more dislike)
pref_cost = {(p,t): 1 for p in people for t in tasks}

# R[t] : Rotation switch (1 = no consecutive hours on that task)
rotation = {t: 1 for t in tasks}

# X_prev[p,t,h,j] : Anchor plan from previous schedule
X_prev = {(p,t,h,j): 0 for p in people for t in tasks for h in hours for j in days}

# --- Priority Weights (tune to your hierarchy) ---
W_COVERAGE  = 100000
W_MANDATE   = 50000
W_STABILITY = 10000
W_EQ_DAY    = 5000
W_EQ_TOTAL  = 1000
W_ROTATION  = 500
W_SOCIAL    = 100
W_GAP       = 50
W_QUOTA     = 10
W_PREF      = 5

# ============================================================
# 1.  CREATE THE MODEL
# ============================================================
model = LpProblem("Staff_Scheduling", LpMinimize)

# ============================================================
# 2.  DECISION VARIABLES
# ============================================================

# x[p,t,h,j] — assignment (binary)
x = LpVariable.dicts("x", ((p,t,h,j) for p in people for t in tasks for h in hours for j in days), cat=LpBinary)

# m[t,h,j] — number of missing staff
m = LpVariable.dicts("m", ((t,h,j) for t in tasks for h in hours for j in days), lowBound=0, cat=LpInteger)

# u[p,t,h,j] — unfulfilled mandate
u = LpVariable.dicts("u", ((p,t,h,j) for p in people for t in tasks for h in hours for j in days), lowBound=0, cat=LpInteger)

# n_max[j], n_min[j] — daily equity bounds
n_max = LpVariable.dicts("n_max", days, lowBound=0, cat=LpInteger)
n_min = LpVariable.dicts("n_min", days, lowBound=0, cat=LpInteger)

# z_max, z_min — global equity bounds
z_max = LpVariable("z_max", lowBound=0, cat=LpInteger)
z_min = LpVariable("z_min", lowBound=0, cat=LpInteger)

# c[p,t,h,j] — consecutive-hour penalty (binary)
# Only for hours that have a successor and tasks with R=1
consec_keys = [
    (p,t,h,j)
    for p in people for t in tasks for h in hours[:-1] for j in days
    if rotation[t] == 1
]

c = LpVariable.dicts("c", consec_keys, cat=LpBinary) if consec_keys else {}

# v[p,p',t,h,j] — social violation (together / friends)
friend_pairs = [(p1,p2) for (p1,p2), val in social.items() if val == 1]
v = LpVariable.dicts(
    "v",
    ((p1,p2, t,h,j) for (p1,p2) in friend_pairs for t in tasks for h in hours for j in days),
    lowBound=0, cat=LpContinuous,
) if friend_pairs else {}

# w[p,p',t,h,j] — social violation (separate / enemies)
enemy_pairs = [(p1,p2) for (p1,p2), val in social.items() if val == -1]

w = LpVariable.dicts(
    "w",
    ((p1,p2, t,h,j) for (p1,p2) in enemy_pairs for t in tasks for h in hours for j in days),
    lowBound=0, cat=LpContinuous,
) if enemy_pairs else {}

# q[p,t] — quota missed (binary)
q = LpVariable.dicts("q", ((p,t) for p in people for t in tasks), cat=LpBinary)

# d[p,t,h,j] — deviation flag (binary)
d = LpVariable.dicts("d", ((p,t,h,j) for p in people for t in tasks for h in hours for j in days), cat=LpBinary)

# s[p,h,j] — start flag (binary)
s = LpVariable.dicts("s", ((p,h,j) for p in people for h in hours for j in days), cat=LpBinary)

# f[p,h,j] — finish flag (binary)
f = LpVariable.dicts("f", ((p,h,j) for p in people for h in hours for j in days), cat=LpBinary)

# g[p,h,j] — gap flag (binary)
g = LpVariable.dicts("g", ((p,h,j) for p in people for h in hours for j in days), cat=LpBinary)

# ============================================================
# 3.  OBJECTIVE FUNCTION
# ============================================================

obj = []

# 1. Coverage penalty
obj += [W_COVERAGE * m[(t,h,j)] for t in tasks for h in hours for j in days]

# 2. Mandate penalty
obj += [W_MANDATE * u[(p,t,h,j)] for p in people for t in tasks for h in hours for j in days]

# 3. Stability penalty
obj += [W_STABILITY * d[(p,t,h,j)] for p in people for t in tasks for h in hours for j in days]

# 4. Daily equity penalty
obj += [W_EQ_DAY * (n_max[j] - n_min[j]) for j in days]

# 5. Global equity penalty
obj.append(W_EQ_TOTAL * (z_max - z_min))

# 6. Rotation fatigue penalty
if consec_keys:
    obj += [W_ROTATION * c[k] for k in consec_keys]

# 7. Social penalties
if friend_pairs:
    obj += [W_SOCIAL * v[(p1,p2, t,h,j)] for (p1,p2) in friend_pairs for t in tasks for h in hours for j in days]
if enemy_pairs:
    obj += [W_SOCIAL * w[(p1,p2, t,h,j)] for (p1,p2) in enemy_pairs for t in tasks for h in hours for j in days]

# 8. Gap penalty
obj += [W_GAP * g[(p,h,j)] for p in people for h in hours for j in days]

# 9. Quota penalty
obj += [W_QUOTA * q[(p,t)] for p in people for t in tasks]

# 10. Preference cost
obj += [W_PREF * pref_cost[(p,t)] * x[(p,t,h,j)] for p in people for t in tasks for h in hours for j in days]

model += lpSum(obj), "Total_Penalty"

# ============================================================
# 4.  CONSTRAINTS
# ============================================================

# --- A. Task Coverage ---
for t in tasks:
    for h in hours:
        for j in days:
            model += (lpSum(x[(p,t,h,j)] for p in people)+m[(t,h,j)] == demand.get((t,h,j),0), f"Coverage_{t}_{h}_{j}")

# --- B. Manual Mandates ---
for p in people:
    for t in tasks:
        for h in hours:
            for j in days:
                if force.get((p,t,h,j),0) == 1:
                    model += (1 - x[(p,t,h,j)] <= u[(p,t,h,j)], f"Mandate_{p}_{t}_{h}_{j}")

# --- C. Physical Availability (Hard) ---
for p in people:
    for h in hours:
        for j in days:
            model += (lpSum(x[(p,t,h,j)] for t in tasks) <= availability.get((p,h,j),0), f"Avail_{p}_{h}_{j}")

# --- D. Skill Filter (Hard) ---
for p in people:
    for t in tasks:
        if skills.get((p,t),0) == 0:
            for h in hours:
                for j in days:
                    model += (x[(p,t,h,j)] == 0, f"Skill_{p}_{t}_{h}_{j}")

# --- E. Double-Squeeze Equity ---
for j in days:
    for p in people:
        td = lpSum(x[(p,t,h,j)] for t in tasks for h in hours)
        model += (td<=n_max[j], f"DayMax_{p}_{j}")
        model += (td>=n_min[j], f"DayMin_{p}_{j}")

for p in people:
    tg = lpSum(x[(p,t,h,j)] for t in tasks for h in hours for j in days)
    model += (tg<=z_max, f"GlobalMax_{p}")
    model += (tg>=z_min, f"GlobalMin_{p}")

# --- F. Rotation Fatigue ---
for key in consec_keys:
    p,t,h,j = key
    model += (x[(p,t,h,j)]+x[(p,t,next_hour[h],j)] - c[key]<=1, f"Rotation_{p}_{t}_{h}_{j}")

# --- G. Social Constraints ---
# G.1 Together (Friends)
for (p1,p2) in friend_pairs:
    for t in tasks:
        for h in hours:
            for j in days:
                model += (x[(p1,t,h,j)] - x[(p2,t,h,j)] <= v[(p1,p2,t,h,j)], f"TogetherA_{p1}_{p2}_{t}_{h}_{j}")
                model += (x[(p2,t,h,j)] - x[(p1,t,h,j)] <= v[(p1,p2,t,h,j)], f"TogetherB_{p1}_{p2}_{t}_{h}_{j}")

# G.2 Separate (Enemies)
for (p1,p2) in enemy_pairs:
    for t in tasks:
        for h in hours:
            for j in days:
                model += (x[(p1,t,h,j)]+x[(p2,t,h,j)] - w[(p1,p2,t,h,j)] <= 1, f"Separate_{p1}_{p2}_{t}_{h}_{j}")

    # --- H. Minimum Quota ---
    for p in people:
        for t in tasks:
            for j in days:
                model += (lpSum(x[(p,t,h,j)] for h in hours) + q[(p,t)] >= min_quota_wish.get((p,t),0), f"Quota_{p}_{t}_{j}")

# --- I. Stability / Deviation ---
for p in people:
    for t in tasks:
        for h in hours:
            for j in days:
                prev = X_prev.get((p,t,h,j),0)
                model += (d[(p,t,h,j)] >= prev-x[(p,t,h,j)], f"DevA_{p}_{t}_{h}_{j}")
                model += (d[(p,t,h,j)] >= x[(p,t,h,j)]-prev, f"DevB_{p}_{t}_{h}_{j}")

# --- J. Workday Envelope (Gap Detection) NOT USEFUL ANYMORE ---
#for p in people:
#    for j in days:
#        for idx, h in enumerate(hours):
#            work_h = lpSum(x[(p,t,h,j)] for t in tasks)

#            # J.1  Start flag
#            model += (s[(p,h,j)] >= work_h,f"StartWork_{p}_{h}_{j}")
#            if idx > 0:
#                h_prev = hours[idx - 1]
#                model += (s[(p,h,j)] >= s[(p,h_prev, j)], f"StartCarry_{p}_{h}_{j}")

#            # J.2  Finish flag
#            model += (f[(p,h,j)] >= work_h,f"FinWork_{p}_{h}_{j}")
#            if idx < len(hours) - 1:
#                h_next = hours[idx + 1]
#                model += (f[(p,h,j)] >= f[(p,h_next,j)], f"FinCarry_{p}_{h}_{j}")

#            # J.3  Gap sensor
#            model += (g[(p,h,j)] >= s[(p,h,j)] + f[(p,h,j)] - 1 - work_h,f"Gap_{p}_{h}_{j}")




# ============================================================
# 5.  SOLVE
# ============================================================
status = model.solve(GUROBI(msg=True, threads=12))

# ============================================================
# 6.  OUTPUT
# ============================================================
print("=" * 60)
print(f"Status : {LpStatus[status]}")
print(f"Objective Value : {value(model.objective):.2f}")
print("=" * 60)

# --- Scheduled assignments ---
print("\n--- SCHEDULE ---")
for j in days:
    print(f"\n  *** Day {j} ***")
    for h in hours:
        assignments = []
        for p in people:
            for t in tasks:
                if x[(p,t,h,j)].varValue and x[(p,t,h,j)].varValue > 0.5:
                    assignments.append(f"{p} → {t}")
        if assignments:
            print(f"    {h}  |  {', '.join(assignments)}")

# --- Missing staff ---
print("\n--- MISSING STAFF (m > 0) ---")
any_missing = False
for t in tasks:
    for h in hours:
        for j in days:
            val = m[(t,h,j)].varValue
            if val and val > 0.01:
                print(f"    {t} @ {h}, day {j}  →  {val:.0f} missing")
                any_missing = True
if not any_missing:
    print("    None — all demand covered!")

# --- Equity summary ---
print("\n--- WORKLOAD EQUITY ---")
for p in people:
    total = sum(
        x[(p,t,h,j)].varValue
        for t in tasks for h in hours for j in days
        if x[(p,t,h,j)].varValue
    )
    print(f"    {p:>8s} : {total:.0f} total hours")
print(f"    Global range: z_max={z_max.varValue:.1f}, z_min={z_min.varValue:.1f}")

# --- Gaps ---
print("\n--- GAPS DETECTED ---")
any_gap = False
for p in people:
    for j in days:
        for h in hours:
            if g[(p,h,j)].varValue and g[(p,h,j)].varValue > 0.5:
                print(f"    {p}, day {j}, {h}")
                any_gap = True
if not any_gap:
    print("    None — compact schedules!")

# --- Social violations ---
print("\n--- SOCIAL VIOLATIONS ---")
any_social = False
for (p1,p2) in friend_pairs:
    for t in tasks:
        for h in hours:
            for j in days:
                val = v[(p1,p2, t,h,j)].varValue
                if val and val > 0.5:
                    print(f"    Friends {p1}&{p2} separated @ {t}, {h}, {j}")
                    any_social = True
for (p1,p2) in enemy_pairs:
    for t in tasks:
        for h in hours:
            for j in days:
                val = w[(p1,p2, t,h,j)].varValue
                if val and val > 0.5:
                    print(f"    Enemies {p1}&{p2} together @ {t}, {h}, {j}")
                    any_social = True
if not any_social:
    print("    None — social preferences respected!")

print("\nDone.")