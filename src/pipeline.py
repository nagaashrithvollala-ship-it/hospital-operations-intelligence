"""
pipeline.py  —  Hospital Operations Intelligence Platform
=========================================================
    synthetic data  ->  Python ETL  ->  SQLite  ->  SQL KPIs  ->  ML  ->  exports

  1. generate_data()    Deterministic synthetic encounters -> data/*.csv
  2. load_to_sqlite()   ETL into SQLite
  3. run_sql()          Named KPI queries in sql/analytics.sql
  4. forecast_volume()  LinearRegression weekly admissions forecast (8 weeks)
  5. predict_readmit()  GradientBoosting 30-day readmission classifier
  6. occupancy()        Bed occupancy per department
  7. exec_summary()     Data-driven "AI" executive summary
  8. export()           metrics.json, dashboard_data.js, dashboard_preview.png

Run:  python src/pipeline.py
Data is illustrative and generated deterministically (numpy seed = 11).
No real or restricted patient data (e.g. MIMIC) is used.
"""

import os, json, sqlite3, datetime as dt, textwrap
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

RNG = np.random.default_rng(11)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
SQL  = os.path.join(ROOT, "sql", "analytics.sql")
os.makedirs(DATA, exist_ok=True)

DEPTS = [   # name, beds, benchmark LOS (days), readmit base, cost base, perdiem
    ("Emergency",        16, 0.5, .09,  900, 480),
    ("ICU",               8, 5.5, .22, 6800, 3200),
    ("Cardiology",       10, 4.2, .18, 4200, 1500),
    ("Orthopedics",       9, 3.8, .10, 3800, 1300),
    ("Oncology",          8, 6.0, .21, 5200, 1700),
    ("Pediatrics",        7, 2.9, .07, 2100, 1100),
    ("General Surgery",   9, 4.5, .13, 4600, 1600),
    ("Neurology",         7, 4.0, .16, 4000, 1500),
]


# ---------------------------------------------------------------- 1. DATA GEN
def generate_data():
    departments = pd.DataFrame(
        [(f"D{i+1}", n, b, bl, rb, cb, pd_) for i, (n, b, bl, rb, cb, pd_) in enumerate(DEPTS)],
        columns=["dept_id", "name", "beds", "benchmark_los", "readmit_base", "cost_base", "perdiem"])

    start = dt.date(2024, 8, 1)
    n = 5200
    rows = []
    for k in range(n):
        di = RNG.integers(0, len(DEPTS))
        d = departments.iloc[di]
        admit = start + dt.timedelta(days=int(RNG.integers(0, 360)))
        month = admit.month
        # winter flu-season volume/pressure bump raises ER wait
        winter = 1.0 + (.35 if month in (12, 1, 2) else 0.0)
        age = int(np.clip(RNG.normal(54, 22), 0, 98))
        acuity = int(np.clip(round(RNG.normal(3, 1.1)), 1, 5))   # 1 = most urgent
        prior_admits = int(RNG.poisson(0.8))

        los = float(max(0.2, RNG.lognormal(np.log(max(d["benchmark_los"], .5)), .38)))
        er_wait = float(max(4, RNG.normal(38 + (6 - acuity) * 14, 18) * winter
                            + (30 if d["name"] == "Emergency" else 0)))
        cost = float(d["cost_base"] + los * d["perdiem"] * RNG.uniform(.85, 1.2)
                     + RNG.normal(0, 400))
        satisfaction = float(np.clip(5.0 - er_wait / 120 - max(0, los - d["benchmark_los"]) * .12
                                     + RNG.normal(0, .35), 1, 5))

        # 30-day readmission probability (learnable signal)
        p = (d["readmit_base"] + (age - 54) / 400 + (los - d["benchmark_los"]) * .025
             + prior_admits * .045 + (5 - acuity) * -.012 + RNG.normal(0, .03))
        p = float(np.clip(p, .01, .9))
        readmit = int(RNG.random() < p)

        rows.append({"encounter_id": f"E{k+1:05d}", "admit_date": admit.isoformat(),
                     "dept_id": d["dept_id"], "age": age, "acuity": acuity,
                     "prior_admits": prior_admits, "length_of_stay": round(los, 2),
                     "er_wait_min": round(er_wait, 1), "cost": round(cost, 0),
                     "satisfaction": round(satisfaction, 2), "readmit_30d": readmit,
                     "admit_month": month})
    encounters = pd.DataFrame(rows)

    # weekly admissions volume (for forecast)
    weeks = pd.date_range("2024-08-05", periods=52, freq="W-MON")
    base, trend = 96, 0.4
    demand = []
    for i, wk in enumerate(weeks):
        winter = 1 + (.22 if wk.month in (12, 1, 2) else 0.0)
        vol = int((base + trend * i) * winter + RNG.normal(0, 6))
        demand.append({"week": wk.date().isoformat(), "admissions": vol})
    demand = pd.DataFrame(demand)

    for name, df in [("departments", departments), ("encounters", encounters),
                     ("weekly_admissions", demand)]:
        df.to_csv(os.path.join(DATA, f"{name}.csv"), index=False)
    return dict(departments=departments, encounters=encounters, weekly_admissions=demand)


# ---------------------------------------------------------------- 2/3. ETL + SQL
def load_to_sqlite(tables):
    conn = sqlite3.connect(":memory:")
    for name, df in tables.items():
        df.to_sql(name, conn, index=False, if_exists="replace")
    return conn


def parse_named_sql(path):
    blocks, name, buf = {}, None, []
    for line in open(path):
        if line.strip().lower().startswith("-- name:"):
            if name:
                blocks[name] = "".join(buf).strip()
            name, buf = line.split(":", 1)[1].strip(), []
        elif name:
            buf.append(line)
    if name:
        blocks[name] = "".join(buf).strip()
    return blocks


def run_sql(conn):
    return {n: pd.read_sql_query(s.split(";")[0], conn)
            for n, s in parse_named_sql(SQL).items()}


# ---------------------------------------------------------------- 4. FORECAST
def forecast_volume(demand, horizon=8):
    d = demand.copy()
    d["idx"] = np.arange(len(d))
    d["month"] = pd.to_datetime(d["week"]).dt.month
    X = pd.concat([d[["idx"]], pd.get_dummies(d["month"], prefix="m")], axis=1)
    y = d["admissions"].values
    model = LinearRegression().fit(X, y)
    fut_idx = np.arange(len(d), len(d) + horizon)
    fut_weeks = pd.date_range(pd.to_datetime(d["week"]).iloc[-1], periods=horizon + 1, freq="W-MON")[1:]
    fut = pd.DataFrame({"idx": fut_idx, "month": fut_weeks.month})
    Xf = pd.concat([fut[["idx"]], pd.get_dummies(fut["month"], prefix="m")], axis=1).reindex(
        columns=X.columns, fill_value=0)
    pred = model.predict(Xf).round().astype(int)
    last, nxt = int(y[-4:].mean()), int(pred[:4].mean())
    return dict(history=[{"week": w, "admissions": int(u)} for w, u in zip(d["week"], y)],
                forecast=[{"week": w.date().isoformat(), "admissions": int(p)}
                          for w, p in zip(fut_weeks, pred)],
                next_month_change_pct=round(100 * (nxt - last) / last, 1))


# ---------------------------------------------------------------- 5. READMISSION MODEL
def predict_readmit(enc):
    df = enc.copy()
    df["dept_code"] = df["dept_id"].str[1:].astype(int)
    feats = ["age", "acuity", "prior_admits", "length_of_stay", "er_wait_min", "dept_code"]
    X, y = df[feats], df["readmit_30d"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=11, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=240, learning_rate=.05,
                                     max_depth=3, subsample=.85, random_state=11).fit(Xtr, ytr)
    auc = round(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), 3)
    imp = sorted(zip(feats, clf.feature_importances_), key=lambda t: -t[1])
    return dict(roc_auc=auc, n_train=int(len(Xtr)), n_test=int(len(Xte)),
                feature_importance=[{"feature": f, "importance": round(float(v), 3)} for f, v in imp])


# ---------------------------------------------------------------- 6. OCCUPANCY
def occupancy(enc, depts):
    period = (pd.to_datetime(enc["admit_date"]).max() - pd.to_datetime(enc["admit_date"]).min()).days or 1
    pat_days = enc.groupby("dept_id")["length_of_stay"].sum()
    rows = []
    for _, d in depts.iterrows():
        occ = round(100 * pat_days.get(d["dept_id"], 0) / (d["beds"] * period), 1)
        rows.append({"department": d["name"], "beds": int(d["beds"]),
                     "occupancy_pct": occ})
    overall = round(100 * pat_days.sum() / (depts["beds"].sum() * period), 1)
    return pd.DataFrame(rows).sort_values("occupancy_pct", ascending=False), overall


# ---------------------------------------------------------------- 7. EXEC SUMMARY
def exec_summary(kpi, dept, occ_df, occ_overall, fc, model, icu_change):
    worst = dept.iloc[0]     # highest LOS over benchmark
    hot = occ_df.iloc[0]
    d = "increased" if icu_change >= 0 else "decreased"
    dirw = "rise" if fc["next_month_change_pct"] >= 0 else "drop"
    return (
        f"Average ER wait is {kpi['avg_er_wait_min']} min with bed occupancy at "
        f"{occ_overall}% and a 30-day readmission rate of {kpi['readmission_rate_pct']}%. "
        f"ICU readmissions {d} {abs(icu_change)}% month-over-month. "
        f"{worst['department']} average length of stay exceeds its benchmark by "
        f"{worst['los_vs_benchmark']} days. {hot['department']} is the most pressured unit "
        f"at {hot['occupancy_pct']}% occupancy. Admissions are forecast to {dirw} "
        f"{abs(fc['next_month_change_pct'])}% next month; the readmission-risk model "
        f"(ROC-AUC {model['roc_auc']}) ranks length of stay, age and prior admissions as "
        f"the strongest drivers — the shortlist for discharge-planning intervention."
    )


def icu_mom(enc):
    e = enc.copy(); e["m"] = pd.to_datetime(e["admit_date"]).dt.strftime("%Y-%m")
    icu = e[e["dept_id"] == "D2"].groupby("m")["readmit_30d"].mean() * 100
    if len(icu) < 2:
        return 0.0
    return round(icu.iloc[-1] - icu.iloc[-2], 1)


# ---------------------------------------------------------------- 8. EXPORT
def export(kpi, sql_out, occ_df, occ_overall, fc, model, summary):
    metrics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "kpi": {**kpi, "bed_occupancy_pct": occ_overall},
        "department_performance": sql_out["department_performance"].to_dict("records"),
        "occupancy": occ_df.to_dict("records"),
        "monthly_trend": sql_out["monthly_trend"].to_dict("records"),
        "er_wait_by_acuity": sql_out["er_wait_by_acuity"].to_dict("records"),
        "forecast": fc,
        "readmit_model": model,
        "ai_summary": summary,
    }
    json.dump(metrics, open(os.path.join(ROOT, "metrics.json"), "w"), indent=2)
    open(os.path.join(ROOT, "dashboard_data.js"), "w").write(
        "window.HOI_DATA = " + json.dumps(metrics) + ";")
    render_preview(metrics)
    return metrics


def render_preview(m):
    BG, PANEL, INK, MUT, LINE = "#0a1222", "#152036", "#e7eefc", "#93a3c6", "#26324f"
    ACC = ["#22d3ee", "#34d399", "#f472b6", "#fbbf24"]
    fig = plt.figure(figsize=(13, 8.4), facecolor=BG)
    fig.suptitle("Hospital Operations Intelligence — Executive Overview",
                 x=.065, ha="left", color=INK, fontsize=19, fontweight="bold", y=.975)
    fig.text(.065, .935, "ER · beds · readmissions · cost — predictive operations · synthetic data",
             color=MUT, fontsize=11)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[.85, 2, 1.15],
                           hspace=.55, wspace=.32, left=.06, right=.965, top=.9, bottom=.055)

    def panel(ax): ax.set_facecolor(PANEL); [s.set_visible(False) for s in ax.spines.values()]
    k = m["kpi"]
    cards = [("AVG ER WAIT", f"{k['avg_er_wait_min']:.0f} min", ACC[0]),
             ("BED OCCUPANCY", f"{k['bed_occupancy_pct']}%", ACC[1]),
             ("READMISSION RATE", f"{k['readmission_rate_pct']}%", ACC[2]),
             ("AVG COST / PATIENT", f"${k['avg_cost_per_patient']:,.0f}", ACC[3])]
    for i, (lbl, val, c) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i]); panel(ax); ax.set_xticks([]); ax.set_yticks([])
        ax.text(.08, .66, lbl, transform=ax.transAxes, color=MUT, fontsize=9)
        ax.text(.08, .26, val, transform=ax.transAxes, color=c, fontsize=23, fontweight="bold")

    # department LOS vs benchmark
    axd = fig.add_subplot(gs[1, 0:2]); panel(axd)
    dp = m["department_performance"][::-1]
    names = [d["department"] for d in dp]
    delta = [d["los_vs_benchmark"] for d in dp]
    axd.barh(names, delta, color=[ACC[2] if v > 0 else ACC[1] for v in delta])
    axd.axvline(0, color=MUT, lw=.8)
    axd.set_title("Length of stay vs. benchmark (days)", color=INK, fontsize=12, loc="left", pad=8)
    axd.tick_params(colors=MUT, labelsize=9); axd.grid(axis="x", color=LINE, ls=":", alpha=.6)
    for s in axd.spines.values(): s.set_color(LINE)

    # occupancy bars
    axo = fig.add_subplot(gs[1, 2:4]); panel(axo)
    oc = m["occupancy"]
    axo.bar([o["department"][:5] for o in oc], [o["occupancy_pct"] for o in oc],
            color=[ACC[2] if o["occupancy_pct"] >= 90 else ACC[3] if o["occupancy_pct"] >= 75 else ACC[0] for o in oc])
    axo.set_title("Bed occupancy by unit (%)", color=INK, fontsize=12, loc="left", pad=8)
    axo.tick_params(colors=MUT, labelsize=8); axo.grid(axis="y", color=LINE, ls=":", alpha=.6)
    for s in axo.spines.values(): s.set_color(LINE)

    axs = fig.add_subplot(gs[2, :]); panel(axs); axs.set_xticks([]); axs.set_yticks([])
    axs.text(.015, .82, "AI EXECUTIVE SUMMARY", transform=axs.transAxes,
             color=ACC[0], fontsize=12, fontweight="bold")
    axs.text(.015, .60, "\n".join(textwrap.wrap(m["ai_summary"], 118)),
             transform=axs.transAxes, color=MUT, fontsize=10.3, va="top")
    fig.savefig(os.path.join(ROOT, "dashboard_preview.png"), dpi=130, facecolor=BG)
    plt.close(fig)


def main():
    t = generate_data()
    conn = load_to_sqlite(t)
    sql_out = run_sql(conn)
    kpi = sql_out["kpi_summary"].iloc[0].to_dict()
    fc = forecast_volume(t["weekly_admissions"])
    model = predict_readmit(t["encounters"])
    occ_df, occ_overall = occupancy(t["encounters"], t["departments"])
    icu_change = icu_mom(t["encounters"])
    summary = exec_summary(kpi, sql_out["department_performance"], occ_df, occ_overall, fc, model, icu_change)
    export(kpi, sql_out, occ_df, occ_overall, fc, model, summary)

    print("=== Hospital Operations Intelligence — pipeline complete ===")
    print(f"Encounters         : {int(kpi['encounters']):,}")
    print(f"Avg ER wait (min)  : {kpi['avg_er_wait_min']}   Bed occupancy: {occ_overall}%")
    print(f"Readmission rate % : {kpi['readmission_rate_pct']}   Avg LOS: {kpi['avg_los_days']}")
    print(f"Readmit ROC-AUC    : {model['roc_auc']}")
    print(f"ICU readmit MoM    : {icu_change:+.1f}%   Volume next month: {fc['next_month_change_pct']:+.1f}%")
    print("Exports: metrics.json, dashboard_data.js, dashboard_preview.png")


if __name__ == "__main__":
    main()
