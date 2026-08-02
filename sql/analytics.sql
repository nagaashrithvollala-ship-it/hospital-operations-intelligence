-- analytics.sql — Hospital Operations Intelligence Platform
-- Named queries; the pipeline splits on "-- name:" and runs each block.
-- Dialect: SQLite (portable to Postgres / Snowflake).

-- name: kpi_summary
-- Headline operational KPIs across all encounters.
SELECT
    COUNT(*)                                   AS encounters,
    ROUND(AVG(er_wait_min), 1)                 AS avg_er_wait_min,
    ROUND(AVG(length_of_stay), 2)              AS avg_los_days,
    ROUND(100.0 * AVG(readmit_30d), 1)         AS readmission_rate_pct,
    ROUND(AVG(satisfaction), 2)                AS avg_satisfaction,
    ROUND(AVG(cost), 0)                        AS avg_cost_per_patient
FROM encounters;

-- name: department_performance
-- Per-department clinical & operational scorecard vs. length-of-stay benchmark.
SELECT
    d.dept_id,
    d.name                                     AS department,
    d.benchmark_los,
    COUNT(e.encounter_id)                      AS encounters,
    ROUND(AVG(e.length_of_stay), 2)            AS avg_los,
    ROUND(AVG(e.length_of_stay) - d.benchmark_los, 2) AS los_vs_benchmark,
    ROUND(100.0 * AVG(e.readmit_30d), 1)       AS readmit_pct,
    ROUND(AVG(e.satisfaction), 2)              AS satisfaction,
    ROUND(AVG(e.cost), 0)                      AS avg_cost
FROM departments d
JOIN encounters e ON e.dept_id = d.dept_id
GROUP BY d.dept_id, d.name, d.benchmark_los
ORDER BY los_vs_benchmark DESC;

-- name: monthly_trend
-- Monthly ER wait and readmission trend.
SELECT
    strftime('%Y-%m', admit_date)              AS month,
    ROUND(AVG(er_wait_min), 1)                 AS avg_er_wait_min,
    ROUND(100.0 * AVG(readmit_30d), 1)         AS readmission_rate_pct,
    COUNT(*)                                   AS admissions
FROM encounters
GROUP BY month
ORDER BY month;

-- name: er_wait_by_acuity
-- Average ER wait by triage acuity (1 = most urgent .. 5 = least).
SELECT
    acuity,
    COUNT(*)                                   AS encounters,
    ROUND(AVG(er_wait_min), 1)                 AS avg_er_wait_min
FROM encounters
GROUP BY acuity
ORDER BY acuity;
