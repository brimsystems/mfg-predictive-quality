-- Individuals and moving-range charts on every shot, per job (limits re-established at each
-- first-shot approval) x sensor x value. Limits come from the first 500 shots after approval,
-- with sigma from their standard deviation: cavity pressure values are autocorrelated from
-- shot to shot (lag-1 around 0.8), and the moving-range estimate would put the limits inside
-- the process's own short-term wander.
-- Rules, as the QMS runs them on the audit charts:
--   WE1  one point beyond 3 sigma
--   WE2  two of three beyond 2 sigma on one side
--   WE4  four of five beyond 1 sigma on one side
--   WE5  eight in a row on one side of the center line
--   MR   moving range above its upper limit (3.267 x mean moving range)
{% set metrics = ['fill_integral', 'pack_integral', 'cycle_integral', 'end_of_fill_pressure_bar', 'gate_seal_time_s'] %}
with s as (
    select n.shot_id, n.sensor_id, n.mold_id, n.press_id, n.shot_ts, c.job_id, c.after_approval,
           {{ metrics | join(', ') }}
    from {{ ref('stg_monitoring__shot_summary') }} n
    join {{ ref('int_shot_context') }} c using (shot_id)
    where c.after_approval and not n.sensor_dropout
),
long as (
    {% for m in metrics %}
    select shot_id, sensor_id, mold_id, press_id, shot_ts, job_id, '{{ m }}' as metric, {{ m }} as value
    from s where {{ m }} is not null
    {% if not loop.last %}union all{% endif %}
    {% endfor %}
),
seq as (
    select *,
           row_number() over w as point_no,
           abs(value - lag(value) over w) as moving_range
    from long
    window w as (partition by job_id, sensor_id, metric order by shot_ts)
),
limits as (
    select job_id, sensor_id, metric,
           avg(value) as center,
           avg(moving_range) as mr_bar,
           -- shot-to-shot values are autocorrelated, so the moving range understates the spread;
           -- sigma is the baseline window's standard deviation
           stddev_samp(value) as sigma,
           avg(moving_range) / 1.128 as sigma_short_term
    from seq
    where point_no <= {{ var('spc_baseline_shots') }}
    group by 1, 2, 3
),
z as (
    select q.*, l.center, l.mr_bar, l.sigma, l.sigma_short_term, (q.value - l.center) / nullif(l.sigma, 0) as z
    from seq q join limits l using (job_id, sensor_id, metric)
),
r as (
    select *,
        abs(z) > 3 as we1,
        greatest(sum((z > 2)::int) over w3, sum((z < -2)::int) over w3) >= 2 as we2,
        greatest(sum((z > 1)::int) over w5, sum((z < -1)::int) over w5) >= 4 as we4,
        (sum((z > 0)::int) over w8 = 8) or (sum((z < 0)::int) over w8 = 8) as we5,
        moving_range > 3.267 * mr_bar as mr_rule
    from z
    window w3 as (partition by job_id, sensor_id, metric order by shot_ts rows between 2 preceding and current row),
           w5 as (partition by job_id, sensor_id, metric order by shot_ts rows between 4 preceding and current row),
           w8 as (partition by job_id, sensor_id, metric order by shot_ts rows between 7 preceding and current row)
)
select shot_id, sensor_id, mold_id, press_id, shot_ts, job_id, metric, point_no, value, moving_range,
       center, sigma, sigma_short_term, mr_bar, z, we1, we2, we4, we5, mr_rule,
       point_no > {{ var('spc_baseline_shots') }} as is_monitoring
from r
