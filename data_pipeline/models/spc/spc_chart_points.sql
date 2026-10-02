-- Control charts on every shot, per job (limits re-established at each first-shot approval) x sensor x value.
--
-- Cavity pressure values are strongly autocorrelated from shot to shot (lag-1 around 0.8): the melt, the mold and
-- the material change slowly, so each shot looks like the last. On such data, limits drawn around the raw values
-- flag the process's own slow wander as special causes, whichever way sigma is estimated. The charts therefore plot
-- the residual of an AR(1) fit: each shot against what the previous shot predicts. The residuals are close to
-- independent, so rules 1 and 2 keep their textbook false-alarm rates; slow movement is left to the EWMA and CUSUM
-- drift detectors. Each residual is centred on the mean of the previous 500 shots rather than a fixed baseline mean, so
-- a sudden shift is flagged when it happens and a level the process has settled at is left to the EWMA and CUSUM.
-- The AR(1) mean and coefficient come from this run's stable baseline (shots 200 to 1,000 after
-- approval, past setup convergence). The residual sigma comes from a longer baseline: every earlier run of the same
-- mold, press and sensor, pooled, so it reflects the variation the process shows across whole runs, not only its first
-- hours. A mold and press's first run uses its own baseline.
--
-- For comparison the model also carries two raw-value versions:
--   textbook  individuals limits with sigma from the mean moving range over the first 500 shots
--   baseline  individuals limits with sigma from the standard deviation of the first 500 shots
-- Rules, as the QMS runs them on the audit charts (WE1 and WE2 at the deployed limits, the settings rule1_limit and
-- rule2_limit; the textbook values are 3 and 2 sigma):
--   WE1  one point beyond 3 sigma          WE2  two of three beyond 2 sigma on one side
--   WE4  four of five beyond 1 sigma        WE5  eight in a row on one side of the center line
{% set metrics = ['fill_integral', 'pack_integral', 'cycle_integral', 'end_of_fill_pressure_bar', 'gate_seal_time_s'] %}
with s as (
    select n.shot_id, n.sensor_id, n.mold_id, n.press_id, n.shot_ts, c.job_id, c.shots_since_approval,
           {{ metrics | join(', ') }}
    from {{ ref('stg_monitoring__shot_summary') }} n
    join {{ ref('int_shot_context') }} c using (shot_id)
    where c.after_approval and not n.sensor_dropout
),
long as (
    {% for m in metrics %}
    select shot_id, sensor_id, mold_id, press_id, shot_ts, job_id, shots_since_approval, '{{ m }}' as metric, {{ m }} as value
    from s where {{ m }} is not null
    {% if not loop.last %}union all{% endif %}
    {% endfor %}
),
seq as (
    select *,
           row_number() over w as point_no,
           lag(value) over w as prev_value,
           avg(value) over (partition by job_id, sensor_id, metric order by shot_ts rows between 500 preceding and 1 preceding) as recent_mean,
           count(value) over (partition by job_id, sensor_id, metric order by shot_ts rows between 500 preceding and 1 preceding) as recent_n,
           abs(value - lag(value) over w) as moving_range
    from long
    window w as (partition by job_id, sensor_id, metric order by shot_ts)
),
raw_limits as (
    select job_id, sensor_id, metric, avg(value) as center, avg(moving_range) as mr_bar,
           avg(moving_range) / 1.128 as sigma_textbook, stddev_samp(value) as sigma_baseline
    from seq where point_no <= {{ var('spc_baseline_shots') }}
    group by 1, 2, 3
),
ar_base as (
    -- stable baseline; short runs fall back to whatever lies past the first 200 shots
    select q.* from seq q
    where q.shots_since_approval between 200 and 1000 and q.prev_value is not null
),
ar_fit as (
    select job_id, sensor_id, metric, avg(value) as ar_mean,
           greatest(least(corr(value, prev_value), 0.98), 0.0) as phi
    from ar_base group by 1, 2, 3
),
resid as (
    select q.*, r.center, r.mr_bar, r.sigma_textbook, r.sigma_baseline, a.ar_mean, a.phi,
           -- centred on the mean of the previous 500 shots, so a level the process has settled at is left to drift detection
           (q.value - m.level) - a.phi * (q.prev_value - m.level) as residual
    from seq q join raw_limits r using (job_id, sensor_id, metric) left join ar_fit a using (job_id, sensor_id, metric),
    lateral (select case when q.recent_n >= 100 then q.recent_mean else a.ar_mean end as level) m
),
job_var as (
    -- residual variance of each run past setup (shots after 200), per mold x press x sensor x value
    select job_id, any_value(mold_id) as mold_id, any_value(press_id) as press_id, sensor_id, metric, min(shot_ts) as job_start,
           sum(residual * residual) as ss, count(*) as n,
           stddev_samp(residual) filter (where shots_since_approval between 200 and 1000) as sd_base
    from resid where shots_since_approval > 200 and residual is not null
    group by job_id, sensor_id, metric
),
ar_sigma as (
    -- residual sigma pooled over every earlier run of the same mold, press and sensor (as of this run's start);
    -- a mold and press's first run falls back to its own stable baseline
    select job_id, sensor_id, metric,
           coalesce(sqrt(sum(ss) over prior / nullif(sum(n) over prior, 0)), sd_base) as sigma_resid
    from job_var
    window prior as (partition by mold_id, press_id, sensor_id, metric order by job_start rows between unbounded preceding and 1 preceding)
),
z as (
    select q.*, s.sigma_resid,
           q.residual / nullif(s.sigma_resid, 0) as z,
           (q.value - q.center) / nullif(q.sigma_textbook, 0) as z_textbook,
           (q.value - q.center) / nullif(q.sigma_baseline, 0) as z_baseline
    from resid q left join ar_sigma s using (job_id, sensor_id, metric)
),
r as (
    select *,
        abs(z) > {{ var('rule1_limit') }} as we1,
        greatest(sum((z > {{ var('rule2_limit') }})::int) over w3, sum((z < -{{ var('rule2_limit') }})::int) over w3) >= 2 as we2,
        greatest(sum((z > 1)::int) over w5, sum((z < -1)::int) over w5) >= 4 as we4,
        (sum((z > 0)::int) over w8 = 8) or (sum((z < 0)::int) over w8 = 8) as we5,
        moving_range > 3.267 * mr_bar as mr_rule,
        abs(z_textbook) > 3 as we1_textbook,
        greatest(sum((z_textbook > 2)::int) over w3, sum((z_textbook < -2)::int) over w3) >= 2 as we2_textbook,
        abs(z_baseline) > 3 as we1_baseline,
        greatest(sum((z_baseline > 2)::int) over w3, sum((z_baseline < -2)::int) over w3) >= 2 as we2_baseline
    from z
    window w3 as (partition by job_id, sensor_id, metric order by shot_ts rows between 2 preceding and current row),
           w5 as (partition by job_id, sensor_id, metric order by shot_ts rows between 4 preceding and current row),
           w8 as (partition by job_id, sensor_id, metric order by shot_ts rows between 7 preceding and current row)
)
select shot_id, sensor_id, mold_id, press_id, shot_ts, job_id, metric, point_no, shots_since_approval, value, prev_value, moving_range,
       center, ar_mean, phi, residual, sigma_resid as sigma, sigma_textbook, sigma_baseline, mr_bar, z, z_textbook, z_baseline,
       coalesce(we1, false) as we1, coalesce(we2, false) as we2, coalesce(we4, false) as we4, coalesce(we5, false) as we5, mr_rule,
       we1_textbook, we2_textbook, we1_baseline, we2_baseline,
       point_no > {{ var('spc_baseline_shots') }} as is_monitoring
from r
