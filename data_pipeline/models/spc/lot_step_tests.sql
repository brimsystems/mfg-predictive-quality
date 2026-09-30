-- At each resin lot change during a job: the post-gate fill integral and peak pressure over the
-- 200 shots after the load against the 200 before, in units of the job's shot-to-shot spread.
with c as (
    select c.shot_id, c.job_id, c.press_id, c.mold_id, c.shot_ts, c.resin_lot_id, c.lot_change_ts, c.after_approval,
           s.pg_fill_dev, s.pg_peak_dev
    from {{ ref('int_shot_context') }} c join {{ ref('int_shot_summary') }} s using (shot_id)
    where c.after_approval
),
changes as (
    select distinct job_id, lot_change_ts, resin_lot_id
    from c where lot_change_ts > (select min(shot_ts) from c c2 where c2.job_id = c.job_id)
),
w as (
    select ch.job_id, ch.lot_change_ts, ch.resin_lot_id,
           c.shot_ts >= ch.lot_change_ts as after,
           row_number() over (partition by ch.job_id, ch.lot_change_ts, c.shot_ts >= ch.lot_change_ts
                              order by abs(datediff('second', ch.lot_change_ts, c.shot_ts))) as k,
           c.pg_fill_dev, c.pg_peak_dev, c.shot_id, c.press_id, c.mold_id, c.shot_ts
    from changes ch join c on c.job_id = ch.job_id
),
agg as (
    select job_id, lot_change_ts, any_value(resin_lot_id) as resin_lot_id,
           any_value(press_id) as press_id, any_value(mold_id) as mold_id,
           min(case when after and k = 1 then shot_id end) as shot_id,
           min(case when after and k = 1 then shot_ts end) as shot_ts,
           avg(case when after then pg_fill_dev end) - avg(case when not after then pg_fill_dev end) as fill_step,
           avg(case when after then pg_peak_dev end) - avg(case when not after then pg_peak_dev end) as peak_step,
           stddev(pg_fill_dev) as fill_sd,
           count(*) filter (where after) as n_after, count(*) filter (where not after) as n_before
    from w where k <= 200
    group by 1, 2
)
select *,
       -- shot-to-shot autocorrelation shrinks the effective sample to about one in eight shots
       fill_step / nullif(fill_sd * sqrt(8.0 * (1.0 / n_after + 1.0 / n_before)), 0) as step_t
from agg
where n_after >= 50 and n_before >= 50
