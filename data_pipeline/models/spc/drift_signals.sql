-- One row per drift signal: the first time a detector fires in a segment between resets
-- (approval, a documented drift correction, vent cleaning, PM, ring replacement, a lot change).
-- A signal stays open until the next reset, so repeated firings inside a segment are one signal.
{% set detectors = ['ewma_pack', 'ewma_fill', 'cusum_eof', 'cusum_gate_seal', 'cusum_pack_var'] %}
with d as (
    select d.*, c.job_id, c.press_id, c.mold_id, c.shot_ts,
           sum(d.drift_reset::int) over (partition by c.job_id order by c.shot_ts
                                         rows between unbounded preceding and current row) as segment_no
    from {{ ref('drift_shot_state') }} d join {{ ref('int_shot_context') }} c using (shot_id)
),
fired as (
    {% for k in detectors %}
    select shot_id, job_id, press_id, mold_id, shot_ts, segment_no, '{{ k }}' as detector, {{ k }} as statistic
    from d where {{ k }}_signal
    {% if not loop.last %}union all{% endif %}
    {% endfor %}
)
select shot_id, job_id, press_id, mold_id, shot_ts, detector, statistic
from fired
qualify row_number() over (partition by job_id, segment_no, detector order by shot_ts) = 1
union all
select shot_id, job_id, press_id, mold_id, shot_ts, 'lot_step' as detector, step_t as statistic
from {{ ref('lot_step_tests') }}
where abs(step_t) > 3
