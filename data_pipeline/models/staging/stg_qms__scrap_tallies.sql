select
    tally_id, job_id, press_id, cast(shift_date as date) as shift_date, shift, hour_bucket,
    cast(cavity_id as integer) as cavity_id, defect_code, qty, entered_by, cast(entered_ts as timestamp) as entered_ts,
    -- the hour the tally covers; the C shift runs past midnight into the next calendar day
    cast(shift_date as date) + to_hours(hour_bucket)
        + case when hour_bucket < 6 then interval 1 day else interval 0 day end as tally_hour
from {{ source('qms', 'scrap_tallies') }}
