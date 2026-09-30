select
    change_id, press_id, job_id, cast(change_ts as timestamp) as change_ts, parameter,
    old_value, new_value, new_value - old_value as delta, technician_id,
    nullif(reason_code, 'unknown') as reason_code, approved_by
from {{ source('qms', 'setpoint_changes') }}
