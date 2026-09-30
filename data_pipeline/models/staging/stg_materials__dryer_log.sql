select dryer_id, cast(ts as timestamp) as ts, setpoint_temp_c, actual_temp_c, dew_point_c, hopper_level_pct, alarm_code
from {{ source('materials', 'dryer_log') }}
