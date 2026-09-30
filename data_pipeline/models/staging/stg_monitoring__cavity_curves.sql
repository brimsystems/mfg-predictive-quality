-- Retained curves: 100 readings per second for the full cycle.
select shot_id, sensor_id, t_ms, pressure_bar, mold as mold_id
from {{ source('monitoring', 'cavity_curves') }}
