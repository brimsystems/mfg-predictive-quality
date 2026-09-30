select mold_id, cast(event_ts as timestamp) as event_ts, event_type, shots_at_event, technician_id, notes
from {{ source('toolroom', 'mold_maintenance') }}
