select press_id, equipment, cast(event_ts as timestamp) as event_ts, event_type
from {{ source('toolroom', 'equipment_service') }}
