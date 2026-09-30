select press_id, press_type, tonnage, install_year, opc_ua, cavity_pressure_unit,
       coalesce(cell = 'instrumented', false) as is_instrumented
from {{ source('mes', 'presses') }}
