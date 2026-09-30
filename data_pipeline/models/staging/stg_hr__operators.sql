select operator_id, cast(hire_date as date) as hire_date, shift from {{ source('hr', 'operators') }}
