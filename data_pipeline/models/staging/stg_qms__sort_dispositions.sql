select
    shot_id, cast(reviewed_ts as timestamp) as reviewed_ts, inspector_id, pieces_reviewed,
    pieces_confirmed_defective, defect_codes, pieces_good
from {{ source('qms', 'sort_dispositions') }}
