-- Reject-bin reviews. Medical molds sort into an indexed tray and each sorted shot is reviewed with its
-- shot (review_mode = indexed_tray); the other molds' bins are reviewed once per shift by job, with no shot
-- (review_mode = per_shift). Codes are recorded with counts, as code:qty pairs.
select
    review_id, review_mode, cast(shot_id as bigint) as shot_id, job_id, press_id, mold_id,
    cast(shift_date as date) as shift_date, shift, cast(reviewed_ts as timestamp) as reviewed_ts, inspector_id,
    sorted_shots, pieces_reviewed, pieces_confirmed_defective, defect_codes,
    array_to_string(list_transform(string_split(defect_codes, ';'), lambda x: split_part(x, ':', 1)), ';') as defect_code_list,
    pieces_good
from {{ source('qms', 'sort_dispositions') }}
