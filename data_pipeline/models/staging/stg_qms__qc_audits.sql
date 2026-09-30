select
    audit_id, job_id, press_id, mold_id, cast(audit_ts as timestamp) as audit_ts, inspector_id, sample_size,
    cast(sampled_shot_cycle_no as bigint) as sampled_shot_cycle_no, cavity_ids_sampled,
    audit_rule_violations, disposition
from {{ source('qms', 'qc_audits') }}
