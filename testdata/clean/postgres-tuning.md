# Postgres Checkpoint Tuning

Set checkpoint_timeout to 15 minutes and max_wal_size to 4GB before touching
anything else. Most write stalls we see in production trace back to
checkpoints firing every 30 seconds because max_wal_size was left at the
1GB default while the workload writes 40MB per second.

Watch the checkpoints_req counter in pg_stat_bgwriter. If requested
checkpoints outnumber timed ones, the WAL ceiling is still too low and the
database is checkpointing under pressure instead of on schedule. After
raising the ceiling, recovery time grows: budget roughly one minute of
replay per 2GB of WAL when sizing.

Leave checkpoint_completion_target at 0.9. Spreading the flush across the
whole interval keeps the I/O curve flat, and the sharp fsync spike people
blame on autovacuum is usually this setting left at its old default.
