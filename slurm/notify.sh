#!/usr/bin/env bash
# Job-outcome notifications. Sourced by every job script, after env.sh.
#
# `dxenv_notify_on_exit <label>` installs traps so the job reports itself whatever
# happens: clean exit, a non-zero exit from `set -e`, or the SIGTERM SLURM sends before a
# wall-clock kill. That last one is the case worth having -- a run that dies on the wall
# clock is the one you most want to hear about, and it is also the one that produces no
# error message at all.
#
# Every notification path is best-effort. A notifier that can turn a successful
# twenty-hour run into a failed one because an HTTPS call timed out is worse than none, so
# nothing here is allowed to change the job's exit status.

dxenv_notify() {
    python "$DXENV_REPO/scripts/notify.py" "$@" >/dev/null 2>&1 || true
}

dxenv_notify_on_exit() {
    local label="${1:-job}"
    local log="${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-local}.out"
    export DXENV_LABEL="$label" DXENV_LOG="$log" DXENV_T0=$SECONDS

    _DXENV_REPORTED=0
    _dxenv_report() {
        local code=$1 reason=$2
        # The TERM trap reports and then exits, which re-enters the EXIT trap. Without
        # this guard a wall-clock kill sends the same alert twice, and an alert channel
        # that duplicates is one you start ignoring.
        [[ "$_DXENV_REPORTED" == "1" ]] && return 0
        _DXENV_REPORTED=1
        local mins=$(( (SECONDS - DXENV_T0) / 60 ))
        local host="${SLURMD_NODENAME:-$(hostname)}"
        if [[ "$code" == "0" ]]; then
            dxenv_notify --status ok --title "$DXENV_LABEL finished" \
                --text "job ${SLURM_JOB_ID:-local} on ${host} · ${mins} min" \
                --log "$DXENV_LOG" --log-lines 20
        else
            dxenv_notify --status fail --title "$DXENV_LABEL FAILED ($reason)" \
                --text "job ${SLURM_JOB_ID:-local} on ${host} · ${mins} min · exit ${code}" \
                --log "$DXENV_LOG" --log-lines 40
        fi
    }

    # SIGTERM first: SLURM sends it before the wall-clock kill, and without this the job
    # simply vanishes with no notification at all.
    trap '_dxenv_report 143 "wall clock / cancelled"; exit 143' TERM
    trap '_dxenv_report $? "exit code"' EXIT

    dxenv_notify --status start --title "$DXENV_LABEL started" \
        --text "job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}"
}
