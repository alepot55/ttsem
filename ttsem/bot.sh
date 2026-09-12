#!/bin/bash
# One cycle of the bot: the newest commit of Triton main, an incremental build, the whole
# test_core.py under the validator at TTGIR, and the diff against the previous cycle.
# Loop it: `while true; do bash bot.sh; sleep 1800; done` inside a terminal multiplexer.
#
# Environment:
#   TRITON_SRC   a checkout of triton-lang/triton (built in place, incrementally)
#   BOT          directory the cycles are written to (one subdirectory per commit)
#   PY           the python of the virtualenv that has ttsem and this Triton installed
#   GPU          value for CUDA_VISIBLE_DEVICES (default 0)
#   JOBS         pytest-xdist workers (default 8)
set -u
TRITON_SRC=${TRITON_SRC:?set TRITON_SRC to a triton checkout}
BOT=${BOT:?set BOT to the output directory}
PY=${PY:-python}
GPU=${GPU:-0}
JOBS=${JOBS:-8}
mkdir -p "$BOT"
cd "$TRITON_SRC" || exit 2
git fetch -q origin main || { echo "FETCH_FAILED $(date +%FT%T)"; exit 2; }
NEW=$(git rev-parse --short=9 origin/main)
if [ -f "$BOT/$NEW/done" ]; then exit 0; fi
PREV=$(ls -t "$BOT" 2>/dev/null | while read -r d; do [ -f "$BOT/$d/done" ] && echo "$d" && break; done)
echo "CYCLE_START $NEW $(date +%FT%T)"
git checkout -q --detach "$NEW" || { echo "CHECKOUT_FAILED $NEW"; exit 2; }
mkdir -p "$BOT/$NEW"
if ! MAX_JOBS=${MAX_JOBS:-8} "$PY" -m pip install -q --no-build-isolation -e . \
    > "$BOT/$NEW/build.log" 2>&1; then
  echo "BUILD_FAILED $NEW"; exit 2
fi
TOPT=$(ls -d "$TRITON_SRC"/build/cmake.*/bin/triton-opt 2>/dev/null | head -1)
for stage in ttgir; do
  rm -rf "$BOT/$NEW/$stage"
  CUDA_VISIBLE_DEVICES=$GPU TTSEM_OUT="$BOT/$NEW/$stage" TTSEM_STAGE=$stage \
    TTSEM_TRITON_OPT="$TOPT" TTSEM_DUMP="$BOT/$NEW/$stage-ir" \
    "$PY" -m pytest -q -p ttsem.pytest_ttsem python/test/unit/language/test_core.py \
    --device cuda -n "$JOBS" -p no:cacheprovider --timeout 900 --timeout-method signal \
    > "$BOT/$NEW/$stage.pytest.log" 2>&1
  tail -n 1 "$BOT/$NEW/$stage.pytest.log"
  "$PY" -m ttsem.bot summary "$BOT/$NEW/$stage"
  if [ -n "$PREV" ] && [ -d "$BOT/$PREV/$stage" ]; then
    "$PY" -m ttsem.bot diff "$BOT/$PREV/$stage" "$BOT/$NEW/$stage" > "$BOT/$NEW/$stage.diff.md"
    rc=$?
    grep -E "^new bad|^no longer bad" "$BOT/$NEW/$stage.diff.md"
    [ $rc -ne 0 ] && echo "NEW_BAD $NEW $stage (see $BOT/$NEW/$stage.diff.md)"
  fi
done
# level 3: every distinct final-TTGIR module of the suite, Membar's barriers in, then out
if [ -d "$BOT/$NEW/ttgir-ir" ]; then
  "$PY" -m ttsem.races_corpus "$BOT/$NEW/ttgir-ir" --triton-opt "$TOPT" \
    --json "$BOT/$NEW/races.json" > "$BOT/$NEW/races.log" 2> "$BOT/$NEW/races.err"
  tail -n 1 "$BOT/$NEW/races.log"
  n_with=$(grep -c "^RACE-WITH" "$BOT/$NEW/races.log")
  [ "$n_with" -gt 0 ] && echo "RACE_WITH_BARRIERS $NEW: $n_with modules (see $BOT/$NEW/races.log)"
fi
touch "$BOT/$NEW/done"
echo "CYCLE_DONE $PREV..$NEW $(date +%FT%T)"
