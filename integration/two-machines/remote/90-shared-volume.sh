#!/usr/bin/env bash
# Lo scenario `shared` su hardware vero: due container distinti, un solo
# filesystem sotto entrambi.
#
#   bash $KIT_ROOT/kit/90-shared-volume.sh run
#   bash $KIT_ROOT/kit/90-shared-volume.sh report
#
# Ravex non lo assume: lo verifica scrivendo e cercando. Su una box sola la
# verifica non poteva sbagliarsi — stesso processo, stesso mount. Qui i due
# rank sono in mount namespace separati e vedono lo stesso volume solo perche'
# la macchina lo monta in entrambi, che e' il caso vero di GPU-59.
#
# Con lo storage condiviso la replica deve essere **spenta**: una copia sullo
# stesso filesystem non protegge da niente e costerebbe banda.
. "$(dirname "$0")/lib.sh"

SHARED="${SHARED:-/workspace/ravex-shared}"
NAME=shared
DIR="$WORK/$NAME"

case "${1:-report}" in
run)
    section "storage condiviso su $SHARED"
    mkdir -p "$DIR" "$SHARED"
    # Solo un rank pulisce, e prima che l'altro parta: e' un filesystem solo.
    [ "$NODE_RANK" = "0" ] && rm -rf "${SHARED:?}"/* && echo "  ripulito da node 0"
    cat > "$DIR/ravex.yaml" <<YAML
checkpoint_every: 4
backend: moonclip
keep_last: 2
sharded_checkpoints: per_rank
replicate_every: 10
storage:
  type: local
  path: $SHARED
log_file: ./ravex.log
log_level: INFO
YAML
    cd "$DIR"
    launch $KIT_ROOT/kit/run_train.py -- \
        --params "${PARAMS}" --hidden "$HIDDEN" --steps "${STEPS:-12}" \
        --measure none 2>&1 | tee "$OUT/shared.node$NODE_RANK.out" || true
    cp -f "$DIR/ravex.log" "$OUT/shared.node$NODE_RANK.log" 2>/dev/null || true
    section "cosa c'e' sul volume condiviso"
    ls -d "$SHARED"/rank_* "$SHARED"/replica/rank_* 2>/dev/null | sed 's/^/  /' || echo "  (niente)"
    ;;
report)
    section "cosa ha detto ravex, node $NODE_RANK"
    grep -hE "shared across|This job spans|Resumed at|Replicated step|from scratch|No space" \
        "$OUT/shared.node$NODE_RANK.log" 2>/dev/null \
        | sed -E 's/^[0-9-]+ [0-9:,]+ \[ravex\] [A-Z]+ //' | sed 's/^/  /' || echo "  (niente)"
    ;;
*) echo "usage: $0 {run|report}" >&2; exit 2 ;;
esac
