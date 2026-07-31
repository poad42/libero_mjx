#!/usr/bin/env bash
# tlbfix.sh - safe build / verify / load / unload helper for mempool_tlbfix.ko
#
# Production ergonomics for the gfx1201 hipMallocAsync mempool TLB-flush fix.
# Does NOT embed any credential: privilege escalation is via ${SUDO:-sudo}, so
# the operator (or CI) supplies the password / runs under a privileged context.
#
# Subcommands:
#   check                 verify live amdgpu srcversion == header, and (if
#                         pahole is present) re-derive the 6 struct offsets from
#                         the live amdgpu.ko and compare them to the header.
#   build                 out-of-tree build against /lib/modules/$(uname -r)/build
#   load [COMM] [MODE] [WAIT_MS]
#                         run `check`, then insmod with onlycomm=COMM mode=MODE
#                         wait_ms=WAIT_MS. COMM defaults to empty (node-wide);
#                         pass a substring to confine the behaviour change to
#                         your own process(es) on a shared node. MODE=syncmap.
#   unload                rmmod (prints the unload counters from dmesg)
#   status                lsmod + recent tlbfix dmesg lines
#
# Examples:
#   ./tlbfix.sh check
#   ./tlbfix.sh build
#   sudo -v && ./tlbfix.sh load mywork syncmap 0     # confine to comm~="mywork"
#   ./tlbfix.sh load '' syncmap 0                     # node-wide (only when safe)
#   ./tlbfix.sh unload
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HDR="$HERE/amdgpu_vm_offsets.h"
KO="$HERE/mempool_tlbfix.ko"
SUDO="${SUDO:-sudo}"

hdr_val() { grep -E "^#define[[:space:]]+$1[[:space:]]" "$HDR" | head -1 | awk '{print $3}' | tr -d '"'; }

check() {
  local rc=0
  local expect live
  expect="$(hdr_val AMDGPU_SRCVERSION_EXPECT)"
  live="$(cat /sys/module/amdgpu/srcversion 2>/dev/null)"
  echo "amdgpu srcversion: live='$live' expected='$expect'"
  if [ "$live" != "$expect" ]; then
    echo "  MISMATCH -> module will refuse to load (rebuild + re-pahole for this build)"; rc=1
  else
    echo "  OK"
  fi

  if command -v pahole >/dev/null 2>&1; then
    local amdko src work
    amdko="$(find /lib/modules/$(uname -r) -name 'amdgpu.ko*' 2>/dev/null | head -1)"
    work="$(mktemp -d)"; src="$amdko"
    case "$amdko" in
      *.zst) zstd -d -f -o "$work/amdgpu.ko" "$amdko" >/dev/null 2>&1 && src="$work/amdgpu.ko";;
      *.xz)  xz -dkc "$amdko" > "$work/amdgpu.ko" 2>/dev/null && src="$work/amdgpu.ko";;
      *.gz)  gzip -dc "$amdko" > "$work/amdgpu.ko" 2>/dev/null && src="$work/amdgpu.ko";;
    esac
    echo "re-deriving struct amdgpu_vm offsets from: $src"
    pahole -C amdgpu_vm "$src" 2>/dev/null > "$work/avm.txt"
    check_off() { # field  header_macro
      local got want
      got="$(grep -E "[ *]$1;" "$work/avm.txt" | grep -oE '/\* +[0-9]+' | grep -oE '[0-9]+' | head -1)"
      want="$(hdr_val "$2")"
      if [ -n "$got" ] && [ "$got" = "$want" ]; then echo "  OK    $1 = $got"; else echo "  DIFF  $1 live=$got header=$want"; rc=1; fi
    }
    if [ -s "$work/avm.txt" ]; then
      check_off "tlb_seq"              AVM_OFF_TLB_SEQ
      check_off "last_tlb_flush"       AVM_OFF_LAST_TLB_FLUSH
      check_off "kfd_last_flushed_seq" AVM_OFF_KFD_LAST_FLUSHED_SEQ
      check_off "pasid"               AVM_OFF_PASID
      check_off "is_compute_context"   AVM_OFF_IS_COMPUTE_CONTEXT
      check_off "need_tlb_fence"       AVM_OFF_NEED_TLB_FENCE
    else
      echo "  (pahole produced no amdgpu_vm output; relying on srcversion guard)"
    fi
    rm -rf "$work"
  else
    echo "(pahole not installed; relying on srcversion guard only)"
  fi
  return $rc
}

case "${1:-}" in
  check)  check ;;
  build)  make -C "$HERE" ;;
  load)
    COMM="${2:-}"; MODE="${3:-syncmap}"; WAIT_MS="${4:-0}"
    if ! check; then echo "ABORT: srcversion/offset check failed"; exit 1; fi
    echo "loading: mode=$MODE onlycomm='$COMM' wait_ms=$WAIT_MS"
    $SUDO insmod "$KO" mode="$MODE" onlycomm="$COMM" wait_ms="$WAIT_MS" && echo "  insmod OK"
    dmesg | grep -E "tlbfix:" | tail -4
    ;;
  unload)
    $SUDO rmmod mempool_tlbfix && echo "  rmmod OK"
    dmesg | grep -E "tlbfix: unloaded" | tail -1
    ;;
  status)
    lsmod | grep mempool_tlbfix || echo "(not loaded)"
    dmesg | grep -E "tlbfix:" | tail -8
    ;;
  *)
    sed -n '2,40p' "${BASH_SOURCE[0]}"
    ;;
esac
