#!/usr/bin/env bash
# Reproduces a real Supervisor add-on install/build/start against a live HAOS
# instance, using VM 120 ("hatest", 192.168.1.201) on the zeratul Proxmox host —
# see INFRA/servers.md. That VM intentionally runs Proxmox's default "kvm64" CPU
# type (no x86-64-v2/SSE4.2), which is what caught the numpy>=2.4 wheel
# incompatibility behind GitHub issue #9: `pip install` and unit tests alone
# can't catch this class of bug because they don't exercise the real Alpine
# base image + real Supervisor build pipeline + real (non-"host") CPU features.
#
# Usage: tests/haos_vm_install_test.sh [--keep]
#   --keep   leave the add-on installed/running on the VM for manual inspection
#            (default: uninstalled and cleaned up on exit, success or failure)
#
# Requires: ssh access to `zeratul` (see INFRA/access.md), and this machine
# reachable from VM 120 over LAN (serves the repo tarball over plain HTTP).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VMID=120
HTTP_PORT=8931
ADDON_DIR_NAME=energy_optimizer_ci_test
LOCAL_ADDON_SLUG="local_${ADDON_DIR_NAME}"
KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"; [[ -n "${HTTP_PID:-}" ]] && kill "$HTTP_PID" 2>/dev/null || true' EXIT

qexec() { ssh zeratul "qm guest exec $VMID -- $*"; }

echo "==> Packing working tree"
tar czf "$SCRATCH/repo.tgz" --exclude=.git -C "$REPO_ROOT" .

echo "==> Serving tarball for the VM to pull"
( cd "$SCRATCH" && python3 -m http.server "$HTTP_PORT" --bind 0.0.0.0 >/dev/null 2>&1 & echo $! > "$SCRATCH/http.pid" )
HTTP_PID="$(cat "$SCRATCH/http.pid")"
MY_IP="$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1)"

cleanup_addon() {
  qexec ha apps stop "$LOCAL_ADDON_SLUG" >/dev/null 2>&1 || true
  qexec ha apps uninstall "$LOCAL_ADDON_SLUG" >/dev/null 2>&1 || true
  qexec "rm -rf /mnt/data/supervisor/apps/local/$ADDON_DIR_NAME /tmp/repo.tgz" >/dev/null 2>&1 || true
  qexec ha store reload >/dev/null 2>&1 || true
}
[[ "$KEEP" -eq 0 ]] && trap 'cleanup_addon' EXIT || true
cleanup_addon  # start from a clean slate too

echo "==> Copying repo into the VM as a local add-on"
qexec "curl -s -o /tmp/repo.tgz http://${MY_IP}:${HTTP_PORT}/repo.tgz"
qexec "mkdir -p /mnt/data/supervisor/apps/local/$ADDON_DIR_NAME"
qexec "tar xzf /tmp/repo.tgz -C /mnt/data/supervisor/apps/local/$ADDON_DIR_NAME"
qexec ha store reload

echo "==> Building (this runs the real Dockerfile against the real HAOS base image)"
if ! qexec ha apps install "$LOCAL_ADDON_SLUG" --no-progress; then
  echo "BUILD FAILED — supervisor build log:" >&2
  qexec ha supervisor logs >&2
  exit 1
fi

echo "==> Starting and checking for a clean boot"
qexec ha apps start "$LOCAL_ADDON_SLUG"
sleep 5
LOGS="$(qexec ha apps logs "$LOCAL_ADDON_SLUG")"
echo "$LOGS"

if ! grep -q "Scheduler started" <<<"$LOGS"; then
  echo "Add-on did not reach a running state (no 'Scheduler started' in logs)" >&2
  exit 1
fi

echo "==> OK: build + start succeeded on real HAOS (VM $VMID)"
[[ "$KEEP" -eq 1 ]] && echo "(--keep: add-on left installed as $LOCAL_ADDON_SLUG)"
