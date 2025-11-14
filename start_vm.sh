#!/usr/bin/env bash
# start_vm.sh — Run as root. Idempotent setup for a RunPod Ubuntu/Debian VM.
# - Creates user "tvr" with passwordless sudo
# - Copies root's authorized_keys to tvr
# - Prepares /workspace and hands it to tvr
# - Installs toolchain + Python stack
# - Installs virtualenv + virtualenvwrapper and wires a robust block into ~tvr/.bashrc
# - Ensures login shells load .bashrc and verifies mkvirtualenv/workon

set -euo pipefail

# -----------------------------
# Config
# -----------------------------
USERNAME="tvr"
USERHOME="/home/${USERNAME}"
WORKSPACE="/workspace"
SUDOERS_D="/etc/sudoers.d/90-${USERNAME}"
DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"

# -----------------------------
# Helpers
# -----------------------------
fail() { echo "ERROR: $*" >&2; exit 1; }
need_root() { [[ "${EUID}" -eq 0 ]] || fail "Run as root."; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }

trap 'echo "start_vm.sh failed at line $LINENO" >&2' ERR

need_root

echo "==> Setting up RunPod VM environment for user '${USERNAME}'"

# -----------------------------
# Base packages
# -----------------------------
if have_cmd apt-get; then
  echo "==> Installing base packages via apt-get"
  apt-get update -y
  apt-get install -y \
    sudo bash \
    python3 python3-venv python3-pip python3-dev \
    build-essential git curl ca-certificates \
    pkg-config unzip \
    virtualenv virtualenvwrapper
else
  fail "This script expects an Ubuntu/Debian base (apt-get not found)."
fi

# -----------------------------
# Create user
# -----------------------------
if ! id -u "${USERNAME}" >/dev/null 2>&1; then
  echo "==> Creating user ${USERNAME}"
  if have_cmd adduser; then
    adduser --disabled-password --gecos "" "${USERNAME}"
  else
    useradd -m -s /bin/bash "${USERNAME}"
  fi
else
  echo "==> User ${USERNAME} already exists (skipping creation)"
fi

# Ensure default shell is bash
if have_cmd chsh; then
  chsh -s /bin/bash "${USERNAME}" || true
fi

# -----------------------------
# Sudo privileges (NOPASSWD)
# -----------------------------
echo "==> Granting sudo to ${USERNAME}"
usermod -aG sudo "${USERNAME}"
echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "${SUDOERS_D}"
chmod 440 "${SUDOERS_D}"

# -----------------------------
# SSH keys
# -----------------------------
if [[ -f /root/.ssh/authorized_keys ]]; then
  echo "==> Installing root's authorized_keys for ${USERNAME}"
  install -d -m 700 -o "${USERNAME}" -g "${USERNAME}" "${USERHOME}/.ssh"
  install -m 600 -o "${USERNAME}" -g "${USERNAME}" /root/.ssh/authorized_keys "${USERHOME}/.ssh/authorized_keys"
else
  echo "==> /root/.ssh/authorized_keys not found; skipping SSH key copy"
fi

# -----------------------------
# Workspace
# -----------------------------
echo "==> Preparing ${WORKSPACE}"
mkdir -p "${WORKSPACE}"
chown -R "${USERNAME}:${USERNAME}" "${WORKSPACE}"

# -----------------------------
# Python tooling (system-wide)
# -----------------------------
echo "==> Upgrading pip/setuptools/wheel (system Python)"
python3 -m pip install --upgrade pip setuptools wheel

# -----------------------------
# Install user-site tools as tvr
# -----------------------------
echo "==> Installing virtualenv + virtualenvwrapper for ${USERNAME} (user site)"
sudo -u "${USERNAME}" bash -lc 'python3 -m pip install --user --upgrade virtualenv virtualenvwrapper || true'

# -----------------------------
# Shell init files (.bashrc/.bash_profile)
# -----------------------------
echo "==> Ensuring ${USERNAME}'s shell init files source virtualenvwrapper"

# Ensure home exists and owned
install -d -o "${USERNAME}" -g "${USERNAME}" "${USERHOME}"

# 1) Ensure .bashrc exists
BASHRC="${USERHOME}/.bashrc"
touch "${BASHRC}"
chown "${USERNAME}:${USERNAME}" "${BASHRC}"

# 2) Ensure login shells source .bashrc (Debian uses .profile; we'll add .bash_profile for clarity)
BASH_PROFILE="${USERHOME}/.bash_profile"
if [[ ! -f "${BASH_PROFILE}" ]]; then
  cat > "${BASH_PROFILE}" <<'EOF'
# ~/.bash_profile: executed by login shells.
# Load ~/.bashrc if present.
if [ -f "$HOME/.bashrc" ]; then
  . "$HOME/.bashrc"
fi
EOF
  chown "${USERNAME}:${USERNAME}" "${BASH_PROFILE}"
fi

# 3) Write robust virtualenvwrapper block into .bashrc if missing
ROBUST_BLOCK_TAG="virtualenvwrapper setup (v2)"
if ! sudo -u "${USERNAME}" bash -lc "grep -q '${ROBUST_BLOCK_TAG}' '${BASHRC}'"; then
  sudo -u "${USERNAME}" bash -lc "cat >> '${BASHRC}' <<'EOF'

# --- ${ROBUST_BLOCK_TAG} ---
# Ensure user-local bin is on PATH (pip --user installs scripts here)
case \":$PATH:\" in *\":$HOME/.local/bin:\"*) ;; *)
  export PATH=\"\$HOME/.local/bin:\$PATH\" ;;
esac

export WORKON_HOME=\"\$HOME/.virtualenvs\"
export VIRTUALENVWRAPPER_PYTHON=\"\$(command -v python3)\"

# Prefer distro-installed script, then user-site locations
for _p in \
  \"/usr/local/bin/virtualenvwrapper.sh\" \
  \"/usr/share/virtualenvwrapper/virtualenvwrapper.sh\" \
  \"\$HOME/.local/bin/virtualenvwrapper.sh\" \
  \"\$(python3 -c 'import site,os; print(os.path.join(site.getuserbase(),\"bin\",\"virtualenvwrapper.sh\"))')\"; do
  if [ -f \"\$_p\" ]; then
    export VIRTUALENVWRAPPER_SCRIPT=\"\$_p\"
    # shellcheck disable=SC1090
    . \"\$_p\"
    break
  fi
done
unset _p
EOF"
fi

# -----------------------------
# Verification (non-fatal)
# -----------------------------
echo "==> Verifying virtualenvwrapper for ${USERNAME} (non-fatal diagnostics)"
set +e
sudo -u "${USERNAME}" bash -lc '
  echo "-- PATH = $PATH"
  echo "-- Candidate virtualenvwrapper.sh files:"
  python3 - <<PY
import os, site, inspect
cands = [
  os.path.join(os.path.expanduser("~"), ".local", "bin", "virtualenvwrapper.sh"),
  os.path.join(site.getuserbase(), "bin", "virtualenvwrapper.sh"),
  "/usr/local/bin/virtualenvwrapper.sh",
  "/usr/bin/virtualenvwrapper.sh",
  "/usr/share/virtualenvwrapper/virtualenvwrapper.sh",
]
try:
    import virtualenvwrapper as v
    cands.append(os.path.join(os.path.dirname(inspect.getfile(v)), "virtualenvwrapper.sh"))
except Exception:
    pass
for p in cands:
    print(("FOUND  " if os.path.isfile(p) else "MISSING") + "  " + p)
PY

  echo "-- Sourcing login environment to load .bashrc..."
  # Simulate a login shell reading .bash_profile -> .bashrc
  set -a
  if [ -f "$HOME/.bash_profile" ]; then . "$HOME/.bash_profile"; fi
  set +a

  if command -v mkvirtualenv >/dev/null 2>&1 && command -v workon >/dev/null 2>&1; then
    echo "OK: mkvirtualenv/workon available."
    # smoke test: create and remove a temp venv
    ENAME="___vm_smoketest___"
    mkvirtualenv "$ENAME" >/dev/null 2>&1 && echo "OK: created $ENAME" || echo "WARN: could not create $ENAME"
    deactivate 2>/dev/null || true
    rmvirtualenv "$ENAME" >/dev/null 2>&1 || true
  else
    echo "WARN: mkvirtualenv/workon not on PATH after sourcing init files."
    echo "      You can source the script directly, e.g.:"
    echo "        source /usr/local/bin/virtualenvwrapper.sh"
    echo "        # or"
    echo "        source /usr/share/virtualenvwrapper/virtualenvwrapper.sh"
  fi
'
RC=$?
set -e

echo "==> All done."
echo "Switch to the user with:  su - ${USERNAME}"
echo "Create a venv with:       mkvirtualenv demo && workon demo"
