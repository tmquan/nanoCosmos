# Server setup

Bring up a fresh Colossus lease and leave two tmux sessions ready.
Session `0` is training. Session `1` is TensorBoard. The train command
is started by hand in session `0` after this setup finishes.

The lease host changes every reprovision. Read it from the lease; do
not keep using a previous `umb-b300-dp-*` name. Home on this image is
`/localhome/local-tranminhq`. Nothing in this file is a password.

Hosts with a problem on 2026-11-01. Do not reuse them:

| Host | What failed |
| --- | --- |
| `umb-b300-dp-171` | NVLink Xid 145 on GPU 1 (`0000:48:00`), link 14 and then link 17. Training died twice with `cudaErrorNvlinkUncorrectable` before a step. Lease cancelled. |
| `umb-b300-dp-206` | Provision `SSH_CHECK_FAILURE` after 40 SSH attempts (19:09). Stayed `POWER_OFF` / `AVAILABLE`. Lease expired. |
| `umb-b300-dp-161` | Provision `SSH_CHECK_FAILURE` after 40 SSH attempts (20:15). Host `POWER_ON` / `ERROR`, no IP. BMC answered, lease secrets were not configured, and the IPMI reboot failed. |

## 1. SSH key

```bash
colossus bm lease list --status ACTIVE --json
colossus bm lease list --lease-id <id> --show-creds --json
```

Use `local_accountname` (it is `local-tranminhq`) and the FQDN from that
lease. `local_accountpassword` is only for `ssh-copy-id`; do not write
it into this repo, a log, or the shell history you keep.

After a clean reprovision the old host key is wrong. Clear it, then
install `~/.ssh/id_ed25519.pub`:

```bash
ssh-keygen -R <fqdn>
ssh-keygen -R <ip>
ssh-copy-id -i ~/.ssh/id_ed25519.pub local-tranminhq@<fqdn>
```

Add the short name to `~/.ssh/config.colossus` (it is included from
`~/.ssh/config`) so later commands can use the short host:

```
Host <short-name>
    HostName <fqdn>
    User local-tranminhq
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

```bash
ssh -o BatchMode=yes local-tranminhq@<short-name>
```

`BatchMode` must succeed before the rest. Sudo on this image is
passwordless (`sudo -n true`).

## 2. Clone and bootstrap

`setup.sh` installs CUDA 13.4.3, cuDNN, and Anaconda at `$HOME/anaconda3`.
Run it once. Wait until it prints `Setup complete`. Do not start a
second copy.

```bash
git clone https://github.com/tmquan/nanoCosmos.git "$HOME/nanoCosmos"
cd "$HOME/nanoCosmos"
sudo ./setup.sh
```

## 3. Environment

Blackwell and Rubin use the same lockfile, `requirements.txt`:
Python 3.14 and the PyTorch 2.15 CUDA 13.4 release wheels
(`torch==2.15.0+cu134`, `torchvision==0.29.2+cu134`) from
`https://download.pytorch.org/whl/cu134`. Leave
`TORCH_DISABLE_NATIVE_JIT` unset. `pip install -e . --no-deps` is
required: without it `python scripts/train.py` raises
`ModuleNotFoundError: No module named 'nanocosmos'`.

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n nanocosmos python=3.14 -y
conda activate nanocosmos
pip install uv
cd "$HOME/nanoCosmos"
uv pip install -r requirements.txt --prerelease=allow --index-strategy unsafe-best-match
pip install -e . --no-deps
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

Expect `2.15.0+cu134 True 8`.

## 4. Scratch

The CIFS username is `tranminhq`. The share password is typed into a
mode-0600 file and the file is deleted after the extra copy mounts
exist. It is not stored here. `uid`/`gid` are the name `local-tranminhq` (the numeric
id changes per lease). The mount is not in fstab; it is gone after
reboot. `cifs-utils` is already on the image (`/usr/sbin/mount.cifs`).

The EM tree on this share is `/scratch/EM`. There is no
`/scratch/data/EM`.

```bash
umask 077
cred=$(mktemp)
cat > "$cred" << 'EOF'
username=tranminhq
password=TYPE_THE_CIFS_PASSWORD_HERE
EOF
share=//ipp1-cdot01-col01.nvidia.com/scratch.tranminhq_colossus
opts="credentials=$cred,uid=local-tranminhq,gid=local-tranminhq,vers=3.1.1,cache=loose,file_mode=0644,dir_mode=0755"
sudo mkdir -p /scratch
sudo mount -t cifs -o "$opts" "$share" /scratch
df -h /scratch
ls /scratch/EM
```

Keep `$cred` until the copy mounts below exist, then delete it. The
server negotiates `rsize=65536` and does not advertise SMB
multichannel, so this mount is one TCP session.

## 5. Data and outputs

About 310G. One rsync of `/scratch` shares that single session.
Raising parallelism on one mount stays near 1.1 Gbit/s. Extra mounts
need `nosharesock` or they reuse the same socket. Cold reads on the
50 Gbit/s NIC climbed through 8 sessions (about 4.3 Gbit/s) and were
no faster at 16. The block below doubles sessions until the rate gains
less than 10% or passes 85% of the link, then stripes the tree across
the winning count with 16 rsyncs each. Run it in the same shell as the
mount. `rsync` can be re-run; it skips files that already match.

```bash
nic=$(ip route get 10.117.145.13 | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}')
link_Bps=$(( $(sudo ethtool "$nic" | awk '/Speed:/ {print $2+0}') * 1000000 / 8 ))
mkdir -p "$HOME/data" "$HOME/outputs"
python3 - "$opts" "$share" "$nic" "$link_Bps" "$HOME/data" << 'PY'
import subprocess, sys, threading, time
opts, share, nic, link_bps, dest = sys.argv[1:]
link_bps = int(link_bps)

def sh(args):
    subprocess.run(args, check=True)

def mounted(path):
    return subprocess.run(["findmnt", path], stdout=subprocess.DEVNULL).returncode == 0

def ensure_mount(i):
    path = f"/mnt/emsrc/{i}"
    if not mounted(path):
        sh(["sudo", "mkdir", "-p", path])
        sh(["sudo", "mount", "-t", "cifs", "-o", opts + ",nosharesock", share, path])
    return path

def nic_rx():
    for line in open("/proc/net/dev"):
        if line.split(":")[0].strip() == nic:
            return int(line.split(":")[1].split()[0])
    raise SystemExit(f"NIC {nic} not in /proc/net/dev")

raw = subprocess.check_output(
    ["find", "/scratch/EM", "-type", "f", "-printf", "%s\\t%P\\0"])
rows = []
for rec in raw.split(b"\0"):
    if not rec:
        continue
    sz, rel = rec.split(b"\t", 1)
    rows.append((int(sz), rel.decode()))
rows.sort(reverse=True)
files = [rel for _, rel in rows]
probe = files[:8]

def read_some(path):
    left = 32 << 20
    with open(path, "rb", buffering=0) as fh:
        while left:
            got = fh.read(min(1 << 20, left))
            if not got:
                break
            left -= len(got)

def bench(n):
    for i in range(1, n + 1):
        ensure_mount(i)
    sh(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
    threads = []
    for i in range(1, n + 1):
        for rel in probe:
            threads.append(threading.Thread(
                target=read_some, args=(f"/mnt/emsrc/{i}/EM/{rel}",)))
    before, t0 = nic_rx(), time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return (nic_rx() - before) / (time.perf_counter() - t0)

best_n, best_rate, prev = 1, 0.0, 0.0
for n in (1, 2, 4, 8, 16):
    rate = bench(n)
    print(f"sessions={n} Bps={rate:.0f}", flush=True)
    if rate > best_rate:
        best_n, best_rate = n, rate
    if rate >= 0.85 * link_bps or (prev and rate < prev * 1.10):
        break
    prev = rate
print(f"using_sessions={best_n}", flush=True)
for i in range(best_n + 1, 17):
    path = f"/mnt/emsrc/{i}"
    if mounted(path):
        sh(["sudo", "umount", path])

lists = [[] for _ in range(best_n)]
for i, rel in enumerate(files):
    lists[i % best_n].append(rel)
procs = []
for i, rels in enumerate(lists, 1):
    payload = b"".join(f"/mnt/emsrc/{i}/EM/./{rel}\0".encode() for rel in rels)
    proc = subprocess.Popen(
        ["xargs", "-0", "-P", "16", "-I", "{}",
         "rsync", "-a", "--partial", "--relative", "--", "{}", dest + "/"],
        stdin=subprocess.PIPE)
    proc.stdin.write(payload)
    proc.stdin.close()
    procs.append(proc)
rc = 0
for proc in procs:
    rc = proc.wait() or rc
raise SystemExit(rc)
PY
rm -f "$cred"
find /tmp /root /home /localhome -name '*cifs*' -o -name '.smbcredentials*' 2>/dev/null | head
ln -sfn "$HOME/data" "$HOME/nanoCosmos/data"
ln -sfn "$HOME/outputs" "$HOME/nanoCosmos/outputs"
```

Confirm the credentials file is gone. Training writes to
`$HOME/outputs` (a timestamped run directory). Leave `$HOME/data`,
`/scratch`, and the `/mnt/emsrc/*` mounts in place.

## 6. tmux

`tmux` is `/usr/bin/tmux`. Name the sessions `0` and `1`. Target them
as `0:` and `1:` (`-t 0` is window 0 of whichever session is current).

Session `1` watches `$HOME/outputs`. The event files live under
`$HOME/outputs/<timestamp>_<experiment>/logs/...`, so the logdir is
`$HOME/outputs`, not `$HOME/outputs/logs`. "TensorFlow installation
not found" from TensorBoard is harmless.

```bash
tmux has-session -t 0 2>/dev/null || tmux new-session -d -s 0 -c "$HOME/nanoCosmos"
tmux has-session -t 1 2>/dev/null || tmux new-session -d -s 1 -c "$HOME"
tmux send-keys -t 1: 'source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate nanocosmos && tensorboard --logdir "$HOME/outputs" --host 0.0.0.0 --port 6006' C-m
tmux ls
```

From the laptop, `ssh -L 6006:127.0.0.1:6006 local-tranminhq@<short-name>`
and open `http://127.0.0.1:6006/`.

## 7. Train

Run this yourself in session `0` when data rsync has exited 0. Do not
`source /etc/profile.d/cuda.sh` first. `scripts/train.py` re-execs once
so the wheel cuDNN wins over `/usr/local/cuda`; a line
`Restarting so cuDNN loads from .../nvidia/cudnn/lib` is expected.

```bash
tmux send-keys -t 0: 'source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate nanocosmos && cd "$HOME/nanoCosmos" && python scripts/train.py --config-name nanocosmos-4B-pretrain' C-m
```

The recipe is `configs/nanocosmos-4B-pretrain.yaml`: 8 GPUs, bf16,
`gradient_checkpointing: [dit, decode]`, 500 steps per epoch.
`diffusion_steps: 1` is only the `pred/Euler` preview. The first launch
also downloads `nvidia/Cosmos3-Edge` into the Hugging Face cache.
