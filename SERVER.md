# Server setup

Bring up a fresh Colossus lease and leave two tmux sessions ready.
Session `0` is training. Session `1` is TensorBoard. The train command
is started by hand in session `0` after this setup finishes.

The lease host changes every reprovision. Read it from the lease; do
not keep using a previous `umb-b300-dp-*` name. Home on this image is
`/localhome/local-tranminhq`. Nothing in this file is a password.

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

The lockfile is a cu134 nightly for Python 3.13
(`torch==2.16.0.dev20261027+cu134`). The stable
`https://download.pytorch.org/whl/cu134` index has no cp313 wheel; do
not install torch from that URL. `pip install -e . --no-deps` is
required: without it `python scripts/train.py` raises
`ModuleNotFoundError: No module named 'nanocosmos'`.

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n nanocosmos python=3.13 -y
conda activate nanocosmos
pip install uv
cd "$HOME/nanoCosmos"
uv pip install -r requirements.txt --prerelease=allow --index-strategy unsafe-best-match
pip install -e . --no-deps
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

Expect `2.16.0.dev20261027+cu134 True 8`.

## 4. Scratch

The CIFS username is `tranminhq`. The share password is typed into a
mode-0600 file and the file is deleted after the mount. It is not
stored here. `uid`/`gid` are the name `local-tranminhq` (the numeric
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
sudo mkdir -p /scratch
sudo mount -t cifs \
  -o credentials="$cred",uid=local-tranminhq,gid=local-tranminhq,vers=3.1.1,file_mode=0644,dir_mode=0755 \
  //ipp1-cdot01-col01.nvidia.com/scratch.tranminhq_colossus /scratch
rm -f "$cred"
find /tmp /root /home /localhome -name '*cifs*' -o -name '.smbcredentials*' 2>/dev/null | head
```

Confirm the credentials file is gone, then:

```bash
df -h /scratch
ls /scratch/EM
```

## 5. Data and outputs

About 310G. `rsync` can be re-run; it skips files that already match.

```bash
mkdir -p "$HOME/data" "$HOME/outputs"
rsync -a --partial /scratch/EM/ "$HOME/data/"
ln -sfn "$HOME/data" "$HOME/nanoCosmos/data"
ln -sfn "$HOME/outputs" "$HOME/nanoCosmos/outputs"
```

Training writes to `$HOME/outputs` (a timestamped run directory). Leave
`$HOME/data` and `/scratch` in place.

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
