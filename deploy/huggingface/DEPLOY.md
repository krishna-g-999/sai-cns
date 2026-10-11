# Deploying SAI-CNS to a free public URL

The constraint is the panel. It is roughly 0.85 GB of fitted estimators on disk after compression,
and a warmed process holds **2.57 GB resident**: 2.31 GB once every model is loaded, rising to 2.57 GB
after the applicability-domain reference and the read-across index are also in memory. That measured
figure, not the size on disk, decides the options, and it is the number to check against any host's
limit. Measure it again after any retrain with `psutil` rather than trusting this line.

| Host | Memory | Disk | Verdict |
|---|---|---|---|
| **Hugging Face Spaces** | 16 GB | 50 GB | works, with 6x headroom over the measured 2.57 GB, but **no longer free for this application**. See below |
| Google Cloud Run | configurable | image | works, scales to zero, but needs a billing account attached even to stay inside the free tier |
| Oracle Cloud Always Free | 24 GB | 200 GB | works, genuinely free with no time limit, but you administer the machine |
| Streamlit Community Cloud | 1 GB | small | **does not fit.** The panel exceeds the memory limit once loaded |
| Render / Fly.io free tiers | 256 MB - 1 GB | small | do not fit |

**Hugging Face changed its terms, and this was verified by hitting them rather than by reading the
marketing page.** Two things are now true that were not when this file was first written:

1. **The Streamlit SDK has been retired.** Not merely hidden in the creation form: the API rejects
   it outright with `Invalid option: expected one of "gradio"|"docker"|"static"`. The
   `huggingface_hub` client still lists `streamlit` in `SPACES_SDK_TYPES`, so the client validates a
   value the server refuses, and the error arrives only from the Hub.
2. **Only Static Spaces are free.** Creating a Docker Space returns `402 Payment Required`, with the
   server stating that "Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on
   free cpu-basic requires a PRO subscription". PRO is $9/month and lists "Host ZeroGPU, Gradio &
   Docker Spaces" among its benefits.

A Static Space serves HTML and CSS and cannot execute Python, so it cannot host this application at
any price. The route is therefore a **Docker Space on a PRO account**, with Streamlit running inside
the container, or one of the alternatives below. The application itself is unchanged; only the
wrapper differs.

Weigh one thing before subscribing. A journal server should stay reachable for years, and a Space on
a personal subscription stops when the subscription does. That is an argument for institutional
hosting or a self-administered virtual machine, not against PRO, but it should be a decision rather
than an accident.

## Before deploying: shrink the panel

Do this first, or the upload is three times larger than it needs to be.

```bash
python tools/compress_models.py            # report what would be saved, change nothing
python tools/compress_models.py --apply    # recompress in place, verifying each file
python src/sai_cns/models/package_models.py 1.1   # regenerate the manifest
```

joblib writes an uncompressed pickle unless asked otherwise. The binder models were saved with
`compress=3` and average 3 MB; the calibrated classifiers were saved without and average 117 MB.
Recompressing takes the panel from 2.05 GB to 0.85 GB, and the tool verifies every file it rewrites:
each is reloaded and scored against the original, and replaced only if predictions agree to 1e-12.
Anything that cannot be verified is left exactly as it was.

`models_rf/holdout/` is another 155 MB and is not needed to serve predictions. It holds the
scaffold-split twins used for validation. Exclude it.

## Creating the Space under your institute organisation

You have already joined the organisation, which is the part people usually get stuck on. What
follows assumes the Space belongs to the organisation rather than to you personally, so that it
survives you changing accounts and carries the institute's name in the URL.

**1. Check you can write to the organisation.** Open
`https://huggingface.co/organizations/<org>/settings/members` and confirm your role is `write` or
`admin`. With `read` you can see the organisation but the Owner dropdown in step 2 will not offer
it, and nothing later will work. If you only have `read`, an admin has to raise it.

**2. Create the Space, and not through the web form.** As of August 2026 the page at
https://huggingface.co/new-space **no longer offers Streamlit**. It states that you may "choose
between Gradio, Docker, or Static", and Docker is marked paid. The SDK a Space runs under is set by
the `sdk:` field of the README's YAML block, and `streamlit` remains valid there and in the API,
which still lists `['gradio', 'streamlit', 'docker', 'static']`. Only the creation form dropped it.

Create it through the client instead, which sets the SDK explicitly:

```bash
brainsafe_env/Scripts/hf.exe auth login          # paste a Write token; it is stored, not typed into git
brainsafe_env/Scripts/hf.exe repo create sai-cns --repo-type space --space_sdk docker
```

`--space_sdk streamlit` is no longer accepted by the Hub and `docker` requires PRO, as set out
above. With Docker the SDK line in the Space card becomes `sdk: docker`, and a `Dockerfile` at the
repository root installs the runtime requirements and launches
`streamlit run app.py --server.port 7860 --server.address 0.0.0.0`. Port 7860 is the port a Space
exposes.

Two further traps in the web form, if you use it for anything else. The Owner dropdown defaults to
your personal account, so creating under an organisation is a deliberate change and easy to miss.
And **hardware defaults to ZeroGPU**, which is a GPU tier with different constraints and is wrong
for this application: it needs CPU Basic, 2 vCPU and 16 GB, against a measured 2.57 GB resident.

The URL is then `https://huggingface.co/spaces/<owner>/sai-cns`. That is the address for the
manuscript.

**On whether to use an organisation at all.** A public Space is readable by the entire internet, so
organisation membership is not a confidentiality question; everything here is published on purpose.
It is a question of control. Every member of an organisation with the write or admin role can
modify or delete the Space that a published paper points at, and an audit log to establish who did
so requires a paid plan. Before choosing an organisation, read
`https://huggingface.co/organizations/<org>/settings/members` and check both the roles and the
default role assigned to new members. A personal namespace is a perfectly ordinary address for a
NAR web server, and a Space can be transferred to an organisation later, with the old URL
redirecting.

**3. Get a token that can write to the organisation.** A personal token is not automatically an
organisational one. At https://huggingface.co/settings/tokens create a token with **Write** access,
and if you choose a fine-grained token, tick the organisation and give it write permission on
repositories. Copy it; it is shown once.

**4. Clone the Space and fill it.**

```bash
git clone https://huggingface.co/spaces/<org>/sai-cns
python deploy/huggingface/prepare_space.py --out sai-cns
```

`prepare_space.py` copies only what answers a query: the app, `src/`, `assets/`, `results/`,
`docs/`, the four `data/` subdirectories the server reads, and `models_rf/` without `holdout/`. It
also writes the Space card and the git-LFS rules. That is 0.81 GB. Copying the repository instead
would be 1.36 GB, most of it raw pulls and API caches that are never opened to serve a prediction.

It ships `deploy/huggingface/requirements.txt`, not the repository's. The root file is the
environment that trains and validates, and carries matplotlib, python-docx, `pypandoc_binary` and
xgboost, none of which is imported to answer a query; `pypandoc_binary` alone pulls a pandoc binary
of over a hundred megabytes. The runtime set is nine packages, pinned to the versions the estimators
were fitted under. streamlit is deliberately not among them, because the Space card's `sdk_version`
installs it and naming it twice invites a conflict. Keep `sdk_version` in the card equal to the
streamlit version this project validates against; it is 1.58.0 today.

**Verify the assembled directory before pushing.**

```bash
python deploy/huggingface/verify_space.py sai-cns
```

This runs the app with the Space as the working directory and the repository's source roots removed
from the path, loads every model, and predicts for three compounds chosen to exercise different
paths. It exists because a file that failed to travel is invisible in a directory listing and
obvious to the first visitor. It also checks that `results/tables/external_novelty_strata.csv`
arrived, since the interface quotes an expected recall from it and would drop that row without
comment if it were missing.

**5. Track LFS before adding anything.** This ordering matters more than anything else here, and the
rule is stricter than the usual one about large files. The Hub rejects a push containing **any**
binary that is not in LFS, whatever its size: `Your push was rejected because it contains binary
files`. A 104 kB logo is refused on the same rule as a 58 MB forest, so a `.gitattributes` naming
only the model formats is not enough. `prepare_space.py` writes patterns for every binary extension
the Space can contain.

It also writes a `.gitignore` for `__pycache__`, which matters more than it sounds. Running
`verify_space.py` imports the application from inside the assembled directory and leaves compiled
bytecode behind, so a `.pyc` that was absent at assembly time is present at commit time and fails
the push. Once a binary is in a commit, adding LFS afterwards does not fix that commit; the commit
has to be redone.

```bash
cd sai-cns
git lfs install
git add .gitattributes && git commit -m "Track model files with LFS"
git add -A && git commit -m "SAI-CNS"
git push
```

When prompted, the username is your Hugging Face username and the **password is the token** from
step 3, not your account password. Alternatively authenticate once and let git use the stored
credential, which avoids pasting the token into a terminal prompt where it may be logged:

```bash
brainsafe_env/Scripts/hf.exe auth login
```

The token is entered into the Hugging Face client, which stores it under `~/.cache/huggingface`.

The push moves 0.8 GB and takes a while. The first build then takes several minutes because the
scientific stack is large.

**6. Silence the model-fetch warning.** `model_fetch.py` compares what is on disk against
`models_manifest.json`, which lists the hold-out files that were deliberately not shipped, so the
log opens with "102 of 252 model files missing". The app runs correctly regardless. To keep the log
clean, add a Space variable under Settings, Variables and secrets:

    SAI_CNS_SKIP_MODEL_FETCH = 1

## If the login fails with CERTIFICATE_VERIFY_FAILED

On an institutional network this is the first thing that happens, and the error names the wrong
culprit. It is not a broken `certifi`, and the fix is not to disable verification.

The cause is TLS interception. A filtering appliance terminates the connection, inspects it, and
re-signs it with its own certificate authority. Confirm it by reading the certificate actually
presented rather than guessing:

```powershell
$tcp = New-Object Net.Sockets.TcpClient('huggingface.co', 443)
$ssl = New-Object Net.Security.SslStream($tcp.GetStream(), $false, {$true})
$ssl.AuthenticateAsClient('huggingface.co')
$ssl.RemoteCertificate.Issuer
```

An issuer naming the site itself is a genuine certificate. An issuer naming a security vendor is
interception. On this network it returns
`CN=Sophos SSL CA_C320ABPCBJ8BQ6B, O=Sophos`.

The browser accepts it because Windows trusts that CA; Python does not, because it uses `certifi`'s
bundle rather than the Windows store. So give Python a bundle containing both. Export the
interception roots from the Windows store and concatenate them onto `certifi`'s:

```powershell
Get-ChildItem Cert:\LocalMachine\Root | Where-Object { $_.Subject -like "*Sophos*" } |
  ForEach-Object {
    "-----BEGIN CERTIFICATE-----"
    [Convert]::ToBase64String($_.RawData, 'InsertLineBreaks')
    "-----END CERTIFICATE-----"
  } | Set-Content "$HOME\.certs\sophos-roots.pem" -Encoding ascii
```

```python
import certifi, pathlib
d = pathlib.Path.home() / ".certs"
(d / "ca-bundle.pem").write_text(
    pathlib.Path(certifi.where()).read_text() + (d / "sophos-roots.pem").read_text())
```

Then point the tools at it. Both are needed: the Hub client uses `httpx`, and git-LFS does its own
TLS and ignores the Python variables entirely, so a push fails the same way after the login
succeeds.

```powershell
$env:SSL_CERT_FILE = "$HOME\.certs\ca-bundle.pem"
$env:REQUESTS_CA_BUNDLE = "$HOME\.certs\ca-bundle.pem"
git config --global http.sslCAInfo "$HOME\.certs\ca-bundle.pem"
```

Verify before spending a token on it. A 401 is the correct answer here: it means TLS succeeded and
the request arrived unauthenticated.

```python
import httpx; print(httpx.get("https://huggingface.co/api/whoami-v2").status_code)
```

**What this means for the token, which is the part worth pausing on.** Interception is not a
transport detail. The appliance holds the plaintext of every request, so a token pasted into
`hf auth login` is readable by whoever administers it, as is anything else sent over that network.
That is normal on a managed network and is not a reason to stop, but it is a reason to scope the
token to writing your own repositories and nothing else, and to revoke it at
https://huggingface.co/settings/tokens once the Space is pushed. A token that can write to an entire
organisation should never cross a connection someone else can read.

## Constraints the Space card must satisfy

The card's YAML block is validated by the Hub on push, not on save, so a mistake here costs a
rejected upload rather than an error in an editor. Two rules cost a push each here:

- `short_description` must be **60 characters or fewer**. The natural one-line summary of this
  project is 81, and it is rejected with `"short_description" length must be less than or equal to
  60 characters long`.
- `sdk` must be one of `gradio`, `docker` or `static`, and Docker additionally needs `app_port`.

Check the length before pushing, rather than after uploading most of a gigabyte:

```python
import re, pathlib
y = pathlib.Path("README.md").read_text().split("---")[1]
d = re.search(r"short_description:\s*(.+)", y).group(1).strip()
print(len(d), d)
```

## Two things to check after it is live

**The models load.** The first prediction is slow because the panel is read into memory once; every
prediction after it is fast. If the Space restarts on the first query, the memory limit was hit and
the panel needs trimming further, though at 2.57 GB measured against 16 GB available that should not
happen.

**The expected-recall row appears.** Submit a compound far from the training chemistry, a steroidal
natural product will do, and confirm the result carries both an applicability-domain distance and
the measured recall at that distance. If the recall row is missing, the novelty-strata table did not
travel and `verify_space.py` was skipped.

**The URL goes in the manuscript.** NAR requires the server address in the abstract. It is currently
`[SERVER URL TO BE SUPPLIED]` in `manuscript/NAR_WebServer_SAI-CNS_draft.md`, and the manuscript
must be rebuilt after it is filled in.

## The alternative, if the Space memory is ever a problem

`model_fetch.py` already downloads the panel from a URL recorded in `models_manifest.json` and
verifies it against the committed SHA-256 of the archive and of every file inside it. Publishing the
archive as a Hugging Face model repository and recording that URL keeps the Space small and the
models versioned separately, which is also what the manuscript's data-availability statement
describes. That is the tidier arrangement long term; putting the models in the Space is the faster
one to get a working URL today.
