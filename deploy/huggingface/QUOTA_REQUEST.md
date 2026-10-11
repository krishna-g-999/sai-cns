# Request to Hugging Face for CPU-basic quota

The Space `Krishnag999/sai-cns` is complete, public, and holds its own models in git LFS, but it
cannot start. The API reports the reason exactly:

```
GET  https://huggingface.co/api/spaces/Krishnag999/sai-cns
  "stage": "PAUSED",
  "errorMessage": "Quota exceeded for flavor cpu-basic (requested=1): current=0, limit=0"
```

`limit=0` with `current=0` means the account is allowed zero free CPU-basic Spaces and is running
none. Restart, Resume and Factory rebuild were all tried and all return HTTP 403 with the same
message, so this is not the stale-container desync reported on the forum, and no action inside the
Space can clear it.

Two routes are free. Send the email below, and separately apply through the "community GPU grant"
button on the Space settings page, which is the Hub's own channel for research projects. They are
independent and can be pursued at the same time.

---

## Email

**To:** website@huggingface.co
**Subject:** Request for CPU-basic quota, academic Space Krishnag999/sai-cns

Dear Hugging Face team,

I am a PhD student at the Sri Sathya Sai Institute of Higher Learning in Puttaparthi, India. I
maintain the Space `Krishnag999/sai-cns`, which hosts SAI-CNS, an open-source web server that
predicts blood-brain-barrier penetration and molecular target engagement for small molecules. The
code is MIT licensed and the Space is public and requires no registration to use.

The Space is paused and cannot be restarted. The API reports:

    Quota exceeded for flavor cpu-basic (requested=1): current=0, limit=0

This is my only Space, and `current=0` confirms I am running no others, so there is nothing for me
to pause. Restart, Resume and Factory rebuild each return HTTP 403 with the same message.

Could you grant this account CPU-basic quota so the Space can run? The application needs 2.35 GB of
resident memory, which fits CPU basic comfortably, and it holds its 150 model files in the Space's
own git LFS, so starting it requires no external download.

The reason for asking rather than simply upgrading is that this server is being submitted to the
Nucleic Acids Research Web Server Issue, which requires a publicly accessible URL with no login,
maintained for at least five years. A stable free academic Space would let us cite the Hub as the
server's permanent home. I am happy to add any acknowledgement you would like on the landing page.

Account: Krishnag999
Space: https://huggingface.co/spaces/Krishnag999/sai-cns
Licence: MIT
Institution: Sri Sathya Sai Institute of Higher Learning, Puttaparthi, India

Thank you for considering this.

Krishnasalini Gunanathan
krishnasalinig@sssihl.edu.in

---

## If both free routes fail

The paid fallbacks, measured against the server's actual 2.35 GB footprint:

| option | cost | note |
|---|---|---|
| HF PRO | $9/month | quota becomes non-zero, and unlocks a custom domain. Everything else is already in place, so this is live within minutes. |
| Cloud Run, 4 GB, scale to zero | free within the always-free tier | about 25 hours of serving per month free, but a cold start reloads 2.35 GB of estimators and takes roughly a minute, which a reviewer may read as a broken link. |
| Cloud Run or Cloudflare Containers, kept warm | $25 to $31/month | no cold starts, but the most expensive option by a wide margin. |

HF PRO is the cheapest reliable route, because 2.35 GB of always-on memory is not something any
other provider gives away.
