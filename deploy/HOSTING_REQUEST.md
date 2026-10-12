# Request for institutional hosting of SAI-CNS

Draft email to SSSIHL, for the guide to review and forward. Everything stated below is measured
or quoted, not estimated. Sources are given at the end so any figure can be checked.

---

**Subject:** Institutional hosting for SAI-CNS, a web server being submitted to Nucleic Acids Research

Dear Sir / Madam,

I am writing to request institutional hosting for **SAI-CNS**, a computational tool developed in
our group that we are preparing to submit to the *Nucleic Acids Research* (NAR) Web Server Issue.
The request is specific and modest, and I set out below exactly what the journal requires, what we
have already built and verified, and what we need from the Institute.

## What SAI-CNS does

SAI-CNS predicts, from a chemical structure, whether a small molecule reaches the human brain, what
molecular targets it engages there, and which neurological conditions those mechanisms touch. It
covers 54 molecular targets, nine ADME and exposure endpoints, blood-brain-barrier penetration and
a cardiac safety endpoint. It is intended for researchers prioritising compounds in early CNS drug
discovery. It is open source under the MIT licence, and developed under the SAI-Net initiative at
SSSIHL.

## Why hosting is a journal requirement, not a convenience

This is the part I would particularly ask you to note. For this class of publication, a working
public server is not supplementary material: **it is the object being peer reviewed.** The journal's
own instructions require that the server:

- is **functional at the time of submission**, and reviewers will use it;
- is reached **through a web browser over HTTPS on port 443**;
- requires **no login, no registration and no email address** from users;
- provides **sample data** so a reviewer can try it immediately;
- displays a **standard licence** on its landing page;
- is **maintained for at least five years**.

The five-year maintenance expectation is the reason a personal or student account is not an
appropriate home. The commitment is institutional in nature, and reviewers read it that way.

This is also how comparable tools are hosted. KEGG runs on Kyoto University's Institute for
Chemical Research supercomputer facility. SwissTargetPrediction is operated by the University of
Lausanne together with the Swiss Institute of Bioinformatics, whose stated position is that such
resources "need to be hosted on dedicated computational infrastructures and supported by
professional staff". MetaboAnalyst runs on dedicated cloud infrastructure with additional compute
nodes at McGill University. None of these runs on a free or personal service.

## What we have already completed

The software is finished and independently verified. Nothing below is pending.

| | |
|---|---|
| Server | built, running, and tested end to end on the institute HPC |
| Correctness | 3,476 output values compared against the validated reference: **largest difference 0** |
| Accuracy | **AUROC 0.8011** on 247 compounds absent from every training set, against 0.8010 published |
| Test suite | 92 tests and 79 subtests passing; 252 of 252 model checksums verified |
| Source code | public and MIT licensed at github.com/krishna-g-999/sai-cns |
| Trained models | published as a verifiable archive with per-file checksums |
| Manuscript | drafted, with the one-page proposal the journal requires |

We have also already satisfied most of the journal's server requirements: no login, sample data on
the landing page, and the licence displayed. **The only requirement we cannot meet ourselves is a
public HTTPS address.**

## What we are asking for

The resource requirement is small. Measured, not estimated:

| | |
|---|---|
| Memory | **2.5 GB** resident while serving |
| CPU | 2 cores is sufficient; it is not a parallel workload |
| Disk | about 5 GB, including the trained models |
| Traffic | low; this is a research tool, not a consumer service |

Concretely, we would ask the Institute for either:

1. **A small virtual machine that stays powered on**, with 4 GB of RAM and 2 cores, together with a
   public hostname under the Institute domain and a TLS certificate, for example
   `saicns.sssihl.edu.in`; **or**
2. if a public address cannot be allocated, **permission to run an outbound-only secure tunnel**
   from an internal machine. This requires no public IP, no inbound firewall change and no open
   port, as the connection is established from inside the network outward.

We are currently running the server on an HPC node for testing only. I want to be explicit that we
do not regard that as an appropriate long-term home: it is a shared login node, a permanent web
service there would compete with other users' work, and it is not reachable from outside the
Institute network in any case.

## What the Institute gains

The server would carry an Institute address and would be cited by that address in a journal whose
Web Server Issue is among the most widely used resources in computational biology. The maintenance
commitment would rest with the Institute rather than with an individual student account, which is
both what the journal expects and what makes the resource durable after I complete my degree.

I would be glad to present the tool, to answer any technical questions from the IT team, and to
work within whatever security arrangements they consider appropriate.

Thank you for your consideration.

Yours sincerely,

Krishnasalini Gunanathan
Research Scholar, SSSIHL
krishnasalinig@sssihl.edu.in

---

## Notes for the guide, not for sending

**Where the journal requirements come from.** Oxford Academic, "Submitting to the Web Server Issue"
(academic.oup.com/nar/pages/Submission_Webserver), and the 2026 Web Server Issue editorial
(doi.org/10.1093/nar/gkag669). Two points changed recently and are worth knowing:

- Proposals are no longer tied to a December deadline. They can now be submitted **at any time**,
  and after approval the manuscript is due **two months later**. There is no acceptance deadline.
- A **one-page proposal must be approved before a manuscript can be submitted.** It is emailed to
  the editors, not uploaded to a portal. Ours is written and builds to exactly one page.

**Why the timing matters for this request.** The server must be live and public *when the proposal
is sent*, because the editors check it. Hosting is therefore the one blocker between us and
starting the submission process, not something that can follow acceptance.

**What it would cost commercially, for comparison.** A 4 GB always-on container is roughly $25 to
$35 per month on Google Cloud Run or Microsoft Azure, so about $1,500 to $2,100 over the five-year
maintenance period. Free tiers were tested and do not fit: Cloudflare Workers caps an instance at
128 MB, Streamlit Community Cloud at 2.7 GB which is below what loading completes with, and Hugging
Face's free allowance for this account is now zero. An Azure for Students subscription was obtained
and tried; its $100 credit would cover roughly three months, and it blocks the cloud build service
the deployment needs.

**Fallback if the Institute declines.** The tool can be hosted commercially for the review period
and migrated later, but the five-year commitment would then sit with a personal account, which
weakens the submission. It is worth making the institutional case first.
