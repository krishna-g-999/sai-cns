# Request for institutional hosting of SAI-CNS

For the supervisors to review and forward to the Institute's IT department, alongside the manuscript
and the one-page NAR proposal. Every figure below was measured on a running deployment on
26 October 2026, not estimated. Section 7 gives the sources.

---

**Subject:** Institutional hosting and a subdomain for SAI-CNS, a web server being submitted to
Nucleic Acids Research

Dear Sir / Madam,

We write to request a hosted address under the Institute domain for **SAI-CNS**, a computational
research tool developed in our group and being prepared for submission to the *Nucleic Acids
Research* Web Server Issue. The request is small and specific. Set out below is what the journal
requires, what we have already built and verified, and the exact provisioning we are asking for.

## 1. What SAI-CNS does

Given a chemical structure, SAI-CNS predicts which of 54 protein targets the molecule is likely to
engage, whether it reaches brain tissue, and what the engaged proteins imply mechanistically. It
covers 54 molecular targets, nine ADME and exposure endpoints, blood-brain barrier penetration and a
cardiac safety endpoint. It is intended for researchers prioritising compounds in early discovery,
is open source under the MIT licence, and is developed under the SAI-Net initiative at SSSIHL.

**Team.** Krishnasalini Gunanathan (Biosciences, research scholar); Prof. R. Raghunatha Sarma and
Dr. Sai Shyam (Mathematics and Computer Science); Dr. Ramya E. M. and Prof. S. Venketesh, Head of
Department (Biosciences).

## 2. Why hosting is a journal requirement rather than a convenience

For this class of publication the working public server is not supplementary material. **It is the
object being peer reviewed.** The journal's instructions require that the server:

- **"be functional on the date of the proposal submission"**, and reviewers test it themselves,
  *"preferably on at least two commonly used web browsers"*;
- **"use the HTTPS protocol and run on the http standard port (443)"**;
- not **"require users to log in, to register, or to leave an email address"**;
- provide sample data loadable in one click, and help pages explaining how to read the output;
- display a standard licence on the landing page;
- be **"maintained for at least five years"**.

That last expectation is why a personal or student account is not a suitable long-term home, and it
is the main reason for this request. Comparable resources are institutionally hosted: KEGG runs at
Kyoto University's Institute for Chemical Research, and SwissTargetPrediction is operated by the
University of Lausanne with the Swiss Institute of Bioinformatics, whose stated position is that
such resources need dedicated infrastructure and professional staff.

## 3. What is already complete and verified

Nothing in this table is pending or projected.

| | |
| --- | --- |
| Server | built, containerised, and **running now** on a cloud instance, verified end to end |
| Deployment integrity | all **252 model files** checksum-verified at image build time |
| Test suite | **97 tests and 94 subtests passing** |
| Accuracy | mean **AUROC 0.917** and average precision 0.899 against compounds experimentally tested and found inactive at the same target; **21 of 47** target models at AUROC 0.95 or better |
| Independent validation | four hold-out regimes: random, scaffold-grouped, by publication date, and by curator |
| Compliance | no login, no cookies set, no third-party resources loaded, sample data in one click, MIT licence shown on the landing page |
| Source code | public at github.com/krishna-g-999/sai-cns |
| Trained models | published as a verifiable archive with per-file checksums |
| Manuscript | drafted, with the one-page proposal the journal requires |

**The only journal requirement we cannot satisfy ourselves is a durable public HTTPS address under
an institutional domain.**

## 4. What we are asking for

The resource requirement is modest, and these are measurements from the running service rather than
estimates.

| | |
| --- | --- |
| Memory | **2.4 GB resident** while serving; 4 GB allocated is comfortable |
| CPU | **2 cores**, sufficient; this is not a parallel workload |
| Disk | about **8 GB**, including the container image and trained models |
| Throughput | 300 structures profiled across the full panel in **59.5 seconds** on 2 cores |
| Traffic | low. A research tool, not a consumer service |
| Inbound ports | **443 only**, or none at all under option B below |

We would be grateful for either of the following.

**Option A, a hosted virtual machine.** 2 cores, 4 GB RAM, about 8 GB disk, kept powered on, with a
public hostname under the Institute domain such as **`saicns.sssihl.edu.in`** and a TLS certificate.
The application ships as a single container, so deployment is one command and the IT team need
install nothing else.

**Option B, a subdomain only, with the service running on hardware we already have.** If allocating
a public address or a virtual machine is difficult, a DNS record for `saicns.sssihl.edu.in` pointing
to a secure outbound tunnel would be sufficient. This needs **no public IP, no inbound firewall
change and no open port**, because the connection is established from inside the network outward.
This is the lighter option for the Institute and we are glad to take it.

Option B alone would unblock the submission.

## 5. What the Institute gains

The server would carry an Institute address and be cited by that address in a journal whose Web
Server Issue is among the most used resources in computational biology, and which is indexed in
PubMed on acceptance. The maintenance commitment would rest with the Institute rather than with an
individual account, which is both what the journal expects and what makes the resource outlive any
one student's enrolment.

We are glad to demonstrate the tool, to answer any technical questions from the IT team, and to work
within whatever security arrangements they consider appropriate.

Thank you for your consideration.

Yours sincerely,

Krishnasalini Gunanathan\
Research Scholar, Department of Biosciences, SSSIHL\
krishnasalinig@sssihl.edu.in

Endorsed by Prof. S. Venketesh, Head, Department of Biosciences\
svenketesh@sssihl.edu.in, +91 8555 287235

---

## 6. Notes for the supervisors, not for forwarding

**The interim position.** The server is live now on a cloud instance paid from a student credit of
about 35 US dollars a month, which gives roughly three months of runway. That is enough to carry the
proposal and the first review round, and it is not a five-year answer. A free permanent address is
also available from the cloud provider and needs one firewall rule, so the submission is not blocked
on the Institute; the Institute address is what makes the five-year commitment credible.

**Why not simply buy a domain.** A commercial domain costs about ten US dollars a year and would
work, but a server cited in a journal under a personal domain carries the same durability problem as
a personal account. An Institute subdomain is what a reviewer reads as a maintained resource.

**What this costs commercially, for comparison.** A 4 GB always-on container is roughly 25 to 35 US
dollars a month on major cloud providers, so about 1,500 to 2,100 dollars over the five-year
maintenance period. Free tiers were tested and do not fit: Cloudflare Workers caps an instance far
below what is needed, Streamlit Community Cloud at 2.7 GB is below what loading completes with, and
this account's Hugging Face allowance is currently zero.

## 7. Sources

Journal requirements quoted from Oxford Academic, "Submitting to the Web Server Issue",
academic.oup.com/nar/pages/Submission_Webserver, read 26 October 2026, and the 2026 Web Server Issue
editorial, doi.org/10.1093/nar/gkag669. Proposals are emailed to the editors rather than uploaded,
can be sent at any time, and the manuscript is due two months after approval.
