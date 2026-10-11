"""Verified HTTPS on a network whose interception certificate is malformed but trusted.

The problem this solves, precisely. Some institutional networks terminate TLS with their own CA,
installed in the machine's trust store by an administrator. Python 3.13 enables
`ssl.VERIFY_X509_STRICT` by default, which enforces RFC 5280 formatting on every certificate in the
chain, and some of these CAs do not mark `basicConstraints` critical as the RFC requires. The chain
builds, the signatures verify, the hostname matches, and the handshake is then refused over a
formatting rule.

What this module does, and what it deliberately does not do. It builds a trust store from certifi
plus the certificates the operating system already trusts, and clears one verification flag:
`VERIFY_X509_STRICT`. Everything else stays on. Measured against `www.ebi.ac.uk` on the network this
was written for:

    certifi only,  strict ON   FAIL  unable to get local issuer certificate
    OS store,      strict ON   FAIL  Basic Constraints of CA cert not marked critical
    OS store,      strict OFF  OK    peer verified
    certifi only,  strict OFF  FAIL  unable to get local issuer certificate

The fourth line is the one that matters. With strictness cleared but the OS store absent, an
unknown certificate is still refused. Chain building, signature verification, validity dates and
hostname checking all remain in force, so this is not `verify=False`: an attacker still needs a key
chaining to a CA an administrator installed on this machine. `verify=False` would need none of that,
and would let anyone on the path substitute a molecule and have the tool report on it confidently.
That is recorded as BS-M-25 in the audit and is not what happens here.

It is opt-in, because silently relaxing a verification default is how these things become invisible.
Set SAI_CNS_ALLOW_NONSTRICT_TLS=1 to enable it; without that the session verifies normally and
fails on such a network, which is the correct default.

Run:  python src/sai_cns/data/_tls.py     (reports what the current network allows)
"""
from __future__ import annotations

import os
import ssl

import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

ENV_FLAG = "SAI_CNS_ALLOW_NONSTRICT_TLS"
# The former name, still honoured so that a shell or CI job written before the project was
# renamed does not quietly lose the opt-in and fail on the network it was configured for.
ENV_FLAG_LEGACY = "BRAINSAFE_ALLOW_NONSTRICT_TLS"


def os_trust_context(allow_nonstrict: bool) -> ssl.SSLContext:
    """A context trusting certifi plus the operating system's own roots."""
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    loaded = 0
    for store in ("ROOT", "CA"):
        try:
            certs = ssl.enum_certificates(store)
        except (AttributeError, OSError):
            continue          # not Windows, or the store is unavailable
        for der, encoding, _trust in certs:
            if encoding != "x509_asn":
                continue
            try:
                ctx.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(der))
                loaded += 1
            except ssl.SSLError:
                pass          # a malformed entry in the store is skipped, not fatal
    if allow_nonstrict:
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    ctx._brainsafe_os_certs = loaded  # noqa: SLF001  (recorded for the run log)
    return ctx


class _ContextAdapter(HTTPAdapter):
    def __init__(self, context: ssl.SSLContext, **kw):
        self._ctx = context
        super().__init__(**kw)

    def init_poolmanager(self, connections, maxsize, block=False, **kw):
        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize,
                                       block=block, ssl_context=self._ctx, **kw)


def session() -> requests.Session:
    """A requests session that verifies, using the OS trust store where one exists.

    Announces itself when the strictness flag has been cleared, so a run log records that the
    network required it rather than leaving it to be discovered.
    """
    allow = os.environ.get(ENV_FLAG, os.environ.get(ENV_FLAG_LEGACY, "")) == "1"
    ctx = os_trust_context(allow)
    s = requests.Session()
    s.mount("https://", _ContextAdapter(ctx))
    if allow:
        print(f"[tls] verifying against certifi plus {ctx._brainsafe_os_certs} operating-system "  # noqa: SLF001
              f"certificates, with VERIFY_X509_STRICT cleared. Chain, signature, validity and "
              f"hostname checks all remain in force; this is not verify=False.", flush=True)
    return s


if __name__ == "__main__":
    url = "https://www.ebi.ac.uk/chembl/api/data/status.json"
    for allow in (False, True):
        os.environ[ENV_FLAG] = "1" if allow else "0"
        label = "strict cleared" if allow else "default (strict)"
        try:
            r = session().get(url, timeout=30)
            print(f"  {label:16s} OK {r.status_code} {r.json().get('chembl_db_version')}")
        except Exception as exc:
            print(f"  {label:16s} FAIL {type(exc).__name__}: {str(exc)[:80]}")
