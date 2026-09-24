"""TLS trust configuration.

Python verifies certificates against the ``certifi`` bundle, which contains only
public certificate authorities. Machines running TLS-inspecting software, such
as corporate proxies or antivirus HTTPS scanning, present certificates signed by
a locally installed root instead. That root is trusted by the operating system
but absent from ``certifi``, so every outbound HTTPS call fails to verify.

``truststore`` resolves this by verifying against the operating system's own
trust store, which already contains both the public authorities and any locally
installed roots. Verification stays fully enabled.
"""

import ssl

from orbital_recon.core.logging import get_logger

logger = get_logger(__name__)


def create_ssl_context() -> ssl.SSLContext:
    """Return an SSL context that honours the operating system trust store.

    Falls back to the default context when ``truststore`` is unavailable, which
    still works on machines without TLS interception.
    """
    try:
        import truststore
    except ImportError:
        logger.debug("truststore_unavailable", fallback="certifi")
        return ssl.create_default_context()

    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
