from unittest.mock import patch

import rag.vectorstore


def _client_headers(client_id, client_secret, token):
    with patch.object(rag.vectorstore, "QDRANT_CF_ACCESS_CLIENT_ID", client_id), \
         patch.object(rag.vectorstore, "QDRANT_CF_ACCESS_CLIENT_SECRET", client_secret), \
         patch.object(rag.vectorstore, "QDRANT_CF_ACCESS_TOKEN", token), \
         patch.object(rag.vectorstore, "QdrantClient") as fake_client:
        rag.vectorstore.get_client()
    return fake_client.call_args.kwargs["headers"]


def test_no_headers_without_cf_access_env():
    """Sem QDRANT_CF_ACCESS_* o QdrantClient vê headers=None, não um dict vazio."""
    assert _client_headers(None, None, None) is None


def test_service_token_headers():
    assert _client_headers("id123", "secret456", None) == {
        "CF-Access-Client-Id": "id123",
        "CF-Access-Client-Secret": "secret456",
    }


def test_access_token_header():
    """App com Managed OAuth (não aceita service token) usa Cf-Access-Token
    de `cloudflared access token -app=<url>` em vez disso."""
    assert _client_headers(None, None, "tok789") == {"Cf-Access-Token": "tok789"}


def test_half_configured_service_token_sends_nothing():
    """CLIENT_ID sem CLIENT_SECRET (ou o inverso) não manda header nenhum —
    meia credencial só viraria 403 do Cloudflare."""
    assert _client_headers("id123", None, None) is None
    assert _client_headers(None, "secret456", None) is None


def test_service_token_and_access_token_both_sent():
    assert _client_headers("id123", "secret456", "tok789") == {
        "CF-Access-Client-Id": "id123",
        "CF-Access-Client-Secret": "secret456",
        "Cf-Access-Token": "tok789",
    }
