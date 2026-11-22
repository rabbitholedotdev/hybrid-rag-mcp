import os
import importlib

os.environ.setdefault("MCP_AUTH_TOKEN", "token-de-teste-16c")
os.environ.setdefault("MCP_ALLOWED_HOSTS", "localhost")
from starlette.testclient import TestClient

TOKEN = os.environ["MCP_AUTH_TOKEN"]
AUTH = {"Authorization": f"Bearer {TOKEN}"}
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}


def _app():
    mod = importlib.import_module("rag.mcp_server")
    return mod.build_app() if hasattr(mod, "build_app") else mod.app


def test_healthz_sem_token():
    assert TestClient(_app()).get("/healthz").status_code == 200


def test_landing_sem_token():
    r = TestClient(_app()).get("/")
    assert r.status_code == 200
    assert "rag-mcp" in r.text


def test_mcp_sem_authorization_401():
    r = TestClient(_app()).post("/mcp", json=INIT)
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"


def test_mcp_bearer_errado_401():
    r = TestClient(_app()).post(
        "/mcp", json=INIT, headers={"Authorization": "Bearer token-errado-16xx"}
    )
    assert r.status_code == 401


def test_mcp_token_na_query_nao_autentica():
    r = TestClient(_app()).post(f"/mcp?token={TOKEN}", json=INIT)
    assert r.status_code == 401


def test_mcp_token_no_caminho_nao_autentica():
    r = TestClient(_app()).post(f"/{TOKEN}/mcp", json=INIT)
    assert r.status_code in (401, 404)


def test_mcp_bearer_correto_passa_da_auth():
    # lifespan starts FastMCP's streamable-http task group
    with TestClient(_app()) as client:
        r = client.post("/mcp", json=INIT, headers=AUTH)
    assert r.status_code != 401
