
from app.core.config import get_settings
from app.services.sommelier_service import SommelierService


def service(tmp_path, **kw):
    return SommelierService(get_settings().resolved_sommelier_csv_path, tmp_path / "feedback.jsonl", **kw)


def test_loads_catalog_from_csv(tmp_path):
    svc = service(tmp_path)
    assert svc.available
    assert len(svc.wines) > 0


def test_unavailable_when_csv_missing(tmp_path):
    svc = SommelierService(tmp_path / "missing.csv", tmp_path / "feedback.jsonl")
    assert not svc.available
    assert svc.wines == []


def test_session_reused_by_id(tmp_path):
    svc = service(tmp_path)
    sid, s1 = svc.session(None)
    sid2, s2 = svc.session(sid)
    assert sid == sid2
    assert s1 is s2


def test_unknown_session_id_starts_new_session(tmp_path):
    svc = service(tmp_path)
    sid, s1 = svc.session("client-chosen-id")
    assert sid == "client-chosen-id"
    _, s2 = svc.session("client-chosen-id")
    assert s1 is s2


def test_reset_clears_context_and_reports_unknown(tmp_path):
    svc = service(tmp_path)
    sid, sommelier = svc.session(None)
    sommelier.ask("стейк рибай")
    assert sommelier.ctx.dish is not None
    assert svc.reset(sid) is True
    assert sommelier.ctx.dish is None
    assert svc.reset("no-such-session") is False


def test_session_eviction_respects_max_sessions(tmp_path):
    svc = service(tmp_path, max_sessions=2, session_ttl_seconds=3600)
    ids = [svc.session(None)[0] for _ in range(3)]
    assert len(svc._sessions) == 2
    assert ids[0] not in svc._sessions  # oldest evicted first


def test_wine_lookup_is_independent_from_dialog_sessions(tmp_path):
    svc = service(tmp_path)
    _, dialog = svc.session(None)
    lookup = svc.wine_lookup()
    assert lookup is not dialog
    some_id = svc.wines[0]["id"]
    result = lookup.about_wine(some_id)
    assert result["kind"] == "wine_info"
    assert result["id"] == some_id
