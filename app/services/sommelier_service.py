from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from threading import Lock

from app.sommelier.catalog import build as build_catalog
from app.sommelier.sommelier import Sommelier

logger = logging.getLogger(__name__)


class SommelierService:
    """Держит общий иммутабельный каталог вин и по одной диалоговой сессии `Sommelier` на клиента.

    Каталог собирается из `csv_path` один раз при старте (доли секунды на ~2000 строк) и не пишется на диск —
    app/ смонтирован read-only в проде. Сессии живут в памяти процесса с TTL и верхним лимитом (это демо-сервис
    без внешнего состояния; при перезапуске API все диалоги начинаются заново).
    """

    def __init__(self, csv_path: Path, feedback_path: Path, max_sessions: int = 500, session_ttl_seconds: int = 3600):
        self.csv_path = csv_path
        self.feedback_path = feedback_path
        self.wines: list[dict] = []
        if csv_path.is_file():
            self.wines = build_catalog(csv_path=str(csv_path), out=None)
        else:
            logger.warning("sommelier catalog CSV not found at %s; sommelier endpoints will report unavailable", csv_path)
        self._max_sessions = max_sessions
        self._ttl = session_ttl_seconds
        self._sessions: dict[str, tuple[Sommelier, float]] = {}
        self._lock = Lock()

    @property
    def available(self) -> bool:
        return bool(self.wines)

    def _evict_locked(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, (_, seen) in self._sessions.items() if now - seen > self._ttl]
        for sid in expired:
            del self._sessions[sid]
        overflow = len(self._sessions) - self._max_sessions
        if overflow > 0:
            oldest = sorted(self._sessions.items(), key=lambda kv: kv[1][1])[:overflow]
            for sid, _ in oldest:
                del self._sessions[sid]

    def _new_sommelier(self) -> Sommelier:
        return Sommelier(wines=self.wines, feedback_log=str(self.feedback_path))

    def session(self, session_id: str | None) -> tuple[str, Sommelier]:
        """Возвращает (session_id, Sommelier) для диалога: существующая сессия или новая с указанным/новым id."""
        with self._lock:
            self._evict_locked()
            if session_id and session_id in self._sessions:
                sommelier, _ = self._sessions[session_id]
                self._sessions[session_id] = (sommelier, time.monotonic())
                return session_id, sommelier
            new_id = session_id or uuid.uuid4().hex
            sommelier = self._new_sommelier()
            self._sessions[new_id] = (sommelier, time.monotonic())
            self._evict_locked()  # применяем лимит и после вставки — иначе переполнение на 1 сессию не отловить
            return new_id, sommelier

    def reset(self, session_id: str) -> bool:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return False
            entry[0].reset()
            self._sessions[session_id] = (entry[0], time.monotonic())
            return True

    def wine_lookup(self) -> Sommelier:
        """Разовый (без сохранения сессии) доступ к движку — для справки по вину, распознанному на этикетке."""
        return self._new_sommelier()
