#!/usr/bin/env python3
"""Консольный диалог с AI-сомелье в обход HTTP API (для ручной проверки движка).

    python scripts/sommelier_cli.py
    python scripts/sommelier_cli.py "стейк рибай, не люблю дуб"

Требует установленный пакет проекта (`pip install -e .`) — движку из `app.sommelier` нужен только `pandas`.
HTTP-точка входа для интеграции с остальным пайплайном — `/api/sommelier/*`, см. docs/SOMMELIER.md.
"""
import sys

from app.sommelier import Sommelier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    s = Sommelier()
    if len(sys.argv) > 1:
        print(s.ask(" ".join(sys.argv[1:]))["text"])
        return
    print(f"AI-сомелье (каталог: {len(s.wines)} вин). Пустая строка — выход, /new — новый разговор.")
    while True:
        try:
            q = input("\nВы: ").strip()
        except EOFError:
            break
        if not q:
            break
        if q == "/new":
            s.reset()
            print("Начали заново.")
            continue
        print("\nСомелье:", s.ask(q)["text"])


if __name__ == "__main__":
    main()
