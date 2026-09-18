#!/usr/bin/env python3
"""Clear database products, embeddings, import records, and uploaded media files."""

from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from sqlalchemy import text


async def clear_all(media_dir: Path) -> None:
    from app.core.config import get_settings
    from app.db.session import create_engine_and_session_factory

    settings = get_settings()
    engine, session_factory = create_engine_and_session_factory(settings.database_url)

    print("Cleaning up database tables...")
    async with session_factory() as session:
        await session.execute(text("TRUNCATE TABLE import_items CASCADE;"))
        await session.execute(text("TRUNCATE TABLE import_jobs CASCADE;"))
        await session.execute(text("TRUNCATE TABLE product_embeddings CASCADE;"))
        await session.execute(text("TRUNCATE TABLE products CASCADE;"))
        await session.commit()
    print("Database tables truncated successfully.")

    products_media = media_dir / "products"
    if products_media.is_dir():
        print(f"Removing media directory: {products_media}...")
        for item in products_media.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif item.is_file():
                item.unlink()
        print("Media files removed.")

    await engine.dispose()
    print("Clean-up finished successfully!")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear all database products, embeddings, and media.")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion without interactive prompt")
    parser.add_argument("--media-dir", type=Path, default=Path("media"), help="Media directory path")
    args = parser.parse_args()

    # Determine media dir in container or local
    target_media = Path("/media") if Path("/media").is_dir() else args.media_dir

    if not args.yes:
        confirm = input("Are you sure you want to delete ALL products, embeddings and media? [y/N]: ")
        if confirm.lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    asyncio.run(clear_all(target_media))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
