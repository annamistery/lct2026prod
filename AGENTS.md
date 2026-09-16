# LCT2026 production rules

- Apache proxies only `/api/` to FastAPI on port 8030. Every externally reachable backend endpoint must start with `/api/`.
- Only search pipelines are versioned: `/api/v1/search` and `/api/v1/search-from-crop`. Health, products, and media routes are unversioned.
- Routes are declared in their feature section and included through `app/api/router.py`.
- Shared core, database models, repositories, and ML services are not versioned.
- PostgreSQL with pgvector is the only product and embedding store. FAISS is not used.
- Schema changes require SQLAlchemy declarative models and Alembic revisions. Never call `create_all` at runtime.
- Resolve paths from configuration; never hard-code machine-specific absolute paths.
- Models are read-only at runtime. Product media is written only below the configured media root.
- Use bound SQLAlchemy expressions and validated typed inputs. Never interpolate request values into SQL, shell commands, paths, templates, or headers.

## Delivery workflow

- The owner deploys, starts, and validates the project on a separate Ubuntu GPU server. Local work must not assume access to that server or claim that server-side checks have passed.
- Every completed, verified increment must be committed and pushed to `origin` immediately. Do not leave completed changes only in the local working tree.
- Before each commit, inspect `git status`, `git diff`, and recent commit style; run all locally available relevant checks and report checks that require the external server.
- Never commit secrets, `.env`, PostgreSQL data, uploaded media, caches, or untracked model artifacts. Model files explicitly covered by Git LFS are allowed.
- Never push broken or unverified work knowingly. If a required check can run only on the external server, push the locally verified increment and provide the exact server commands from `docs/SERVER_COMMANDS.md`.
- Use concise commits that describe one completed increment. Push the current branch after every successful commit.
