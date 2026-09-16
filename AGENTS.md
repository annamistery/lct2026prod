# LCT2026 production rules

- Apache proxies only `/api/` to FastAPI on port 8030. Every externally reachable backend endpoint must start with `/api/`.
- Only search pipelines are versioned: `/api/v1/search` and `/api/v1/search-from-crop`. Health, products, and media routes are unversioned.
- Routes are declared in their feature section and included through `app/api/router.py`.
- Shared core, database models, repositories, and ML services are not versioned.
- PostgreSQL with pgvector is the only product and embedding store. FAISS is not used.
- Product metadata belongs in `products`; reusable reference vectors belong in `product_embeddings`. A product may have multiple catalog, real, or customer embeddings. Search must select the nearest embedding per unique product before SIFT reranking.
- Catalog images, crops, datasets, and legacy indexes are not stored in Git. They are loaded from the customer dataset into PostgreSQL and the configured media storage.
- Large customer archives are uploaded to `imports/inbox`, safely extracted to `imports/staging/<batch_id>`, and imported through durable unversioned `/api/imports` jobs. API input may select only a validated batch ID and manifest filename, never an arbitrary server path.
- Schema changes require SQLAlchemy declarative models and Alembic revisions. Never call `create_all` at runtime.
- Resolve paths from configuration; never hard-code machine-specific absolute paths.
- Models are read-only at runtime. Product media is written only below the configured media root. Docker API UID/GID must match the owner of the host media bind mount.
- Use bound SQLAlchemy expressions and validated typed inputs. Never interpolate request values into SQL, shell commands, paths, templates, or headers.

## Delivery workflow

- The owner deploys, starts, and validates the project on a separate Ubuntu GPU server. Local work must not assume access to that server or claim that server-side checks have passed.
- Every completed, verified increment must be committed and pushed to `origin` immediately. Do not leave completed changes only in the local working tree.
- Before each commit, inspect `git status`, `git diff`, and recent commit style; run all locally available relevant checks and report checks that require the external server.
- Never commit secrets, `.env`, PostgreSQL data, uploaded media, caches, or untracked model artifacts. Model files explicitly covered by Git LFS are allowed.
- Never push broken or unverified work knowingly. If a required check can run only on the external server, push the locally verified increment and provide the exact server commands from `docs/SERVER_COMMANDS.md`.
- Use concise commits that describe one completed increment. Push the current branch after every successful commit.
