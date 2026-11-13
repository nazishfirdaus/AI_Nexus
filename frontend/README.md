# ANexus frontend

React + TypeScript client for the ANexus API (Vite build).

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173, proxies /api -> http://localhost:8000
npm run build    # type-check + bundle into frontend/dist
npm run preview  # serve the production build
```

Run the API alongside it:

```bash
uvicorn src.api.main:app --reload --port 8000
```

## Configuration

| Variable | Purpose |
| --- | --- |
| `VITE_API_KEY` | Optional. Sent as `X-API-Key` on every `/api` call. Set it to `FRONTEND_API_KEY` from the server `.env` when the API key middleware is enabled. |

The tenant is sent as `X-User-Id`; it is chosen in **Settings → Tenant** and stored
in `localStorage`. When unset the API applies its own default (`ANEXUS_DEV_USER_ID`,
normally `dev`).

## Wiring

Every screen reads live data; no sample documents, answers or counts are bundled.

| Screen | Endpoint(s) |
| --- | --- |
| Chat | `POST /api/chat/stream` (SSE `meta`/`delta`/`done`/`error`), `GET /api/conversations/{id}` |
| Feedback | `POST /api/messages/{id}/feedback` |
| Conversations | `GET/POST /api/conversations`, `PATCH/DELETE /api/conversations/{id}` |
| Library | `GET/POST /api/documents`, `DELETE /api/documents/{id}` |
| Settings | `GET /api/health` |

Streaming uses `fetch` + a `ReadableStream` reader rather than `EventSource`,
because the endpoint is a `POST` with a JSON body.

Presentation-only metadata (format, icon, kind) is derived from each document's
filename extension: `DocumentOut` does not carry a format field.

## Serving the build

`src/api/main.py` mounts `frontend/dist` when it exists: `/assets/*` comes from
 disk and any other unknown path falls back to `index.html` so client-side routes
work. `/api`, `/docs` and `/openapi.json` always win over the fallback.

## Layout

```
frontend/
  index.html
  vite.config.ts
  src/
    main.tsx                 React root + AuthProvider
    App.tsx                  shell, data loading, streaming, toasts
    context/AuthContext.tsx  tenant identity
    components/
      ConversationSidebar.tsx
      ChatView.tsx
      LibraryView.tsx
      SettingsView.tsx
      EvidencePanel.tsx
      Icon.tsx
    utils/
      api.ts                 typed client (mirrors src/api/schemas.py)
      documents.ts           corpus stats + filename-derived presentation
    styles/main.css          design tokens and components
```