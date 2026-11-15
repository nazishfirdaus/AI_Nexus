import { useState } from "react"
import type { Health } from "../utils/api"
import { Icon } from "./Icon"

export type ThemeMode = "light" | "dark" | "system"

interface Props {
  health: Health | null
  tenant: string
  theme: ThemeMode
  onTheme: (mode: ThemeMode) => void
  onTenant: (tenant: string) => void
}

const THEMES: { key: ThemeMode; name: string; hint: string; preview: string }[] = [
  { key: "light", name: "Light", hint: "Ivory & charcoal", preview: "preview--light" },
  { key: "dark", name: "Dark", hint: "Charcoal & gold", preview: "preview--dark" },
  { key: "system", name: "System", hint: "Follow device", preview: "preview--system" },
]

export function SettingsView({ health, tenant, theme, onTheme, onTenant }: Props) {
  const [draftTenant, setDraftTenant] = useState(tenant)

  return (
    <section className="view view--settings is-active" aria-label="Settings">
      <div className="page page--narrow">
        <div className="page-head">
          <div>
            <span className="eyebrow">Workspace</span>
            <h2 className="display">Settings</h2>
            <p className="page-sub">
              Tenant identity and appearance. Documents and conversations are scoped
              to the tenant below.
            </p>
          </div>
        </div>

        <section className="card">
          <div className="card-head">
            <div>
              <h3>Tenant</h3>
              <p>
                Sent as <span className="mono">X-User-Id</span> on every request and
                used to pick the Qdrant collection and SQLite database. Letters,
                digits, “-” and “_” only.
              </p>
            </div>
            <span className="badge badge--gold">
              <span className="pip pip--live" />
              {tenant || "default"}
            </span>
          </div>

          <div className="field-grid">
            <label className="field">
              <span className="field-label">Tenant id</span>
              <input
                className="input input--mono"
                value={draftTenant}
                onChange={(event) => setDraftTenant(event.target.value)}
                placeholder="dev"
              />
            </label>
            <div className="field">
              <span className="field-label">Apply</span>
              <button
                className="btn btn--ghost"
                type="button"
                onClick={() => onTenant(draftTenant.trim())}
              >
                <Icon name="check" />
                <span>Switch tenant</span>
              </button>
            </div>
          </div>
        </section>

        <section className="card">
          <div className="card-head">
            <div>
              <h3>API status</h3>
              <p>Reported by <span className="mono">GET /api/health</span>.</p>
            </div>
          </div>

          <div className="toggle-list">
            <div className="toggle-row">
              <div>
                <span className="toggle-title">Service</span>
                <span className="toggle-sub">{health?.service ?? "not queried"}</span>
              </div>
              <span className="env mono">
                {health ? health.environment : "—"}
              </span>
              <span className={`status ${health ? "status--ok" : "status--muted"}`}>
                <span className="pip" />
                {health ? health.status : "offline"}
              </span>
            </div>

            <div className="toggle-row">
              <div>
                <span className="toggle-title">Collection</span>
                <span className="toggle-sub">
                  {health?.collection ?? "unknown until the API responds"}
                </span>
              </div>
              <span className="env mono">qdrant</span>
              <span
                className={`status ${
                  health?.qdrant_configured ? "status--ok" : "status--warn"
                }`}
              >
                <span className="pip" />
                {health?.qdrant_configured ? "configured" : "missing key"}
              </span>
            </div>
          </div>
        </section>

        <section className="card">
          <div className="card-head">
            <div>
              <h3>Appearance</h3>
              <p>Warm ivory by day, charcoal by night. Stored in this browser.</p>
            </div>
          </div>

          <div className="theme-grid" role="group" aria-label="Theme">
            {THEMES.map((item) => (
              <button
                key={item.key}
                type="button"
                className={`theme-card${theme === item.key ? " is-active" : ""}`}
                aria-pressed={theme === item.key}
                onClick={() => onTheme(item.key)}
              >
                <span className={`theme-preview ${item.preview}`} aria-hidden="true">
                  <span className="tp-rail" />
                  <span className="tp-line" />
                  <span className="tp-card" />
                </span>
                <span className="theme-name">
                  {item.name}
                  <small>{item.hint}</small>
                </span>
                <span className="theme-check">
                  <Icon name="check" size={10} />
                </span>
              </button>
            ))}
          </div>
        </section>
      </div>
    </section>
  )
}