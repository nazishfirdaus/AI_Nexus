import { createContext, useCallback, useContext, useEffect, useState } from "react"
import type { ReactNode } from "react"

interface AuthState {
  /** Tenant id sent as ``X-User-Id``. */
  user: string | null
  loading: boolean
  login: (tenant: string) => void
  logout: () => void
}

const STORAGE_KEY = "anexus-tenant"

const AuthContext = createContext<AuthState | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let stored: string | null = null
    try {
      stored = window.localStorage.getItem(STORAGE_KEY)
    } catch {
      // storage may be blocked
    }
    // The API falls back to its own default tenant when the header is absent,
    // so an unset value is not an error here.
    setUser(stored && /^[A-Za-z0-9_-]{1,64}$/.test(stored) ? stored : null)
    setLoading(false)
  }, [])

  const login = useCallback((tenant: string) => {
    setUser(tenant)
    try {
      window.localStorage.setItem(STORAGE_KEY, tenant)
    } catch {
      // storage may be blocked
    }
  }, [])

  const logout = useCallback(() => {
    setUser(null)
    try {
      window.localStorage.removeItem(STORAGE_KEY)
    } catch {
      // storage may be blocked
    }
  }, [])

  return (
    <AuthContext.Provider value={{ user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error("useAuth must be used inside <AuthProvider>.")
  }
  return context
}