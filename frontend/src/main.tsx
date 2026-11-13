import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { App } from "./App"
import { AuthProvider } from "./context/AuthContext"
import "./styles/main.css"

const container = document.getElementById("root")
if (!container) throw new Error("Root container #root is missing from index.html")

createRoot(container).render(
  <StrictMode>
    <AuthProvider>
      <App />
    </AuthProvider>
  </StrictMode>
)