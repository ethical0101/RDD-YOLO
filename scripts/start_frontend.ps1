# Start the Vite dev server on http://localhost:5173 (proxies /api and /files to the backend).
$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "frontend")
if (-not (Test-Path "node_modules")) { npm install }
npm run dev
