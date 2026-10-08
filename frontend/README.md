# RDD-YOLO dashboard

React 19 + Vite + TypeScript + Tailwind CSS 4, React-Leaflet (OpenStreetMap), Leaflet.markercluster,
Leaflet.heat and Recharts.

```powershell
npm install
npm run dev     # http://localhost:5173, proxies /api and /files to the backend on :8000
npm run build   # production build in dist/ (served by the FastAPI backend at :8000)
```

Pages: Dashboard, Image Detection, Video Detection, Live Camera, Damage Map, Analytics, Detection History,
Model, Training & Experiments. See the project README for the full system.
