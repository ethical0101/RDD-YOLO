import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import './index.css'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Detect from './pages/Detect'
import VideoPage from './pages/VideoPage'
import Live from './pages/Live'
import MapPage from './pages/MapPage'
import Analytics from './pages/Analytics'
import History from './pages/History'
import ModelPage from './pages/ModelPage'
import Training from './pages/Training'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="detect" element={<Detect />} />
          <Route path="video" element={<VideoPage />} />
          <Route path="live" element={<Live />} />
          <Route path="map" element={<MapPage />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="history" element={<History />} />
          <Route path="model" element={<ModelPage />} />
          <Route path="training" element={<Training />} />
          <Route path="*" element={<Dashboard />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
