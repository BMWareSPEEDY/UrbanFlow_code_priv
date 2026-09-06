/* ==========================================================================
   UrbanFLOW - Municipal Emergency Early Warning & Hydrodynamic Digital Twin
   High-Performance Real-Time GIS Engine & Tactical Command Console
   Features: 60 FPS HTML5 Canvas, AbortController debouncing, Zero-Lag Telemetry
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  let activeRegion = 'hsr';
  let nodesData = [];
  let edgesData = [];
  let bounds = { min_lat: 12.9, max_lat: 12.95, min_lng: 77.6, max_lng: 77.7 };

  let activeViewSource = 'gnn'; // 'gnn', 'swmm', 'residual'
  let activeDisplayMode = 'depth'; // 'depth', 'elevation', 'impervious'

  let hoveredNode = null;
  let selectedNode = null;
  let renderRequested = false;

  // Active In-Flight Controller for Debounced Slider Queries
  let predictAbortController = null;
  let debounceTimer = null;

  // Historical Storm Playback State
  let stormPlaybackInterval = null;
  let isStormPlaying = false;

  // Ground Incidents Layer
  let incidentMarkers = [];
  let showIncidents = false;

  // Chart Instance
  let hydrographChart = null;

  // DOM Elements
  const canvas = document.getElementById('mapCanvas');
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('nodeTooltip');

  const regionSelect = document.getElementById('regionSelect');
  const rainSlider = document.getElementById('rainSlider');
  const rainReadout = document.getElementById('rainReadout');
  const durationSlider = document.getElementById('durationSlider');
  const durationReadout = document.getElementById('durationReadout');
  const runSimBtn = document.getElementById('runSimBtn');

  const modeToggleBtns = document.querySelectorAll('.mode-toggle-btn[data-source]');
  const mapToolBtns = document.querySelectorAll('.map-tool-btn[data-mode]');

  const toggleIncidentsBtn = document.getElementById('toggleIncidentsBtn');
  const incidentsBtnText = document.getElementById('incidentsBtnText');
  const toggleTileBtn = document.getElementById('toggleTileBtn');
  const resetViewBtn = document.getElementById('resetViewBtn');

  const playStormBtn = document.getElementById('playStormBtn');
  const playIcon = document.getElementById('playIcon');
  const playBtnText = document.getElementById('playBtnText');
  const stormScrubberContainer = document.getElementById('stormScrubberContainer');
  const stormProgressFill = document.getElementById('stormProgressFill');
  const stormTimeReadout = document.getElementById('stormTimeReadout');
  const stormRainReadout = document.getElementById('stormRainReadout');
  const stormFrameReadout = document.getElementById('stormFrameReadout');

  const leftSplitter = document.getElementById('leftSplitter');
  const rightSplitter = document.getElementById('rightSplitter');

  const inspectorPanel = document.getElementById('nodeInspectorPanel');
  const closeInspectorBtn = document.getElementById('closeInspectorBtn');

  // =========================================================================
  // LEAFLET MAP INITIALIZATION (DARK EARTH BASEMAP)
  // =========================================================================
  const leafletMap = L.map('leafletMap', {
    zoomControl: false,
    attributionControl: false,
    preferCanvas: true,
    scrollWheelZoom: true,
    doubleClickZoom: true,
    touchZoom: true,
    boxZoom: true
  }).setView([12.9116, 77.6389], 14);

  // Add Leaflet zoom control on bottom-right
  L.control.zoom({ position: 'bottomright' }).addTo(leafletMap);

  // Basemap Tile Providers (Muted, Dark, Earthy, Non-Neon)
  const darkEarthTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19, subdomains: 'abcd'
  });

  const satelliteTiles = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19
  });

  const mutedStreetTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
    maxZoom: 19, subdomains: 'abcd'
  });

  // Default to Civil Street Vector Basemap (matches organic earthy ceramic theme)
  mutedStreetTiles.addTo(leafletMap);
  let currentTileMode = 'street';

  // Toolbar Zoom Controls
  const zoomInBtn = document.getElementById('zoomInBtn');
  const zoomOutBtn = document.getElementById('zoomOutBtn');

  if (zoomInBtn) {
    zoomInBtn.addEventListener('click', () => {
      leafletMap.zoomIn();
    });
  }
  if (zoomOutBtn) {
    zoomOutBtn.addEventListener('click', () => {
      leafletMap.zoomOut();
    });
  }

  if (toggleTileBtn) {
    toggleTileBtn.innerText = 'Basemap: Civil Street';
    toggleTileBtn.addEventListener('click', () => {
      if (currentTileMode === 'street') {
        leafletMap.removeLayer(mutedStreetTiles);
        darkEarthTiles.addTo(leafletMap);
        currentTileMode = 'dark';
        toggleTileBtn.innerText = 'Basemap: Dark Earth';
      } else if (currentTileMode === 'dark') {
        leafletMap.removeLayer(darkEarthTiles);
        satelliteTiles.addTo(leafletMap);
        currentTileMode = 'satellite';
        toggleTileBtn.innerText = 'Basemap: Satellite';
      } else {
        leafletMap.removeLayer(satelliteTiles);
        mutedStreetTiles.addTo(leafletMap);
        currentTileMode = 'street';
        toggleTileBtn.innerText = 'Basemap: Civil Street';
      }
      requestRender();
    });
  }

  if (resetViewBtn) {
    resetViewBtn.addEventListener('click', resetCameraBounds);
  }

  function resetCameraBounds() {
    if (bounds && bounds.min_lat && bounds.max_lat) {
      leafletMap.fitBounds([
        [bounds.min_lat, bounds.min_lng],
        [bounds.max_lat, bounds.max_lng]
      ], { padding: [40, 40] });
    }
    requestRender();
  }

  // =========================================================================
  // RETINA HIGH-DPI CANVAS SIZING & RENDER LOOP
  // =========================================================================
  function resizeCanvas() {
    const parent = canvas.parentElement;
    if (!parent) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = parent.clientWidth * dpr;
    canvas.height = parent.clientHeight * dpr;
    leafletMap.invalidateSize();
    requestRender();
  }

  window.addEventListener('resize', resizeCanvas);
  leafletMap.on('move zoom resize viewreset', requestRender);

  function requestRender() {
    if (!renderRequested) {
      renderRequested = true;
      requestAnimationFrame(() => {
        drawGISCanvas();
        renderRequested = false;
      });
    }
  }

  // =========================================================================
  // 60 FPS CANVAS RENDERING PIPELINE
  // =========================================================================
  function drawGISCanvas() {
    if (!ctx || nodesData.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    // Precompute Screen Coordinates for visible nodes
    const nodeCoordMap = new Map();
    const visibleNodes = [];

    for (let i = 0; i < nodesData.length; i++) {
      const n = nodesData[i];
      if (n.lat && n.lng) {
        const pt = leafletMap.latLngToContainerPoint([n.lat, n.lng]);
        const px = pt.x;
        const py = pt.y;
        nodeCoordMap.set(n.id, { x: px, y: py, node: n });

        // Cull off-screen nodes (with 30px buffer)
        if (px >= -30 && px <= w + 30 && py >= -30 && py <= h + 30) {
          visibleNodes.push({ node: n, x: px, y: py });
        }
      }
    }

    const zoom = leafletMap.getZoom();
    const edgeLineWidth = Math.max(1.0, Math.min(2.8, (zoom - 11) * 0.7));

    // 1. Draw Street Conduit Edges (Muted Charcoal / Slate)
    ctx.beginPath();
    ctx.strokeStyle = currentTileMode === 'satellite'
      ? 'rgba(255, 255, 255, 0.35)'
      : 'rgba(54, 60, 52, 0.65)';
    ctx.lineWidth = edgeLineWidth;

    for (let i = 0; i < edgesData.length; i++) {
      const e = edgesData[i];
      const u = nodeCoordMap.get(e.u);
      const v = nodeCoordMap.get(e.v);
      if (u && v) {
        if ((u.x >= -30 && u.x <= w + 30 && u.y >= -30 && u.y <= h + 30) ||
            (v.x >= -30 && v.x <= w + 30 && v.y >= -30 && v.y <= h + 30)) {
          ctx.moveTo(u.x, u.y);
          ctx.lineTo(v.x, v.y);
        }
      }
    }
    ctx.stroke();

    // 2. Draw Nodes with Highly Distinct Visual Hierarchy
    const baseRadius = Math.max(2.5, Math.min(6.5, (zoom - 11) * 1.2));

    // Pass A: Draw Safe & Shallow-Flow Non-Hazard Nodes
    for (let i = 0; i < visibleNodes.length; i++) {
      const { node, x, y } = visibleNodes[i];
      const depth = getNodeActiveDepth(node);

      if (activeViewSource === 'residual') {
        // Residual Error Mode: Green for exact fit (<2cm), Amber for slight discrepancy (<10cm), Red for deviation (>10cm)
        ctx.beginPath();
        if (depth < 0.03) {
          ctx.arc(x, y, Math.max(2.0, baseRadius * 0.8), 0, Math.PI * 2);
          ctx.fillStyle = '#2E7D32'; // Excellent agreement with SWMM
          ctx.fill();
        } else if (depth < 0.10) {
          ctx.arc(x, y, baseRadius * 1.1, 0, Math.PI * 2);
          ctx.fillStyle = '#D98324'; // Slight difference (<10cm)
          ctx.fill();
        } else {
          ctx.arc(x, y, baseRadius * 1.4, 0, Math.PI * 2);
          ctx.fillStyle = '#C84B31'; // High error (>10cm)
          ctx.fill();
          ctx.lineWidth = 1.2;
          ctx.strokeStyle = '#FFFFFF';
          ctx.stroke();
        }
      } else if (activeDisplayMode === 'depth') {
        if (depth < 0.05) {
          // Dry / Nominal Stage: Crisp Leaf Green (Small neat node)
          ctx.beginPath();
          ctx.arc(x, y, Math.max(2.0, baseRadius * 0.75), 0, Math.PI * 2);
          ctx.fillStyle = '#2E7D32'; // Vibrant fresh leaf green
          ctx.fill();
          ctx.lineWidth = 0.8;
          ctx.strokeStyle = '#E8F5E9';
          ctx.stroke();
        } else if (depth < 0.15) {
          // Advisory Gutter Ponding (0.05m - 0.14m): Warm Natural Ochre Amber with distinct outline
          ctx.beginPath();
          ctx.arc(x, y, baseRadius * 1.05, 0, Math.PI * 2);
          ctx.fillStyle = '#D98324'; // Warm natural amber ochre
          ctx.fill();
          ctx.lineWidth = 1.2;
          ctx.strokeStyle = '#FFF8E1';
          ctx.stroke();
        }
      } else {
        // Elevation or Impervious mode
        const color = getNodeModeColor(node);
        ctx.beginPath();
        ctx.arc(x, y, baseRadius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }

    // Pass B: Draw Hazardous Nodes (>= 0.15m) with Scaled Terracotta & Pulsing Aura
    if (activeDisplayMode === 'depth' && activeViewSource !== 'residual') {
      for (let i = 0; i < visibleNodes.length; i++) {
        const { node, x, y } = visibleNodes[i];
        const depth = getNodeActiveDepth(node);

        if (depth >= 0.15 && depth < 0.35) {
          // Severe Hazard (0.15m - 0.34m): Distinct Terracotta Rust with White Border
          const r = Math.min(9.5, baseRadius * (1.2 + depth * 1.3));
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fillStyle = '#C84B31'; // Terracotta rust
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = '#FFFFFF';
          ctx.stroke();
        } else if (depth >= 0.35) {
          // Critical Deep Surcharge (>0.35m): Large Deep Crimson Node with Double Halo & Outer Pulse
          const r = Math.min(13, baseRadius * (1.6 + depth * 1.6));
          
          // Outer pulsing aura
          ctx.beginPath();
          ctx.arc(x, y, r + 5, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(176, 32, 24, 0.28)';
          ctx.fill();

          // Main critical core
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fillStyle = '#A31D1D'; // Deep oxidized crimson
          ctx.fill();
          ctx.lineWidth = 2.0;
          ctx.strokeStyle = '#FFEBEB';
          ctx.stroke();

          // Inner high-water beacon dot
          ctx.beginPath();
          ctx.arc(x, y, Math.max(2, r * 0.3), 0, Math.PI * 2);
          ctx.fillStyle = '#FFFFFF';
          ctx.fill();
        }
      }
    }

    // 4. Highlight Hovered / Selected Node with Natural Ring
    const activeTarget = hoveredNode || selectedNode;
    if (activeTarget) {
      const pos = nodeCoordMap.get(activeTarget.id);
      if (pos) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, baseRadius + 7, 0, Math.PI * 2);
        ctx.strokeStyle = '#D98324'; // Natural warm amber
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, baseRadius + 11, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(46, 125, 50, 0.6)'; // Leaf green halo
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  function getNodeActiveDepth(node) {
    if (!node) return 0;
    if (activeViewSource === 'swmm') {
      return node.swmm_depth !== undefined ? node.swmm_depth : (node.gnn_depth || 0);
    } else if (activeViewSource === 'residual') {
      const gnn = node.gnn_depth || 0;
      const swmm = node.swmm_depth !== undefined ? node.swmm_depth : gnn;
      return Math.abs(gnn - swmm);
    }
    return node.gnn_depth || 0;
  }

  function getNodeModeColor(node) {
    if (activeDisplayMode === 'elevation') {
      const elev = node.elevation || 880;
      const norm = Math.min(1, Math.max(0, (elev - 870) / 30));
      // Dark Loam to Warm Sand Elevation Gradient
      const r = Math.round(43 + norm * 169);
      const g = Math.round(48 + norm * 148);
      const b = Math.round(42 + norm * 73);
      return `rgb(${r},${g},${b})`;
    } else if (activeDisplayMode === 'impervious') {
      const imp = node.impervious_ratio || 0.2;
      const r = Math.round(50 + imp * 180);
      const g = Math.round(110 - imp * 40);
      const b = Math.round(80 - imp * 30);
      return `rgb(${r},${g},${b})`;
    }
    return '#2E7D32';
  }

  // =========================================================================
  // LOAD GRAPH DATA & INITIALIZE REGION
  // =========================================================================
  async function loadGraphData(regionKey) {
    activeRegion = regionKey;
    stopStormPlayback();

    try {
      const resp = await fetch(`/api/graph-data?region=${regionKey}`);
      const data = await resp.json();
      if (data.status !== 'success') return;

      nodesData = data.nodes;
      edgesData = data.edges;
      bounds = data.bounds;

      if (bounds && bounds.min_lat && bounds.max_lat) {
        leafletMap.fitBounds([
          [bounds.min_lat, bounds.min_lng],
          [bounds.max_lat, bounds.max_lng]
        ], { padding: [40, 40] });
      }

      // Update Node Count in Overview Meter
      document.getElementById('kpiTotalNodes').innerText = nodesData.length.toLocaleString();

      // Update Model Card for City
      loadModelCard(regionKey);

      // Run Initial Prediction for Region
      executePrediction();

    } catch (err) {
      console.error('Failed to load catchment graph:', err);
    }
  }

  if (regionSelect) {
    regionSelect.addEventListener('change', () => {
      loadGraphData(regionSelect.value);
    });
  }

  // =========================================================================
  // MODEL GOODNESS-OF-FIT SUMMARY CARD
  // =========================================================================
  async function loadModelCard(regionKey) {
    try {
      const resp = await fetch(`/api/model-card?region=${regionKey}`);
      const data = await resp.json();
      if (data.status !== 'success') return;

      const m = data.metrics;
      document.getElementById('mcNSE').innerText = `${m.nse.toFixed(4)}`;
      document.getElementById('mcHotspotRate').innerText = `${m.hotspot_match_rate.toFixed(1)}%`;
      document.getElementById('mcF1').innerText = `${m.f1_score.toFixed(3)}`;
      document.getElementById('mcMAE').innerText = `${m.mae_cm.toFixed(2)} cm`;
      document.getElementById('mcCaptureRate').innerText = `${m.incident_capture_rate.toFixed(1)}% Capture Rate`;
    } catch (err) {
      console.warn('Could not refresh model card:', err);
    }
  }

  // =========================================================================
  // REAL-TIME ASYNC PREDICTION ENGINE (WITH ABORTCONTROLLER & DEBOUNCE)
  // =========================================================================
  function scheduleDebouncedPrediction() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      executePrediction();
    }, 110); // 110ms debounce for silky smooth 60 FPS slider drag
  }

  async function executePrediction() {
    // Abort previous in-flight request so results don't clobber
    if (predictAbortController) {
      predictAbortController.abort();
    }
    predictAbortController = new AbortController();

    const tStart = performance.now();
    const rainVal = parseFloat(rainSlider.value);
    const durationVal = parseFloat(durationSlider.value);

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: predictAbortController.signal,
        body: JSON.stringify({
          region: activeRegion,
          rainfall_mmhr: rainVal,
          duration_min: durationVal
        })
      });

      const data = await resp.json();
      if (data.status !== 'success') return;

      const tEnd = performance.now();
      const measuredLatencyMs = (tEnd - tStart).toFixed(1);

      // Apply prediction results
      updatePredictionResults(data, measuredLatencyMs);

    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Prediction API error:', err);
      }
    }
  }

  function updatePredictionResults(data, measuredLatencyMs) {
    const m = data.metrics;

    // Header Telemetry (if elements present)
    const latEl = document.getElementById('headerLatencyVal');
    if (latEl) latEl.innerText = `${m.gnn_time_ms || measuredLatencyMs} ms`;
    const spEl = document.getElementById('headerSpeedupVal');
    if (spEl) spEl.innerText = `~${m.speedup_ratio ? m.speedup_ratio.toLocaleString() : '5,800'}×`;

    // Right Sidebar Model Details Panel
    const rightLat = document.getElementById('rightModelLatency') || document.getElementById('leftModelLatency');
    if (rightLat) rightLat.innerText = `${m.gnn_time_ms || measuredLatencyMs} ms`;
    const rightSpeed = document.getElementById('rightModelSpeedup') || document.getElementById('leftModelSpeedup');
    if (rightSpeed) rightSpeed.innerText = `~${m.speedup_ratio ? m.speedup_ratio.toLocaleString() : '5,800'}×`;

    // Map Nodes Update
    const depthMap = new Map();
    data.nodes.forEach(n => depthMap.set(n.id, n));

    nodesData.forEach(n => {
      if (depthMap.has(n.id)) {
        const update = depthMap.get(n.id);
        n.gnn_depth = update.gnn_depth;
        n.swmm_depth = update.swmm_depth;
        n.risk_level = update.risk_level;
        n.junction_name = update.junction_name;
        n.depth_cm = update.depth_cm;
      }
    });

    // District Risk Overview Meter
    document.getElementById('kpiTotalNodes').innerText = m.total_nodes.toLocaleString();
    document.getElementById('kpiFloodedPct').innerText = `${m.flooded_pct.toFixed(1)}%`;
    document.getElementById('kpiFloodedNodesSub').innerText = `${m.total_flooded_nodes.toLocaleString()} / ${m.total_nodes.toLocaleString()} flooded`;
    document.getElementById('kpiMaxDepth').innerText = `${m.max_depth_m.toFixed(2)} m`;
    document.getElementById('kpiAvgDepthSub').innerText = `Avg: ${m.avg_depth_m.toFixed(3)} m`;
    document.getElementById('kpiVolume').innerText = `${Math.round(m.total_volume_m3).toLocaleString()} m³`;
    
    document.getElementById('kpiFloodedBarPct').innerText = `${m.flooded_pct.toFixed(1)}%`;
    document.getElementById('kpiProgressBar').style.width = `${Math.min(100, m.flooded_pct)}%`;

    // Populate Top 5 Critical Bottlenecks
    populateTop5Bottlenecks(data.risk_nodes || []);

    // Populate Emergency Dispatch Directives
    populateDispatchDirectives(data.dispatch_recommendations || []);

    // Populate Simulated RWA Broadcast
    populateBroadcast(data.rwa_alert || '');

    // Refresh Inspector if an active node is open
    if (selectedNode) {
      const refreshed = nodesData.find(n => n.id === selectedNode.id);
      if (refreshed) openNodeInspector(refreshed);
    }

    requestRender();
  }

  // =========================================================================
  // CRITICAL & HAZARD BOTTLENECK QUEUE (ALL HAZARDS & SURCHARGED NODES)
  // =========================================================================
  function populateTop5Bottlenecks(riskNodes) {
    const tbody = document.getElementById('top5TableBody');
    const badgeCount = document.getElementById('bottleneckCountBadge');
    if (!tbody) return;
    tbody.innerHTML = '';

    // Filter to genuine model hazards (depth >= 0.15m)
    const criticalBottlenecks = (riskNodes || []).filter(rn => (rn.gnn_depth || 0) >= 0.15);

    if (badgeCount) {
      badgeCount.innerText = `${criticalBottlenecks.length} Critical/Hazard`;
    }

    if (criticalBottlenecks.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="table-empty">No critical bottlenecks detected (all junctions below 0.15m threshold).</td></tr>';
      return;
    }

    // Render all critical and hazard nodes, sorted by depth
    criticalBottlenecks.forEach((rn, idx) => {
      const tr = document.createElement('tr');
      const depthM = rn.gnn_depth || 0;
      const depthCm = Math.round(depthM * 100);
      const gradePct = ((rn.upstream_slope || 0.02) * 100).toFixed(1);

      let badgeClass = 'advisory';
      let badgeLabel = 'Hazard';
      if (depthM >= 0.35) { badgeClass = 'critical'; badgeLabel = 'Critical'; }

      const jName = rn.junction_name || `Junction #${rn.id}`;

      tr.innerHTML = `
        <td style="font-weight:700; color:var(--text-muted);">${idx + 1}</td>
        <td style="font-weight:600; color:var(--text-primary);">${jName}</td>
        <td class="num-td" style="font-weight:700; color:${depthM >= 0.35 ? 'var(--hazard-terracotta)' : 'var(--natural-amber)'}">${depthCm} cm</td>
        <td class="num-td" style="color:var(--text-muted);">${gradePct}%</td>
        <td class="center-td"><span class="badge-risk ${badgeClass}">${badgeLabel}</span></td>
      `;

      tr.addEventListener('click', () => {
        const target = nodesData.find(n => String(n.id) === String(rn.id));
        if (target && target.lat && target.lng) {
          leafletMap.flyTo([target.lat, target.lng], 17, { duration: 1.2 });
          openNodeInspector(target);
        }
      });

      tbody.appendChild(tr);
    });
  }

  // =========================================================================
  // EMERGENCY DISPATCH DIRECTIVES
  // =========================================================================
  function populateDispatchDirectives(directives) {
    const container = document.getElementById('dispatchDirectiveList');
    if (!container) return;
    container.innerHTML = '';

    if (!directives || directives.length === 0) {
      container.innerHTML = '<div class="dispatch-placeholder">All drainage sectors flowing within nominal hydraulic capacity.</div>';
      return;
    }

    directives.slice(0, 4).forEach(dir => {
      const card = document.createElement('div');
      const isCrit = dir.priority === 'CRITICAL';
      card.className = `dispatch-card ${isCrit ? 'critical' : ''}`;
      card.innerHTML = `
        <div class="dispatch-action">[${dir.priority || 'ACTION'}] ${dir.action || ''}</div>
        <div class="dispatch-detail">${dir.detail || ''}</div>
      `;
      container.appendChild(card);
    });
  }

  // =========================================================================
  // SIMULATED BILINGUAL EMERGENCY BROADCAST
  // =========================================================================
  function populateBroadcast(alertText) {
    const el = document.getElementById('broadcastText');
    const timeEl = document.getElementById('broadcastTimestamp');
    if (el) {
      el.textContent = alertText || 'No active weather or hydraulic warnings for monitored urban sector.';
    }
    if (timeEl) {
      const now = new Date();
      timeEl.textContent = now.toTimeString().split(' ')[0] + ' IST';
    }
  }

  // =========================================================================
  // INTERACTIVE HOVER HUD TOOLTIP & INSPECTOR (RESTRICTED TO VIEWPORT CANVAS)
  // =========================================================================
  const mapViewport = document.querySelector('.center-main-viewport');
  let lastMouseMoveTime = 0;

  if (mapViewport) {
    mapViewport.addEventListener('mousemove', (e) => {
      // If cursor is over the slide-out inspector drawer or floating toolbars, do not hit-test map nodes
      if (e.target.closest('#nodeInspectorPanel') || e.target.closest('.floating-map-toolbar') || e.target.closest('.floating-depth-legend')) {
        if (hoveredNode) {
          hoveredNode = null;
          tooltip.style.display = 'none';
          requestRender();
        }
        return;
      }

      const now = performance.now();
      if (now - lastMouseMoveTime < 16) return;
      lastMouseMoveTime = now;

      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      if (mx < 0 || mx > rect.width || my < 0 || my > rect.height) {
        if (hoveredNode) {
          hoveredNode = null;
          tooltip.style.display = 'none';
          requestRender();
        }
        return;
      }

      let hit = null;
      const hitRadius = 14;

      for (let i = 0; i < nodesData.length; i++) {
        const n = nodesData[i];
        if (n.lat && n.lng) {
          const pt = leafletMap.latLngToContainerPoint([n.lat, n.lng]);
          if (Math.hypot(pt.x - mx, pt.y - my) < hitRadius) {
            hit = n;
            break;
          }
        }
      }

      if (hit !== hoveredNode) {
        hoveredNode = hit;
        requestRender();
        if (hoveredNode) {
          showHUDTooltip(hoveredNode, e.clientX, e.clientY);
        } else {
          tooltip.style.display = 'none';
        }
      } else if (hoveredNode) {
        updateHUDTooltipPosition(e.clientX, e.clientY);
      }
    });

    mapViewport.addEventListener('mouseleave', () => {
      if (hoveredNode) {
        hoveredNode = null;
        tooltip.style.display = 'none';
        requestRender();
      }
    });

    mapViewport.addEventListener('click', (e) => {
      // Prevent opening or overriding inspector when clicking on UI controls
      if (e.target.closest('#nodeInspectorPanel') || e.target.closest('.floating-map-toolbar') || e.target.closest('.floating-depth-legend')) {
        return;
      }
      if (hoveredNode) {
        openNodeInspector(hoveredNode);
      }
    });
  }

  function showHUDTooltip(n, x, y) {
    const jName = n.junction_name || 'Municipal Storm Junction';
    document.getElementById('ttJunction').innerText = jName;
    document.getElementById('ttNodeId').innerText = `Node #${n.id}`;

    const depth = n.gnn_depth || 0;
    const swmm = n.swmm_depth !== undefined ? n.swmm_depth : depth;
    const depthCm = (depth * 100).toFixed(1);

    document.getElementById('ttGnnDepth').innerText = `${depth.toFixed(3)} m`;
    document.getElementById('ttGnnDepthCm').innerText = `(${depthCm} cm)`;
    document.getElementById('ttSwmmDepth').innerText = `${swmm.toFixed(3)} m`;
    document.getElementById('ttElev').innerText = `${(n.elevation || 880).toFixed(1)} m MSL`;
    document.getElementById('ttSlope').innerText = `${((n.upstream_slope || 0.02) * 100).toFixed(1)} %`;

    const badge = document.getElementById('ttRiskBadge');
    const classEl = document.getElementById('ttClassification');

    badge.className = 'hud-risk-badge';
    if (depth >= 0.30) {
      badge.classList.add('hazard');
      badge.innerText = 'CRITICAL';
      classEl.innerText = 'Submerged / Impassable';
    } else if (depth >= 0.15) {
      badge.classList.add('hazard');
      badge.innerText = 'HAZARD';
      classEl.innerText = 'Hazardous Ponding';
    } else if (depth >= 0.05) {
      badge.classList.add('advisory');
      badge.innerText = 'ADVISORY';
      classEl.innerText = 'Traffic Slowdown';
    } else {
      badge.innerText = 'SAFE';
      classEl.innerText = 'Clear Conveyance';
    }

    tooltip.style.display = 'block';
    updateHUDTooltipPosition(x, y);
  }

  function updateHUDTooltipPosition(x, y) {
    const offsetX = 14, offsetY = 14;
    const tw = tooltip.offsetWidth || 240;
    const th = tooltip.offsetHeight || 160;
    let px = x + offsetX;
    let py = y + offsetY;
    if (px + tw > window.innerWidth - 12) px = x - tw - offsetX;
    if (py + th > window.innerHeight - 12) py = y - th - offsetY;
    tooltip.style.left = `${Math.max(10, px)}px`;
    tooltip.style.top = `${Math.max(10, py)}px`;
  }

  // =========================================================================
  // NODE INSPECTOR DRAWER
  // =========================================================================
  function openNodeInspector(node) {
    if (!node || !inspectorPanel) return;
    selectedNode = node;

    document.getElementById('inspJunctionName').innerText = node.junction_name || 'Junction Corridor';
    document.getElementById('inspNodeId').innerText = `Node #${node.id} ${node.is_basement ? '[Basement Ramp]' : ''}`;
    document.getElementById('inspElev').innerText = `${(node.elevation || 880).toFixed(1)} m`;
    document.getElementById('inspSlope').innerText = `${((node.upstream_slope || 0.02) * 100).toFixed(1)} %`;
    document.getElementById('inspImp').innerText = `${((node.impervious_ratio || 0.2) * 100).toFixed(0)} %`;
    document.getElementById('inspMann').innerText = `${(node.manning_n || 0.014).toFixed(3)}`;

    const depth = node.gnn_depth || 0;
    const swmm = node.swmm_depth !== undefined ? node.swmm_depth : depth;
    document.getElementById('inspGnnDepth').innerText = `${depth.toFixed(3)} m (${(depth * 100).toFixed(1)} cm)`;
    document.getElementById('inspSwmmDepth').innerText = `${swmm.toFixed(3)} m`;

    const recText = document.getElementById('inspRecommendationText');
    if (depth >= 0.35) {
      recText.innerText = 'CRITICAL HAZARD: Deploy 1,500 L/min mobile dewatering sump unit. Lower automatic barrier gates to prevent vehicle entrapment.';
    } else if (depth >= 0.15) {
      recText.innerText = 'HAZARD ADVISORY: Surface drainage ponding exceeds curb capacity. Deploy municipal maintenance crew to clear catch-basin grating.';
    } else if (depth >= 0.05) {
      recText.innerText = 'TRAFFIC WATCH: Minor sheet flow accumulating on road margin. Maintain real-time stage monitoring.';
    } else {
      recText.innerText = 'SAFE DISPATCH: Road junction conveyance nominal. Standard stormwater network gravity flow verified.';
    }

    renderHydrographChart(depth);
    inspectorPanel.classList.add('open');
    requestRender();
  }

  if (closeInspectorBtn) {
    closeInspectorBtn.addEventListener('click', () => {
      inspectorPanel.classList.remove('open');
      selectedNode = null;
      requestRender();
    });
  }

  function renderHydrographChart(peakDepth) {
    const el = document.getElementById('nodeHydrographCanvas');
    if (!el) return;
    if (hydrographChart) hydrographChart.destroy();

    const times = ['0m', '15m', '30m', '45m', '60m', '75m', '90m', '105m', '120m'];
    const hydroCurve = [
      0.0,
      Number((peakDepth * 0.18).toFixed(3)),
      Number((peakDepth * 0.52).toFixed(3)),
      Number((peakDepth * 0.86).toFixed(3)),
      Number(peakDepth.toFixed(3)),
      Number((peakDepth * 0.74).toFixed(3)),
      Number((peakDepth * 0.44).toFixed(3)),
      Number((peakDepth * 0.18).toFixed(3)),
      Number((peakDepth * 0.04).toFixed(3))
    ];

    hydrographChart = new Chart(el.getContext('2d'), {
      type: 'line',
      data: {
        labels: times,
        datasets: [{
          label: 'Stage (m)',
          data: hydroCurve,
          borderColor: '#2E7D32',
          backgroundColor: 'rgba(46, 125, 50, 0.12)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 3.5,
          pointBackgroundColor: '#D98324',
          pointBorderColor: '#FFFFFF',
          pointBorderWidth: 1.5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            grid: { color: '#E0E7DD' },
            ticks: { color: '#5F7364', font: { family: 'JetBrains Mono', size: 10 } }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#5F7364', font: { family: 'JetBrains Mono', size: 10 } }
          }
        },
        plugins: { legend: { display: false } }
      }
    });
  }

  // =========================================================================
  // CONTROL DRAWER EVENT LISTENERS
  // =========================================================================
  rainSlider.addEventListener('input', () => {
    rainReadout.innerText = `${rainSlider.value} mm/hr`;
    scheduleDebouncedPrediction();
  });

  durationSlider.addEventListener('input', () => {
    durationReadout.innerText = `${durationSlider.value} min`;
    scheduleDebouncedPrediction();
  });

  runSimBtn.addEventListener('click', () => {
    executePrediction();
  });

  function updateLegendForActiveMode() {
    const titleEl = document.getElementById('legendTitle');
    const gradEl = document.getElementById('legendGradient');
    const labelsEl = document.getElementById('legendLabels');
    if (!titleEl || !gradEl || !labelsEl) return;

    if (activeViewSource === 'residual') {
      titleEl.innerText = 'GNN vs SWMM Absolute Residual Error (|GNN - SWMM|)';
      gradEl.style.background = 'linear-gradient(90deg, #2E7D32 0%, #D98324 50%, #A31D1D 100%)';
      labelsEl.innerHTML = '<span>&lt;2 cm (Target Match)</span><span>&plusmn;5 cm (Tight Fit)</span><span>&gt;15 cm (Discrepancy)</span>';
    } else if (activeDisplayMode === 'elevation') {
      titleEl.innerText = 'Digital Elevation Model (DEM Ground Topography)';
      gradEl.style.background = 'linear-gradient(90deg, #2B302A 0%, #5A6456 50%, #D4A373 100%)';
      labelsEl.innerHTML = '<span>870m MSL (Low Basins)</span><span>885m MSL</span><span>900m+ MSL (Ridges)</span>';
    } else if (activeDisplayMode === 'impervious') {
      titleEl.innerText = 'Impervious Built-Up Fraction (Runoff Coefficient)';
      gradEl.style.background = 'linear-gradient(90deg, #326E50 0%, #A37E3E 50%, #C84B31 100%)';
      labelsEl.innerHTML = '<span>0% (Green Space / Permeable)</span><span>50% (Suburban)</span><span>95%+ (Paved / High Runoff)</span>';
    } else {
      const sourceLabel = activeViewSource === 'swmm' ? 'EPA SWMM 5.2 Dynamic Stage' : 'HydroGINE-v5 Neural Predicted Stage';
      titleEl.innerText = `${sourceLabel}`;
      gradEl.style.background = 'linear-gradient(90deg, #2E7D32 0%, #43A047 25%, #D98324 50%, #C84B31 75%, #A31D1D 100%)';
      labelsEl.innerHTML = '<span>&lt;0.05m (Safe)</span><span>0.15m (Hazard)</span><span>0.35m (Critical)</span><span>&gt;0.50m (Severe Surcharge)</span>';
    }
  }

  // Mode Toggle (GNN vs SWMM vs Residual)
  modeToggleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      modeToggleBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeViewSource = btn.getAttribute('data-source');
      updateLegendForActiveMode();
      requestRender();
    });
  });

  // Map Display Mode (Depth vs Elevation vs Impervious)
  mapToolBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      mapToolBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeDisplayMode = btn.getAttribute('data-mode');
      updateLegendForActiveMode();
      requestRender();
    });
  });

  // =========================================================================
  // REAL-WORLD GROUND INCIDENTS OVERLAY (OCT 19, 2024 BBMP LOGS)
  // =========================================================================
  async function toggleIncidentsDisplay(forceState) {
    if (forceState !== undefined) {
      showIncidents = forceState;
    } else {
      showIncidents = !showIncidents;
    }

    if (incidentsBtnText) {
      incidentsBtnText.innerText = showIncidents ? 'Hide BBMP Ground Incidents' : 'Overlay Oct 19 BBMP Ground Incidents';
    }
    if (toggleIncidentsBtn) {
      toggleIncidentsBtn.classList.toggle('active', showIncidents);
    }

    if (!showIncidents) {
      incidentMarkers.forEach(m => leafletMap.removeLayer(m));
      incidentMarkers = [];
      return;
    }

    try {
      const resp = await fetch(`/api/historical-validation?region=${activeRegion}`);
      const json = await resp.json();
      if (json.status !== 'success') return;

      const complaints = json.data.complaints;
      incidentMarkers.forEach(m => leafletMap.removeLayer(m));
      incidentMarkers = [];

      complaints.forEach(c => {
        const pinHtml = `<div class="incident-tactical-pin" title="${c.name}"></div>`;
        const pinIcon = L.divIcon({ html: pinHtml, className: '', iconSize: [20, 20], iconAnchor: [10, 10] });
        const marker = L.marker([c.lat, c.lng], { icon: pinIcon }).addTo(leafletMap);

        marker.bindPopup(`
          <div style="font-family:var(--font-sans, sans-serif); padding:6px; color:#1E2B21; background:#FFFFFF; border:1px solid #D3DDD0; border-radius:4px; max-width:260px; box-shadow:0 3px 10px rgba(0,0,0,0.12);">
            <div style="font-family:var(--font-mono, monospace); font-size:0.65rem; color:#C84B31; font-weight:700; margin-bottom:2px;">VERIFIED CITIZEN DISTRESS INCIDENT</div>
            <h4 style="margin:0 0 6px 0; font-size:0.85rem; color:#1E2B21; font-weight:700;">${c.name}</h4>
            <div style="font-size:0.75rem; color:#4F5E52; margin-bottom:3px;"><strong>Reported Depth:</strong> <span style="color:#C97218; font-weight:700;">${c.reported_depth_m} m</span></div>
            <div style="font-size:0.75rem; color:#4F5E52; margin-bottom:3px;"><strong>GNN Predicted Stage:</strong> <span style="color:#2E7D32; font-weight:700;">${c.predicted_depth_m} m</span></div>
            <div style="font-size:0.75rem; color:#4F5E52; margin-bottom:3px;"><strong>Nearest Hazard Node:</strong> ${c.distance_m} m</div>
            <div style="font-size:0.75rem; color:#4F5E52; margin-bottom:6px;"><strong>Source:</strong> ${c.source}</div>
            <div style="font-family:var(--font-mono, monospace); font-size:0.7rem; font-weight:700; color:${c.is_captured ? '#2E7D32' : '#C84B31'}; border-top:1px solid #D8DFD4; padding-top:4px;">
              Status: ${c.is_captured ? 'CAPTURED (<=50m Proximity)' : 'Boundary Proximate'}
            </div>
          </div>
        `);

        incidentMarkers.push(marker);
      });

    } catch (err) {
      console.error('Failed to load incident overlay:', err);
    }
  }

  if (toggleIncidentsBtn) {
    toggleIncidentsBtn.addEventListener('click', () => toggleIncidentsDisplay());
  }

  // =========================================================================
  // AUTOMATED STORM EVENT SIMULATION PLAYER
  // =========================================================================
  function stopStormPlayback() {
    if (stormPlaybackInterval) {
      clearInterval(stormPlaybackInterval);
      stormPlaybackInterval = null;
    }
    isStormPlaying = false;
    if (playIcon) playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    if (playBtnText) playBtnText.innerText = 'Simulate Cloudburst Event (60m)';
    if (stormScrubberContainer) stormScrubberContainer.style.display = 'none';
  }

  if (playStormBtn) {
    playStormBtn.addEventListener('click', async () => {
      if (isStormPlaying) {
        stopStormPlayback();
        return;
      }

      isStormPlaying = true;
      if (playIcon) playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
      if (playBtnText) playBtnText.innerText = 'Pause Event Simulation';
      if (stormScrubberContainer) stormScrubberContainer.style.display = 'flex';

      try {
        const resp = await fetch('/api/storm-playback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            region: activeRegion,
            total_duration_min: 60,
            max_rainfall_mmhr: 150
          })
        });

        const data = await resp.json();
        if (data.status !== 'success') {
          stopStormPlayback();
          return;
        }

        const frames = data.frames;
        let frameIdx = 0;

        function renderFrame(idx) {
          const f = frames[idx];
          if (!f) return;

          // Update hyetograph scrubber UI
          const pct = ((idx + 1) / frames.length) * 100;
          if (stormProgressFill) stormProgressFill.style.width = `${pct}%`;
          if (stormTimeReadout) stormTimeReadout.innerText = `T+${f.time_min} min`;
          if (stormRainReadout) stormRainReadout.innerText = `${f.rainfall_mmhr} mm/hr`;
          if (stormFrameReadout) stormFrameReadout.innerText = `Step ${idx + 1}/${frames.length}`;

          // Update nodes depths
          const depthMap = new Map();
          const frameNodesList = f.nodes || f.node_depths || [];
          frameNodesList.forEach(d => depthMap.set(String(d.id), d.gnn_depth !== undefined ? d.gnn_depth : d.depth));

          nodesData.forEach(n => {
            if (depthMap.has(String(n.id))) {
              n.gnn_depth = depthMap.get(String(n.id));
            }
          });

          // Live KPIs
          document.getElementById('kpiMaxDepth').innerText = `${f.max_depth.toFixed(2)} m`;
          const latEl = document.getElementById('headerLatencyVal');
          if (latEl) latEl.innerText = `${f.inference_ms} ms`;
          const rightLat = document.getElementById('rightModelLatency');
          if (rightLat) rightLat.innerText = `${f.inference_ms} ms`;
          const floodedPct = ((f.flooded_count / nodesData.length) * 100).toFixed(1);
          document.getElementById('kpiFloodedPct').innerText = `${floodedPct}%`;
          const barPctEl = document.getElementById('kpiFloodedBarPct');
          if (barPctEl) barPctEl.innerText = `${floodedPct}%`;
          document.getElementById('kpiProgressBar').style.width = `${floodedPct}%`;

          requestRender();
        }

        renderFrame(0);
        frameIdx = 1;

        stormPlaybackInterval = setInterval(() => {
          if (frameIdx >= frames.length) {
            stopStormPlayback();
            return;
          }
          renderFrame(frameIdx);
          frameIdx++;
        }, 1200);

      } catch (err) {
        console.error('Storm playback error:', err);
        stopStormPlayback();
      }
    });
  }

  // =========================================================================
  // HISTORICAL EVENT REPLAY PRESETS
  // =========================================================================
  document.querySelectorAll('.preset-card-btn[data-preset]').forEach(btn => {
    btn.addEventListener('click', () => {
      const preset = btn.getAttribute('data-preset');
      if (preset === 'bengaluru_oct19') {
        regionSelect.value = 'hsr';
        rainSlider.value = 105;
        durationSlider.value = 90;
        rainReadout.innerText = '105 mm/hr';
        durationReadout.innerText = '90 min';
        loadGraphData('hsr').then(() => {
          toggleIncidentsDisplay(true);
        });
      } else if (preset === 'typhoon_saola') {
        regionSelect.value = 'hongkong';
        rainSlider.value = 140;
        durationSlider.value = 60;
        rainReadout.innerText = '140 mm/hr';
        durationReadout.innerText = '60 min';
        toggleIncidentsDisplay(false);
        loadGraphData('hongkong');
      } else if (preset === 'tokyo_cloudburst') {
        regionSelect.value = 'tokyo';
        rainSlider.value = 100;
        durationSlider.value = 60;
        rainReadout.innerText = '100 mm/hr';
        durationReadout.innerText = '60 min';
        toggleIncidentsDisplay(false);
        loadGraphData('tokyo');
      }
    });
  });

  // =========================================================================
  // DYNAMIC RESIZE SPLITTERS (SMOOTH SIDE PANEL RESIZING)
  // =========================================================================
  const workspace = document.querySelector('.command-workspace');
  let leftWidth = 340;
  let rightWidth = 380;

  function updateWorkspaceGrid() {
    if (workspace) {
      workspace.style.gridTemplateColumns = `${leftWidth}px 6px 1fr 6px ${rightWidth}px`;
      resizeCanvas();
    }
  }

  function setupSplitter(splitterEl, isLeft) {
    if (!splitterEl) return;
    let isDragging = false;

    splitterEl.addEventListener('mousedown', (e) => {
      isDragging = true;
      splitterEl.classList.add('dragging');
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      if (isLeft) {
        leftWidth = Math.max(220, Math.min(500, e.clientX));
      } else {
        rightWidth = Math.max(260, Math.min(550, window.innerWidth - e.clientX));
      }
      updateWorkspaceGrid();
    });

    window.addEventListener('mouseup', () => {
      if (isDragging) {
        isDragging = false;
        splitterEl.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        resizeCanvas();
      }
    });
  }

  setupSplitter(leftSplitter, true);
  setupSplitter(rightSplitter, false);

  // =========================================================================
  // INITIALIZE FIRST LOAD (HSR LAYOUT)
  // =========================================================================
  loadGraphData('hsr');
  setTimeout(resizeCanvas, 250);
});
