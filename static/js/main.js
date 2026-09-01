/* ==========================================================================
   UrbanFlow - High-Performance Digital Twin Engine
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  let nodesData = [];
  let edgesData = [];
  let bounds = { min_lat: 12.9, max_lat: 12.95, min_lng: 77.6, max_lng: 77.7 };

  let activeViewMode = 'depth';
  let currentMetrics = {};
  let currentSoilMoisture = 'dry';
  let swmmComparisonData = null;
  let showSwmmOverlay = false;

  let hoveredNode = null;
  let selectedNode = null;
  let renderRequested = false;
  let flowAnimationProgress = 0;
  let pulseTime = 0;
  let stormPlaybackInterval = null;

  const canvas = document.getElementById('mapCanvas');
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('nodeTooltip');

  const rainSlider = document.getElementById('rainSlider');
  const rainValue = document.getElementById('rainValue');
  const durationSlider = document.getElementById('durationSlider');
  const durationValue = document.getElementById('durationValue');
  const runSimBtn = document.getElementById('runSimBtn');

  const bioswaleSlider = document.getElementById('bioswaleSlider');
  const bioswaleValue = document.getElementById('bioswaleValue');
  const drainSlider = document.getElementById('drainSlider');
  const drainValue = document.getElementById('drainValue');
  const gardenSlider = document.getElementById('gardenSlider');
  const gardenValue = document.getElementById('gardenValue');
  const applyMitigationBtn = document.getElementById('applyMitigationBtn');

  const toolBtns = document.querySelectorAll('.tool-btn[data-mode]');
  const resetMapBtn = document.getElementById('resetMapBtn');
  const toggleTileBtn = document.getElementById('toggleTileBtn');

  const soilBtns = document.querySelectorAll('.soil-btn');
  const playStormBtn = document.getElementById('playStormBtn');
  const stormProfileSelect = document.getElementById('stormProfileSelect');
  const stormProgress = document.getElementById('stormProgress');
  const stormProgressFill = document.getElementById('stormProgressFill');
  const stormTimeLabel = document.getElementById('stormTimeLabel');
  const stormRainLabel = document.getElementById('stormRainLabel');
  const stormFrameLabel = document.getElementById('stormFrameLabel');

  let benchmarkChart = null;
  let nodeHydrographChart = null;

  const inspectorPanel = document.getElementById('nodeInspectorPanel');
  const closeInspectorBtn = document.getElementById('closeInspectorBtn');

  if (closeInspectorBtn) {
    closeInspectorBtn.addEventListener('click', () => {
      if (inspectorPanel) inspectorPanel.classList.remove('open');
      selectedNode = null;
      requestRender();
    });
  }

  // Initialize Leaflet Map
  const leafletMap = L.map('leafletMap', {
    zoomControl: false,
    attributionControl: false,
    preferCanvas: true
  }).setView([12.9116, 77.6389], 14);

  // Basemap Tiles
  const lightTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
    maxZoom: 19, subdomains: 'abcd'
  });

  const satelliteTiles = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19
  });

  const darkTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19, subdomains: 'abcd'
  });

  // Default to Street Vector Basemap (Earthy Bright Eco-Tech)
  lightTiles.addTo(leafletMap);
  let currentTileMode = 'light';

  if (toggleTileBtn) {
    toggleTileBtn.innerText = 'Basemap: Street Vector';
    toggleTileBtn.addEventListener('click', () => {
      if (currentTileMode === 'light') {
        leafletMap.removeLayer(lightTiles);
        satelliteTiles.addTo(leafletMap);
        currentTileMode = 'satellite';
        toggleTileBtn.innerText = 'Basemap: Satellite';
      } else if (currentTileMode === 'satellite') {
        leafletMap.removeLayer(satelliteTiles);
        darkTiles.addTo(leafletMap);
        currentTileMode = 'dark';
        toggleTileBtn.innerText = 'Basemap: Cyber Dark';
      } else {
        leafletMap.removeLayer(darkTiles);
        lightTiles.addTo(leafletMap);
        currentTileMode = 'light';
        toggleTileBtn.innerText = 'Basemap: Street Vector';
      }
      requestRender();
    });
  }

  function stopStormPlayback() {
    if (stormPlaybackInterval) {
      clearInterval(stormPlaybackInterval);
      stormPlaybackInterval = null;
    }
    if (playStormBtn) {
      playStormBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Play Storm Event (60-min timeline)`;
      playStormBtn.disabled = false;
    }
    if (stormProgress) stormProgress.style.display = 'none';
  }

  function requestRender() {
    if (!renderRequested) {
      renderRequested = true;
      requestAnimationFrame(() => {
        drawMap();
        renderRequested = false;
      });
    }
  }

  // HiDPI Retina Canvas Sizing
  function resizeCanvas() {
    const parent = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = parent.clientWidth * dpr;
    canvas.height = parent.clientHeight * dpr;
    leafletMap.invalidateSize();
    requestRender();
  }

  window.addEventListener('resize', resizeCanvas);
  leafletMap.on('move zoom resize viewreset', requestRender);

  const regionSelect = document.getElementById('regionSelect');
  if (regionSelect) {
    regionSelect.addEventListener('change', () => {
      stopStormPlayback();
      loadGraphData(regionSelect.value);
    });
  }

  async function loadGraphData(regionKey) {
    try {
      const resp = await fetch(`/api/graph-data?region=${regionKey}`);
      const data = await resp.json();
      if (data.status !== 'success') return;

      nodesData = data.nodes;
      edgesData = data.edges;
      bounds = data.bounds;

      swmmComparisonData = null;
      showSwmmOverlay = false;

      if (bounds.min_lat && bounds.max_lat) {
        leafletMap.fitBounds([
          [bounds.min_lat, bounds.min_lng],
          [bounds.max_lat, bounds.max_lng]
        ], { padding: [40, 40] });
      }

      if (!benchmarkChart) initCharts();

      const riskNodes = nodesData
        .filter(n => (n.gnn_depth || 0) > 0.08)
        .sort((a, b) => (b.gnn_depth || 0) - (a.gnn_depth || 0));
      populateRiskTable(riskNodes);
      requestRender();
    } catch (err) {
      console.error('Failed to load graph data:', err);
    }
  }

  function resetCamera() {
    if (bounds.min_lat && bounds.max_lat) {
      leafletMap.fitBounds([
        [bounds.min_lat, bounds.min_lng],
        [bounds.max_lat, bounds.max_lng]
      ], { padding: [40, 40] });
    }
    requestRender();
  }

  if (resetMapBtn) resetMapBtn.addEventListener('click', resetCamera);

  function getNodeColor(node) {
    if (activeViewMode === 'depth') {
      const d = node.gnn_depth || 0;
      if (d < 0.02) return '#059669';
      if (d < 0.15) return '#22C55E';
      if (d < 0.30) return '#F59E0B';
      if (d < 0.50) return '#EA580C';
      return '#DC2626';
    } else if (activeViewMode === 'elevation') {
      const elev = node.elevation || 880;
      const norm = Math.min(1, Math.max(0, (elev - 872) / 25));
      const r = Math.round(146 * (1 - norm) + 5 * norm);
      const g = Math.round(64 * (1 - norm) + 150 * norm);
      const b = Math.round(14 * (1 - norm) + 105 * norm);
      return `rgb(${r},${g},${b})`;
    } else if (activeViewMode === 'impervious') {
      const imp = node.impervious_ratio || 0.2;
      const r = Math.round(2 + 240 * imp);
      const g = Math.round(132 * (1 - imp));
      const b = Math.round(199 * (1 - imp));
      return `rgb(${r},${g},${b})`;
    }
    return '#0284C7';
  }

  function getNodeRadius(node, baseRadius) {
    if (activeViewMode === 'depth') {
      const d = node.gnn_depth || 0;
      if (d >= 0.30) return baseRadius * 2.0;
      if (d >= 0.15) return baseRadius * 1.5;
    }
    return baseRadius;
  }

  function drawMap() {
    if (!ctx || nodesData.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    if (currentTileMode === 'light') {
      ctx.strokeStyle = '#CBD5E1';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      for (let x = 0; x < w; x += 20) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
      for (let y = 0; y < h; y += 20) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
      ctx.stroke();
    }

    const nodeCoordMap = new Map();
    const visibleNodes = [];

    for (let i = 0; i < nodesData.length; i++) {
      const n = nodesData[i];
      let px = 0, py = 0;
      if (n.lat && n.lng) {
        const pt = leafletMap.latLngToContainerPoint([n.lat, n.lng]);
        px = pt.x; py = pt.y;
      }
      nodeCoordMap.set(n.id, { x: px, y: py, node: n });
      if (px >= -20 && px <= w + 20 && py >= -20 && py <= h + 20) {
        visibleNodes.push({ node: n, x: px, y: py });
      }
    }

    const zoomLevel = leafletMap.getZoom();
    const edgeWidth = Math.max(1.5, Math.min(3.5, (zoomLevel - 11) * 0.8));

    // Draw Graph Edges
    ctx.beginPath();
    ctx.strokeStyle = currentTileMode === 'satellite' ? 'rgba(255,255,255,0.4)' : (currentTileMode === 'dark' ? 'rgba(148, 163, 184, 0.25)' : 'rgba(15,23,42,0.35)');
    ctx.lineWidth = edgeWidth;
    for (let i = 0; i < edgesData.length; i++) {
      const e = edgesData[i];
      const uPos = nodeCoordMap.get(e.u);
      const vPos = nodeCoordMap.get(e.v);
      if (uPos && vPos) {
        if ((uPos.x >= -30 && uPos.x <= w + 30 && uPos.y >= -30 && uPos.y <= h + 30) ||
            (vPos.x >= -30 && vPos.x <= w + 30 && vPos.y >= -30 && vPos.y <= h + 30)) {
          ctx.moveTo(uPos.x, uPos.y);
          ctx.lineTo(vPos.x, vPos.y);
        }
      }
    }
    ctx.stroke();

    // Flow direction vectors along edges
    if (activeViewMode === 'depth' && nodesData.some(n => (n.gnn_depth || 0) > 0.02)) {
      for (let i = 0; i < edgesData.length; i++) {
        const e = edgesData[i];
        const uNode = nodesData.find(n => n.id === e.u);
        const vNode = nodesData.find(n => n.id === e.v);
        if (!uNode || !vNode) continue;
        const ud = uNode.gnn_depth || 0;
        const vd = vNode.gnn_depth || 0;
        if (ud < 0.02 && vd < 0.02) continue;

        const uPos = nodeCoordMap.get(e.u);
        const vPos = nodeCoordMap.get(e.v);
        if (!uPos || !vPos) continue;

        const screenDx = vPos.x - uPos.x;
        const screenDy = vPos.y - uPos.y;
        const screenLen = Math.sqrt(screenDx * screenDx + screenDy * screenDy);
        if (screenLen < 12) continue;

        const animOffset = (flowAnimationProgress % 1.0) * screenLen;
        const arrowX = uPos.x + (screenDx / screenLen) * animOffset;
        const arrowY = uPos.y + (screenDy / screenLen) * animOffset;
        const angle = Math.atan2(screenDy, screenDx);
        const arrowSize = 6;

        ctx.save();
        ctx.translate(arrowX, arrowY);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.moveTo(arrowSize, 0);
        ctx.lineTo(-arrowSize, -arrowSize * 0.6);
        ctx.lineTo(-arrowSize, arrowSize * 0.6);
        ctx.closePath();
        ctx.fillStyle = ud > 0.15 ? 'rgba(220,38,38,0.5)' : 'rgba(5,150,105,0.45)';
        ctx.fill();
        ctx.restore();
      }
    }

    const baseRadius = Math.max(3.0, (zoomLevel - 11) * 1.5);
    const colorGroups = new Map();
    const highlightNodes = [];

    for (let i = 0; i < visibleNodes.length; i++) {
      const vn = visibleNodes[i];
      const n = vn.node;
      const isHovered = hoveredNode && String(hoveredNode.id) === String(n.id);
      const isSelected = selectedNode && String(selectedNode.id) === String(n.id);
      const d = n.gnn_depth || 0;
      const isHazard = activeViewMode === 'depth' && d >= 0.15;

      if (isHovered || isSelected || isHazard) {
        highlightNodes.push(vn);
        continue;
      }

      const color = getNodeColor(n);
      if (!colorGroups.has(color)) colorGroups.set(color, []);
      colorGroups.get(color).push(vn);
    }

    ctx.shadowBlur = 0;
    ctx.strokeStyle = currentTileMode === 'satellite' ? '#FFFFFF' : '#0F172A';
    ctx.lineWidth = 1.0;

    colorGroups.forEach((ptList, color) => {
      ctx.beginPath();
      ctx.fillStyle = color;
      for (let i = 0; i < ptList.length; i++) {
        const pt = ptList[i];
        ctx.moveTo(pt.x + baseRadius, pt.y);
        ctx.arc(pt.x, pt.y, baseRadius, 0, 2 * Math.PI);
      }
      ctx.fill();
      ctx.stroke();
    });

    for (let i = 0; i < highlightNodes.length; i++) {
      const vn = highlightNodes[i];
      const n = vn.node;
      const isHovered = hoveredNode && String(hoveredNode.id) === String(n.id);
      const isSelected = selectedNode && String(selectedNode.id) === String(n.id);
      const d = n.gnn_depth || 0;
      const radius = getNodeRadius(n, isHovered || isSelected ? baseRadius * 2.2 : baseRadius * 1.4);
      const color = getNodeColor(n);

      if (d >= 0.30 && activeViewMode === 'depth') {
        const pulseScale = 1.0 + 0.4 * Math.sin(pulseTime * 3.0 + i * 0.5);
        ctx.beginPath();
        ctx.arc(vn.x, vn.y, radius * 2.5 * pulseScale, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(220, 38, 38, 0.20)';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(vn.x, vn.y, radius * 3.5 * pulseScale, 0, 2 * Math.PI);
        ctx.strokeStyle = `rgba(220, 38, 38, ${0.15 + 0.1 * Math.sin(pulseTime * 4.0 + i)})`;
        ctx.lineWidth = 2.0;
        ctx.stroke();
      } else if (d >= 0.15 && activeViewMode === 'depth') {
        ctx.beginPath();
        ctx.arc(vn.x, vn.y, radius * 2.0, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(220, 38, 38, 0.15)';
        ctx.fill();
      }

      if (isSelected) {
        ctx.beginPath();
        ctx.arc(vn.x, vn.y, radius * 3.2, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(2, 132, 199, 0.35)';
        ctx.fill();
        ctx.strokeStyle = '#0284C7';
        ctx.lineWidth = 2.0;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(vn.x, vn.y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = isHovered || isSelected ? 16 : (d >= 0.30 ? 12 : 6);
      ctx.fill();

      if (isHovered || isSelected) {
        ctx.strokeStyle = isSelected ? '#0284C7' : '#FFFFFF';
        ctx.lineWidth = 3.0;
        ctx.stroke();
      }
    }

    if (showSwmmOverlay && swmmComparisonData) {
      for (let i = 0; i < swmmComparisonData.length; i++) {
        const sc = swmmComparisonData[i];
        const pos = nodeCoordMap.get(sc.id);
        if (!pos || pos.x < -20 || pos.x > w + 20 || pos.y < -20 || pos.y > h + 20) continue;
        const swmmR = Math.max(3, baseRadius * 1.2);
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, swmmR, 0, 2 * Math.PI);
        ctx.strokeStyle = '#0ea5e9';
        ctx.lineWidth = 2.0;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    ctx.restore();

    requestAnimationFrame(() => {
      pulseTime += 0.016;
      flowAnimationProgress = (flowAnimationProgress + 0.008) % 1.0;
      const hasAnimatedNodes = activeViewMode === 'depth' && nodesData.some(n => (n.gnn_depth || 0) >= 0.15);
      if (hasAnimatedNodes) requestRender();
    });
  }

  function openNodeInspector(node) {
    if (!node || !inspectorPanel) return;
    selectedNode = node;

    document.getElementById('inspNodeTitle').innerText = `Node #${node.id}`;
    document.getElementById('inspElev').innerText = `${node.elevation.toFixed(1)} m`;
    document.getElementById('inspSlope').innerText = `${((node.upstream_slope || 0) * 100).toFixed(1)} %`;
    document.getElementById('inspImp').innerText = `${((node.impervious_ratio || 0.2) * 100).toFixed(1)} %`;
    document.getElementById('inspMann').innerText = `${(node.manning_n || 0.013).toFixed(3)}`;
    document.getElementById('inspGnnDepth').innerText = `${(node.gnn_depth || 0).toFixed(3)} m`;

    const d = node.gnn_depth || 0;
    const riskEl = document.getElementById('inspRisk');
    if (d >= 0.30) { riskEl.innerText = 'CRITICAL'; riskEl.style.color = '#DC2626'; }
    else if (d >= 0.15) { riskEl.innerText = 'ADVISORY'; riskEl.style.color = '#f59e0b'; }
    else if (d >= 0.05) { riskEl.innerText = 'WATCH'; riskEl.style.color = '#22c55e'; }
    else { riskEl.innerText = 'SAFE'; riskEl.style.color = '#059669'; }

    const recText = document.getElementById('inspRecText');
    const recCard = document.getElementById('inspRecCard');
    if (d > 0.30) {
      recText.innerText = `CRITICAL HAZARD: Deploy mobile dewatering sump pump. Activate automated underpass barrier gates. Close road access immediately.`;
      recCard.style.borderColor = '#dc2626';
      recCard.style.background = 'rgba(239,68,68,0.12)';
    } else if (d > 0.15) {
      recText.innerText = `ADVISORY: Dispatch drain maintenance crew. Clear debris from storm drain inlets. Issue RWA advisory.`;
      recCard.style.borderColor = '#f59e0b';
      recCard.style.background = 'rgba(245,158,11,0.12)';
    } else if (d > 0.05) {
      recText.innerText = `WATCH: Monitor water levels. De-silt stormwater catch basin.`;
      recCard.style.borderColor = '#22c55e';
      recCard.style.background = 'rgba(34,197,94,0.12)';
    } else {
      recText.innerText = `SAFE: Low surface runoff. Standard routine maintenance.`;
      recCard.style.borderColor = '#059669';
      recCard.style.background = 'rgba(5,150,105,0.12)';
    }

    renderNodeHydrograph(d);
    inspectorPanel.classList.add('open');
    requestRender();
  }

  function renderNodeHydrograph(peakDepth) {
    const el = document.getElementById('nodeHydrographChart');
    if (!el) return;
    if (nodeHydrographChart) nodeHydrographChart.destroy();

    const times = ['0m', '15m', '30m', '45m', '60m', '75m', '90m', '105m', '120m'];
    const curve = [
      0.0,
      round(peakDepth * 0.2, 3),
      round(peakDepth * 0.55, 3),
      round(peakDepth * 0.88, 3),
      round(peakDepth, 3),
      round(peakDepth * 0.72, 3),
      round(peakDepth * 0.45, 3),
      round(peakDepth * 0.20, 3),
      round(peakDepth * 0.05, 3)
    ];

    nodeHydrographChart = new Chart(el.getContext('2d'), {
      type: 'line',
      data: {
        labels: times,
        datasets: [{
          label: 'Depth (m)',
          data: curve,
          borderColor: '#06b6d4',
          backgroundColor: 'rgba(6, 182, 212, 0.15)',
          fill: true,
          tension: 0.35,
          pointRadius: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { grid: { color: '#E2E8F0' }, ticks: { color: '#64748B', font: { size: 9 } } },
          x: { grid: { display: false }, ticks: { color: '#64748B', font: { size: 9 } } }
        },
        plugins: { legend: { display: false } }
      }
    });
  }

  function round(val, decimals) {
    return Number(Math.round(val + 'e' + decimals) + 'e-' + decimals);
  }

  let lastMouseMoveTime = 0;
  window.addEventListener('mousemove', (e) => {
    const now = performance.now();
    if (now - lastMouseMoveTime < 16) return;
    lastMouseMoveTime = now;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (mouseX < 0 || mouseX > rect.width || mouseY < 0 || mouseY > rect.height) {
      if (hoveredNode) {
        hoveredNode = null;
        tooltip.style.display = 'none';
        requestRender();
      }
      return;
    }

    let found = null;
    const hitRadius = 14;

    for (let i = 0; i < nodesData.length; i++) {
      const n = nodesData[i];
      if (n.lat && n.lng) {
        const pt = leafletMap.latLngToContainerPoint([n.lat, n.lng]);
        if (Math.hypot(pt.x - mouseX, pt.y - mouseY) < hitRadius) {
          found = n;
          break;
        }
      }
    }

    if (found !== hoveredNode) {
      hoveredNode = found;
      requestRender();
      if (hoveredNode) {
        showTooltip(hoveredNode, e.clientX, e.clientY);
      } else {
        tooltip.style.display = 'none';
      }
    } else if (hoveredNode) {
      updateTooltipPosition(e.clientX, e.clientY);
    }
  });

  document.getElementById('leafletMap').addEventListener('click', () => {
    if (hoveredNode) openNodeInspector(hoveredNode);
  });

  function showTooltip(n, x, y) {
    document.getElementById('ttNodeId').innerText = `Node #${n.id}`;
    document.getElementById('ttElev').innerText = `${n.elevation.toFixed(1)} m`;
    document.getElementById('ttSlope').innerText = `${((n.upstream_slope || 0) * 100).toFixed(1)} %`;
    const depth = n.gnn_depth || 0;
    document.getElementById('ttGnnDepth').innerText = `${depth.toFixed(3)} m`;
    document.getElementById('ttGnnDepthCm').innerText = `${(depth * 100).toFixed(1)} cm`;
    document.getElementById('ttSwmmDepth').innerText = `${(n.swmm_depth || 0).toFixed(3)} m`;

    const riskEl = document.getElementById('ttRisk');
    if (depth >= 0.30) { riskEl.innerText = 'CRITICAL'; riskEl.style.color = '#DC2626'; }
    else if (depth >= 0.15) { riskEl.innerText = 'ADVISORY'; riskEl.style.color = '#f59e0b'; }
    else if (depth >= 0.05) { riskEl.innerText = 'WATCH'; riskEl.style.color = '#22c55e'; }
    else { riskEl.innerText = 'SAFE'; riskEl.style.color = '#059669'; }

    tooltip.style.display = 'block';
    updateTooltipPosition(x, y);
  }

  function updateTooltipPosition(x, y) {
    const offsetX = 12, offsetY = 12;
    const tooltipWidth = tooltip.offsetWidth || 210;
    const tooltipHeight = tooltip.offsetHeight || 140;
    let posX = x + offsetX;
    let posY = y + offsetY;
    if (posX + tooltipWidth > window.innerWidth - 12) posX = x - tooltipWidth - offsetX;
    if (posY + tooltipHeight > window.innerHeight - 12) posY = y - tooltipHeight - offsetY;
    tooltip.style.left = `${Math.max(8, posX)}px`;
    tooltip.style.top = `${Math.max(8, posY)}px`;
  }

  rainSlider.addEventListener('input', () => { rainValue.innerText = `${rainSlider.value} mm/hr`; });
  durationSlider.addEventListener('input', () => { durationValue.innerText = `${durationSlider.value} min`; });
  bioswaleSlider.addEventListener('input', () => { bioswaleValue.innerText = `${bioswaleSlider.value}%`; });
  drainSlider.addEventListener('input', () => { drainValue.innerText = `${drainSlider.value}%`; });
  gardenSlider.addEventListener('input', () => { gardenValue.innerText = `${gardenSlider.value}%`; });

  soilBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      soilBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentSoilMoisture = btn.getAttribute('data-moisture');
    });
  });

  toolBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      toolBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeViewMode = btn.getAttribute('data-mode');
      const titleMap = {
        'depth': 'Water Depth Inundation (Meters)',
        'elevation': 'Digital Elevation Model (Meters)',
        'impervious': 'Built-Up Impervious Ratio (%)'
      };
      document.getElementById('legendTitle').innerText = titleMap[activeViewMode] || 'Inundation Depth';
      requestRender();
    });
  });

  const swmmCompareBtn = document.getElementById('swmmCompareBtn');
  if (swmmCompareBtn) {
    swmmCompareBtn.addEventListener('click', async () => {
      if (showSwmmOverlay) {
        showSwmmOverlay = false;
        swmmCompareBtn.style.background = 'rgba(2,132,199,0.1)';
        requestRender();
        return;
      }

      swmmCompareBtn.innerText = 'Computing SWMM...';
      try {
        const resp = await fetch('/api/swmm-compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            region: regionSelect ? regionSelect.value : 'hsr',
            rainfall_mmhr: parseFloat(rainSlider.value),
            duration_min: parseFloat(durationSlider.value),
            soil_moisture: currentSoilMoisture
          })
        });
        const data = await resp.json();
        if (data.status === 'success') {
          swmmComparisonData = data.nodes;
          showSwmmOverlay = true;
          swmmCompareBtn.style.background = 'rgba(2,132,199,0.3)';
          swmmCompareBtn.innerText = `GNN vs SWMM (MAE: ${data.metrics.mae_m.toFixed(4)}m)`;
          requestRender();
        }
      } catch (err) {
        console.error('SWMM compare error:', err);
        swmmCompareBtn.innerText = 'GNN vs SWMM';
      }
    });
  }

  runSimBtn.addEventListener('click', async () => {
    stopStormPlayback();
    runSimBtn.innerText = 'Evaluating PINN Graph...';
    runSimBtn.disabled = true;

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          region: regionSelect ? regionSelect.value : 'hsr',
          rainfall_mmhr: parseFloat(rainSlider.value),
          duration_min: parseFloat(durationSlider.value),
          soil_moisture: currentSoilMoisture
        })
      });

      const data = await resp.json();
      if (data.status === 'success') {
        bioswaleSlider.value = 0;
        bioswaleValue.innerText = '0%';
        drainSlider.value = 0;
        drainValue.innerText = '0%';
        gardenSlider.value = 0;
        gardenValue.innerText = '0%';

        updateSimulationResults(data);
      }
    } catch (err) {
      console.error('Simulation error:', err);
    } finally {
      runSimBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        Run Neural Digital Twin`;
      runSimBtn.disabled = false;
    }
  });

  if (playStormBtn) {
    playStormBtn.addEventListener('click', async () => {
      if (stormPlaybackInterval) {
        stopStormPlayback();
        return;
      }

      playStormBtn.innerText = 'Loading Storm Frames...';
      playStormBtn.disabled = true;

      try {
        const resp = await fetch('/api/storm-playback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            region: regionSelect ? regionSelect.value : 'hsr',
            total_duration_min: 60.0,
            max_rainfall_mmhr: parseFloat(stormProfileSelect.value),
            soil_moisture: currentSoilMoisture
          })
        });
        const data = await resp.json();
        if (data.status !== 'success') {
          playStormBtn.innerText = 'Play Storm Event (60-min timeline)';
          playStormBtn.disabled = false;
          return;
        }

        stormProgress.style.display = 'block';
        playStormBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg> Stop Storm Playback`;
        playStormBtn.disabled = false;

        let frameIdx = 0;
        const frames = data.frames;

        function applyFrame(idx) {
          const frame = frames[idx];
          stormProgressFill.style.width = `${((idx + 1) / frames.length) * 100}%`;
          stormTimeLabel.innerText = `${frame.time_min} min`;
          stormRainLabel.innerText = `${frame.rainfall_mmhr} mm/hr`;
          stormFrameLabel.innerText = `Frame ${idx + 1}/${frames.length}`;

          const depthMap = new Map();
          frame.nodes.forEach(n => depthMap.set(n.id, n.gnn_depth));

          nodesData.forEach(n => {
            if (depthMap.has(n.id)) {
              n.gnn_depth = depthMap.get(n.id);
            }
          });

          let maxDepth = 0;
          let depthSum = 0;
          let floodedCount = 0;
          let basementRisk = 0;

          nodesData.forEach(n => {
            const d = n.gnn_depth || 0;
            maxDepth = Math.max(maxDepth, d);
            depthSum += d;
            if (d > 0.05) floodedCount++;
            if (n.is_basement && d > 0.12) basementRisk++;
          });

          const avgDepth = depthSum / nodesData.length;
          document.getElementById('maxDepthVal').innerText = `${maxDepth.toFixed(3)} m`;
          document.getElementById('avgDepthSub').innerText = `Avg: ${avgDepth.toFixed(4)} m`;
          document.getElementById('volVal').innerText = `${Math.round(depthSum * 500.0).toLocaleString()} m3`;
          document.getElementById('floodedNodesSub').innerText = `Flooded Nodes: ${floodedCount} / ${nodesData.length}`;
          document.getElementById('basementAlertVal').innerText = `${basementRisk} Complex`;
          document.getElementById('floodedPctVal').innerText = `${(100.0 * floodedCount / nodesData.length).toFixed(1)}%`;
          document.getElementById('floodedCountSub').innerText = `${floodedCount} / ${nodesData.length} nodes inundated`;

          const riskNodes = nodesData
            .filter(n => (n.gnn_depth || 0) > 0.08)
            .sort((a, b) => (b.gnn_depth || 0) - (a.gnn_depth || 0));
          populateRiskTable(riskNodes);
          requestRender();
        }

        applyFrame(0);
        frameIdx = 1;

        stormPlaybackInterval = setInterval(() => {
          if (frameIdx >= frames.length) {
            stopStormPlayback();
            return;
          }
          applyFrame(frameIdx);
          frameIdx++;
        }, 1800);

      } catch (err) {
        console.error('Storm playback error:', err);
        stopStormPlayback();
      }
    });
  }

  applyMitigationBtn.addEventListener('click', async () => {
    applyMitigationBtn.innerText = 'Recalculating...';
    applyMitigationBtn.disabled = true;

    try {
      const resp = await fetch('/api/mitigation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          region: regionSelect ? regionSelect.value : 'hsr',
          rainfall_mmhr: parseFloat(rainSlider.value),
          bioswales_pct: parseFloat(bioswaleSlider.value),
          drain_cleaning_pct: parseFloat(drainSlider.value),
          rain_gardens_pct: parseFloat(gardenSlider.value)
        })
      });

      const data = await resp.json();
      if (data.status === 'success') {
        const depthMap = new Map();
        data.nodes.forEach(n => depthMap.set(n.id, n.mitigated_depth));
        nodesData.forEach(n => { if (depthMap.has(n.id)) n.gnn_depth = depthMap.get(n.id); });

        let maxDepth = 0, depthSum = 0, floodedNodesCount = 0, basementRiskCount = 0;
        nodesData.forEach(n => {
          const d = n.gnn_depth || 0;
          maxDepth = Math.max(maxDepth, d);
          depthSum += d;
          if (d > 0.05) floodedNodesCount++;
          if (n.is_basement && d > 0.12) basementRiskCount++;
        });

        const avgDepth = depthSum / nodesData.length;
        document.getElementById('maxDepthVal').innerText = `${maxDepth.toFixed(3)} m`;
        document.getElementById('avgDepthSub').innerText = `Avg: ${avgDepth.toFixed(4)} m`;
        document.getElementById('volVal').innerText = `${Math.round(depthSum * 500.0).toLocaleString()} m3`;
        document.getElementById('floodedNodesSub').innerText = `Flooded Nodes: ${floodedNodesCount} / ${nodesData.length}`;
        document.getElementById('basementAlertVal').innerText = `${basementRiskCount} Complex`;
        document.getElementById('floodedPctVal').innerText = `${(100.0 * floodedNodesCount / nodesData.length).toFixed(1)}%`;
        document.getElementById('floodedCountSub').innerText = `${floodedNodesCount} / ${nodesData.length} nodes inundated`;

        const riskNodes = nodesData.filter(n => (n.gnn_depth || 0) > 0.08)
          .sort((a, b) => (b.gnn_depth || 0) - (a.gnn_depth || 0));
        requestRender();
        populateRiskTable(riskNodes);
      }
    } catch (err) {
      console.error('Mitigation error:', err);
    } finally {
      applyMitigationBtn.innerText = 'Apply NBS & Recalculate Runoff';
      applyMitigationBtn.disabled = false;
    }
  });

  function updateSimulationResults(data) {
    const m = data.metrics;
    currentMetrics = m;

    document.getElementById('gnnTimeVal').innerText = `${m.gnn_time_ms} ms`;
    document.getElementById('inferLatVal').innerText = `${m.gnn_time_ms} ms`;
    document.getElementById('speedupVal').innerText = `${m.speedup_ratio}x`;
    document.getElementById('speedupHeaderVal').innerText = `~${m.speedup_ratio}x`;
    document.getElementById('speedupSub').innerText = `vs SWMM: ${(m.swmm_time_ms / 1000).toFixed(1)}s`;

    document.getElementById('maxDepthVal').innerText = `${m.max_depth_m} m`;
    document.getElementById('avgDepthSub').innerText = `Avg: ${m.avg_depth_m} m`;
    document.getElementById('volVal').innerText = `${m.total_volume_m3.toLocaleString()} m3`;
    document.getElementById('floodedNodesSub').innerText = `Flooded Nodes: ${m.total_flooded_nodes} / ${m.total_nodes}`;
    document.getElementById('floodedPctVal').innerText = `${m.flooded_pct}%`;
    document.getElementById('floodedCountSub').innerText = `${m.total_flooded_nodes} / ${m.total_nodes} nodes inundated`;
    document.getElementById('basementAlertVal').innerText = `${m.basement_risk_count} Complex`;

    const depthMap = new Map();
    data.nodes.forEach(n => depthMap.set(n.id, n));
    nodesData.forEach(n => {
      if (depthMap.has(n.id)) {
        const r = depthMap.get(n.id);
        n.gnn_depth = r.gnn_depth;
        n.swmm_depth = r.swmm_depth;
      }
    });

    populateRiskTable(data.risk_nodes);
    populateDispatch(data.dispatch_recommendations || []);
    populateAlert(data.rwa_alert || '');
    updateBenchmarkChart(m.gnn_time_ms, m.swmm_time_ms);
    requestRender();
  }

  function populateRiskTable(riskNodes) {
    const tbody = document.getElementById('riskTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!riskNodes || riskNodes.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="table-empty">No high-risk inundated junctions detected at current threshold.</td></tr>';
      return;
    }

    riskNodes.slice(0, 30).forEach(n => {
      const tr = document.createElement('tr');
      const d = n.gnn_depth || 0;
      let badgeClass = 'badge-safe', riskTxt = 'Safe';
      if (d >= 0.30) { badgeClass = 'badge-severe'; riskTxt = 'Critical'; }
      else if (d >= 0.15) { badgeClass = 'badge-high'; riskTxt = 'Advisory'; }
      else if (d >= 0.05) { badgeClass = 'badge-moderate'; riskTxt = 'Watch'; }

      const fullIdStr = String(n.id);
      const shortLabel = fullIdStr.length > 6 ? `#${fullIdStr.slice(-5)}` : `#${fullIdStr}`;

      tr.innerHTML = `
        <td class="clickable-node-cell" data-id="${n.id}" title="Click to inspect Node #${n.id}" style="font-weight:700; color:#0284c7; cursor:pointer; text-decoration:underline;">${shortLabel} ${n.is_basement ? '<span style="font-size:0.65rem; color:#dc2626; font-weight:700;">[B]</span>' : ''}</td>
        <td class="num-col" style="color:#475569;">${(n.elevation || 880).toFixed(1)}</td>
        <td class="num-col" style="font-weight:700; color:${d >= 0.30 ? '#dc2626' : (d >= 0.15 ? '#ea580c' : '#0284c7')}">${d.toFixed(3)}</td>
        <td class="text-center"><span class="badge ${badgeClass}">${riskTxt}</span></td>
        <td class="text-center"><button class="locate-btn" data-id="${n.id}">Locate</button></td>
      `;
      tbody.appendChild(tr);
    });

    const triggerSelect = (e) => {
      const id = e.currentTarget.getAttribute('data-id');
      const target = nodesData.find(n => String(n.id) === String(id));
      if (target) {
        selectedNode = target;
        if (target.lat && target.lng) {
          leafletMap.flyTo([target.lat, target.lng], 17, { duration: 1.2 });
        }
        openNodeInspector(target);
        requestRender();
      }
    };

    document.querySelectorAll('.locate-btn, .clickable-node-cell').forEach(el => {
      el.addEventListener('click', triggerSelect);
    });
  }

  function populateDispatch(recs) {
    const container = document.getElementById('dispatchList');
    if (!container) return;
    container.innerHTML = '';
    if (recs.length === 0) {
      container.innerHTML = '<p class="empty-state-text">No emergency dispatch recommendations at current risk level.</p>';
      return;
    }
    recs.forEach(rec => {
      const div = document.createElement('div');
      div.className = 'dispatch-card';
      const color = rec.priority === 'CRITICAL' ? '#DC2626' : (rec.priority === 'HIGH' ? '#f59e0b' : '#22c55e');
      div.style.borderLeft = `4px solid ${color}`;
      div.innerHTML = `
        <div class="dispatch-priority" style="color:${color}">[${rec.priority}]</div>
        <div class="dispatch-action">${rec.action}</div>
        <div class="dispatch-detail">${rec.detail}</div>
      `;
      container.appendChild(div);
    });
  }

  function populateAlert(alertText) {
    const el = document.getElementById('rwaAlertText');
    if (el) el.textContent = alertText;
  }

  // Ground Truth Validation Overlay
  const validationModeBtn = document.getElementById('validationModeBtn');
  let historicalMarkers = [];
  let isValidationActive = false;

  if (validationModeBtn) {
    validationModeBtn.addEventListener('click', async () => {
      if (isValidationActive) {
        historicalMarkers.forEach(m => leafletMap.removeLayer(m));
        historicalMarkers = [];
        isValidationActive = false;
        validationModeBtn.style.background = 'rgba(220,38,38,0.1)';
        validationModeBtn.innerText = 'Ground Truth (Oct 2024)';
        return;
      }

      validationModeBtn.innerText = 'Fetching Complaint Logs...';
      try {
        const resp = await fetch('/api/historical-validation');
        const json = await resp.json();

        if (json.status === 'success') {
          const valData = json.data;
          const complaints = valData.complaints;
          const metrics = valData.metrics;

          isValidationActive = true;
          validationModeBtn.style.background = '#dc2626';
          validationModeBtn.style.color = '#ffffff';
          validationModeBtn.innerText = `Ground Truth Active (F1: ${(metrics.f1_score * 100).toFixed(1)}%)`;

          historicalMarkers.forEach(m => leafletMap.removeLayer(m));
          historicalMarkers = [];

          complaints.forEach(c => {
            const iconHtml = `<div style="background:#dc2626; border:2px solid #ffffff; width:16px; height:16px; border-radius:50%; box-shadow:0 0 12px rgba(220,38,38,0.6); cursor:pointer;"></div>`;
            const customIcon = L.divIcon({ html: iconHtml, className: 'complaint-pin-icon', iconSize: [16, 16] });
            const marker = L.marker([c.lat, c.lng], { icon: customIcon }).addTo(leafletMap);
            marker.bindPopup(`
              <div style="font-family:sans-serif; padding:4px; color:#0f172a;">
                <h4 style="margin:0 0 4px 0; font-size:0.85rem; color:#dc2626; font-weight:800;">Geotagged Incident Log</h4>
                <p style="margin:0 0 4px 0; font-weight:700; font-size:0.8rem;">${c.name}</p>
                <p style="margin:0 0 2px 0; font-size:0.75rem; color:#475569;"><strong>Reported Depth:</strong> ${c.reported_depth_m} m</p>
                <p style="margin:0 0 2px 0; font-size:0.75rem; color:#475569;"><strong>GNN Predicted:</strong> ${c.predicted_depth_m} m</p>
                <p style="margin:0 0 2px 0; font-size:0.75rem; color:#475569;"><strong>Source:</strong> ${c.source}</p>
                <p style="margin:0; font-size:0.75rem; color:${c.is_correctly_flagged ? '#059669' : '#d97706'}; font-weight:700;"><strong>Status:</strong> ${c.is_correctly_flagged ? 'Verified True Positive' : 'Moderate Detection'}</p>
              </div>
            `);
            historicalMarkers.push(marker);
          });
        }
      } catch (err) {
        console.error('Validation error:', err);
        validationModeBtn.innerText = 'Ground Truth (Oct 2024)';
      }
    });
  }

  // Tabs Switching Handler
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');
      if (tabId === 'tab-benchmark' && benchmarkChart) {
        benchmarkChart.resize();
      }
    });
  });

  function initCharts() {
    const el1 = document.getElementById('benchmarkChart');
    if (el1) {
      benchmarkChart = new Chart(el1.getContext('2d'), {
        type: 'bar',
        data: {
          labels: ['PINN-GNN', 'EPA SWMM', '3D CFD'],
          datasets: [{
            label: 'Runtime (ms)',
            data: [8.4, 12500, 450000],
            backgroundColor: ['#059669', '#0284C7', '#DC2626'],
            borderRadius: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { type: 'logarithmic', grid: { color: '#E2E8F0' }, ticks: { color: '#64748B', font: { family: 'JetBrains Mono' } } },
            x: { grid: { display: false }, ticks: { color: '#64748B' } }
          },
          plugins: { legend: { display: false } }
        }
      });
    }
  }

  function updateBenchmarkChart(gnnMs, swmmMs) {
    if (benchmarkChart) {
      benchmarkChart.data.datasets[0].data = [gnnMs, swmmMs, 450000];
      benchmarkChart.update();
    }
  }

  loadGraphData('hsr');
  setTimeout(resizeCanvas, 300);
});
