/**
 * Charts and Data Visualizations Helper - Light Minimalist Styling
 */

export function renderConfusionMatrix(canvasId, matrix, labels = ["Negative", "Positive"]) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width = 280;
  const h = canvas.height = 240;

  ctx.clearRect(0, 0, w, h);
  
  const cellW = (w - 70) / 2;
  const cellH = (h - 60) / 2;
  const maxVal = Math.max(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]);

  // Labels
  ctx.fillStyle = "#475569";
  ctx.font = "600 11px Inter, sans-serif";
  ctx.fillText("Pred: 0", 70 + cellW/2 - 20, 20);
  ctx.fillText("Pred: 1", 70 + 3*cellW/2 - 20, 20);

  ctx.save();
  ctx.translate(16, 40 + cellH);
  ctx.rotate(-Math.PI/2);
  ctx.fillText("Actual Class", 0, 0);
  ctx.restore();

  for (let r = 0; r < 2; r++) {
    ctx.fillText(`Act: ${r}`, 20, 40 + r*cellH + cellH/2 + 4);
    for (let c = 0; c < 2; c++) {
      const val = matrix[r][c];
      const intensity = val / (maxVal || 1);
      const x = 70 + c * cellW;
      const y = 30 + r * cellH;

      ctx.fillStyle = (r === c) 
        ? `rgba(2, 132, 199, ${0.12 + intensity * 0.75})` 
        : `rgba(220, 38, 38, ${0.12 + intensity * 0.75})`;
      ctx.fillRect(x, y, cellW - 4, cellH - 4);
      ctx.strokeStyle = "#E2E8F0";
      ctx.strokeRect(x, y, cellW - 4, cellH - 4);

      ctx.fillStyle = (r === c && intensity > 0.4) ? "#FFFFFF" : "#0F172A";
      ctx.font = "bold 14px Outfit, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`${val}`, x + cellW/2 - 2, y + cellH/2 + 5);
      ctx.textAlign = "left";
    }
  }
}

export function renderRadarChart(canvasId, factors) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width = 300;
  const h = canvas.height = 260;
  const cx = w / 2;
  const cy = h / 2 + 10;
  const radius = 75;

  ctx.clearRect(0, 0, w, h);

  const keys = Object.keys(factors);
  const numPoints = keys.length;
  const angleStep = (Math.PI * 2) / numPoints;

  // Background Web
  for (let level = 1; level <= 4; level++) {
    const r = (radius / 4) * level;
    ctx.beginPath();
    ctx.strokeStyle = "#E2E8F0";
    ctx.lineWidth = 1;
    for (let i = 0; i < numPoints; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }

  // Spoke Lines & Labels
  ctx.font = "500 10px Inter, sans-serif";
  ctx.fillStyle = "#475569";
  ctx.textAlign = "center";
  for (let i = 0; i < numPoints; i++) {
    const angle = i * angleStep - Math.PI / 2;
    const x = cx + radius * Math.cos(angle);
    const y = cy + radius * Math.sin(angle);

    ctx.beginPath();
    ctx.strokeStyle = "#CBD5E1";
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();

    const lx = cx + (radius + 20) * Math.cos(angle);
    const ly = cy + (radius + 14) * Math.sin(angle);
    ctx.fillText(keys[i], lx, ly);
  }

  // Draw Data Polygon
  ctx.beginPath();
  for (let i = 0; i < numPoints; i++) {
    const angle = i * angleStep - Math.PI / 2;
    const valRatio = Math.min(Math.max(factors[keys[i]], 0), 1);
    const r = radius * valRatio;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(2, 132, 199, 0.2)";
  ctx.fill();
  ctx.strokeStyle = "#0284C7";
  ctx.lineWidth = 2;
  ctx.stroke();
}
