const startButton = document.getElementById("startButton");
const importMediaButton = document.getElementById("importMediaButton");
const mediaInput = document.getElementById("mediaInput");
const stopButton = document.getElementById("stopButton");
const liveFrame = document.getElementById("liveFrame");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const monitoringLabel = document.getElementById("monitoringLabel");
const cameraConnection = document.getElementById("cameraConnection");
const runtimeLabel = document.getElementById("runtimeLabel");
const fpsValue = document.getElementById("fpsValue");
const resolutionValue = document.getElementById("resolutionValue");
const riskRing = document.getElementById("riskRing");
const riskPercent = document.getElementById("riskPercent");
const riskLabel = document.getElementById("riskLabel");
const riskLevel = document.getElementById("riskLevel");
const riskBadge = document.getElementById("riskBadge");
const riskDetail = document.getElementById("riskDetail");
const assessmentStatus = document.getElementById("assessmentStatus");
const activityList = document.getElementById("activityList");
const statusHero = document.getElementById("statusHero");
const statusTitle = document.getElementById("statusTitle");
const statusDetail = document.getElementById("statusDetail");
const repairCameraButton = document.getElementById("repairCameraButton");
const cameraStatus = document.getElementById("cameraStatus");
const modelStatus = document.getElementById("modelStatus");
const confidenceStatus = document.getElementById("confidenceStatus");
const environmentStatus = document.getElementById("environmentStatus");
const canvas = document.getElementById("riskChart");
const chart = canvas.getContext("2d");

let riskHistory = [];
let startedAt = null;
let isUploadingMedia = false;
let lastMediaError = "";
let cameraRepairMessage = "";

let frameTimer = null;

// ---- Sidebar toggle ----
const SIDEBAR_KEY = "fallguard_sidebar";
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");

function toggleSidebar() {
  const collapsed = sidebar.classList.toggle("collapsed");
  try { localStorage.setItem(SIDEBAR_KEY, collapsed ? "collapsed" : "open"); } catch (_) {}
}

sidebarToggle.addEventListener("click", () => {
  toggleSidebar();
});

// Restore sidebar state
(function initSidebar() {
  let state = "open";
  try { state = localStorage.getItem(SIDEBAR_KEY) || "open"; } catch (_) {}
  if (state === "collapsed") {
    sidebar.classList.add("collapsed");
  }
})();

// ---- Theme ----
const THEME_KEY = "fallguard_theme";
let systemThemeQuery = window.matchMedia("(prefers-color-scheme: dark)");

function applyTheme(mode) {
  // mode: "light" | "dark" | "system"
  if (mode === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", mode);
  }
  try { localStorage.setItem(THEME_KEY, mode); } catch (_) {}
}

function getEffectiveTheme() {
  if (document.documentElement.hasAttribute("data-theme")) {
    return document.documentElement.getAttribute("data-theme");
  }
  return systemThemeQuery.matches ? "dark" : "light";
}

// Listen for system theme changes
systemThemeQuery.addEventListener("change", () => {
  const saved = (() => { try { return localStorage.getItem(THEME_KEY); } catch (_) { return "system"; } })();
  if (saved === "system" || !saved) {
    // Re-trigger CSS variable switch by forcing a repaint
    document.documentElement.removeAttribute("data-theme");
  }
});

// ---- Profile pill (in sidebar) ----
const profilePill = document.getElementById("profilePill");
const profileName = document.getElementById("profileName");

// ---- Settings button (opens separate window) ----
const settingsButton = document.getElementById("settingsButton");

function openSettingsWindow() {
  // Use backend API to open settings — this is the most reliable approach
  // for pywebview where window.open may not create a native window.
  fetch("/api/open-settings")
    .then((r) => r.json())
    .then((data) => {
      if (!data.ok) {
        // Backend failed, try window.open as fallback
        const w = window.open("/settings", "fallguard-settings",
          "width=640,height=700");
        if (!w) {
          console.error("Failed to open settings window via both methods");
        }
      }
    })
    .catch(() => {
      // Network error, try window.open as last resort
      window.open("/settings", "fallguard-settings",
        "width=640,height=700");
    });
}

settingsButton.addEventListener("click", (e) => {
  e.stopPropagation();
  openSettingsWindow();
});

// Clicking profile pill also opens settings window
profilePill.addEventListener("click", (e) => {
  e.stopPropagation();
  openSettingsWindow();
});
const fallHistoryList = document.getElementById("fallHistoryList");
let activeProfile = null;
let allProfiles = [];

// Load profiles from backend and update UI
async function loadProfiles() {
  try {
    const resp = await fetch("/api/profiles");
    const data = await resp.json();
    allProfiles = data.profiles || [];
    activeProfile = data.activeProfile || null;
    renderProfilePill();
    renderFallHistory();
  } catch (_) {
    // Keep defaults
  }
}

// Update profile pill in sidebar
function renderProfilePill() {
  if (profileName && activeProfile) {
    profileName.textContent = activeProfile.name;
  }
}

// Render fall history for active profile
function renderFallHistory() {
  if (!fallHistoryList) return;
  const events = activeProfile && activeProfile.fallEvents ? activeProfile.fallEvents : [];
  if (!events.length) {
    fallHistoryList.innerHTML = `<div class="empty-state">${i18n.t("activity.noFallHistory", "No falls recorded for this profile.")}</div>`;
    return;
  }
  fallHistoryList.innerHTML = events
    .slice()
    .reverse()
    .map((e) => {
      const ts = formatFallTime(e.timestamp);
      const stateLabel = e.state === "Fall"
        ? i18n.t("risk.statusFall", "Fall Detected")
        : i18n.t("risk.statusPrefall", "Pre-fall Risk");
      const level = e.state === "Fall" ? "danger" : "warning";
      return `
        <div class="activity-row">
          <span class="activity-mark ${level}"></span>
          <strong>${escapeHtml(stateLabel)}</strong>
          <span>${escapeHtml(ts)}</span>
          <span>${i18n.t("risk.riskScorePrefix", "Risk Score:")} ${e.risk_score}%</span>
        </div>
      `;
    }).join("");
}

function formatFallTime(isoString) {
  try {
    const d = new Date(isoString);
    return d.toLocaleString(undefined, {
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  } catch (_) {
    return isoString || "--";
  }
}

// ---- Buttons ----
startButton.addEventListener("click", async () => {
  await fetch("/api/start", { method: "POST" });
});

stopButton.addEventListener("click", async () => {
  await fetch("/api/stop", { method: "POST" });
  stopFrameRefresh();
});

importMediaButton.addEventListener("click", async () => {
  await pickMedia();
});

repairCameraButton.addEventListener("click", async () => {
  repairCameraButton.disabled = true;
  repairCameraButton.textContent = i18n.t("buttons.repairing", "Repairing...");
  try {
    const response = await fetch("/api/camera/repair", { method: "POST" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || i18n.t("errors.cameraRepairFailed", "Camera repair failed."));
    }
    cameraRepairMessage = payload.detail || i18n.t("errors.cameraAccessReset", "Camera access was reset. Quit and reopen FallGuard, then allow camera access.");
  } catch (error) {
    cameraRepairMessage = error.message || i18n.t("errors.cameraRepairFailed", "Camera repair failed.");
  } finally {
    repairCameraButton.disabled = false;
    repairCameraButton.textContent = i18n.t("buttons.repairCamera", "Repair Camera Access");
    await pollStatus();
  }
});

mediaInput.addEventListener("change", async () => {
  const files = Array.from(mediaInput.files || []);
  if (!files.length) return;
  await uploadMedia(files);
  mediaInput.value = "";
});

async function uploadMedia(files) {
  isUploadingMedia = true;
  lastMediaError = "";
  updateImportButtons({ running: true, state: "Uploading" });

  try {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("media", file, file.webkitRelativePath || file.name);
    });
    const response = await fetch("/api/media/import", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || i18n.t("errors.mediaImportFailed", "Media import failed."));
    }
    await pollStatus();
  } catch (error) {
    lastMediaError = error.message || i18n.t("errors.mediaImportFailed", "Media import failed.");
    showMediaImportError(lastMediaError);
  } finally {
    isUploadingMedia = false;
    await pollStatus();
  }
}

async function pickMedia() {
  isUploadingMedia = true;
  lastMediaError = "";
  updateImportButtons({ running: true, state: "Uploading" });

  try {
    const response = await fetch("/api/media/pick", { method: "POST" });
    const payload = await response.json().catch(() => ({}));
    if (payload.canceled) return;
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || i18n.t("errors.mediaImportFailed", "Media import failed."));
    }
    await pollStatus();
  } catch (error) {
    isUploadingMedia = false;
    if (error.message && error.message.includes("Native picker unavailable")) {
      mediaInput.click();
      return;
    }
    lastMediaError = error.message || i18n.t("errors.mediaImportFailed", "Media import failed.");
    showMediaImportError(lastMediaError);
  } finally {
    isUploadingMedia = false;
    await pollStatus();
  }
}

function startFrameRefresh() {
  stopFrameRefresh();
  // Use a cache-busting query parameter so WebKit always re-fetches
  videoPlaceholder.style.display = "none";
  liveFrame.src = `/frame.jpg?ts=${Date.now()}`;
  frameTimer = setInterval(() => {
    liveFrame.src = `/frame.jpg?ts=${Date.now()}`;
  }, 67);  // ~15 fps
}

function stopFrameRefresh() {
  if (frameTimer) {
    clearInterval(frameTimer);
    frameTimer = null;
  }
  liveFrame.removeAttribute("src");
  videoPlaceholder.style.display = "flex";
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const status = await response.json();
    updateUI(status);
    // Stop frame refresh when monitoring stops
    if (!status.running && !status.loading && frameTimer) {
      stopFrameRefresh();
    }
  } catch (_error) {
    monitoringLabel.textContent = i18n.t("monitoring.disconnected", "Disconnected");
  }
}

function updateUI(status) {
  const mediaJob = status.mediaJob || status.videoJob || {};
  const mediaBusy = isMediaBusy(mediaJob);
  startButton.disabled = status.running || status.loading || mediaBusy || isUploadingMedia;
  stopButton.disabled = !status.running && !status.loading;
  updateImportButtons(mediaJob, status);

  // Update profile info from backend
  if (status.activeProfile) {
    activeProfile = status.activeProfile;
    renderProfilePill();
    renderFallHistory();
  }

  // Monitoring label — status capsule with class-based styling
  monitoringLabel.textContent = status.running
    ? i18n.t("monitoring.monitoring", "Monitoring")
    : status.loading
      ? i18n.t("monitoring.starting", "Starting")
      : mediaBusy
        ? i18n.t("monitoring.processing", "Processing")
        : i18n.t("monitoring.idle", "Idle");

  // Apply CSS class for visual state
  monitoringLabel.className = "monitor-tag";
  if (status.running) {
    monitoringLabel.classList.add("active");
  } else if (status.loading) {
    monitoringLabel.classList.add("warning");
  }

  // Camera connection — update text and pill class
  cameraConnection.textContent = status.cameraConnected
    ? i18n.t("monitoring.cameraConnected", "Camera Connected")
    : i18n.t("monitoring.cameraReady", "Camera Ready");

  const connPill = document.getElementById("connectionPill");
  if (connPill) {
    if (status.cameraConnected) {
      connPill.classList.remove("idle");
    } else {
      connPill.classList.add("idle");
    }
  }

  fpsValue.textContent = status.fps ? status.fps.toFixed(1) : "--";
  resolutionValue.textContent = status.resolution || "--";

  const statusCard = getStatusCard(status, mediaJob);
  // statusCard.title/detail can be backend-originated — try dynamic translation first
  statusTitle.textContent = i18n.tDynamic(statusCard.title);
  statusDetail.textContent = cameraRepairMessage || i18n.tDynamic(statusCard.detail);
  repairCameraButton.hidden = !isCameraAccessError(status);

  // Status list items — backend-originated values, use tDynamic
  cameraStatus.textContent = status.cameraConnected
    ? i18n.t("status.connected", "Connected")
    : i18n.t("status.ready", "Ready");
  modelStatus.textContent = status.modelActive
    ? i18n.t("status.active", "Active")
    : mediaBusy
      ? i18n.t("monitoring.processing", "Processing")
      : status.loading
        ? i18n.t("status.loading", "Loading")
        : i18n.t("status.ready", "Ready");
  confidenceStatus.textContent = status.confidencePercent ? `${status.confidencePercent}%` : "--";
  environmentStatus.textContent = i18n.tDynamic(status.environment || "Waiting");

  if (status.startedAt && status.running) {
    if (!startedAt) startedAt = Date.now();
    runtimeLabel.textContent = formatDuration(Math.floor((Date.now() - startedAt) / 1000));
  } else {
    startedAt = null;
    runtimeLabel.textContent = "00:00";
  }

  if (!status.running && !status.loading && !status.cameraConnected) {
    stopFrameRefresh();
  } else if (status.cameraConnected && !frameTimer) {
    startFrameRefresh();
  }

  updateRisk(status);
  updateStatusTone(status, mediaJob);
  renderActivities(status.activities || []);
}

function isMediaBusy(mediaJob) {
  return Boolean(mediaJob.running) || mediaJob.state === "Uploading" || mediaJob.state === "Processing";
}

function updateImportButtons(mediaJob, status = {}) {
  const mediaBusy = isMediaBusy(mediaJob);
  const disabled = Boolean(status.running || status.loading || mediaBusy || isUploadingMedia);
  importMediaButton.disabled = disabled;
  importMediaButton.textContent = isUploadingMedia
    ? i18n.t("media.importing", "Importing...")
    : mediaBusy
      ? i18n.t("media.processing", "Processing...")
      : i18n.t("buttons.importMedia", "Import Media");
}

function isCameraAccessError(status) {
  const detail = `${status.title || ""} ${status.detail || ""} ${status.error || ""}`.toLowerCase();
  return status.state === "Error" && detail.includes("camera");
}

function getStatusCard(status, mediaJob) {
  // Show backend error messages — camera permission denied, model load failure, etc.
  if (status.state === "Error") {
    return {
      title: i18n.tDynamic(status.title || "Setup Needed"),
      detail: i18n.tDynamic(status.detail || status.error || "An error occurred. Check the terminal log for details."),
    };
  }

  if (status.running || status.loading) {
    return {
      title: i18n.tDynamic(status.title || i18n.t("monitoring.monitoring", "Monitoring")),
      detail: i18n.tDynamic(status.detail || i18n.t("monitoring.runningNormally", "System is running normally.")),
    };
  }

  if (isUploadingMedia) {
    return {
      title: i18n.t("media.importingMedia", "Importing Media"),
      detail: i18n.t("media.preparing", "Preparing the selected files."),
    };
  }

  if (mediaJob.running || mediaJob.state === "Complete" || mediaJob.state === "Error") {
    return {
      title: i18n.tDynamic(mediaJob.title || i18n.t("media.processingMedia", "Processing Media")),
      detail: i18n.tDynamic(mediaJob.detail || i18n.t("media.working", "Working on the imported media.")),
    };
  }

  if (lastMediaError) {
    return {
      title: i18n.t("media.importFailed", "Import Failed"),
      detail: lastMediaError,
    };
  }

  return {
    title: i18n.t("status.ready", "Ready"),
    detail: i18n.t("status.systemWaiting", "System is waiting to start."),
  };
}

function showMediaImportError(message) {
  statusHero.classList.add("danger");
  statusTitle.textContent = i18n.t("media.importFailed", "Import Failed");
  statusDetail.textContent = message;
}

function updateRisk(status) {
  const risk = Math.max(0, Math.min(100, Number(status.riskPercent || 0)));
  riskHistory.push(risk);
  if (riskHistory.length > 48) riskHistory.shift();

  let color = "var(--green)";
  let label = i18n.t("risk.lowRisk", "Low Risk");
  let level = i18n.t("risk.levelLow", "Low");
  let badge = i18n.t("risk.badgeNormal", "Normal");
  let badgeClass = "risk-badge low";
  if (risk >= 65 || status.state === "Fall") {
    color = "var(--red)";
    label = i18n.t("risk.highRisk", "High Risk");
    level = i18n.t("risk.levelHigh", "High");
    badge = i18n.t("risk.badgeTakeCare", "Take Care");
    badgeClass = "risk-badge high";
  } else if (risk >= 35 || status.state === "Pre-fall") {
    color = "var(--amber)";
    label = i18n.t("risk.mediumRisk", "Medium Risk");
    level = i18n.t("risk.levelMedium", "Medium");
    badge = i18n.t("risk.badgeWatch", "Watch");
    badgeClass = "risk-badge medium";
  }

  const ringBg = getComputedStyle(document.documentElement).getPropertyValue("--ring-bg").trim() || "#edf2f7";
  riskRing.style.background = `conic-gradient(${color} ${risk * 3.6}deg, ${ringBg} 0deg)`;
  riskRing.querySelector("strong").style.color = color;
  riskRing.querySelector("span").style.color = color;
  riskPercent.textContent = `${risk}%`;
  riskLabel.textContent = label;
  riskLevel.textContent = level;
  riskLevel.style.color = color;
  riskBadge.textContent = badge;
  riskBadge.className = badgeClass;

  // Assessment status — map the model's internal state to a user-friendly label
  const modelState = status.state || "";
  const stateMap = {
    "Normal": { key: "risk.statusNormal", color: "var(--green)" },
    "Pre-fall": { key: "risk.statusPrefall", color: "var(--amber)" },
    "Fall": { key: "risk.statusFall", color: "var(--red)" },
    "Unknown": { key: "risk.statusUnknown", color: "var(--muted)" },
  };
  const mapped = stateMap[modelState] || { key: "risk.statusNormal", color: "var(--muted)" };
  assessmentStatus.textContent = i18n.t(mapped.key, modelState || "Normal");
  assessmentStatus.style.color = mapped.color;

  riskDetail.textContent = i18n.tDynamic(status.detail || i18n.t("monitoring.runningNormally", "System is running normally."));
  drawChart(color);
}

function updateStatusTone(status, mediaJob = {}) {
  statusHero.classList.remove("danger", "warning");
  if (mediaJob.state === "Error" || status.state === "Fall" || status.riskPercent >= 65 || status.state === "Error") {
    statusHero.classList.add("danger");
  } else if (status.state === "Pre-fall" || status.riskPercent >= 35) {
    statusHero.classList.add("warning");
  }
}

function renderActivities(items) {
  if (!items.length) {
    activityList.innerHTML = `<div class="empty-state">${i18n.t("activity.noActivity", "No activity yet.")}</div>`;
    return;
  }

  activityList.innerHTML = items
    .slice()
    .reverse()
    .map((item) => {
      const level = item.level || "muted";
      const title = i18n.tDynamic(item.title || "Activity");
      return `
        <div class="activity-row">
          <span class="activity-mark ${level}"></span>
          <strong>${escapeHtml(title)}</strong>
          <span>${escapeHtml(item.time || "--")}</span>
          <span>${i18n.t("risk.riskScorePrefix", "Risk Score:")} ${Number(item.risk || 0)}%</span>
        </div>
      `;
    })
    .join("");
}

function drawChart(color) {
  const width = canvas.width;
  const height = canvas.height;
  chart.clearRect(0, 0, width, height);

  const styles = getComputedStyle(document.documentElement);
  chart.strokeStyle = styles.getPropertyValue("--chart-line").trim() || "#e5ebf3";
  chart.lineWidth = 1;
  chart.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  chart.fillStyle = styles.getPropertyValue("--chart-text").trim() || "#7a8798";

  for (let i = 0; i <= 4; i += 1) {
    const y = 18 + i * 32;
    chart.beginPath();
    chart.moveTo(40, y);
    chart.lineTo(width - 12, y);
    chart.stroke();
  }

  if (!riskHistory.length) return;

  const values = riskHistory.length === 1 ? [0, riskHistory[0]] : riskHistory;
  const step = (width - 56) / Math.max(values.length - 1, 1);
  const points = values.map((value, index) => ({
    x: 40 + index * step,
    y: 146 - (value / 100) * 128,
  }));

  chart.beginPath();
  chart.moveTo(points[0].x, height - 18);
  points.forEach((point) => chart.lineTo(point.x, point.y));
  chart.lineTo(points[points.length - 1].x, height - 18);
  chart.closePath();
  chart.fillStyle = color === "var(--red)" ? "rgba(234,67,53,0.14)" : color === "var(--amber)" ? "rgba(251,188,4,0.14)" : "rgba(52,168,83,0.14)";
  chart.fill();

  chart.beginPath();
  points.forEach((point, index) => {
    if (index === 0) chart.moveTo(point.x, point.y);
    else chart.lineTo(point.x, point.y);
  });
  chart.strokeStyle = color === "var(--red)" ? "#EA4335" : color === "var(--amber)" ? "#FBBC04" : "#34A853";
  chart.lineWidth = 2.5;
  chart.stroke();

  const last = points[points.length - 1];
  chart.beginPath();
  chart.arc(last.x, last.y, 5, 0, Math.PI * 2);
  chart.fillStyle = chart.strokeStyle;
  chart.fill();
}

function formatDuration(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, "0");
  const seconds = (totalSeconds % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ---- Init ----
// Apply saved theme before anything renders
(function initTheme() {
  let saved = "system";
  try { saved = localStorage.getItem(THEME_KEY) || "system"; } catch (_) {}
  applyTheme(saved);
})();

drawChart("var(--green)");
i18n.init().then(() => {
  loadProfiles().then(() => pollStatus());
});

// Poll status frequently
setInterval(pollStatus, 700);

// Poll settings less frequently (for changes from settings window)
async function pollSettings() {
  try {
    const resp = await fetch("/api/settings");
    const data = await resp.json();
    // Sync theme
    if (data.theme) {
      const current = (() => { try { return localStorage.getItem(THEME_KEY); } catch (_) { return "system"; } })();
      if (data.theme !== current) {
        applyTheme(data.theme);
      }
    }
    // Sync language
    if (data.lang) {
      const current = (() => { try { return localStorage.getItem("fallguard_lang"); } catch (_) { return "en"; } })();
      if (data.lang !== current) {
        i18n.setLanguage(data.lang);
      }
    }
  } catch (_) {}
}
setInterval(pollSettings, 2000);
