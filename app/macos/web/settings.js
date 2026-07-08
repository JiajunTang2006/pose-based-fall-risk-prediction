// ---- FallGuard Settings Window ----

const languageSelect = document.getElementById("languageSelect");
const themeSelect = document.getElementById("themeSelect");
const sensitivitySelect = document.getElementById("sensitivitySelect");
const settingsVersion = document.getElementById("settingsVersion");
const profileList = document.getElementById("profileList");
const newProfileName = document.getElementById("newProfileName");
const addProfileButton = document.getElementById("addProfileButton");

let activeProfile = null;
let allProfiles = [];

// ---- Theme ----
const THEME_KEY = "fallguard_theme";
const systemThemeQuery = window.matchMedia("(prefers-color-scheme: dark)");

function applyTheme(mode) {
  if (mode === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", mode);
  }
  try { localStorage.setItem(THEME_KEY, mode); } catch (_) {}

  // Notify main window of theme change
  try {
    const mainWin = window.opener;
    if (mainWin) {
      mainWin.postMessage({ type: "themeChanged", mode: mode }, "*");
    }
  } catch (_) {}

  // Also save via API so main window can pick it up
  fetch("/api/settings/theme", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  }).catch(() => {});
}

systemThemeQuery.addEventListener("change", () => {
  const saved = (() => { try { return localStorage.getItem(THEME_KEY); } catch (_) { return "system"; } })();
  if (saved === "system" || !saved) {
    document.documentElement.removeAttribute("data-theme");
  }
});

// Init theme
(function initTheme() {
  let saved = "system";
  try { saved = localStorage.getItem(THEME_KEY) || "system"; } catch (_) {}
  applyTheme(saved);
  if (themeSelect) themeSelect.value = saved;
})();

// ---- Event handlers ----
themeSelect.addEventListener("change", () => {
  applyTheme(themeSelect.value);
});

languageSelect.addEventListener("change", () => {
  i18n.setLanguage(languageSelect.value);
  // Notify main window
  try {
    const mainWin = window.opener;
    if (mainWin) {
      mainWin.postMessage({ type: "languageChanged", lang: languageSelect.value }, "*");
    }
  } catch (_) {}
  fetch("/api/settings/language", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lang: languageSelect.value }),
  }).catch(() => {});
});

sensitivitySelect.addEventListener("change", async () => {
  await fetch("/api/settings/sensitivity", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ level: sensitivitySelect.value }),
  });
});

// ---- Load settings ----
async function loadSettings() {
  try {
    const resp = await fetch("/api/settings");
    const data = await resp.json();
    if (data.sensitivity && sensitivitySelect) {
      sensitivitySelect.value = data.sensitivity;
    }
    if (data.theme && themeSelect) {
      themeSelect.value = data.theme;
    }
    if (data.lang && languageSelect) {
      languageSelect.value = data.lang;
    }
    if (data.version && settingsVersion) {
      settingsVersion.textContent = "v" + data.version;
    }
  } catch (_) {}
}

// ---- Profile management ----
async function loadProfiles() {
  try {
    const resp = await fetch("/api/profiles");
    const data = await resp.json();
    allProfiles = data.profiles || [];
    activeProfile = data.activeProfile || null;
    renderProfileList();
  } catch (_) {}
}

function renderProfileList() {
  if (!profileList) return;
  if (!allProfiles.length) {
    profileList.innerHTML = `<div class="settings-item" style="color:var(--muted);font-size:13px">${i18n.t("settings.noProfiles", "No profiles")}</div>`;
    return;
  }
  profileList.innerHTML = allProfiles.map((p) => {
    const isActive = activeProfile && p.id === activeProfile.id;
    return `
      <div class="profile-list-item${isActive ? " active" : ""}" data-id="${p.id}">
        <span>${esc(p.name)}</span>
        <div style="display:flex;align-items:center;gap:8px">
          ${isActive ? `<span class="profile-active-badge">${i18n.t("settings.activeBadge", "✓ Active")}</span>` : ""}
          ${allProfiles.length > 1 ? `<button class="profile-delete-btn" data-action="delete" data-id="${p.id}" title="${i18n.t("settings.deleteProfile", "Delete")}">×</button>` : ""}
        </div>
      </div>
    `;
  }).join("");

  profileList.querySelectorAll(".profile-list-item").forEach((item) => {
    item.addEventListener("click", async (e) => {
      if (e.target.closest("[data-action='delete']")) return;
      const id = item.dataset.id;
      if (id && activeProfile && id !== activeProfile.id) {
        await fetch(`/api/profiles/${id}/activate`, { method: "POST" });
        await loadProfiles();
      }
    });
  });

  profileList.querySelectorAll(".profile-delete-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      if (id) {
        await fetch(`/api/profiles/${id}`, { method: "DELETE" });
        await loadProfiles();
      }
    });
  });
}

addProfileButton.addEventListener("click", async () => {
  const name = newProfileName.value.trim();
  if (!name) return;
  await fetch("/api/profiles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  newProfileName.value = "";
  await loadProfiles();
});

newProfileName.addEventListener("keydown", (e) => {
  if (e.key === "Enter") addProfileButton.click();
});

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ---- Init ----
i18n.init().then(() => {
  loadSettings().then(() => loadProfiles());
});
