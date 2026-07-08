// FallGuard i18n — lightweight translation engine
// Loads en.json + zh.json at init time, switches text via data-i18n + t() calls.

const i18n = (() => {
  const STORAGE_KEY = "fallguard_lang";
  let translations = {};   // merged: { en: {...}, zh: {...} }
  let currentLang = "en";
  let ready = false;

  // Resolve a dotted key like "buttons.startMonitoring" → value
  function resolve(langObj, path) {
    const parts = path.split(".");
    let node = langObj;
    for (const part of parts) {
      if (node == null) return undefined;
      node = node[part];
    }
    return typeof node === "string" ? node : undefined;
  }

  // Public: translate a key to the current language
  function t(key, fallback) {
    if (!ready) return fallback || key;
    const value = resolve(translations[currentLang], key);
    if (value !== undefined) return value;
    // Fallback to English
    const enValue = resolve(translations["en"], key);
    if (enValue !== undefined) return enValue;
    return fallback || key;
  }

  // Public: translate a backend English string (reverse lookup via backend map)
  function tDynamic(englishString) {
    if (!ready || !englishString || currentLang === "en") return englishString;
    // The backend map is a flat dict: English string → i18n dotted key
    const enBackend = translations["en"]?.backend || {};
    const lookupKey = enBackend[englishString];
    if (lookupKey) {
      // lookupKey is something like "status.ready" — resolve it in current language
      return resolve(translations[currentLang], lookupKey) || englishString;
    }
    return englishString;
  }

  // Update all [data-i18n] elements and elements with data-i18n-placeholder
  function updateStaticElements() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (key) {
        const translated = t(key, el.textContent);
        if (el.tagName === "TITLE") {
          document.title = translated;
        } else {
          el.textContent = translated;
        }
      }
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      const key = el.getAttribute("data-i18n-placeholder");
      if (key) {
        el.placeholder = t(key, el.placeholder);
      }
    });
    document.querySelectorAll("[data-i18n-title]").forEach((el) => {
      const key = el.getAttribute("data-i18n-title");
      if (key) {
        el.title = t(key, el.title);
      }
    });
    document.querySelectorAll("[data-i18n-alt]").forEach((el) => {
      const key = el.getAttribute("data-i18n-alt");
      if (key) {
        el.alt = t(key, el.alt);
      }
    });
  }

  // Apply language: save, update all static text, trigger UI refresh
  function setLanguage(lang) {
    if (lang !== "en" && lang !== "zh") return;
    currentLang = lang;
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch (_) {
      // localStorage unavailable
    }
    updateStaticElements();
    // Update the language select dropdown
    const sel = document.getElementById("languageSelect");
    if (sel) sel.value = lang;
    // Re-render dynamic content by re-polling status
    if (typeof pollStatus === "function") {
      pollStatus();
    }
  }

  // Load JSON files and initialize
  async function init() {
    try {
      const [enResp, zhResp] = await Promise.all([
        fetch("/static/locales/en.json"),
        fetch("/static/locales/zh.json"),
      ]);
      translations["en"] = await enResp.json();
      translations["zh"] = await zhResp.json();
    } catch (_) {
      // If locale files fail to load, translations stays empty and t() returns fallback
      translations["en"] = {};
      translations["zh"] = {};
    }

    // Restore saved language
    let saved = "en";
    try {
      saved = localStorage.getItem(STORAGE_KEY) || "en";
    } catch (_) {
      // localStorage unavailable
    }
    if (saved !== "en" && saved !== "zh") saved = "en";
    currentLang = saved;
    ready = true;

    // Update static elements and language select
    updateStaticElements();
    const sel = document.getElementById("languageSelect");
    if (sel) sel.value = currentLang;
  }

  // Check if i18n is ready
  function isReady() {
    return ready;
  }

  // Get current language
  function getLanguage() {
    return currentLang;
  }

  return { t, tDynamic, setLanguage, init, isReady, getLanguage, updateStaticElements };
})();
