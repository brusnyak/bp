/* Hlas app shell service worker: cache UI shell, never cache API/WS.
   Backend remains the compute; this only makes the control-room installable
   and resilient to flaky venue networks (models/voices still served live). */
const SHELL_CACHE = "hlas-shell-v26-ds";
const SHELL = [
  "/ui/live-speech/live.html?v=20261115-local",
  "/ui/models/models.html?v=20261115-local",
  "/ui/models/models.js?v=20261115-local",
  "/ui/models/models.css?v=20261115-local",
  "/ui/i18n/i18n.js",
  "/ui/i18n/index.json?v=20261115-local",
  "/ui/shared/caps.js",
  "/ui/shared/lang-picker.js",
  "/ui/shared/lang-picker.css",
  "/ui/shared/manifest.json",
  "/ui/live-speech/live-langs.js",
  "/ui/live-speech/live-flow.js",
  "/ui/live-speech/live-style.css?v=20261115-local",
  "/ui/live-speech/live-script.js?v=20261115ds4",
  "/ui/live-speech/session-playback.js",
  "/ui/home/home.html?v=20261115-local",
  "/ui/site.css?v=20261105",
  "/ui/vendor/fonts.css",
  "/ui/vendor/material-symbols.css",
  "/ui/vendor/fonts/Montserrat-latin.woff2",
  "/ui/vendor/fonts/Montserrat-latin-ext.woff2",
  "/ui/vendor/fonts/PlayfairDisplay-latin.woff2",
  "/ui/vendor/fonts/PlayfairDisplay-latin-ext.woff2",
  "/ui/vendor/fonts/Montserrat-cyrillic.woff2",
  "/ui/vendor/fonts/PlayfairDisplay-cyrillic.woff2",
  "/ui/vendor/fonts/MaterialSymbolsOutlined-400.ttf",
  "/ui/home/home.css",
  "/ui/home/home.js?v=20261115-local",
  "/ui/global-styles.css",
  "/ui/tokens.css?v=20261115-ds",
  "/ui/app.css?v=20261115-ds",
  "/ui/theme-toggle.js",
  "/ui/manifest.webmanifest",
  "/ui/images/icon-192.png",
  "/ui/images/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== SHELL_CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  // API, WebSocket upgrades, and audio/voice blobs always go to the network.
  if (url.pathname.startsWith("/api") || url.pathname.startsWith("/ws") ||
      url.pathname.startsWith("/speaker_voices") || url.pathname.startsWith("/voice_qc")) {
    return;
  }
  if (event.request.method !== "GET") return;
  // Generated manifests change when a language or route is added: network first, cache only as the offline fallback.
  if (url.pathname === "/ui/i18n/index.json?v=20261115-local" || url.pathname === "/ui/shared/manifest.json") {
    event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
    return;
  }
  event.respondWith(
    caches.match(event.request).then((hit) => hit || fetch(event.request))
  );
});
