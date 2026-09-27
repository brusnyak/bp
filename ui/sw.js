/* Hlas app shell service worker: cache UI shell, never cache API/WS.
   Backend remains the compute; this only makes the control-room installable
   and resilient to flaky venue networks (models/voices still served live). */
const SHELL_CACHE = "hlas-shell-v2";
const SHELL = [
  "/ui/live-speech/live.html",
  "/ui/live-speech/live-style.css",
  "/ui/live-speech/live-script.js",
  "/ui/home/home.html",
  "/ui/home/home.css",
  "/ui/home/home.js",
  "/ui/auth/auth.html",
  "/ui/auth/auth.css",
  "/ui/auth/auth.js",
  "/ui/voice-lab/lab.html",
  "/ui/voice-lab/lab-style.css",
  "/ui/voice-lab/lab.js",
  "/ui/global-styles.css",
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
  event.respondWith(
    caches.match(event.request).then((hit) => hit || fetch(event.request))
  );
});
