/* Have Hit service worker — app shell caching only.
 *
 * WHY THIS EXISTS
 *   Two reasons, and neither is offline analysis. First, Android Chrome will
 *   not treat a site as installable without a service worker that handles
 *   fetch, so "Add to Home Screen" needs one to be a real install rather than
 *   a bookmark. Second, the shell (markup, script, icons) is static and
 *   identical on every load, so serving it from cache makes a cold open in a
 *   gym with two bars of signal feel instant.
 *
 * WHAT IT DELIBERATELY DOES NOT DO
 *   It never caches /api/*. Every analysis response is specific to one
 *   session, and a stale form score served from cache would be a wrong answer
 *   presented with full confidence -- far worse than an error. The app is
 *   useless without the server anyway: V-JEPA, SAM and Gemini all live there.
 *
 * NETWORK-FIRST, NOT CACHE-FIRST
 *   Cache-first is faster and is how most shells are written. It is also how
 *   a user ends up running last week's JavaScript against this week's API for
 *   days without knowing. The network is tried first and the cache is the
 *   fallback, so the cache is what you get when the network fails rather than
 *   what you get by default.
 */
/* Bumped from v1 to drop every entry cached before app.js became versioned.
   activate() deletes any cache whose key is not VERSION, so raising this
   number is the eviction mechanism -- and it had to be raised, because a v1
   cache holds an unversioned /static/js/app.js from before the UI rebuild. */
const VERSION = "have-hit-v2";

/* app.js is deliberately ABSENT from this list now that it is requested as
   /static/js/app.js?v=<mtime>. Precaching the bare URL would store a copy
   that no page ever asks for again -- dead bytes that still look like a
   working cache entry. The fetch handler below caches whichever version is
   actually requested, which is the one the page will ask for next time. */
const SHELL = [
  "/",
  "/manifest.json",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/icons/apple-touch-icon.png",
];

self.addEventListener("install", e => {
  // addAll rejects the whole install if ANY entry 404s, so failures are
  // swallowed per-entry: a missing icon must not leave the app uninstallable.
  e.waitUntil(caches.open(VERSION).then(c =>
    Promise.all(SHELL.map(u => c.add(u).catch(() => {})))
  ).then(() => self.skipWaiting()));
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== VERSION).map(k => caches.delete(k)))
  ).then(() => self.clients.claim()));
});

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  // Analysis, history and health are never cached. See the note above.
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/coach/") ||
      url.pathname === "/health" || url.pathname === "/analytics") return;

  /* cache: "no-cache" forces a conditional request to the server instead of
     letting the browser answer this "network" fetch out of its own HTTP cache.
     That silent HTTP-cache hit is exactly how a network-first worker managed
     to keep serving a stale app.js: the worker did go to fetch(), and fetch()
     did not go to the network. Revalidation is cheap -- an unchanged file
     comes back 304 with no body. */
  const fresh = new Request(req, { cache: "no-cache" });

  e.respondWith(
    fetch(fresh)
      .then(res => {
        // Only successful, non-partial, same-origin responses are worth
        // storing. A 206 from a video range request is not a whole file and
        // replaying one from cache produces a truncated, unplayable clip.
        if (res && res.status === 200 && res.type === "basic") {
          const copy = res.clone();
          caches.open(VERSION).then(c => c.put(req, copy)).catch(() => {});
        }
        return res;
      })
      .catch(() => caches.match(req).then(hit => hit || caches.match("/")))
  );
});
