async function prepareFoundryLogin(p) {
  const g = globalThis.game;
  if (!g?.world?.id) return {phase: "loading", message: "Waiting for the selected world to launch…"};
  if (g.world.id !== p.world) return {phase: "error", message: "A different Foundry world is open. Return to setup before starting the selected world."};
  const gamePath = new URL(p.base + "/game").pathname;
  if (location.pathname === gamePath) {
    // The game view exposes its user before canvas/system initialization ends.
    // Do not accidentally submit the join form while that view is still loading.
    if (!g.ready || !g.user) return {phase: "loading", message: "Loading the Foundry game…"};
    if (g.user.name === p.user && g.user.isGM) return {phase: "joined"};
    return {phase: "error", message: "Foundry is logged in as a different user. Log out before continuing."};
  }
  // Foundry publishes the World before constructing its join-page Users
  // collection. Missing users during this normal interval is not a bad login.
  if (!Array.isArray(g.users?.contents) || !document.querySelector("select[name='userid']"))
    return {phase: "loading", message: "Waiting for Foundry's user list…"};
  const matches = g.users.contents.filter(user => user.name === p.user);
  if (matches.length !== 1) return {phase: "error", message: "Choose the exact name of an existing Foundry Gamemaster user."};
  const user = matches[0];
  if (!(user.isGM || user.role >= 4)) return {phase: "error", message: "Automatic setup requires a Gamemaster user."};
  if (user.active) return {phase: "error", message: "That Foundry user is already connected. Use that game window or log it out first."};
  if (globalThis.__sarweenJoining) return {phase: "joining", message: "Joining the Foundry game…"};
  globalThis.__sarweenJoining = true;
  try {
    const response = await fetch(p.base + "/join", {
      method: "POST", credentials: "same-origin", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action: "join", userid: user.id, password: p.password}),
    });
    if (!response.ok) {
      globalThis.__sarweenJoining = false;
      return {phase: "error", message: "Foundry rejected the login. Check the user name and password."};
    }
    const result = await response.json();
    if (result.status !== "success") {
      globalThis.__sarweenJoining = false;
      return {phase: "error", message: "Foundry did not accept the login."};
    }
    const next = new URL(result.redirect || p.base + "/game", p.base);
    if (next.origin !== location.origin || next.pathname !== gamePath) {
      globalThis.__sarweenJoining = false;
      return {phase: "error", message: "Foundry returned an unexpected login destination."};
    }
    setTimeout(() => location.assign(next.href), 100);
    return {phase: "joining", message: "Login accepted. Loading the game…"};
  } catch (_) {
    globalThis.__sarweenJoining = false;
    return {phase: "error", message: "The local Foundry login request could not finish."};
  }
}
