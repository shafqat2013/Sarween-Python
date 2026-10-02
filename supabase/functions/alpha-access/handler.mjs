import { importPKCS8, SignJWT } from "jose";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const reply = (status, body) => new Response(JSON.stringify(body), {
  status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
});

export function createHandler(config, { fetcher = fetch, now = () => Math.floor(Date.now() / 1000) } = {}) {
  let signingKey;
  return async function handle(request) {
    if (request.method !== "POST") return reply(405, { error: "method_not_allowed" });
    const bearer = request.headers.get("Authorization") || "";
    if (!/^Bearer [A-Za-z0-9_.-]+$/.test(bearer) || bearer.length > 8192)
      return reply(401, { error: "login_required" });
    if (!request.headers.get("Content-Type")?.startsWith("application/json"))
      return reply(415, { error: "invalid_request" });
    let input;
    try {
      const reader = request.body?.getReader();
      if (!reader) return reply(400, { error: "invalid_request" });
      const chunks = [];
      let size = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.length;
        if (size > 1024) { await reader.cancel(); return reply(413, { error: "invalid_request" }); }
        chunks.push(value);
      }
      const data = new Uint8Array(size);
      let offset = 0;
      for (const chunk of chunks) { data.set(chunk, offset); offset += chunk.length; }
      input = JSON.parse(new TextDecoder().decode(data));
      if (!input || Object.keys(input).length !== 1 || !UUID.test(input.device_id))
        return reply(400, { error: "invalid_request" });
    } catch { return reply(400, { error: "invalid_request" }); }
    if (!config.url || !config.publishableKey || !config.privateKey || !config.keyId)
      return reply(503, { error: "unavailable" });
    try {
      // PostgREST validates the signed user JWT; the RPC independently checks the
      // current server session, banned status, and approval on every renewal.
      const response = await fetcher(config.url + "/rest/v1/rpc/alpha_access_status", {
        method: "POST", headers: {
          "apikey": config.publishableKey, "Authorization": bearer, "Content-Type": "application/json",
        }, body: "{}", signal: AbortSignal.timeout(8000), redirect: "error",
      });
      if (response.status === 401 || response.status === 403)
        return reply(401, { error: "login_required" });
      if (!response.ok) return reply(503, { error: "unavailable" });
      const access = await response.json();
      if (access.status === "login_required") return reply(401, { error: "login_required" });
      if (access.status === "denied") return reply(403, { error: "access_denied" });
      if (access.status !== "approved" || !UUID.test(access.user_id) || !UUID.test(access.session_id))
        return reply(502, { error: "unavailable" });
      const issued = now();
      if (access.not_after !== null && (!Number.isFinite(access.not_after) || access.not_after <= issued))
        return reply(401, { error: "login_required" });
      const expiry = Math.min(issued + 86400, access.not_after ?? Infinity);
      signingKey ??= await importPKCS8(config.privateKey, "EdDSA");
      const permission = await new SignJWT({ sid: access.session_id, device_id: input.device_id })
        .setProtectedHeader({ alg: "EdDSA", typ: "JWT", kid: config.keyId })
        .setIssuer(config.url + "/functions/v1/alpha-access")
        .setAudience("sarween-desktop-alpha").setSubject(access.user_id)
        .setIssuedAt(issued).setNotBefore(issued).setExpirationTime(expiry).sign(signingKey);
      return reply(200, { permission });
    } catch {
      // Do not log requests, bearer tokens, secrets, or upstream response bodies.
      return reply(503, { error: "unavailable" });
    }
  };
}
