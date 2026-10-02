import test from 'node:test';
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
import { generateKeyPair, exportPKCS8, jwtVerify } from 'jose';
import { createHandler } from '../functions/alpha-access/handler.mjs';
globalThis.crypto ??= webcrypto;

const USER = '11111111-1111-4111-8111-111111111111';
const SID = '22222222-2222-4222-8222-222222222222';
const DEVICE = '33333333-3333-4333-8333-333333333333';
const keys = await generateKeyPair('EdDSA', { extractable: true });
const config = { url: 'https://example.supabase.co', publishableKey: 'public-test-key',
  privateKey: await exportPKCS8(keys.privateKey), keyId: 'alpha-v1' };
function request(body = { device_id: DEVICE }, headers = {}) {
  return new Request(config.url + '/functions/v1/alpha-access', {
    method: 'POST', headers: { Authorization: 'Bearer signed.user.token', 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify(body),
  });
}
const approved = { status: 'approved', user_id: USER, session_id: SID, not_after: null };

test('approved current session receives a 24-hour permission bound to account, session and installation', async () => {
  let called;
  const now = Math.floor(Date.now() / 1000);
  const handler = createHandler(config, { now: () => now, fetcher: async (url, options) => {
    called = { url, options };
    return Response.json(approved);
  }});
  const response = await handler(request());
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Cache-Control'), 'no-store');
  const { payload, protectedHeader } = await jwtVerify((await response.json()).permission, keys.publicKey, {
    issuer: config.url + '/functions/v1/alpha-access', audience: 'sarween-desktop-alpha', algorithms: ['EdDSA'],
  });
  assert.equal(payload.exp - payload.iat, 86400);
  assert.equal(payload.sub, USER);
  assert.equal(payload.sid, SID);
  assert.equal(payload.device_id, DEVICE);
  assert.equal(protectedHeader.kid, 'alpha-v1');
  assert.equal(called.url, config.url + '/rest/v1/rpc/alpha_access_status');
  assert.equal(called.options.headers.Authorization, 'Bearer signed.user.token');
  assert.equal(called.options.body, '{}');
});

test('unapproved, revoked, expired and missing sessions receive no offline grant', async () => {
  for (const [access, status] of [[{ status: 'denied' }, 403], [{ status: 'login_required' }, 401]]) {
    const handler = createHandler(config, { fetcher: async () => Response.json(access) });
    const response = await handler(request());
    assert.equal(response.status, status);
    assert.equal((await response.json()).permission, undefined);
  }
  const expired = createHandler(config, { fetcher: async () => Response.json({ ...approved, not_after: 1 }) });
  assert.equal((await expired(request())).status, 401);
});

test('upstream denial and service errors never issue a permission', async () => {
  for (const [upstream, expected] of [[401, 401], [403, 401], [500, 503]]) {
    const handler = createHandler(config, { fetcher: async () => new Response('', { status: upstream }) });
    assert.equal((await handler(request())).status, expected);
  }
  const failed = createHandler(config, { fetcher: async () => { throw new Error('secret must not escape'); } });
  assert.deepEqual(await (await failed(request())).json(), { error: 'unavailable' });
});

test('caller cannot request approval or another identity and request bodies are bounded', async () => {
  let calls = 0;
  const handler = createHandler(config, { fetcher: async () => { calls++; return Response.json(approved); } });
  assert.equal((await handler(request({ device_id: DEVICE, user_id: USER, approved: true }))).status, 400);
  assert.equal((await handler(request({ device_id: 'x'.repeat(2000) }))).status, 413);
  assert.equal((await handler(request({}, { Authorization: '' }))).status, 401);
  assert.equal(calls, 0);
});

test('grant cannot outlive server session expiry', async () => {
  const now = Math.floor(Date.now() / 1000);
  const handler = createHandler(config, { now: () => now,
    fetcher: async () => Response.json({ ...approved, not_after: now + 90 }) });
  const response = await handler(request());
  const { payload } = await jwtVerify((await response.json()).permission, keys.publicKey);
  assert.equal(payload.exp, now + 90);
});
