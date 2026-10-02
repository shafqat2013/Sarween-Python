import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { PGlite } from '@electric-sql/pglite';

const USER = '11111111-1111-4111-8111-111111111111';
const OTHER = '33333333-3333-4333-8333-333333333333';
const SID = '22222222-2222-4222-8222-222222222222';

test('actual migration enforces approval, session lifetime, identity and database privileges', async () => {
  const db = new PGlite();
  try {
    // Minimal Supabase platform contract, supplied only in this isolated database.
    await db.exec(`
      create role anon; create role authenticated; create role service_role;
      grant usage on schema public to anon, authenticated, service_role;
      create schema auth;
      create table auth.users(id uuid primary key, banned_until timestamptz);
      create table auth.sessions(id uuid primary key, user_id uuid, not_after timestamptz);
      create function auth.jwt() returns jsonb language sql as
        $$ select current_setting('request.jwt.claims', true)::jsonb $$;
      create function auth.uid() returns uuid language sql as $$ select (auth.jwt()->>'sub')::uuid $$;
      create function auth.role() returns text language sql as $$ select auth.jwt()->>'role' $$;
      insert into auth.users values ('${USER}', null), ('${OTHER}', null);
      insert into auth.sessions values ('${SID}', '${USER}', null);
    `);
    await db.exec(await readFile(new URL('../migrations/202610010001_alpha_access.sql', import.meta.url), 'utf8'));
    async function status(user = USER, sid = SID) {
      await db.query("select set_config('request.jwt.claims', $1, false)", [JSON.stringify({ sub: user, session_id: sid, role: 'authenticated' })]);
      await db.exec('set role authenticated');
      try { return (await db.query('select public.alpha_access_status() as access')).rows[0].access; }
      finally { await db.exec('reset role'); }
    }
    assert.equal((await status()).status, 'denied');
    await db.query('insert into public.alpha_testers(user_id, approved) values($1, true)', [USER]);
    assert.equal((await status()).status, 'approved');
    assert.equal((await status(OTHER)).status, 'login_required');
    await db.exec('set role authenticated');
    await assert.rejects(db.query('select * from public.alpha_testers'), /permission denied/);
    await assert.rejects(db.query('update public.alpha_testers set approved = true'), /permission denied/);
    await assert.rejects(db.query(`insert into public.alpha_testers(user_id, approved) values('${OTHER}', true)`), /permission denied/);
    await db.exec('reset role; set role anon');
    await assert.rejects(db.query('select public.alpha_access_status()'), /permission denied/);
    await db.exec('reset role');
    await db.query('update public.alpha_testers set approved = false where user_id = $1', [USER]);
    assert.equal((await status()).status, 'denied');
    await db.query('update public.alpha_testers set approved = true where user_id = $1', [USER]);
    await db.exec("update auth.sessions set not_after = now() - interval '1 second'");
    assert.equal((await status()).status, 'login_required');
    await db.exec('update auth.sessions set not_after = null');
    await db.exec("update auth.users set banned_until = now() + interval '1 day'");
    assert.equal((await status()).status, 'login_required');
    await db.exec('update auth.users set banned_until = null; delete from auth.sessions');
    assert.equal((await status()).status, 'login_required');
  } finally { await db.close(); }
});
