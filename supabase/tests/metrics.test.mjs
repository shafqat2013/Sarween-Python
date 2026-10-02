import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { PGlite } from '@electric-sql/pglite';

const USER = '11111111-1111-4111-8111-111111111111';
const OTHER = '33333333-3333-4333-8333-333333333333';
const SID = '22222222-2222-4222-8222-222222222222';
const RUN = '44444444-4444-4444-8444-444444444444';

async function fixture() {
  const db = new PGlite();
  await db.exec(`
    create role anon; create role authenticated; create role service_role;
    grant usage on schema public to anon, authenticated, service_role;
    create schema auth;
    create table auth.users(id uuid primary key, email text, banned_until timestamptz,
      created_at timestamptz default now(), last_sign_in_at timestamptz);
    create table auth.sessions(id uuid primary key, user_id uuid, not_after timestamptz,
      created_at timestamptz default now());
    create function auth.jwt() returns jsonb language sql as
      $$ select current_setting('request.jwt.claims', true)::jsonb $$;
    create function auth.uid() returns uuid language sql as $$ select (auth.jwt()->>'sub')::uuid $$;
    create function auth.role() returns text language sql as $$ select auth.jwt()->>'role' $$;
    insert into auth.users(id,email) values ('${USER}','owner@example.invalid'), ('${OTHER}','other@example.invalid');
    insert into auth.sessions(id,user_id) values ('${SID}','${USER}');
  `);
  for (const file of ['202610010001_alpha_access.sql', '202610020001_alpha_metrics.sql'])
    await db.exec(await readFile(new URL('../migrations/'+file, import.meta.url), 'utf8'));
  await db.exec(`insert into public.alpha_testers(user_id,approved) values('${USER}',true)`);
  return db;
}
async function asUser(db, callback, user=USER) {
  await db.query("select set_config('request.jwt.claims',$1,false)", [JSON.stringify({sub:user,session_id:SID,role:'authenticated'})]);
  await db.exec('set role authenticated');
  try { return await callback(); } finally { await db.exec('reset role'); }
}
function report(extra={}) {
  return {run_id:RUN,activity_date:new Date().toISOString().slice(0,10),app_version:'0.2.0-metrics1',
    app_seconds:300,tracking_seconds:200,foundry_seconds:100,tracking_sessions:1,
    replay_opens:2,demo_opens:0,foundry_connections:1,live_errors:0,...extra};
}
const upload=(db, rows)=>asUser(db,()=>db.query('select public.record_alpha_usage($1::jsonb)',[JSON.stringify(rows)]));

test('owner metrics join emails and approvals; clients cannot read or edit metrics', async()=>{
  const db=await fixture();
  try {
    await asUser(db,async()=>{
      for(const table of ['alpha_usage_reports','alpha_sign_ins','alpha_presence_daily','alpha_metrics_config','alpha_tester_metrics','alpha_metrics_overview'])
        await assert.rejects(db.query('select * from public.'+table),/permission denied/);
      await assert.rejects(db.query(`update public.alpha_testers set approved=true`),/permission denied/);
    });
    await db.exec('set role anon');
    await assert.rejects(db.query("select public.record_alpha_usage('[]')"),/permission denied/);
    await db.exec('reset role');
    const row=(await db.query('select * from public.alpha_tester_metrics where user_id=$1',[USER])).rows[0];
    assert.equal(row.email,'owner@example.invalid');
    assert.equal(row.approved,true);
    assert.equal(row.recorded_sign_ins,1);
    assert.equal(row.app_opens,null); // Unknown until the instrumented build reports.
  } finally { await db.close(); }
});

test('session creation counts sign-ins once; refresh and deletion do not reset history', async()=>{
  const db=await fixture();
  try {
    await db.exec(`insert into auth.sessions(id,user_id) values('${RUN}','${USER}')`);
    await asUser(db,()=>db.query('select public.alpha_access_status()'));
    await asUser(db,()=>db.query('select public.alpha_access_status()'));
    await db.exec(`delete from auth.sessions where id='${RUN}'`);
    assert.equal((await db.query('select count(*) n from public.alpha_sign_ins')).rows[0].n,2);
    assert.equal((await db.query('select approval_checks from public.alpha_presence_daily')).rows[0].approval_checks,2);
    // Even a broken metrics table cannot prevent approval or authentication.
    await db.exec('alter table public.alpha_presence_daily rename to metrics_unavailable');
    assert.equal((await asUser(db,()=>db.query('select public.alpha_access_status() a'))).rows[0].a.status,'approved');
    await db.exec('alter table public.alpha_sign_ins rename to signins_unavailable');
    await db.exec(`insert into auth.sessions(id,user_id) values('${OTHER}','${USER}')`);
  } finally { await db.close(); }
});

test('cumulative retries and out-of-order reports never double count usage', async()=>{
  const db=await fixture();
  try {
    await upload(db,[report()]); await upload(db,[report()]);
    await upload(db,[report({app_seconds:400,tracking_seconds:250})]);
    await upload(db,[report()]);
    const row=(await db.query('select * from public.alpha_usage_reports')).rows[0];
    assert.equal(row.app_seconds,400); assert.equal(row.tracking_seconds,250);
    const summary=(await db.query('select * from public.alpha_tester_metrics where user_id=$1',[USER])).rows[0];
    assert.equal(summary.app_opens,1); assert.equal(summary.tracking_sessions,1);
    assert.equal(summary.active_days_30d,1);
    await db.exec(`delete from auth.users where id='${USER}'`);
    assert.equal((await db.query('select count(*) n from public.alpha_usage_reports')).rows[0].n,0);
  } finally { await db.close(); }
});

test('revocation, expired sessions, identity forgery and malformed counters are rejected', async()=>{
  const db=await fixture();
  try {
    for(const bad of [report({user_id:OTHER}),report({app_seconds:-1}),report({app_seconds:1.5}),
      report({tracking_seconds:301}),report({live_errors:1001}),report({app_version:'secret filename / path'}),
      report({activity_date:'2000-01-01'}),report({demo_opens:null})])
      await assert.rejects(upload(db,[bad]));
    await assert.rejects(upload(db,Array(33).fill(report())));
    await assert.rejects(asUser(db,()=>db.query("select public.record_alpha_usage('[]')"),OTHER),/approved session/);
    await db.exec('update public.alpha_testers set approved=false');
    await assert.rejects(upload(db,[report()]),/approved session/);
    await db.exec("update public.alpha_testers set approved=true; update auth.sessions set not_after=now()-interval '1 second'");
    await assert.rejects(upload(db,[report()]),/approved session/);
    assert.equal((await db.query('select count(*) n from public.alpha_usage_reports')).rows[0].n,0);
  } finally { await db.close(); }
});
