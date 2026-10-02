-- Owner-only engagement reporting. No recruitment import or new approval path.
create table public.alpha_metrics_config (
    singleton boolean primary key default true check (singleton),
    collection_started_at timestamptz not null default now()
);
insert into public.alpha_metrics_config(singleton) values (true);

create table public.alpha_sign_ins (
    session_id uuid primary key,
    user_id uuid not null references auth.users(id) on delete cascade,
    signed_in_at timestamptz not null,
    source text not null check (source in ('existing_session', 'new_session'))
);
create index on public.alpha_sign_ins(user_id, signed_in_at);

-- A session is created by Auth after successful authentication; token refreshes
-- do not create another session. Keep counts after Auth removes old sessions.
insert into public.alpha_sign_ins
select id, user_id, coalesce(created_at, now()), 'existing_session'
from auth.sessions on conflict do nothing;
create function public.alpha_capture_sign_in() returns trigger
language plpgsql security definer set search_path = '' as $$
begin
    begin
        insert into public.alpha_sign_ins values
            (new.id, new.user_id, coalesce(new.created_at, now()), 'new_session')
        on conflict do nothing;
    exception when others then
        -- Analytics must never prevent authentication.
        null;
    end;
    return new;
end;
$$;
revoke all on function public.alpha_capture_sign_in() from public, anon, authenticated;
create trigger alpha_metrics_sign_in after insert on auth.sessions
for each row execute function public.alpha_capture_sign_in();

create table public.alpha_presence_daily (
    user_id uuid not null references auth.users(id) on delete cascade,
    activity_date date not null,
    first_seen_at timestamptz not null,
    last_seen_at timestamptz not null,
    approval_checks bigint not null default 1,
    estimated_online_seconds double precision not null default 0,
    primary key(user_id, activity_date)
);

create table public.alpha_usage_reports (
    user_id uuid not null references auth.users(id) on delete cascade,
    run_id uuid not null,
    activity_date date not null,
    app_version text not null,
    app_seconds integer not null,
    tracking_seconds integer not null,
    foundry_seconds integer not null,
    tracking_sessions integer not null,
    replay_opens integer not null,
    demo_opens integer not null,
    foundry_connections integer not null,
    live_errors integer not null,
    received_at timestamptz not null default now(),
    primary key(user_id, run_id, activity_date),
    check (app_seconds between 0 and 86400),
    check (tracking_seconds between 0 and app_seconds),
    check (foundry_seconds between 0 and app_seconds),
    check (tracking_sessions between 0 and 1000 and replay_opens between 0 and 1000
       and demo_opens between 0 and 1000 and foundry_connections between 0 and 1000
       and live_errors between 0 and 1000)
);

alter table public.alpha_metrics_config enable row level security;
alter table public.alpha_sign_ins enable row level security;
alter table public.alpha_presence_daily enable row level security;
alter table public.alpha_usage_reports enable row level security;
revoke all on public.alpha_metrics_config, public.alpha_sign_ins,
    public.alpha_presence_daily, public.alpha_usage_reports from public, anon, authenticated;
grant select on public.alpha_metrics_config, public.alpha_sign_ins,
    public.alpha_presence_daily, public.alpha_usage_reports to service_role;

-- Retain the exact session, ban and approval checks used by older builds.
create or replace function public.alpha_access_status() returns jsonb
language plpgsql volatile security definer set search_path = '' as $$
declare
    caller uuid := auth.uid();
    sid uuid := nullif(auth.jwt()->>'session_id', '')::uuid;
    deadline timestamptz;
    observed timestamptz := clock_timestamp();
begin
    if caller is null or sid is null or auth.role() <> 'authenticated' then
        return jsonb_build_object('status', 'login_required');
    end if;
    select s.not_after into deadline from auth.sessions s
      where s.id = sid and s.user_id = caller;
    if not found or (deadline is not null and deadline <= now()) then
        return jsonb_build_object('status', 'login_required');
    end if;
    if not exists(select 1 from auth.users u where u.id = caller
        and (u.banned_until is null or u.banned_until <= now())) then
        return jsonb_build_object('status', 'login_required');
    end if;
    if not exists(select 1 from public.alpha_testers t where t.user_id = caller and t.approved) then
        return jsonb_build_object('status', 'denied');
    end if;
    begin
        insert into public.alpha_presence_daily as p
            (user_id, activity_date, first_seen_at, last_seen_at)
        values (caller, (observed at time zone 'UTC')::date, observed, observed)
        on conflict(user_id, activity_date) do update set
            estimated_online_seconds = p.estimated_online_seconds +
                case when excluded.last_seen_at - p.last_seen_at between interval '0 seconds' and interval '6 minutes'
                     then extract(epoch from excluded.last_seen_at - p.last_seen_at) else 0 end,
            last_seen_at = greatest(p.last_seen_at, excluded.last_seen_at),
            approval_checks = p.approval_checks + 1;
    exception when others then null;
    end;
    return jsonb_build_object('status', 'approved', 'user_id', caller,
        'session_id', sid, 'not_after', extract(epoch from deadline));
end;
$$;

-- An authenticated client can submit ONLY its own bounded cumulative counters.
-- Retries/out-of-order delivery merge with greatest(), never add duplicate time.
create function public.record_alpha_usage(reports jsonb) returns boolean
language plpgsql security definer set search_path = '' as $$
declare
    caller uuid := auth.uid();
    sid uuid := nullif(auth.jwt()->>'session_id', '')::uuid;
    item jsonb;
    day date;
    counter text;
begin
    if caller is null or auth.role() <> 'authenticated' or not exists(
        select 1 from auth.sessions s join auth.users u on u.id = s.user_id
        join public.alpha_testers t on t.user_id = u.id
        where s.id = sid and s.user_id = caller and t.approved
        and (s.not_after is null or s.not_after > now())
        and (u.banned_until is null or u.banned_until <= now())
    ) then raise insufficient_privilege using message = 'Current approved session required'; end if;
    if reports is null or jsonb_typeof(reports) <> 'array' or jsonb_array_length(reports) > 32
        or octet_length(reports::text) > 32768 then
        raise invalid_parameter_value using message = 'Invalid usage batch';
    end if;
    for item in select value from jsonb_array_elements(reports) loop
        if jsonb_typeof(item) <> 'object' or
           (select count(*) from jsonb_object_keys(item)) <> 11 or
           not item ?& array['run_id','activity_date','app_version','app_seconds','tracking_seconds',
              'foundry_seconds','tracking_sessions','replay_opens','demo_opens','foundry_connections','live_errors'] then
            raise invalid_parameter_value using message = 'Invalid usage fields';
        end if;
        day := (item->>'activity_date')::date;
        if day is null or day < (now() at time zone 'UTC')::date - 30
            or day > (now() at time zone 'UTC')::date
            or coalesce(item->>'app_version', '') !~ '^[A-Za-z0-9._-]{1,40}$' then
            raise invalid_parameter_value using message = 'Invalid usage date or version';
        end if;
        foreach counter in array array['app_seconds','tracking_seconds','foundry_seconds',
            'tracking_sessions','replay_opens','demo_opens','foundry_connections','live_errors'] loop
            if jsonb_typeof(item->counter) is distinct from 'number'
               or item->>counter !~ '^[0-9]+$' then
                raise invalid_parameter_value using message = 'Invalid usage counter';
            end if;
        end loop;
        insert into public.alpha_usage_reports as r values (
            caller, (item->>'run_id')::uuid, day, item->>'app_version',
            (item->>'app_seconds')::integer, (item->>'tracking_seconds')::integer,
            (item->>'foundry_seconds')::integer, (item->>'tracking_sessions')::integer,
            (item->>'replay_opens')::integer, (item->>'demo_opens')::integer,
            (item->>'foundry_connections')::integer, (item->>'live_errors')::integer, now())
        on conflict(user_id, run_id, activity_date) do update set
            app_seconds = greatest(r.app_seconds, excluded.app_seconds),
            tracking_seconds = greatest(r.tracking_seconds, excluded.tracking_seconds),
            foundry_seconds = greatest(r.foundry_seconds, excluded.foundry_seconds),
            tracking_sessions = greatest(r.tracking_sessions, excluded.tracking_sessions),
            replay_opens = greatest(r.replay_opens, excluded.replay_opens),
            demo_opens = greatest(r.demo_opens, excluded.demo_opens),
            foundry_connections = greatest(r.foundry_connections, excluded.foundry_connections),
            live_errors = greatest(r.live_errors, excluded.live_errors), received_at = now();
    end loop;
    return true;
end;
$$;
revoke all on function public.record_alpha_usage(jsonb) from public, anon;
grant execute on function public.record_alpha_usage(jsonb) to authenticated;

-- Explicitly owner-only views: never grant these to app users or the public.
create view public.alpha_tester_metrics as
with days as (
    select user_id, activity_date from public.alpha_presence_daily
    union select user_id, activity_date from public.alpha_usage_reports
), activity as (
    select user_id, count(*) active_days,
        count(*) filter(where activity_date >= (now() at time zone 'UTC')::date - 6) active_days_7d,
        count(*) filter(where activity_date >= (now() at time zone 'UTC')::date - 29) active_days_30d
    from days group by user_id
), signins as (
    select user_id, count(*) recorded_sign_ins,
        count(*) filter(where signed_in_at >= now() - interval '30 days') sign_ins_30d
    from public.alpha_sign_ins group by user_id
), presence as (
    select user_id, min(first_seen_at) first_seen_online_at, max(last_seen_at) last_seen_online_at,
        sum(approval_checks) approval_checks,
        round((sum(estimated_online_seconds)/3600)::numeric, 2) estimated_online_hours
    from public.alpha_presence_daily group by user_id
), usage as (
    select user_id, count(distinct run_id) app_opens,
        count(distinct run_id) filter(where activity_date >= (now() at time zone 'UTC')::date - 29) app_runs_active_30d,
        round(sum(app_seconds)::numeric/3600, 2) app_open_hours,
        round(sum(tracking_seconds)::numeric/3600, 2) tracking_hours,
        round(sum(foundry_seconds)::numeric/3600, 2) foundry_connected_hours,
        sum(tracking_sessions) tracking_sessions, sum(replay_opens) replay_opens,
        sum(demo_opens) demo_opens, sum(foundry_connections) foundry_connections,
        sum(live_errors) live_errors, max(received_at) last_usage_upload_at
    from public.alpha_usage_reports group by user_id
)
select u.email, coalesce(t.approved, false) approved, u.id user_id,
    u.created_at account_created_at, u.last_sign_in_at,
    s.recorded_sign_ins, s.sign_ins_30d,
    p.first_seen_online_at, p.last_seen_online_at,
    coalesce(a.active_days,0) active_days, coalesce(a.active_days_7d,0) active_days_7d,
    coalesce(a.active_days_30d,0) active_days_30d,
    p.approval_checks, p.estimated_online_hours,
    g.app_opens, g.app_runs_active_30d, g.app_open_hours, g.tracking_hours,
    g.foundry_connected_hours, g.tracking_sessions, g.replay_opens, g.demo_opens,
    g.foundry_connections, g.live_errors, g.last_usage_upload_at,
    t.note, c.collection_started_at
from auth.users u left join public.alpha_testers t on t.user_id=u.id
left join signins s on s.user_id=u.id left join presence p on p.user_id=u.id
left join activity a on a.user_id=u.id left join usage g on g.user_id=u.id
cross join public.alpha_metrics_config c;

create view public.alpha_metrics_overview as
select count(*) accounts, count(*) filter(where approved) approved_testers,
    count(*) filter(where last_seen_online_at >= now()-interval '24 hours') online_users_24h,
    count(*) filter(where active_days_7d > 0) active_users_7d,
    count(*) filter(where active_days_30d > 0) active_users_30d,
    count(*) filter(where active_days_30d > 1) returning_users_30d,
    count(*) filter(where tracking_sessions > 0) testers_who_tracked,
    sum(recorded_sign_ins) recorded_sign_ins, sum(app_opens) app_opens,
    sum(tracking_sessions) tracking_sessions, sum(tracking_hours) tracking_hours,
    min(collection_started_at) collection_started_at
from public.alpha_tester_metrics;

revoke all on public.alpha_tester_metrics, public.alpha_metrics_overview from public, anon, authenticated;
grant select on public.alpha_tester_metrics, public.alpha_metrics_overview to service_role;
comment on view public.alpha_tester_metrics is
  'Owner-only. UTC activity dates. Recorded sign-ins include retained sessions at setup plus new sessions; not lifetime history. Null client metrics mean no instrumented app data. App-open time is not active play; online hours are heartbeat estimates.';
comment on view public.alpha_metrics_overview is
  'Owner-only engagement overview. Returning means activity on at least two UTC dates in the last 30 calendar days. Detailed client metrics begin with the metrics-enabled build.';
