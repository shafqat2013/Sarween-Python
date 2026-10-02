-- Recruitment/Airtable is deliberately not an input to this database.
create table public.alpha_testers (
    user_id uuid primary key references auth.users(id) on delete cascade,
    approved boolean not null default false,
    note text,
    updated_at timestamptz not null default now()
);
alter table public.alpha_testers enable row level security;
revoke all on public.alpha_testers from anon, authenticated;
grant all on public.alpha_testers to service_role;

create function public.alpha_access_status()
returns jsonb
language plpgsql stable security definer
set search_path = ''
as $$
declare
    caller uuid := auth.uid();
    sid uuid := nullif(auth.jwt() ->> 'session_id', '')::uuid;
    deadline timestamptz;
begin
    if caller is null or sid is null or auth.role() <> 'authenticated' then
        return jsonb_build_object('status', 'login_required');
    end if;
    select s.not_after into deadline from auth.sessions s
      where s.id = sid and s.user_id = caller;
    if not found or (deadline is not null and deadline <= now()) then
        return jsonb_build_object('status', 'login_required');
    end if;
    if not exists (select 1 from auth.users u where u.id = caller
                   and (u.banned_until is null or u.banned_until <= now())) then
        return jsonb_build_object('status', 'login_required');
    end if;
    if not exists (select 1 from public.alpha_testers t
                    where t.user_id = caller and t.approved) then
        return jsonb_build_object('status', 'denied');
    end if;
    return jsonb_build_object('status', 'approved', 'user_id', caller,
             'session_id', sid, 'not_after', extract(epoch from deadline));
end;
$$;
revoke all on function public.alpha_access_status() from public, anon;
grant execute on function public.alpha_access_status() to authenticated;

comment on table public.alpha_testers is
  'Owner-managed desktop alpha approval. Set approved=false to revoke. Never expose write access to app users.';
