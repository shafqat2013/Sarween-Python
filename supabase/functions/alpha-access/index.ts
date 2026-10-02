import { createHandler } from "./handler.mjs";

const publicKeys = JSON.parse(Deno.env.get("SUPABASE_PUBLISHABLE_KEYS") || "{}");
Deno.serve(createHandler({
  url: Deno.env.get("SUPABASE_URL"),
  publishableKey: publicKeys.default || Deno.env.get("SUPABASE_ANON_KEY"),
  privateKey: Deno.env.get("ALPHA_PERMISSION_PRIVATE_KEY_B64")
    ? atob(Deno.env.get("ALPHA_PERMISSION_PRIVATE_KEY_B64")!) : undefined,
  keyId: Deno.env.get("ALPHA_PERMISSION_KEY_ID"),
}));
