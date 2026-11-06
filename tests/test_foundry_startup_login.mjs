import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import test from "node:test";

const source = fs.readFileSync(new URL("../foundry_startup_login.js", import.meta.url), "utf8");
const parameters = {world: "testing", user: "Gamemaster", password: "", base: "http://localhost:30000"};
const gm = {id: "gm-1", name: "Gamemaster", role: 4, isGM: true, active: false};

function harness({game = {}, path = "/join", form = true, response} = {}) {
  const posts = [], timers = [], navigation = [];
  const context = vm.createContext({game, URL,
    document: {querySelector() {return form ? {} : null;}},
    location: {pathname: path, origin: "http://localhost:30000", assign(url) {navigation.push(url);}},
    setTimeout(callback) {timers.push(callback);},
    async fetch(url, options) {
      posts.push({url, options});
      return response || {ok: true, async json() {return {status: "success", redirect: "/game"};}};
    },
  });
  const prepare = vm.runInContext("(" + source + ")", context);
  return {context, prepare, posts, timers, navigation};
}

test("normal world-before-user-list loading never submits a login", async () => {
  const h = harness({game: {world: {id: "testing"}}});
  assert.equal((await h.prepare(parameters)).phase, "loading");
  assert.equal(h.posts.length, 0);
});

test("user list before the login form is rendered keeps waiting", async () => {
  const h = harness({game: {world: {id: "testing"}, users: {contents: [gm]}}, form: false});
  assert.equal((await h.prepare(parameters)).phase, "loading");
  assert.equal(h.posts.length, 0);
});

test("logged-in game still initializing does not submit another login", async () => {
  const h = harness({game: {world: {id: "testing"}, user: {...gm, active: true},
    users: {contents: [{...gm, active: true}]}, ready: false}, path: "/game"});
  assert.equal((await h.prepare(parameters)).phase, "loading");
  assert.equal(h.posts.length, 0);
  h.context.game.ready = true;
  assert.equal((await h.prepare(parameters)).phase, "joined");
});

test("ordinary join submits once across repeated startup polls and preserves session cookies", async () => {
  const h = harness({game: {world: {id: "testing"}, users: {contents: [gm]}}});
  assert.equal((await h.prepare(parameters)).phase, "joining");
  assert.equal((await h.prepare(parameters)).phase, "joining");
  assert.equal(h.posts.length, 1);
  assert.equal(h.posts[0].url, "http://localhost:30000/join");
  assert.equal(h.posts[0].options.credentials, "same-origin");
  assert.deepEqual(JSON.parse(h.posts[0].options.body), {action: "join", userid: "gm-1", password: ""});
  assert.equal(h.timers.length, 1);
  h.timers[0]();
  assert.deepEqual(h.navigation, ["http://localhost:30000/game"]);
});

test("different world, user, active user and non-GM are explicit errors without a POST", async () => {
  for (const setup of [
    {game: {world: {id: "another"}}},
    {game: {world: {id: "testing"}, ready: true, user: {...gm, name: "Someone else"}}, path: "/game"},
    {game: {world: {id: "testing"}, users: {contents: [{...gm, active: true}]}}},
    {game: {world: {id: "testing"}, users: {contents: [{...gm, isGM: false, role: 1}]}}},
  ]) {
    const h = harness(setup);
    assert.equal((await h.prepare(parameters)).phase, "error");
    assert.equal(h.posts.length, 0);
  }
});

test("login rejection and unexpected redirects never navigate or expose the password", async () => {
  for (const response of [
    {ok: false},
    {ok: true, async json() {return {status: "success", redirect: "https://example.com/game"};}},
  ]) {
    const h = harness({game: {world: {id: "testing"}, users: {contents: [gm]}}, response});
    const result = await h.prepare({...parameters, password: "not-for-logs"});
    assert.equal(result.phase, "error");
    assert.equal(JSON.stringify(result).includes("not-for-logs"), false);
    assert.equal(h.timers.length, 0);
    assert.equal(h.navigation.length, 0);
  }
});
