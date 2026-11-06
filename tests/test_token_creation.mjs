import assert from "node:assert/strict";
import test from "node:test";
import {createModuleHarness} from "./module_harness.mjs";

const values = (extra = {}) => ({actorId: "goblin", name: "Scout", placement: "cell", row: 3, col: 2,
  creationId: "creation_00000001", ...extra});
const turn = () => new Promise(resolve => setImmediate(resolve));

async function harness() {
  const h = await createModuleHarness();
  h.game.world = {id: "testing", title: "Testing"};
  h.game.user = {id: "gm", name: "Gamemaster", isGM: true, can: () => true};
  const prototype = {width: 1, height: 1, texture: {src: "goblin.png"}, sight: {enabled: true, range: 40},
    actorLink: false, flags: {other: {preserved: true}}};
  h.prototypeCalls = [];
  const actor = {id: "goblin", name: "Goblin", type: "npc", testUserPermission: () => true,
    async getTokenDocument(data, options) {
      h.prototypeCalls.push({data, options});
      await h.prototypeBehavior?.();
      return {toObject: () => ({...JSON.parse(JSON.stringify(prototype)), ...data, _id: "unsaved-prototype"})};
    }};
  h.actor = actor;
  h.game.actors = new Map([[actor.id, actor]]);
  h.creationWrites = [];
  h.scene.canUserModify = () => true;
  h.scene.createEmbeddedDocuments = async (kind, data, options) => {
    assert.equal(kind, "Token");
    h.creationWrites.push({kind, data: JSON.parse(JSON.stringify(data)), options});
    const created = data.map(item => h.addToken({...item, id: `created-${h.creationWrites.length}`}));
    await h.afterCreation?.();
    return created;
  };
  return h;
}

test("state exposes current world, user, absolute pause and permitted world actors", async () => {
  const h = await harness();
  h.game.paused = true;
  h.game.actors.set("hidden", {id: "hidden", name: "Hidden", getTokenDocument() {}, testUserPermission: () => false});
  h.game.actors.set("invalid", {id: "invalid", name: "No token document"});
  const state = h.api.vttState();
  assert.deepEqual(JSON.parse(JSON.stringify(state.world)), {id: "testing", name: "Testing"});
  assert.deepEqual(JSON.parse(JSON.stringify(state.user)), {id: "gm", name: "Gamemaster", isGM: true});
  assert.equal(state.paused, true);
  assert.deepEqual(JSON.parse(JSON.stringify(state.actors)), [{actorId: "goblin", actorName: "Goblin", type: "npc"}]);
  h.game.user.isGM = false;
  assert.equal(h.api.vttState().actors.length, 0);
});

test("explicit cell creation preserves actor prototype and publishes an assignable scene token", async () => {
  const h = await harness();
  const result = await h.command("token.create", values());
  assert.equal(result.success, true);
  assert.equal(result.result.tokenId, "created-1");
  assert.equal(h.creationWrites.length, 1);
  const token = h.scene.tokens.get(result.result.tokenId);
  assert.equal(token.actorId, "goblin");
  assert.equal(token.name, "Scout");
  assert.equal(token.x, 222); assert.equal(token.y, 354);
  assert.deepEqual(token.texture, {src: "goblin.png"});
  assert.equal(token.sight.range, 40); assert.equal(token.actorLink, false);
  assert.equal(token.flags.other.preserved, true);
  assert.equal(Object.hasOwn(h.creationWrites[0].data[0], "_id"), false);
  assert.equal(result.state.tokens[0].tokenId, token.id);
  assert.equal(h.prototypeCalls[0].options.parent, h.scene);
});

test("viewport placement uses the live inverse transform and prototype dimensions", async () => {
  const h = await harness();
  Object.assign(h.canvas.stage.worldTransform, {a: 2, d: 2, tx: 100, ty: -60});
  const request = values({placement: "viewport", name: ""});
  delete request.row; delete request.col;
  const result = await h.command("token.create", request);
  assert.equal(result.success, true);
  const token = h.scene.tokens.get(result.result.tokenId);
  assert.equal(token.name, "Goblin");
  assert.equal(token.x, 422); assert.equal(token.y, 254);
});

test("same creation ID is idempotent across retries, field ordering and module reload", async () => {
  const h = await harness();
  const first = await h.command("token.create", values());
  const reordered = {creationId: "creation_00000001", col: 2, row: 3, placement: "cell", name: "Scout", actorId: "goblin"};
  const retry = await h.command("token.create", reordered);
  assert.equal(retry.result.tokenId, first.result.tokenId);
  assert.equal(h.creationWrites.length, 1);
  const reloaded = await harness();
  reloaded.scene.tokens.push(h.scene.tokens[0]);
  const afterReload = await reloaded.command("token.create", values());
  assert.equal(afterReload.result.tokenId, first.result.tokenId);
  assert.equal(reloaded.creationWrites.length, 0);
  assert.equal(reloaded.prototypeCalls.length, 0);
  const conflict = await h.command("token.create", values({name: "Different"}));
  assert.equal(conflict.success, false);
  assert.equal(h.creationWrites.length, 1);
});

test("a lost acknowledgement after persistence does not duplicate the token on retry", async () => {
  const h = await harness();
  h.afterCreation = async () => {
    const replacement = new h.socket.constructor("replacement");
    h.api.setSocket(replacement);
  };
  assert.equal(await h.command("token.create", values()), undefined);
  assert.equal(h.creationWrites.length, 1);
  h.afterCreation = null;
  const retry = await h.command("token.create", values());
  assert.equal(retry.success, true);
  assert.equal(retry.result.tokenId, "created-1");
  assert.equal(h.creationWrites.length, 1);
});

test("concurrent duplicate requests serialize to one persisted token", async () => {
  const h = await harness();
  let release;
  h.prototypeBehavior = () => new Promise(resolve => {release = resolve;});
  const first = h.command("token.create", values());
  await turn();
  const second = h.command("token.create", values());
  release();
  const result = await Promise.all([first, second]);
  assert.equal(result[0].result.tokenId, result[1].result.tokenId);
  assert.equal(h.creationWrites.length, 1);
});

test("invalid payload, permissions, capture and out-of-scene placement cause no token write", async () => {
  const h = await harness();
  for (const change of [{actorId: "missing"}, {name: "bad\nname"}, {creationId: "short"}, {row: true},
    {row: -1}, {placement: "other"}, {unknown: true}, {row: 99999}, {placement: "viewport"}]) {
    assert.equal((await h.command("token.create", values(change))).success, false);
  }
  h.actor.testUserPermission = () => false;
  assert.equal((await h.command("token.create", values())).success, false);
  h.actor.testUserPermission = () => true;
  h.game.user.isGM = false;
  assert.equal((await h.command("token.create", values())).success, false);
  h.game.user.isGM = true;
  h.api.setCapture({recording: true, phase: "placement"});
  assert.equal((await h.command("token.create", values())).success, false);
  assert.equal(h.creationWrites.length, 0);
});

test("scene, capture and permission are checked again after prototype async work", async () => {
  for (const change of [h => {h.canvas.scene = {...h.scene, id: "other"};},
                       h => h.api.setCapture({recording: true, phase: "placement"}),
                       h => {h.game.user.isGM = false;}]) {
    const h = await harness();
    h.prototypeBehavior = async () => change(h);
    assert.equal((await h.command("token.create", values())).success, false);
    assert.equal(h.creationWrites.length, 0);
  }
});

test("pause uses an absolute boolean, is idempotent, GM-only and capture protected", async () => {
  const h = await harness();
  h.game.paused = true;
  const calls = [];
  h.game.togglePause = async (paused, options) => {calls.push([paused, options.broadcast]); h.game.paused = paused;};
  assert.equal((await h.command("lifecycle.setPaused", {paused: false})).success, true);
  assert.equal((await h.command("lifecycle.setPaused", {paused: false})).state.paused, false);
  assert.deepEqual(calls, [[false, true], [false, true]]);
  assert.equal((await h.command("lifecycle.setPaused", {paused: "false"})).success, false);
  h.game.user.isGM = false;
  assert.equal((await h.command("lifecycle.setPaused", {paused: true})).success, false);
  h.game.user.isGM = true;
  h.api.setCapture({phase: "Saving"});
  assert.equal((await h.command("lifecycle.setPaused", {paused: true})).success, false);
  assert.equal(calls.length, 2);
});
