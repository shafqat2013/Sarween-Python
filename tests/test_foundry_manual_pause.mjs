import assert from "node:assert/strict";
import test from "node:test";
import {createModuleHarness} from "./module_harness.mjs";

test("manual pause commands set absolute states and repeated requests never toggle", async () => {
  const h = await createModuleHarness();
  const calls = [];
  h.game.paused = false;
  h.game.togglePause = (paused, options) => {
    calls.push({paused, options: {...options}});
    h.game.paused = paused;
    h.hooks.get("pauseGame")(paused);
  };
  for (const desired of [true, true, false, false]) {
    const result = await h.command("lifecycle.setPaused", {paused: desired});
    assert.equal(result.success, true);
    assert.equal(result.state.paused, desired);
    assert.equal(h.sent.findLast(message => message.type === "vttState").paused, desired);
  }
  assert.deepEqual(calls, [true, true, false, false].map(paused => ({paused, options: {broadcast: true}})));
});

test("external pause events synchronize desktop status without a new command", async () => {
  const h = await createModuleHarness();
  await h.api.handlePythonMessage({type: "vttGet"});
  for (const paused of [true, false]) {
    h.game.paused = paused;
    h.hooks.get("pauseGame")(paused);
    const state = h.sent.findLast(message => message.type === "vttState");
    assert.equal(state.paused, paused);
    assert.ok(state.capabilities.includes("lifecycle.setPaused"));
  }
});

test("malformed, failed and unconfirmed pause changes never return success", async () => {
  const h = await createModuleHarness();
  h.game.paused = false;
  let calls = 0;
  h.game.togglePause = () => {calls++;};
  for (const values of [{}, {paused: 1}, {paused: "true"}, {paused: true, toggle: true}]) {
    const result = await h.command("lifecycle.setPaused", values);
    assert.equal(result.success, false);
  }
  assert.equal(calls, 0);
  const unconfirmed = await h.command("lifecycle.setPaused", {paused: true});
  assert.equal(unconfirmed.success, false);
  assert.match(unconfirmed.message, /did not confirm/);
  assert.equal(unconfirmed.state.paused, false);
  h.game.togglePause = () => {throw new Error("Pause rejected");};
  const failed = await h.command("lifecycle.setPaused", {paused: true});
  assert.equal(failed.success, false);
  assert.equal(failed.state.paused, false);
});
