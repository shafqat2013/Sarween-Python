import assert from "node:assert/strict";
import {createModuleHarness} from "./module_harness.mjs";

const move = (harness, token, extra = {}) => harness.api.dispatchPythonMessage({
  type: "moveToken", commandId: "move-ack-test", sceneId: harness.scene.id,
  tokenId: token.id, miniId: "red10", x: 400, y: 500, source: "detection", ...extra,
}, harness.socket);
const reply = harness => harness.sent.filter(message => message.commandId === "move-ack-test").at(-1);

// Foundry v13 Token animation merges its interpolated coordinates into the
// prepared document. Its persisted _source has already reached the target.
{
  const harness = await createModuleHarness();
  const token = harness.addToken({_source: {x: 100, y: 200}, x: 100, y: 200});
  let databaseWrites = 0;
  harness.scene.updateEmbeddedDocuments = async (_type, updates) => {
    databaseWrites++;
    token._source.x = updates[0].x;
    token._source.y = updates[0].y;
    token.x = 150;
    token.y = 250; // Animation has not finished when the update promise resolves.
  };
  await move(harness, token);
  assert.equal(reply(harness).type, "tokenMoveApplied");
  assert.deepEqual([reply(harness).x, reply(harness).y], [400, 500]);
  assert.deepEqual([token.x, token.y], [150, 250]);
  await move(harness, token); // Retry while the prepared document still animates.
  assert.equal(reply(harness).type, "tokenMoveApplied");
  assert.equal(databaseWrites, 1, "Committed move retry must not restart an animation or budget update");
}

// A clipped or rejected update must remain an error even if the prepared
// coordinates happen to look like the requested destination.
for (const reject of [false, true]) {
  const harness = await createModuleHarness();
  const token = harness.addToken({_source: {x: 100, y: 200}, x: 400, y: 500});
  harness.scene.updateEmbeddedDocuments = async () => {
    if (reject) throw new Error("Movement rejected");
    token._source.x = 350;
    token._source.y = 500;
  };
  await move(harness, token);
  assert.equal(reply(harness).type, "tokenMoveError");
  assert.match(reply(harness).error, reject ? /Movement rejected/ : /did not apply/);
}

// Invalid source data must never fall back to matching animated coordinates.
for (const source of [null, undefined, {}, {x: "400", y: 500}, {x: NaN, y: 500}, {x: 400, y: Infinity}]) {
  const harness = await createModuleHarness();
  const token = harness.addToken({_source: source, x: 400, y: 500});
  let databaseWrites = 0;
  harness.scene.updateEmbeddedDocuments = async () => { databaseWrites++; };
  await move(harness, token);
  assert.equal(reply(harness).type, "tokenMoveError");
  assert.match(reply(harness).error, /valid persisted position/);
  assert.equal(databaseWrites, 0);
}

{
  const harness = await createModuleHarness();
  const token = harness.addToken({_source: {x: 100, y: 200}, x: 100, y: 200});
  harness.scene.updateEmbeddedDocuments = async () => {
    token.x = 400;
    token.y = 500;
    token._source = null; // A disappearing source after the await is not an ACK.
  };
  await move(harness, token);
  assert.equal(reply(harness).type, "tokenMoveError");
  assert.match(reply(harness).error, /valid persisted position/);
}

// A real Document-shaped object with missing source is not a plain stand-in.
{
  const harness = await createModuleHarness();
  const token = harness.addToken({x: 400, y: 500});
  Object.setPrototypeOf(token, {constructor: {name: "TokenDocument"}});
  await move(harness, token);
  assert.equal(reply(harness).type, "tokenMoveError");
  assert.match(reply(harness).error, /valid persisted position/);
}

// The public source serialization API is also authoritative if provided.
{
  const harness = await createModuleHarness();
  const token = harness.addToken({x: 100, y: 200,
    toObject(source) { assert.equal(source, true); return {x: 400, y: 500}; }});
  let databaseWrites = 0;
  harness.scene.updateEmbeddedDocuments = async () => { databaseWrites++; };
  await move(harness, token);
  assert.equal(reply(harness).type, "tokenMoveApplied");
  assert.equal(databaseWrites, 0);
}

console.log("PASS: persisted-source acknowledgements survive animation, retries stay idempotent, and invalid/clipped/rejected moves fail");
