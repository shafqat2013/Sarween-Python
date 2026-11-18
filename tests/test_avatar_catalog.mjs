import assert from "node:assert/strict";
import test from "node:test";
import {deflateSync} from "node:zlib";
import {createModuleHarness} from "./module_harness.mjs";

const crc = bytes => {
  let value = 0xFFFFFFFF;
  for (const byte of bytes) {
    value ^= byte;
    for (let bit = 0; bit < 8; bit++) value = value & 1 ? (value >>> 1) ^ 0xEDB88320 : value >>> 1;
  }
  return (value ^ 0xFFFFFFFF) >>> 0;
};
function chunk(name, bytes) {
  const type = Buffer.from(name), length = Buffer.alloc(4), checksum = Buffer.alloc(4);
  length.writeUInt32BE(bytes.length); checksum.writeUInt32BE(crc(Buffer.concat([type, bytes])));
  return Buffer.concat([length, type, bytes, checksum]);
}
function png(width, height, tint) {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width); header.writeUInt32BE(height, 4); header[8] = 8; header[9] = 6;
  const pixels = Buffer.alloc(height * (1 + 4 * width));
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const index = y * (1 + 4 * width) + 1 + x * 4;
    pixels[index] = tint >> 16 & 255; pixels[index + 1] = tint >> 8 & 255; pixels[index + 2] = tint & 255;
    pixels[index + 3] = x > 1 && y > 1 && x < width - 2 && y < height - 2 ? 255 : 0;
  }
  return "data:image/png;base64," + Buffer.concat([Buffer.from("89504e470d0a1a0a", "hex"),
    chunk("IHDR", header), chunk("IDAT", deflateSync(pixels)), chunk("IEND", Buffer.alloc(0))]).toString("base64");
}

async function setup() {
  const h = await createModuleHarness();
  const owned = [], targets = [], renders = [];
  let zoom = 1;
  class Point {constructor(x = 0, y = 0) {this.x = x; this.y = y;} set(x, y) {this.x = x; this.y = y;}}
  class Sprite {
    constructor(texture) {this.texture = texture; this.anchor = new Point(); this.scale = new Point(1, 1); this.rotation = 0; this.tint = 0xFFFFFF; owned.push(this);}
    destroy(options) {assert.equal(options.texture, false); assert.equal(options.baseTexture, false); this.destroyed = true;}
  }
  class Container {
    constructor() {this.children = []; owned.push(this);}
    addChild(sprite) {this.children.push(sprite); return sprite;}
    getLocalBounds() {
      const sprite = this.children[0], w = sprite.texture.orig.width * Math.abs(sprite.scale.x),
        height = sprite.texture.orig.height * Math.abs(sprite.scale.y), c = Math.abs(Math.cos(sprite.rotation)), s = Math.abs(Math.sin(sprite.rotation));
      return {x: -w / 2, y: -height / 2, width: w * c + height * s, height: w * s + height * c};
    }
    destroy(options) {for (const sprite of this.children) sprite.destroy(options); this.destroyed = true;}
  }
  h.context.PIXI = {Sprite, Container};
  h.canvas.app = {renderer: {
    generateTexture(container, options) {
      h.beforeRender?.();
      const sprite = container.children[0];
      renders.push({container, options, sprite: {texture: sprite.texture, tint: sprite.tint,
        rotation: sprite.rotation, scale: {...sprite.scale}, anchor: {...sprite.anchor}}});
      const target = {width: Math.ceil(options.region.width * options.resolution),
        height: Math.ceil(options.region.height * options.resolution), tint: sprite.tint,
        destroy(value) {assert.equal(value, true); this.destroyed = true;}};
      targets.push(target); return target;
    },
    extract: {canvas(target) {
      if (h.rejectExtract) throw new Error("Private asset URL must not be sent: https://private.invalid/token?secret=no");
      return target;
    }},
  }};
  const originalCreate = h.context.document.createElement;
  h.context.document.createElement = tag => {
    if (tag !== "canvas") return originalCreate(tag);
    const result = {width: 0, height: 0};
    result.getContext = () => ({drawImage(source) {result.source = source;}});
    result.toDataURL = type => {
      assert.equal(type, "image/png");
      return png(result.width, result.height, result.source.tint);
    };
    return result;
  };
  const add = (id, options = {}) => {
    const document = h.addToken({id, ...options});
    const token = h.canvas.tokens.get(id);
    const texture = options.texture ?? {uid: 1, _updateID: 1, orig: {width: 128, height: 128},
      frame: {x: 0, y: 0, width: 128, height: 128}, baseTexture: {uid: 1, _updateID: 1, valid: true}};
    const liveTransform = {untouched: true};
    token.visible = token.renderable = true;
    token.mesh = {texture, visible: true, worldVisible: true, transform: liveTransform,
      scale: {x: .5, y: .5}, anchor: {x: .5, y: .5}, rotation: 0, tint: 0xFF00FF, alpha: 1,
      filters: [], shader: {constructor: {name: "PrimaryBaseSamplerShader"}},
      getBounds() {return {x: document.x, y: document.y, width: 64 * zoom, height: 64 * zoom};}};
    return {document, token, mesh: token.mesh, liveTransform};
  };
  return {h, add, renders, targets, owned, zoom(value) {zoom = value;},
    catalog() {return h.sent.findLast(message => message.type === "avatarCatalog");}};
}

test("catalog automatically exports isolated true-alpha tinted sprites without mutating live meshes", async () => {
  const fixture = await setup(); const {h, add, renders, targets, owned} = fixture;
  const {mesh, liveTransform} = add("wizard");
  mesh.rotation = Math.PI / 4; mesh.scale.x = -.5;
  assert.equal(h.api.captureAvatarCatalog(), true);
  const catalog = fixture.catalog();
  assert.equal(catalog.schemaVersion, 1);
  assert.equal(catalog.sceneId, h.scene.id);
  assert.equal(catalog.method, "isolated-token-sprite-v1");
  assert.equal(catalog.status, "available");
  assert.equal(catalog.revision, 1);
  assert.ok(catalog.viewTransform.registration);
  assert.equal(catalog.templates.length, 1);
  const template = catalog.templates[0];
  assert.ok(template.image.startsWith("data:image/png;base64,iVBOR"));
  assert.ok(template.width <= 96 && template.height <= 96);
  assert.equal(template.logicalWidth, 64);
  assert.deepEqual(template.tokenIds, ["wizard"]);
  assert.equal(renders[0].sprite.texture, mesh.texture);
  assert.equal(renders[0].sprite.tint, 0xFF00FF);
  assert.equal(renders[0].sprite.scale.x, -.5);
  assert.equal(renders[0].sprite.rotation, Math.PI / 4);
  assert.equal(mesh.transform, liveTransform);
  assert.ok(owned.every(item => item.destroyed));
  assert.ok(targets.every(target => target.destroyed));
});

test("artwork is cached and deduplicated; position does not trigger encoding or a catalog revision", async () => {
  const fixture = await setup(); const {h, add, renders} = fixture;
  const first = add("first"), second = add("second", {x: 300});
  h.api.captureAvatarCatalog();
  assert.equal(renders.length, 1);
  assert.deepEqual(fixture.catalog().templates[0].tokenIds, ["first", "second"]);
  const count = h.sent.length;
  first.document.x = 450; second.document.y = 550;
  assert.equal(h.api.captureAvatarCatalog(), false);
  assert.equal(h.sent.length, count);
  assert.equal(renders.length, 1);
  await h.api.handlePythonMessage({type: "avatarCatalogGet"});
  assert.equal(h.sent.length, count + 1);
  assert.equal(fixture.catalog().revision, 1);
  assert.equal(renders.length, 1);
});

test("tint changes invalidate the raster while viewport scale only refreshes expected dimensions", async () => {
  const fixture = await setup(); const {h, add, renders} = fixture;
  const {mesh} = add("wizard");
  h.api.captureAvatarCatalog();
  const first = fixture.catalog();
  mesh.tint = 0x00FF00;
  h.api.captureAvatarCatalog();
  const second = fixture.catalog();
  assert.equal(renders.length, 2);
  assert.equal(second.revision, 2);
  assert.notEqual(second.templates[0].image, first.templates[0].image);
  fixture.zoom(2);
  h.api.captureAvatarCatalog();
  assert.equal(renders.length, 2);
  assert.equal(fixture.catalog().templates[0].logicalWidth, 128);
  assert.equal(fixture.catalog().revision, 3);
});

test("PIXI BaseTexture dirtyId updates invalidate cached image pixels", async () => {
  const fixture = await setup(); const {h, add, renders} = fixture;
  const {mesh} = add("wizard");
  mesh.texture.baseTexture.dirtyId = 1;
  h.api.captureAvatarCatalog();
  assert.equal(renders.length, 1);
  mesh.texture.baseTexture.dirtyId++;
  h.api.captureAvatarCatalog();
  assert.equal(renders.length, 2);
  // A re-encode that yields identical pixels need not publish a new catalog.
  assert.equal(fixture.catalog().revision, 1);
});

test("unsupported visual effects abstain and never substitute a background crop", async () => {
  const fixture = await setup(); const {h, add} = fixture;
  add("valid");
  add("ring").token.hasDynamicRing = true;
  add("filter").mesh.filters = [{}];
  add("shader").mesh.shader = {constructor: {name: "DifferentShader"}};
  add("faded").mesh.alpha = .5;
  add("movie").mesh.texture = {orig: {width: 128, height: 128}, baseTexture: {valid: true, resource: {source: {tagName: "VIDEO"}}}};
  h.api.captureAvatarCatalog();
  assert.deepEqual(fixture.catalog().templates.flatMap(row => row.tokenIds), ["valid"]);
  assert.deepEqual(fixture.catalog().unsupportedTokens.map(row => row.tokenId), ["ring", "filter", "shader", "faded", "movie"]);
  assert.equal(fixture.renders.length, 1);
});

test("failed extraction destroys owned objects and clears templates without sending asset URLs", async () => {
  const fixture = await setup(); const {h, add} = fixture;
  add("wizard"); h.rejectExtract = true;
  assert.equal(h.api.captureAvatarCatalog(), true);
  const catalog = fixture.catalog();
  assert.equal(catalog.status, "unsupported");
  assert.deepEqual(catalog.templates, []);
  assert.deepEqual(catalog.unsupportedTokens, [{tokenId: "wizard", reason: "snapshot_failed"}]);
  assert.ok(!JSON.stringify(catalog).includes("private.invalid"));
  assert.ok(fixture.targets.every(target => target.destroyed));
  assert.ok(fixture.owned.every(item => item.destroyed));
});

test("scene replacement during extraction cannot publish stale templates, and teardown clears the catalog", async () => {
  const fixture = await setup(); const {h, add} = fixture;
  add("wizard");
  h.beforeRender = () => {h.canvas.scene = {...h.scene, id: "new-scene", tokens: []};};
  assert.equal(h.api.captureAvatarCatalog(), false);
  assert.equal(fixture.catalog(), undefined);
  h.beforeRender = null;
  assert.equal(h.api.captureAvatarCatalog(), true);
  assert.equal(fixture.catalog().sceneId, "new-scene");
  assert.deepEqual(fixture.catalog().templates, []);
  h.hooks.get("canvasTearDown")();
  assert.equal(fixture.catalog().status, "unavailable");
  assert.equal(fixture.catalog().viewTransform, null);
});

test("reconnect replaces catalog identity and old extraction cannot publish through its replacement", async () => {
  const fixture = await setup(); const {h, add} = fixture;
  add("wizard");
  h.api.captureAvatarCatalog();
  const oldId = fixture.catalog().catalogId;
  h.sent.length = 0;
  h.beforeRender = () => {h.api.connectToPython();};
  h.api.resetAvatarCatalog();
  assert.equal(h.api.captureAvatarCatalog(), false);
  assert.equal(h.sent.some(row => row.type === "avatarCatalog"), false);
  h.beforeRender = null;
  h.api.getSocket().onopen();
  assert.notEqual(fixture.catalog().catalogId, oldId);
  assert.equal(fixture.catalog().revision, 1);
  assert.equal(h.sent.find(row => row.type === "hello").avatarCatalog, true);
});

test("catalog limits bound template encoding and never export hidden tokens", async () => {
  const fixture = await setup(); const {h, add} = fixture;
  for (let index = 0; index < 40; index++) add(`token-${index}`).mesh.tint = 0x100000 + index;
  add("invisible").mesh.visible = false;
  h.api.captureAvatarCatalog();
  const catalog = fixture.catalog();
  assert.equal(catalog.templates.length, 32);
  assert.equal(catalog.unsupportedTokens.length, 8);
  assert.ok(catalog.unsupportedTokens.every(row => row.reason === "catalog_budget"));
  assert.ok(!JSON.stringify(catalog).includes("invisible"));
  assert.ok(JSON.stringify(catalog).length < 1_048_576);
});

test("artwork updates revoke authority until a refreshed mesh exists, with reference recording off", async () => {
  const fixture = await setup(); const {h, add, renders} = fixture;
  const {document, token, mesh} = add("wizard");
  h.api.captureAvatarCatalog();
  assert.equal(h.intervals.size, 0);
  h.hooks.get("updateToken")(document, {texture: {tint: "#00ff00"}});
  assert.equal(fixture.catalog().status, "unavailable");
  const revokedRevision = fixture.catalog().revision;
  await h.api.handlePythonMessage({type: "avatarCatalogGet"});
  await h.advanceTimers(200);
  assert.equal(fixture.catalog().status, "unavailable");
  assert.equal(renders.length, 1);
  mesh.tint = 0x00FF00;
  h.hooks.get("refreshToken")(token, {refreshMesh: true});
  await h.advanceTimers(100);
  assert.equal(fixture.catalog().status, "available");
  assert.ok(fixture.catalog().revision > revokedRevision);
  assert.equal(renders.length, 2);
  assert.equal(renders[1].sprite.tint, 0x00FF00);
  assert.equal(h.intervals.size, 0);
});

test("async texture replacement cannot republish old pixels before draw and refresh finish", async () => {
  const fixture = await setup(); const {h, add, renders} = fixture;
  const {document, token, mesh} = add("wizard");
  h.api.captureAvatarCatalog();
  h.hooks.get("updateToken")(document, {"texture.src": "replacement.webp"});
  h.hooks.get("refreshToken")(token, {refreshState: true});
  await h.advanceTimers(300);
  await h.api.handlePythonMessage({type: "avatarCatalogGet"});
  assert.equal(fixture.catalog().status, "unavailable");
  assert.equal(renders.length, 1);
  mesh.texture = {...mesh.texture, uid: 2, baseTexture: {...mesh.texture.baseTexture, uid: 2}};
  h.hooks.get("drawToken")(token);
  await h.advanceTimers(100);
  assert.equal(fixture.catalog().status, "unavailable");
  h.hooks.get("refreshToken")(token, {refreshMesh: true, refreshVisibility: true});
  await h.advanceTimers(100);
  assert.equal(fixture.catalog().status, "available");
  assert.equal(renders.length, 2);
  assert.equal(renders[1].sprite.texture, mesh.texture);
});

test("hidden, alpha, ring, rotation and size changes refresh independently of reference capture", async () => {
  const cases = [
    {changes: {hidden: true}, flags: {refreshState: true}, change: ({token}) => {token.visible = false;},
      check: catalog => assert.equal(catalog.templates.length, 0)},
    {changes: {alpha: .5}, flags: {refreshMesh: true}, change: ({mesh}) => {mesh.alpha = .5;},
      check: catalog => assert.equal(catalog.unsupportedTokens[0].reason, "translucent_token")},
    {changes: {ring: {colors: {ring: "#ffffff"}}}, flags: {refreshRingVisuals: true},
      change: ({token}) => {token.hasDynamicRing = true;},
      check: catalog => assert.equal(catalog.unsupportedTokens[0].reason, "dynamic_ring_or_padding")},
    {changes: {rotation: 45}, flags: {refreshRotation: true}, change: ({mesh}) => {mesh.rotation = Math.PI / 4;},
      check: (_catalog, fixture) => assert.equal(fixture.renders.at(-1).sprite.rotation, Math.PI / 4)},
    {changes: {width: 2}, flags: {refreshSize: true}, change: (_token, fixture) => fixture.zoom(2),
      check: catalog => assert.equal(catalog.templates[0].logicalWidth, 128)},
  ];
  for (const scenario of cases) {
    const fixture = await setup(); const value = fixture.add("wizard"), {h} = fixture;
    h.api.captureAvatarCatalog();
    h.hooks.get("updateToken")(value.document, scenario.changes);
    assert.equal(fixture.catalog().status, "unavailable");
    scenario.change(value, fixture);
    h.hooks.get("refreshToken")(value.token, scenario.flags);
    await h.advanceTimers(100);
    assert.notEqual(fixture.catalog().status, "unavailable");
    scenario.check(fixture.catalog(), fixture);
    assert.equal(h.intervals.size, 0);
  }
});

test("create waits for actual drawing; delete immediately clears and removes the template", async () => {
  const fixture = await setup(); const {h, add, renders} = fixture;
  add("first"); h.api.captureAvatarCatalog();
  const second = add("second"); second.mesh.tint = 0x00FF00;
  h.hooks.get("createToken")(second.document);
  assert.equal(fixture.catalog().status, "unavailable");
  await h.advanceTimers(300);
  assert.equal(fixture.catalog().status, "unavailable");
  h.hooks.get("drawToken")(second.token);
  h.hooks.get("refreshToken")(second.token, {refreshMesh: true});
  await h.advanceTimers(100);
  assert.equal(fixture.catalog().templates.length, 2);
  const beforeDeleteRenders = renders.length;
  h.scene.tokens.splice(h.scene.tokens.indexOf(second.document), 1);
  h.hooks.get("deleteToken")(second.document);
  assert.equal(fixture.catalog().status, "unavailable");
  await h.advanceTimers(100);
  assert.deepEqual(fixture.catalog().templates.flatMap(item => item.tokenIds), ["first"]);
  assert.equal(renders.length, beforeDeleteRenders);
  assert.equal(h.intervals.size, 0);
});

test("visibility refresh and view geometry refresh catalogs without a reference timer", async () => {
  const fixture = await setup(); const {h, add, renders} = fixture;
  const {token} = add("wizard"); h.api.captureAvatarCatalog();
  fixture.zoom(2);
  h.canvas.stage.worldTransform.a = h.canvas.stage.worldTransform.d = 2;
  h.hooks.get("canvasPan")();
  await h.advanceTimers(200);
  assert.equal(fixture.catalog().templates[0].logicalWidth, 128);
  assert.equal(fixture.catalog().viewTransform.canvasTransform.a, 2);
  assert.equal(renders.length, 1);
  token.mesh.visible = false;
  h.hooks.get("refreshToken")(token, {refreshVisibility: true});
  await h.advanceTimers(100);
  assert.deepEqual(fixture.catalog().templates, []);
  token.mesh.visible = true;
  h.hooks.get("refreshToken")(token, {refreshVisibility: true});
  await h.advanceTimers(100);
  assert.equal(fixture.catalog().templates.length, 1);
  assert.equal(h.intervals.size, 0);
});

test("pure moves preserve artwork authority and do not encode or publish new catalogs", async () => {
  const fixture = await setup(); const {h, add, renders} = fixture;
  const {document, token} = add("wizard"); h.api.captureAvatarCatalog();
  const catalogs = h.sent.filter(item => item.type === "avatarCatalog").length;
  document.x += 100;
  h.hooks.get("updateToken")(document, {x: document.x});
  h.hooks.get("refreshToken")(token, {refreshPosition: true, refreshVisibility: true});
  await h.advanceTimers(500);
  assert.equal(h.sent.filter(item => item.type === "avatarCatalog").length, catalogs);
  assert.equal(fixture.catalog().status, "available");
  assert.equal(renders.length, 1);
});

test("inert rotation and disabled-ring edits do not wait for a render event which Foundry will not emit", async () => {
  const fixture = await setup(); const {h, add, renders} = fixture;
  const {document} = add("wizard");
  document.lockRotation = true; document.ring = {enabled: false};
  h.api.captureAvatarCatalog();
  const revision = fixture.catalog().revision;
  h.hooks.get("updateToken")(document, {rotation: 90});
  h.hooks.get("updateToken")(document, {ring: {colors: {ring: "#ffffff"}}});
  await h.advanceTimers(500);
  assert.equal(fixture.catalog().status, "available");
  assert.equal(fixture.catalog().revision, revision);
  assert.equal(renders.length, 1);
});

test("pending refresh work is scene and connection scoped", async () => {
  const fixture = await setup(); const {h, add} = fixture;
  const {document, token} = add("wizard"); h.api.captureAvatarCatalog();
  const revision = fixture.catalog().revision;
  const otherDocument = {...document, parent: {id: "other-scene"}};
  h.hooks.get("updateToken")(otherDocument, {texture: {src: "other.webp"}});
  h.hooks.get("deleteToken")(otherDocument);
  h.hooks.get("createToken")(otherDocument);
  await h.advanceTimers(100);
  assert.equal(fixture.catalog().revision, revision);
  h.hooks.get("updateToken")(document, {texture: {src: "next.webp"}});
  h.api.resetAvatarCatalog();
  const count = h.sent.length;
  await h.advanceTimers(500);
  assert.equal(h.sent.length, count);
  h.api.captureAvatarCatalog();
  assert.equal(fixture.catalog().status, "available");
  assert.equal(fixture.catalog().revision, 1);
  h.hooks.get("refreshToken")(token, {refreshState: true});
  h.api.connectToPython();
  const reconnectCount = h.sent.length;
  await h.advanceTimers(500);
  assert.equal(h.sent.length, reconnectCount);
});
