import assert from "node:assert/strict";
import {createModuleHarness} from "./module_harness.mjs";

function installRenderer(h, {resolution = 2, fail = false} = {}) {
  const targets = [], renders = [];
  const state = {fail};
  h.context.PIXI = {RenderTexture: {create(options) {
    const target = {...options, destroy(all) {this.destroyed = all;}};
    targets.push(target);
    return target;
  }}};
  h.canvas.app = {stage: h.canvas.stage, renderer: {
    screen: {width: 1920, height: 1080}, resolution,
    render(stage, options) {
      renders.push({stage, options});
      if (state.fail) throw new Error("Simulated render failure");
      options.renderTexture.content = "the-visible-stage-with-fog";
    },
    extract: {canvas(target) {
      assert.ok(target, "Reading the default framebuffer can silently return a cleared image");
      assert.equal(target.content, "the-visible-stage-with-fog");
      return {width: target.width * target.resolution, height: target.height * target.resolution,
        content: target.content};
    }},
  }};
  return {targets, renders, state};
}

{
  const h = await createModuleHarness();
  const {targets, renders} = installRenderer(h);
  const originalTransform = {...h.canvas.stage.worldTransform};
  assert.equal(h.api.captureRenderedReference(), true);
  const sent = h.sent.find(row => row.type === "renderedReference");
  assert.ok(sent);
  assert.equal(renders[0].stage, h.canvas.app.stage);
  assert.equal(renders[0].options.clear, true);
  assert.equal(renders[0].options.renderTexture, targets[0]);
  assert.equal(targets[0].width, 1920);
  assert.equal(targets[0].height, 1080);
  assert.equal(targets[0].resolution, 2);
  assert.equal(targets[0].destroyed, true);
  assert.equal(sent.capture.method, "pixi-stage-render-texture-v1");
  assert.equal(sent.capture.sourceWidth, 3840);
  assert.equal(sent.capture.sourceHeight, 2160);
  assert.equal(sent.capture.sequence, 1);
  const snapshot = h.created.find(canvas => canvas.draws?.some(row => row.kind === "image"));
  const draw = snapshot.draws.find(row => row.kind === "image").args;
  assert.equal(draw[0].content, "the-visible-stage-with-fog");
  assert.deepEqual(draw.slice(1), [56, 56, 3728, 2048, 0, 0, 640, 360]);
  assert.deepEqual(h.canvas.stage.worldTransform, originalTransform);
  assert.equal(h.pans.length, 0);
  assert.equal(h.writes.length, 0);
}

{
  const h = await createModuleHarness();
  const {targets, state} = installRenderer(h, {fail: true});
  assert.equal(h.api.captureRenderedReference(), false);
  assert.equal(targets[0].destroyed, true, "Failed captures must release GPU targets");
  assert.equal(h.sent.at(-1).type, "renderedReferenceStatus");
  assert.match(h.sent.at(-1).message, /Simulated render failure/);
  assert.equal(h.sent.filter(row => row.type === "renderedReference").length, 0);
  state.fail = false;
  assert.equal(h.api.captureRenderedReference(), true, "A failure must release the busy latch");
}

{
  const h = await createModuleHarness();
  const {targets} = installRenderer(h);
  h.canvas.app.renderer.screen = {width: 1e6, height: 1e6};
  assert.equal(h.api.captureRenderedReference(), false);
  assert.equal(targets.length, 0, "Unbounded exports must be rejected before GPU allocation");
}

{
  const h = await createModuleHarness();
  installRenderer(h, {fail: true});
  h.api.setReferenceCapture(true);
  assert.equal(h.intervals.size, 1);
  await h.advanceTime(2000);
  await h.advanceTime(2000);
  assert.equal(h.intervals.size, 0, "Repeated renderer failures must stop polling");
  assert.equal(h.sent.at(-1).stopped, true);
}

console.log("PASS: explicit viewport render target, scaled crop, cleanup, error recovery and bounded polling");
