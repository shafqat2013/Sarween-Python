import fs from "node:fs";
import vm from "node:vm";
import * as captureLogic from "../capture_logic.mjs";
import * as movementLogic from "../movement_logic.mjs";

const source = fs.readFileSync(new URL("../module.js", import.meta.url), "utf8");
const plain = value => JSON.parse(JSON.stringify(value));

export async function createModuleHarness() {
  const hooks = new Map(), once = new Map(), settings = new Map(), registrations = new Map();
  const sent = [], writes = [], pans = [], sockets = [], created = [];
  const timers = new Map(), timerDeadlines = new Map(), intervals = new Map(), elements = new Map(), windowListeners = new Map();
  let timerId = 0, now = 0, reloads = 0, uuid = 0;
  let setBehavior = async () => {}, panBehavior = async () => {}, updateBehavior = async () => {};
  const append = element => {if (element.id) elements.set(element.id, element); return element;};
  const document = {getElementById: id => elements.get(id), head: {appendChild: append}, body: {appendChild: append},
    createElement(tagName) {
      const element = {tagName, style: {}, dataset: {}, children: [], appendChild(child) {this.children.push(child); return append(child);},
        setAttribute(key, value) {this[key] = value;},
        remove() {if (this.id) elements.delete(this.id);}};
      if (tagName === "canvas") {
        element.draws = [];
        element.context2d = {
          setTransform(...values) {element.transform = values;},
          clearRect() {element.draws.length = 0;},
          strokeText(text, x, y) {element.draws.push({kind: "stroke", text, x, y});},
          fillText(text, x, y) {element.draws.push({kind: "fill", text, x, y});},
          drawImage(...args) {element.draws.push({kind: "image", args});},
        };
        element.getContext = name => name === "2d" ? element.context2d : null;
        element.toDataURL = () => "data:image/jpeg;base64,test";
      }
      created.push(element);
      return element;
    }};
  const tokens = [];
  tokens.get = id => tokens.find(token => token.id === id);
  const scene = {id: "scene-1", name: "Battle map", width: 4000, height: 3000, tokenVision: true,
    grid: {type: 1, size: 100, distance: 5, shiftX: 12, shiftY: 34}, tokens,
    async updateEmbeddedDocuments(_type, updates, options) {
      await updateBehavior(updates);
      for (const update of updates) await tokens.get(update._id).update({x: update.x, y: update.y}, options);
    }};
  const placeables = new Map();
  const canvas = {ready: true, scene, dimensions: {sceneX: 10, sceneY: 20},
    stage: {pivot: {x: 789, y: 456}, worldTransform: {a: 1, b: 0, c: 0, d: 1, tx: 0, ty: 0}},
    tokens: {controlled: [], get: id => placeables.get(id), releaseAll() {for (const token of [...this.controlled]) token.release();}},
    grid: {measurePath(points) {return {distance: Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y) / 100 * 5};}},
    async pan(position) {
      pans.push(plain(position)); await panBehavior(position);
      Object.assign(this.stage.worldTransform, {a: position.scale, d: position.scale});
    }};
  canvas.animatePan = position => canvas.pan(position);
  function addToken(values = {}) {
    const token = {id: `token-${tokens.length}`, name: "Red", width: 1, height: 1, x: 100, y: 200,
      sight: {enabled: true, range: 60}, actor: {system: {attributes: {movement: {walk: 30}}}}, ...values,
      async update(changes, options = {}) {Object.assign(this, changes); hooks.get("updateToken")?.(this, changes, options);}};
    const placeable = {id: token.id, document: token,
      control({releaseOthers = true} = {}) {
        if (releaseOthers) canvas.tokens.releaseAll();
        if (!canvas.tokens.controlled.includes(this)) canvas.tokens.controlled.push(this);
        hooks.get("controlToken")?.(token, true);
      },
      release() {canvas.tokens.controlled = canvas.tokens.controlled.filter(value => value !== this); hooks.get("controlToken")?.(token, false);}};
    tokens.push(token); placeables.set(token.id, placeable); return token;
  }
  class Socket {
    static OPEN = 1; static CONNECTING = 0;
    constructor(url) {this.url = url; this.readyState = Socket.OPEN; sockets.push(this);}
    send(raw) {sent.push(JSON.parse(raw));}
    close() {this.readyState = 3;}
  }
  const game = {user: {isGM: true}, modules: new Map([["sarween", {version: "test"}]]), scenes: new Map([[scene.id, scene]]), settings: {
    register(_id, key, registration) {registrations.set(key, registration); settings.set(key, registration.default);},
    get(_id, key) {return settings.get(key);},
    async set(_id, key, value) {
      writes.push({key, value}); await setBehavior(key, value); settings.set(key, value);
      await registrations.get(key)?.onChange?.(value);
    },
  }};
  const context = vm.createContext({
    console: {log() {}, warn() {}, error() {}}, Date: {now: () => now}, crypto: {randomUUID: () => `session-${++uuid}`},
    Hooks: {once(name, callback) {once.set(name, callback);}, on(name, callback) {hooks.set(name, callback);}},
    CONFIG: {Canvas: {minZoom: 1}}, game, canvas, document,
    window: {innerWidth: 1920, innerHeight: 1080, screen: {width: 1920, height: 1080},
      addEventListener(name, callback) {windowListeners.set(name, callback);},
      location: {reload() {reloads++;}}},
    ui: {notifications: {info() {}}}, WebSocket: Socket,
    Dialog: class {constructor() {throw new Error("Sarween must not open a Foundry control dialog");}},
    setTimeout(callback, delay = 0) {
      const id = ++timerId; timers.set(id, callback); timerDeadlines.set(id, now + delay); return id;
    },
    clearTimeout(id) {timers.delete(id); timerDeadlines.delete(id);},
    setInterval(callback) {const id = ++timerId; intervals.set(id, callback); return id;}, clearInterval(id) {intervals.delete(id);},
  });
  vm.runInContext("Math.clamp = (value, min, max) => Math.min(max, Math.max(min, value));", context);
  const module = new vm.SourceTextModule(source + `
    globalThis.moduleTest = {
      setSocket(value) {ws = value;}, getSocket() {return ws;},
      setCapture(value) {guidedCapture = value;}, getCapture() {return guidedCapture;},
      getMovement() {return movementState;},
      handlePythonMessage, dispatchPythonMessage, connectToPython, promptCaptureTarget, vttState,
      captureRenderedReference, setReferenceCapture,
      buildViewTransformPayload, sendViewTransform,
    };`, {context});
  await module.link(specifier => {
    const imported = specifier.endsWith("capture_logic.mjs") ? captureLogic : movementLogic;
    return new vm.SyntheticModule(Object.keys(imported), function () {
      for (const [key, value] of Object.entries(imported)) this.setExport(key, value);
    }, {context});
  });
  await module.evaluate(); once.get("init")(); settings.set("debug", false);
  const socket = new Socket("test"); context.moduleTest.setSocket(socket);
  return {api: context.moduleTest, context, hooks, once, settings, registrations, game, scene, canvas, addToken,
    sent, writes, pans, socket, sockets, created, elements, timers, intervals,
    window: context.window, windowListeners,
    get reloads() {return reloads;},
    setBehavior(callback) {setBehavior = callback;}, panBehavior(callback) {panBehavior = callback;},
    updateBehavior(callback) {updateBehavior = callback;},
    async advanceTime(ms) {now += ms; for (const callback of [...intervals.values()]) await callback();},
    async flushTimers() {
      const pending = [...timers.values()]; timers.clear(); timerDeadlines.clear();
      for (const callback of pending) await callback();
    },
    async advanceTimers(ms) {
      const target = now + ms;
      while (true) {
        const next = [...timerDeadlines].filter(([, deadline]) => deadline <= target)
          .sort((left, right) => left[1] - right[1] || left[0] - right[0])[0];
        if (!next) break;
        const [id, deadline] = next, callback = timers.get(id);
        now = deadline; timers.delete(id); timerDeadlines.delete(id);
        await callback();
      }
      now = target;
    },
    async command(action, values = {}, extra = {}) {
      const id = `vtt-${++uuid}`;
      await context.moduleTest.dispatchPythonMessage({type: "vttCommand", requestId: id, sceneId: scene.id,
        action, values, ...extra}, context.moduleTest.getSocket());
      return sent.find(message => message.type === "vttResult" && message.requestId === id);
    },
    request(settings = {}, extra = {}) {
      return context.moduleTest.dispatchPythonMessage({type: "physicalGridSet", requestId: "grid-1",
        sceneId: scene.id, settings, apply: true, ...extra}, socket);
    }};
}
