import assert from "node:assert/strict";
import {
  addMovementPoint,
  createMovementState,
  fallbackGridDistance,
  movementColor,
  movementOverage,
  remainingMovement,
  resetMovementAt,
  undoMovementPoint,
} from "../movement_logic.mjs";

const a1 = {x: 0, y: 0, row: 0, column: 0};
const c3 = {x: 100, y: 100, row: 2, column: 2};
const g3 = {x: 300, y: 100, row: 2, column: 6};

let state = createMovementState({
  miniId: "red10",
  tokenId: "red-token",
  label: "Red",
  budgetFeet: 30,
  point: a1,
});
state = addMovementPoint(state, c3, fallbackGridDistance(a1, c3, 5));
assert.equal(state.usedFeet, 10);
assert.equal(remainingMovement(state), 20);
assert.equal(movementColor(state), "#ffffff");

state = addMovementPoint(state, g3, fallbackGridDistance(c3, g3, 5));
assert.equal(state.usedFeet, 30);
assert.equal(movementColor(state), "#ffffff");
state = addMovementPoint(state, {x: 350, y: 100, row: 2, column: 7}, 5);
assert.equal(movementOverage(state), 5);
assert.equal(movementColor(state), "#ef4444");

const undone = undoMovementPoint(state);
assert.deepEqual(undone.point, g3);
assert.equal(undone.state.usedFeet, 30);

state = resetMovementAt(undone.state, c3);
assert.equal(state.usedFeet, 0);
assert.deepEqual(state.points, [c3]);
console.log("PASS: movement budget, white/red state, undo and reset");
