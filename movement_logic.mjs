export function createMovementState({miniId, tokenId, label, budgetFeet, point}) {
  const start = normalizePoint(point);
  return {
    miniId: String(miniId),
    tokenId: String(tokenId),
    label: String(label || miniId),
    budgetFeet: Math.max(0, Number(budgetFeet) || 0),
    usedFeet: 0,
    points: [start],
    segments: [],
  };
}

export function addMovementPoint(state, point, distanceFeet) {
  const next = normalizePoint(point);
  const previous = state.points.at(-1);
  if (samePoint(previous, next)) return state;
  const distance = Math.max(0, Number(distanceFeet) || 0);
  return {
    ...state,
    usedFeet: state.usedFeet + distance,
    points: [...state.points, next],
    segments: [...state.segments, {from: previous, to: next, distanceFeet: distance}],
  };
}

export function undoMovementPoint(state) {
  if (!state?.segments?.length) return {state, point: null};
  const segments = state.segments.slice(0, -1);
  const points = state.points.slice(0, -1);
  const removed = state.segments.at(-1);
  return {
    state: {
      ...state,
      usedFeet: Math.max(0, state.usedFeet - removed.distanceFeet),
      points,
      segments,
    },
    point: points.at(-1),
  };
}

export function resetMovementAt(state, point) {
  return {
    ...state,
    usedFeet: 0,
    points: [normalizePoint(point)],
    segments: [],
  };
}

export function remainingMovement(state) {
  return Math.max(0, Number(state?.budgetFeet || 0) - Number(state?.usedFeet || 0));
}

export function movementOverage(state) {
  return Math.max(0, Number(state?.usedFeet || 0) - Number(state?.budgetFeet || 0));
}

export function movementColor(state) {
  return movementOverage(state) > 0 ? "#ef4444" : "#ffffff";
}

export function fallbackGridDistance(from, to, feetPerSquare = 5) {
  const a = normalizePoint(from);
  const b = normalizePoint(to);
  const columns = Math.abs(Number(b.column) - Number(a.column));
  const rows = Math.abs(Number(b.row) - Number(a.row));
  if (Number.isFinite(columns) && Number.isFinite(rows)) {
    return Math.max(columns, rows) * Math.max(0, Number(feetPerSquare) || 0);
  }
  return 0;
}

function normalizePoint(point) {
  if (!point) throw new Error("Movement point is required");
  return {
    x: Number(point.x),
    y: Number(point.y),
    row: Number(point.row),
    column: Number(point.column),
  };
}

function samePoint(a, b) {
  return a && b && a.row === b.row && a.column === b.column;
}
