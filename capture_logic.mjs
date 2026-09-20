export const CAPTURE_MINIS = [
  {route: "A", ringColor: "red", tokenName: "Red", miniId: "red10"},
  {route: "B", ringColor: "blue", tokenName: "Blue", miniId: "blue"},
  {route: "C", ringColor: "yellow", tokenName: "Yellow", miniId: "yellow"},
  {route: "D", ringColor: "green", tokenName: "Green", miniId: "green"},
  {route: "E", ringColor: "white", tokenName: "White", miniId: "white"},
];

function columnName(index) {
  let value = index + 1;
  let name = "";
  while (value > 0) {
    value -= 1;
    name = String.fromCharCode(65 + value % 26) + name;
    value = Math.floor(value / 26);
  }
  return name;
}

export function generateCaptureTargets(bounds, participants) {
  if (!bounds || !participants.length) throw new Error("No visible grid or minis selected");
  if (bounds.firstColumn > bounds.lastColumn || bounds.firstRow > bounds.lastRow) throw new Error("Pan or zoom to show more complete grid cells");
  const used = new Set();
  const targets = [];
  for (let step = 0; step <= 5; step += 1) {
    for (const participant of participants) {
      const colorIndex = CAPTURE_MINIS.findIndex(item => item.route === participant.route);
      if (colorIndex < 0) throw new Error("Unknown mini route");
      const xFraction = 0.17 + 0.66 * step / 5;
      const yFraction = 0.18 + 0.64 * ((colorIndex + step) % 5) / 4;
      const column = Math.round(bounds.firstColumn + (bounds.lastColumn - bounds.firstColumn) * xFraction);
      const row = Math.round(bounds.firstRow + (bounds.lastRow - bounds.firstRow) * yFraction);
      const key = `${row}:${column}`;
      if (used.has(key)) throw new Error("Not enough visible grid space; zoom out or pan to show more of the map");
      used.add(key);
      targets.push({
        ...participant,
        step,
        index: targets.length,
        label: `${participant.route}${step}`,
        row,
        column,
        cell: `${columnName(column)}${row + 1}`,
      });
    }
  }
  return targets;
}
