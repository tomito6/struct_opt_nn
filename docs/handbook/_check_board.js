// Sanity check for board.html: the two inline scripts must parse, and the
// BOARD content must be consistent (unique ids, every card in a frame, every
// arrow between existing cards, every tour step pointing at something real).
//   node docs/handbook/_check_board.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, 'board.html'), 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (scripts.length !== 2) throw new Error(`expected 2 inline scripts, found ${scripts.length}`);
scripts.forEach((s, i) => new vm.Script(s, { filename: `board.html script ${i + 1}` }));

const ctx = {};
vm.runInNewContext(scripts[0] + '\nthis.BOARD = BOARD;', ctx);
const { frames, nodes, edges, tour } = ctx.BOARD;
const problems = [];
const ids = new Set();
for (const n of nodes) {
  if (ids.has(n.id)) problems.push(`duplicate card id ${n.id}`);
  ids.add(n.id);
  if (!frames.some(f => f.id === n.frame)) problems.push(`card ${n.id} in unknown frame ${n.frame}`);
  for (const [p] of n.files || []) {
    if (!fs.existsSync(path.join(__dirname, '..', '..', p))) problems.push(`card ${n.id}: missing file ${p}`);
  }
}
for (const e of edges) {
  if (!ids.has(e.from) || !ids.has(e.to)) problems.push(`arrow ${e.from} -> ${e.to} has an unknown end`);
}
for (const t of tour) {
  if (t.frame && !frames.some(f => f.id === t.frame)) problems.push(`tour step "${t.title}" -> unknown frame`);
  for (const id of t.focus || []) if (!ids.has(id)) problems.push(`tour step "${t.title}" -> unknown card ${id}`);
}
console.log(`${frames.length} frames, ${nodes.length} cards, ${edges.length} arrows, ${tour.length} tour steps`);
if (problems.length) { console.log(problems.join('\n')); process.exit(1); }
console.log('OK');
