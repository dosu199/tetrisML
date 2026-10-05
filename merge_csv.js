/**
 * merge_csv.js - glue value datasets together for a DAgger round.
 *
 *   node merge_csv.js data/values.csv data/dagger1.csv --out data/combined.csv
 *
 * Why not just `cat` the files: every file numbers its decisions from 0, so a
 * naive concatenation would give two unrelated decisions the same
 * decision_id - and the ranking trainer would happily build "pairs" between
 * a board from one game and a board from a completely different one. This
 * shifts each file's ids past the previous file's highest id.
 *
 * Streams line by line, so it does not need the whole dataset in memory.
 */

import { createReadStream, createWriteStream } from 'node:fs';
import { createInterface } from 'node:readline';

const argv = process.argv.slice(2);
const outIdx = argv.indexOf('--out');
const outFile = outIdx === -1 ? 'data/combined.csv' : argv[outIdx + 1];
const inputs = argv.filter((a, i) => !a.startsWith('--') && (outIdx === -1 || i !== outIdx + 1));

if (inputs.length < 2) {
  console.error('usage: node merge_csv.js a.csv b.csv [...] --out combined.csv');
  process.exit(1);
}

const out = createWriteStream(outFile);
let header = null;
let offset = 0;
let total = 0;

for (const file of inputs) {
  const lines = createInterface({ input: createReadStream(file), crlfDelay: Infinity });
  let idCol = -1;
  let maxId = -1;
  let first = true;
  let count = 0;

  for await (const line of lines) {
    if (!line) continue;
    if (first) {
      first = false;
      if (header === null) {
        header = line;
        out.write(line + '\n');
      } else if (line !== header) {
        console.error(`\n${file} has different columns from ${inputs[0]} - refusing to merge.`);
        process.exit(1);
      }
      idCol = line.split(',').indexOf('decision_id');
      if (idCol === -1) {
        console.error(`\n${file} has no decision_id column. Re-collect it with the current collect_values.js.`);
        process.exit(1);
      }
      continue;
    }

    const cells = line.split(',');
    const id = Number(cells[idCol]);
    if (id > maxId) maxId = id;
    cells[idCol] = String(id + offset);
    out.write(cells.join(',') + '\n');
    count++;
  }

  console.log(`  ${file.padEnd(28)} ${count.toLocaleString().padStart(9)} rows, ids shifted by ${offset.toLocaleString()}`);
  offset += maxId + 1;
  total += count;
}

out.end();
await new Promise((r) => out.on('finish', r));
console.log(`  ${'-> ' + outFile} ${total.toLocaleString().padStart(9)} rows\n`);
console.log(`next: python3 train_rank.py --data ${outFile}\n`);
