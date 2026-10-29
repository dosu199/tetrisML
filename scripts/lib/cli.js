/**
 * cli.js - command-line flags for the scripts, on top of Node's built-in
 * util.parseArgs. Every flag is declared with its default, so a typo
 * (--epoch), a missing value (--epochs at the end) or a non-number
 * (--games ten) stops the script with a clear message instead of silently
 * running with NaN or a default.
 *
 *   const opts = parseFlags({ epochs: 3000, out: 'models/dqn-model.json', hold: false, mode: STRING });
 *   opts.epochs   // a number: the default's type decides how the value is read
 *   opts.hold     // a boolean switch: --hold
 *   opts.mode     // a string, or null when not given (NUMBER: the same for a number)
 *
 * Flag names are kebab-case on the command line and camelCase in the result:
 * --eval-games 5  ->  opts.evalGames === 5.
 */

import { parseArgs } from 'node:util';
import { basename } from 'node:path';

/** Default for a text flag that has no default value (reads as null when not given). */
export const STRING = Symbol('text flag');
/** Default for a number flag that has no default value (reads as null when not given). */
export const NUMBER = Symbol('number flag');

const camel = (name) => name.replace(/-([a-z])/g, (_, ch) => ch.toUpperCase());

function describe(spec) {
  return Object.entries(spec)
    .map(([name, d]) => {
      if (d === false) return `  --${name}`;
      const isNumber = typeof d === 'number' || d === NUMBER;
      const hasDefault = d !== STRING && d !== NUMBER;
      return `  --${name} <${isNumber ? 'number' : 'text'}>${hasDefault ? `   (default: ${d})` : ''}`;
    })
    .join('\n');
}

/**
 * Parses the command line against `spec` ({ flagName: default }).
 *
 * With { passThrough: true }, flags not in `spec` are not an error: they are
 * returned untouched in `rest` (as they were typed), for a script that hands
 * them on to another script.
 *
 * @returns {object} the values by camelCase name, plus `rest` (string[]) with passThrough
 */
export function parseFlags(spec, { argv = process.argv.slice(2), passThrough = false } = {}) {
  const options = { help: { type: 'boolean' } };
  for (const [name, d] of Object.entries(spec)) options[name] = { type: d === false ? 'boolean' : 'string' };

  const script = basename(process.argv[1] ?? 'script');
  let parsed;
  try {
    parsed = parseArgs({ args: argv, options, strict: !passThrough, allowPositionals: passThrough, tokens: true });
  } catch (err) {
    fail(`${err.message}\n\nflags of ${script}:\n${describe(spec)}`);
  }
  if (parsed.values.help) {
    console.log(`flags of ${script}:\n${describe(spec)}`);
    process.exit(0);
  }

  const result = {};
  for (const [name, d] of Object.entries(spec)) {
    const raw = parsed.values[name];
    if (d === false) result[camel(name)] = raw === true;
    else if (raw === undefined) result[camel(name)] = d === STRING || d === NUMBER ? null : d;
    else if (raw === true) fail(`--${name} needs a value`);
    else if (typeof d === 'number' || d === NUMBER) result[camel(name)] = toNumber(raw, name);
    else result[camel(name)] = raw;
  }
  if (passThrough) result.rest = unknownArgs(parsed.tokens, spec);
  return result;
}

/** The flags (and their values) that are not in `spec`, exactly as typed. */
function unknownArgs(tokens, spec) {
  const rest = [];
  let lastWasUnknownFlag = false;
  for (const t of tokens) {
    if (t.kind === 'option') {
      lastWasUnknownFlag = !(t.name in spec) && t.name !== 'help';
      if (lastWasUnknownFlag) {
        rest.push(t.rawName);
        if (t.inlineValue) rest.push(t.value);
      }
    } else if (t.kind === 'positional') {
      // parseArgs cannot know whether an unknown flag takes a value, so it
      // reads "--epochs 40" as a switch plus a positional; put them back together
      if (lastWasUnknownFlag) rest.push(t.value);
      else fail(`unexpected argument: ${t.value}`);
      lastWasUnknownFlag = false;
    }
  }
  return rest;
}

function toNumber(raw, name) {
  const n = Number(raw);
  if (raw.trim() === '' || Number.isNaN(n)) fail(`--${name} must be a number, got "${raw}"`);
  return n;
}

/** Stops the script with a message (no stack trace: it is the user's input, not a bug). */
export function fail(message) {
  console.error(`\n${message}\n`);
  process.exit(2);
}

/** Checks that a value is one of `allowed`; returns it. */
export function oneOf(value, name, allowed) {
  if (!allowed.includes(value)) fail(`--${name} must be one of: ${allowed.join(', ')} (got "${value}")`);
  return value;
}

/** A comma-separated list of numbers ("1,2,3") as an array. */
export function numberList(text, name) {
  return String(text)
    .split(',')
    .filter(Boolean)
    .map((part) => toNumber(part, name));
}
