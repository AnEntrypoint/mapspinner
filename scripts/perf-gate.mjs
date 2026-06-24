#!/usr/bin/env node
// perf-gate.mjs -- performance regression gate (zero deps, <80 lines)
import { execFile } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { promisify } from 'node:util';

const exec = promisify(execFile);
const BASELINE_PATH = new URL('../.perf-baseline.json', import.meta.url).pathname.replace(/^\/([A-Z]:)/, '$1');
const THRESHOLD = 1.10;
const UPDATE = process.argv.includes('--update-baseline');

function readBaseline() {
  if (!existsSync(BASELINE_PATH)) return null;
  return JSON.parse(readFileSync(BASELINE_PATH, 'utf8'));
}

function writeBaseline(data) {
  writeFileSync(BASELINE_PATH, JSON.stringify(data, null, 2) + '\n');
  console.log(`[perf-gate] baseline written: ${BASELINE_PATH}`);
  console.log(JSON.stringify(data, null, 2));
}

async function runGlslCheck() {
  console.log('[perf-gate] running lab.mjs glsl-check ...');
  let stdout = '', stderr = '';
  try {
    const r = await exec(process.execPath, ['scripts/lab.mjs', 'glsl-check'], {
      cwd: new URL('..', import.meta.url).pathname.replace(/^\/([A-Z]:)/, '$1'),
      timeout: 300_000,
    });
    stdout = r.stdout; stderr = r.stderr;
  } catch (e) {
    stdout = e.stdout ?? ''; stderr = e.stderr ?? '';
    console.error('[perf-gate] glsl-check failed:\n', stderr || e.message);
    process.exit(1);
  }
  const combined = stdout + '\n' + stderr;
  const m = combined.match(/shaderCompileMs[:\s]+(\d+)/i);
  if (!m) {
    console.error('[perf-gate] could not parse shaderCompileMs from output:\n', combined.slice(0, 1000));
    process.exit(1);
  }
  return parseInt(m[1], 10);
}

async function runParity() {
  console.log('[perf-gate] running lab.mjs parity ...');
  try {
    await exec(process.execPath, ['scripts/lab.mjs', 'parity'], {
      cwd: new URL('..', import.meta.url).pathname.replace(/^\/([A-Z]:)/, '$1'),
      timeout: 300_000,
    });
    console.log('[perf-gate] parity: PASS');
  } catch (e) {
    console.error('[perf-gate] parity FAILED:\n', e.stderr || e.message);
    process.exit(1);
  }
}

async function main() {
  const shaderCompileMs = await runGlslCheck();
  console.log(`[perf-gate] shaderCompileMs = ${shaderCompileMs}`);

  if (UPDATE) {
    writeBaseline({ shaderCompileMs });
    await runParity();
    console.log('[perf-gate] baseline updated. PASS');
    process.exit(0);
  }

  const baseline = readBaseline();
  if (!baseline) {
    console.error('[perf-gate] no baseline found. Run with --update-baseline to create one.');
    process.exit(1);
  }

  const limit = Math.round(baseline.shaderCompileMs * THRESHOLD);
  console.log(`[perf-gate] baseline=${baseline.shaderCompileMs}ms  limit=${limit}ms (+10%)  measured=${shaderCompileMs}ms`);

  if (shaderCompileMs > limit) {
    console.error(`[perf-gate] REGRESSION: ${shaderCompileMs}ms > ${limit}ms (${((shaderCompileMs/baseline.shaderCompileMs-1)*100).toFixed(1)}% over baseline)`);
    process.exit(1);
  }

  await runParity();
  console.log('[perf-gate] PASS');
}

main();
