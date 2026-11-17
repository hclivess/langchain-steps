// langchain-steps web interface. Plain JavaScript, no build step, no libraries.
// The server (lcsteps/server.py) runs LangChain and sends one JSON line per
// step or chain event; this file only draws what arrives.
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, props, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid);
  return e;
}

// ---- talking to the server ----------------------------------------------------
function request(url, body) {
  return fetch(url, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
}
async function json(url, body) {
  const res = await request(url, body);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
// Reads newline-delimited JSON as it arrives: one event per line.
async function* stream(url, body) {
  const res = await request(url, body);
  if (!res.ok) throw new Error((await res.json()).error || res.statusText);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, cut).trim();
      buffer = buffer.slice(cut + 1);
      if (line) yield JSON.parse(line);
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer);
}

// ---- playback: the work takes milliseconds, the display is slowed --------------
const DELAYS = { instant: [0, 0], normal: [450, 70], slow: [1400, 350] };
function pause(kind) {
  const ms = DELAYS[$('#speed').value][kind === 'step' ? 0 : 1];
  return ms ? new Promise((r) => setTimeout(r, ms)) : null;
}

// ---- shared state --------------------------------------------------------------
let files = [];                     // documents listed in the Load card
let selected = 0;
let busy = false;
let template = [];                  // the prompt template, from the server
const ingested = new Map();         // doc index -> chunks as they were ingested
const embedded = new Map();         // doc index -> Set of embedded chunk positions
let queryWords = [];

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const baseName = (p) => String(p).split(/[\\/]/).pop();
const settings = () => ({ chunk_size: +$('#size').value, overlap: +$('#overlap').value });

function setStatus(step, state, text) {
  const card = $('#s-' + step);
  card.classList.remove('run', 'done', 'skip');
  if (state) card.classList.add(state);
  $('.status', card).textContent = text || '';
}
// Mark one step as running; steps that were running become done.
function activate(group, step, text) {
  for (const id of group) {
    const card = $('#s-' + id);
    if (id === step) setStatus(id, 'run', text);
    else if (card.classList.contains('run')) setStatus(id, 'done', 'done');
  }
}
function setBusy(on) {
  busy = on;
  for (const id of ['#start', '#ask', '#reset', '#paste-add']) $(id).disabled = on;
}
function progress(which, fraction, text, isError) {
  $(`#${which}-bar`).style.width = Math.round(Math.min(1, fraction) * 100) + '%';
  const label = $(`#${which}-label`);
  label.textContent = text;
  label.classList.toggle('error', !!isError);
}
function showFacts(state) {
  if (state.embeddings) $('#f-embeddings').textContent = state.embeddings;
  if (state.model) $('#f-model').textContent = state.model;
  if (state.store) $('#f-store').textContent = state.store;
  $('#f-docs').textContent = state.stats.documents;
  $('#f-chunks').textContent = state.stats.chunks;
}

// ---- drawing: vectors ----------------------------------------------------------
// One vector as a small tile: every cell is one dimension, brighter = larger value.
function vectorTile(vector, label) {
  const dim = vector.length;
  const cols = Math.ceil(Math.sqrt(dim * 2));
  const rows = Math.ceil(dim / cols);
  const cell = dim > 2000 ? 2 : 3;
  const canvas = h('canvas', { width: cols * cell, height: rows * cell, title: `${dim} dimensions` });
  const ctx = canvas.getContext('2d');
  const max = vector.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1;
  const plus = css('--accent'), minus = css('--kw');
  vector.forEach((v, i) => {
    if (!v) return;
    ctx.globalAlpha = 0.3 + 0.7 * Math.abs(v) / max;
    ctx.fillStyle = v > 0 ? plus : minus;
    ctx.fillRect((i % cols) * cell, Math.floor(i / cols) * cell, cell, cell);
  });
  return h('div', { class: 'tile' }, canvas, label ? h('span', { text: label }) : null);
}
function tokenChips(ev) {
  if (!ev.tokens) return null;
  const chips = ev.tokens.map((t) => h('span', { class: 'tok' }, t.token + ' → ', h('b', { text: t.bucket })));
  if (ev.token_count > ev.tokens.length) chips.push(h('span', { class: 'tok', text: `+${ev.token_count - ev.tokens.length} more` }));
  return h('div', { class: 'toks' }, chips);
}

// ---- A: build the index --------------------------------------------------------
const INGEST = ['load', 'split', 'embed', 'store'];

function renderFiles() {
  const box = $('#load-live');
  box.replaceChildren();
  if (!files.length) {
    box.append(h('p', { class: 'empty', text: 'No .md or .txt files here. Pick another folder or add your own text.' }));
    return;
  }
  files.forEach((f, i) => {
    const kids = [h('span', { class: 'tick', text: f.loaded ? '✓' : '·' }), h('b', { text: f.title })];
    if (f.loaded) {
      kids.push(h('small', { text: `Document(page_content = ${f.loaded.chars} characters, metadata =` }));
      kids.push(h('dl', { class: 'meta' }, Object.entries(f.loaded.metadata).flatMap(([k, v]) => [h('dt', { text: k }), h('dd', { text: v })])));
    } else {
      kids.push(h('small', { text: f.source }));
      kids.push(h('small', { text: 'not loaded yet' }));
    }
    box.append(h('button', { type: 'button', class: 'file' + (i === selected ? ' sel' : ''), onclick: () => select(i) }, kids));
  });
}

function renderChunks(chunks, note) {
  const box = $('#split-live');
  box.replaceChildren();
  const f = files[selected];
  if (!f) { box.append(h('p', { class: 'empty', text: 'No document selected.' })); return; }
  const done = embedded.get(selected) || new Set();
  box.append(h('div', { class: 'chunk-head', text: `${f.title}: ${chunks.length} chunks · ${note}` }));
  box.append(h('div', { class: 'minimap', title: 'chunk lengths' },
    chunks.map((c) => h('i', { style: `flex: ${c.chars} 1 0`, class: done.has(c.position) ? 'emb' : '', 'data-pos': c.position }))));
  for (const c of chunks) {
    box.append(h('div', { class: 'chunk' + (done.has(c.position) ? ' emb' : ''), 'data-pos': c.position },
      h('div', null, h('span', { text: `chunk ${c.position + 1} · starts at ${c.start}` }),
        h('span', { text: `${c.chars} chars` + (c.overlap ? ` · ${c.overlap} overlap` : '') })),
      h('pre', null, c.overlap ? h('mark', { text: c.text.slice(0, c.overlap) }) : null, c.text.slice(c.overlap))));
  }
}

let previewTicket = 0;
async function preview() {
  const f = files[selected];
  if (!f) { renderChunks([], ''); return; }
  const ticket = ++previewTicket;
  const body = f.pasted ? { text: f.text, ...settings() } : { folder: $('#folder').value, source: f.source, ...settings() };
  try {
    const data = await json('/api/split', body);
    if (ticket === previewTicket) renderChunks(data.chunks, 'live preview, nothing stored yet');
  } catch (err) {
    if (ticket === previewTicket) $('#split-live').replaceChildren(h('p', { class: 'empty', text: String(err.message) }));
  }
}

function select(i) {
  selected = i;
  renderFiles();
  if (ingested.has(i)) renderChunks(ingested.get(i), 'as ingested');
  else preview();
}

async function loadFiles() {
  const pasted = files.filter((f) => f.pasted);
  ingested.clear(); embedded.clear();
  try {
    const data = await json('/api/files', { folder: $('#folder').value });
    files = [...data.files, ...pasted];
    if (!busy) progress('ingest', 0, `${files.length} documents listed. The split card shows a live preview of the selected one.`);
  } catch (err) {
    files = pasted;
    progress('ingest', 0, err.message, true);
  }
  select(0);
}

function resetIngestCards() {
  ingested.clear(); embedded.clear();
  for (const f of files) delete f.loaded;
  for (const id of INGEST) setStatus(id, '', '');
  $('#embed-now').replaceChildren();
  $('#embed-tiles').replaceChildren();
  $('#store-rows').replaceChildren();
  $('#store-live .totals')?.remove();
}

async function onIngest(ev) {
  const total = Math.max(1, files.length);
  switch (ev.step) {
    case 'plan':
      if (!ev.files.length) throw new Error('No documents to ingest: the folder has no .md or .txt files.');
      if (ev.files.length !== files.length) files = ev.files;
      renderFiles();
      break;
    case 'load':
      activate(INGEST, 'load', `${ev.doc + 1} of ${files.length}`);
      files[ev.doc].loaded = { chars: ev.chars, metadata: ev.metadata };
      selected = ev.doc;
      renderFiles();
      progress('ingest', ev.doc / total, `Loading ${ev.source}`);
      await pause('step');
      break;
    case 'split':
      activate(INGEST, 'split', `${ev.chunks.length} chunks`);
      ingested.set(ev.doc, ev.chunks);
      embedded.set(ev.doc, new Set());
      renderChunks(ev.chunks, 'as ingested');
      progress('ingest', (ev.doc + 0.25) / total, `Split ${files[ev.doc].title} into ${ev.chunks.length} chunks`);
      await pause('step');
      break;
    case 'embed': {
      activate(INGEST, 'embed', `${ev.position + 1} of ${ev.of}`);
      embedded.get(ev.doc).add(ev.position);
      const live = $('#split-live');
      for (const node of live.querySelectorAll(`[data-pos="${ev.position}"]`)) node.classList.add('emb');
      const block = $(`.chunk[data-pos="${ev.position}"]`, live);
      if (block) live.scrollTop = block.offsetTop - live.offsetTop - 60;
      $('#embed-now').replaceChildren(
        h('div', { class: 'line', text: `chunk ${ev.position + 1} of ${files[ev.doc].title} → ${ev.dim} numbers, ${ev.nonzero} not zero` }),
        tokenChips(ev));
      const tiles = $('#embed-tiles');
      $('.tile.last', tiles)?.classList.remove('last');
      const tile = vectorTile(ev.vector, `d${ev.doc + 1} c${ev.position + 1}`);
      tile.classList.add('last');
      tiles.append(tile);
      if (tiles.children.length > 300) tiles.firstChild.remove();
      progress('ingest', (ev.doc + 0.25 + 0.65 * (ev.position + 1) / ev.of) / total,
        `Embedding ${files[ev.doc].title}: chunk ${ev.position + 1} of ${ev.of}`);
      await pause('item');
      break;
    }
    case 'store':
      activate(INGEST, 'store', `${ev.rows} added`);
      $('#store-rows').append(h('tr', null,
        h('td', { text: baseName(ev.source) }),
        h('td', { text: ev.rows }),
        h('td', { text: ev.first_id ? ev.first_id.slice(0, 13) + '…' : '–' })));
      showFacts({ stats: ev.stats });
      $('#store-live .totals')?.remove();
      $('#store-live').append(h('div', { class: 'totals',
        text: `In the store now: ${ev.stats.documents} documents, ${ev.stats.chunks} chunks` }));
      progress('ingest', (ev.doc + 1) / total, `Stored ${baseName(ev.source)}`);
      await pause('step');
      break;
    case 'done':
      for (const id of INGEST) setStatus(id, 'done', 'done');
      $('.tile.last')?.classList.remove('last');
      showFacts({ stats: ev.stats });
      progress('ingest', 1, `Done: ${ev.chunks} chunks from ${ev.documents} documents in ${ev.ms} ms (real time, without playback).`);
      break;
    case 'error':
      throw new Error(ev.message);
  }
}

async function runIngest() {
  if (busy) return;
  setBusy(true);
  resetIngestCards();
  renderFiles();
  progress('ingest', 0, 'Starting…');
  const body = {
    folder: $('#folder').value,
    documents: files.filter((f) => f.pasted).map((f) => ({ source: f.source, title: f.title, text: f.text })),
    ...settings(),
  };
  try {
    for await (const ev of stream('/api/ingest', body)) await onIngest(ev);
  } catch (err) {
    for (const id of INGEST) if ($('#s-' + id).classList.contains('run')) setStatus(id, '', 'failed');
    progress('ingest', 0, 'Failed: ' + err.message, true);
  } finally {
    setBusy(false);
  }
}

// ---- B: run the chain ----------------------------------------------------------
const ASK = ['embed_query', 'retrieve', 'format', 'prompt', 'model', 'parser'];
const ASK_NAMES = {
  embed_query: 'Question embedded', retrieve: 'Chunks retrieved', format: 'Context formatted',
  prompt: 'Prompt filled', model: 'Model finished', parser: 'Output parsed',
};
const began = {};                   // card -> ms of its start event
let finished = 0;
let lastRow = null;

function highlighted(text) {
  if (!queryWords.length) return [text];
  const pattern = new RegExp(`\\b(${queryWords.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})\\b`, 'gi');
  return text.split(pattern).map((part, i) => (i % 2 ? h('mark', { class: 'q', text: part }) : part));
}
function slots(text) {
  return text.split(/(\{\w+\})/).map((part, i) => (i % 2 ? h('mark', { text: part }) : part));
}
function messageBox(role, body) {
  return h('div', { class: 'msg' }, h('div', { text: role }), h('pre', null, body));
}
function showTemplate() {
  $('#prompt-live').replaceChildren(
    h('div', { class: 'note', text: 'the template, before it is filled' }),
    ...template.map((m) => messageBox(m.role, slots(m.template))));
}

function resetAskCards() {
  for (const id of ASK) setStatus(id, '', '');
  for (const id of ['embed_query', 'format', 'model', 'parser']) $(`#${id}-live`).replaceChildren();
  $('#retrieve-live .hits').replaceChildren();
  showTemplate();
  $('#log').replaceChildren();
  $('#log-count').textContent = 'what the chain itself reports';
  for (const span of document.querySelectorAll('#code span')) span.classList.remove('run', 'done');
  for (const k of Object.keys(began)) delete began[k];
  finished = 0;
  lastRow = null;
}

// One line per event; a run of stream events from the same component is counted on one line.
function logEvent(ev, count) {
  const log = $('#log');
  const key = ev.event + ' ' + ev.name;
  lastRow = lastRow || {};
  if (ev.phase === 'stream' && lastRow[key]) {          // streams of several components interleave
    lastRow[key].n += 1;
    lastRow[key].count.textContent = '×' + lastRow[key].n;
  } else {
    const countCell = h('em');
    log.append(h('div', { class: 'row ' + ev.phase },
      h('span', { text: ev.ms.toFixed(1) + ' ms' }), h('span', { text: ev.event }), h('span', { text: ev.name }), countCell));
    if (ev.phase === 'stream') lastRow[key] = { n: 1, count: countCell };
  }
  log.scrollTop = log.scrollHeight;
  $('#log-count').textContent = `${count} events`;
}

function codeMark(card, state) {
  const span = $(`#code span[data-card="${card === 'embed_query' ? 'retrieve' : card}"]`);
  if (!span) return;
  span.classList.remove('run', 'done');
  span.classList.add(state);
}

async function onEvent(ev, count) {
  logEvent(ev, count);
  const card = ev.card;
  if (!card) return;
  const d = ev.data;

  if (ev.phase === 'start') {
    began[card] = ev.ms;
    setStatus(card, 'run', 'running');
    codeMark(card, 'run');
    if (card === 'model') {
      $('#model-live').replaceChildren(
        h('div', { class: 'note', text: `received ${d.messages.length} messages (${d.messages.map((m) => m.role).join(' + ')}); streaming:` }),
        h('div', { class: 'stream-out', id: 'tokens' }));
    }
    return;
  }

  if (ev.phase === 'stream') {
    if (card === 'model' && d.token) {
      const out = $('#tokens');
      $('.new', out)?.classList.remove('new');
      out.append(h('span', { class: 'new', text: d.token }));
      setStatus(card, 'run', `${out.children.length} pieces`);
      await pause('item');
    }
    return;
  }

  // ---- end: the component returned ----
  const took = began[card] != null ? ` · ${(ev.ms - began[card]).toFixed(1)} ms` : '';
  if (card === 'embed_query') {
    if (d.tokens) queryWords = d.tokens.map((t) => t.token).filter((t) => !t.includes(' '));
    $('#embed_query-live').replaceChildren(
      h('div', { class: 'now' },
        h('div', { class: 'line', text: `the question → ${d.dim} numbers, ${d.nonzero} not zero` }),
        tokenChips(d)),
      vectorTile(d.vector, 'question vector'));
    setStatus(card, 'done', `${d.dim} dimensions`);
  } else if (card === 'retrieve') {
    const hits = d.output || [];
    $('#retrieve-live .hits').replaceChildren(...(hits.length ? hits.map((hit, i) => h('div', { class: 'hit' },
      h('div', { class: 'meta' },
        h('span', { text: `#${i + 1} · ${baseName(hit.source)} · starts at ${hit.start_index}` }),
        h('span', { text: Number(hit.score).toFixed(4) })),
      h('div', { class: 'meter' }, h('i', { style: `width: ${Math.max(0, Math.min(1, hit.score)) * 100}%` })),
      h('p', null, highlighted(hit.text.length > 240 ? hit.text.slice(0, 240) + '…' : hit.text))))
      : [h('p', { class: 'empty', text: 'The store returned nothing.' })]));
    setStatus(card, 'done', `${hits.length} Documents${took}`);
  } else if (card === 'format') {
    $('#format-live').replaceChildren(
      h('div', { class: 'note', text: `list of Documents → one string, ${d.output.length} characters` }),
      h('pre', { text: d.output }));
    setStatus(card, 'done', `str${took}`);
  } else if (card === 'prompt') {
    $('#prompt-live').replaceChildren(
      h('div', { class: 'note', text: 'the template, filled: these messages go to the model' }),
      ...d.output.map((m) => messageBox(m.role, m.content)));
    setStatus(card, 'done', `${d.output.length} messages${took}`);
  } else if (card === 'model') {
    $('.new', $('#model-live'))?.classList.remove('new');
    if (!$('#tokens').children.length) $('#tokens').textContent = d.content;      // a model that did not stream
    if (d.usage) $('#model-live').append(h('div', { class: 'totals', text: `tokens: ${d.usage.input_tokens} in, ${d.usage.output_tokens} out` }));
    setStatus(card, 'done', `AIMessage${took}`);
  } else if (card === 'parser') {
    $('#parser-live').replaceChildren(
      h('div', { class: 'note', text: 'AIMessage → str: the chain\'s return value' }),
      h('div', { class: 'answer', text: d.output }));
    setStatus(card, 'done', `str, ${d.output.length} characters`);
  }
  codeMark(card, 'done');
  finished += 1;
  progress('ask', finished / ASK.length, ASK_NAMES[card]);
  if (card !== 'parser') await pause('step');
}

async function runAsk() {
  if (busy) return;
  const question = $('#question').value.trim();
  if (!question) { progress('ask', 0, 'Type a question first.', true); return; }
  if ($('#f-chunks').textContent === '0') { progress('ask', 0, 'The store is empty. Press Start ingest first.', true); return; }
  setBusy(true);
  resetAskCards();
  queryWords = (question.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => w.length > 2);
  progress('ask', 0, 'Starting…');
  let count = 0;
  try {
    for await (const ev of stream('/api/ask', { question, k: +$('#k').value })) {
      if (ev.step === 'error') throw new Error(ev.message);
      if (ev.step === 'event') await onEvent(ev, ++count);
      if (ev.step === 'done') progress('ask', 1, `Done: ${ev.events} events in ${ev.ms} ms (real time, without playback). Every card shows what its component returned.`);
    }
  } catch (err) {
    for (const id of ASK) if ($('#s-' + id).classList.contains('run')) setStatus(id, '', 'failed');
    progress('ask', 0, 'Failed: ' + err.message, true);
  } finally {
    setBusy(false);
  }
}

// ---- wiring --------------------------------------------------------------------
function showSettings() {
  const size = $('#size'), overlap = $('#overlap');
  if (+overlap.value >= +size.value) overlap.value = Math.max(0, +size.value - 20);
  $('#size-out').textContent = size.value + ' chars';
  $('#overlap-out').textContent = overlap.value + ' chars';
}
let debounce;
function settingsChanged() {
  showSettings();
  if (busy) return;
  ingested.clear(); embedded.clear();           // the stored chunks no longer match the sliders
  clearTimeout(debounce);
  debounce = setTimeout(preview, 60);
}

async function init() {
  const state = await json('/api/state');
  showFacts(state);
  template = state.template;
  showTemplate();
  $('#folder').value = state.defaults.folder;
  $('#size').value = state.defaults.chunk_size;
  $('#overlap').value = state.defaults.overlap;
  $('#k').value = state.defaults.k;
  showSettings();

  $('#size').addEventListener('input', settingsChanged);
  $('#overlap').addEventListener('input', settingsChanged);
  $('#folder').addEventListener('change', () => { if (!busy) loadFiles(); });
  $('#ingest-form').addEventListener('submit', (e) => { e.preventDefault(); runIngest(); });
  $('#ask-form').addEventListener('submit', (e) => { e.preventDefault(); runAsk(); });
  $('#reset').addEventListener('click', async () => {
    if (busy) return;
    showFacts(await json('/api/reset', {}));
    resetIngestCards();
    renderFiles();
    progress('ingest', 0, 'Store emptied.');
  });
  $('#paste-add').addEventListener('click', () => {
    const text = $('#paste-text').value;
    if (!text.trim() || busy) return;
    const name = $('#paste-name').value.trim() || `pasted-${files.filter((f) => f.pasted).length + 1}`;
    files.push({ source: name, title: name, pasted: true, text });
    $('#paste-text').value = ''; $('#paste-name').value = '';
    ingested.clear(); embedded.clear();
    select(files.length - 1);
  });

  await loadFiles();

  // http://127.0.0.1:8766/?autorun  runs both halves once, without playback delay.
  const params = new URLSearchParams(location.search);
  if (params.has('autorun')) {
    $('#speed').value = 'instant';
    if (params.get('autorun')) $('#question').value = params.get('autorun');
    await runIngest();
    await runAsk();
    document.title = 'ready · ' + document.title;
  }
}

init().catch((err) => progress('ingest', 0, 'Could not reach the server: ' + err.message, true));
