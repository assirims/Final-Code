// Helpers that turn simple content descriptors into docx-js objects (MDPI-like layout).
const fs = require("fs");
const path = require("path");
const {
  Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType, BorderStyle,
  ShadingType, HeadingLevel, TabStopType, VerticalAlign, PageBreak,
} = require("docx");

const FONT = "Palatino Linotype";
const BODY = 20;        // half-points (10 pt)
const SMALL = 18;       // 9 pt
const TINY = 16;        // 8 pt
const PAGE_W = 11906, MARGIN = 1134;              // A4, 2 cm margins
const TEXT_W = PAGE_W - 2 * MARGIN;               // 9638 DXA

// ---------------------------------------------------------------- citations
const refs = require("./refs");
const order = [];
function citeKeys(keys) {
  const nums = keys.map((k) => {
    if (!refs[k]) throw new Error("unknown reference key " + k);
    let i = order.indexOf(k);
    if (i < 0) { order.push(k); i = order.length - 1; }
    return i + 1;
  }).sort((a, b) => a - b);
  // compress ranges 1,2,3 -> 1-3
  const parts = []; let s = nums[0], p = nums[0];
  for (let i = 1; i <= nums.length; i++) {
    if (i < nums.length && nums[i] === p + 1) { p = nums[i]; continue; }
    parts.push(s === p ? `${s}` : (p === s + 1 ? `${s},${p}` : `${s}–${p}`));
    if (i < nums.length) { s = nums[i]; p = nums[i]; }
  }
  return "[" + parts.join(",") + "]";
}
function resolveCites(text) {
  return text.replace(/\[@([^\]]+)\]/g, (_, ks) => citeKeys(ks.split(";").map((s) => s.trim().replace(/^@/, ""))));
}

// ---------------------------------------------------------------- revision highlighting
// Text written as {+ ... +} is new or changed in the revision. With HIGHLIGHT=1 it is shaded yellow (highlighted
// manuscript); otherwise it is printed normally (clean manuscript).
const HL = process.env.HIGHLIGHT === "1";
function runs(text, base = {}) {
  text = String(text);
  if (text.includes("{+")) {
    const out = [];
    const re = /\{\+([\s\S]*?)\+\}/g;
    let last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) out.push(...runs0(text.slice(last, m.index), base));
      out.push(...runs0(m[1], HL ? { ...base, highlight: "yellow" } : base));
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push(...runs0(text.slice(last), base));
    return out;
  }
  return runs0(text, base);
}
const hl = (t) => `{+${String(t).replace(/\{\+|\+\}/g, "")}+}`;

// ---------------------------------------------------------------- inline markup: **bold**, _italic_, ^sup^, ~sub~
function runs0(text, base = {}) {
  text = resolveCites(text).replace(/=\s*<\s*/g, "< ");
  const out = [];
  const re = /(\*\*[^*]+\*\*|(?<![A-Za-z0-9])_[^_]+_(?![A-Za-z0-9])|\^[^^]+\^|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: BODY, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, size: BODY, ...base }));
    else if (t.startsWith("_")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, size: BODY, ...base }));
    else if (t.startsWith("^")) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, font: FONT, size: BODY, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), subScript: true, font: FONT, size: BODY, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: BODY, ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}), alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 80, line: 260 }, indent: opts.noIndent ? undefined : { firstLine: opts.firstLine ?? 0 },
  keepNext: opts.keepNext,
});
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs(text, { bold: true }), spacing: { before: 200, after: 80 }, keepNext: true });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(text, { italics: true }), spacing: { before: 140, after: 60 }, keepNext: true });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: runs(text, { italics: true }), spacing: { before: 100, after: 40 }, keepNext: true });

let eqNo = 0;
const Eq = (text) => {
  eqNo += 1;
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(TEXT_W / 2) }, { type: TabStopType.RIGHT, position: TEXT_W }],
    children: [new TextRun({ text: "\t", font: FONT }), ...runs(text, { font: "Cambria Math" }), new TextRun({ text: `\t(${eqNo})`, font: FONT, size: BODY })],
    spacing: { before: 60, after: 60 },
  });
};

const Bullets = (items, ref = "bullets") => items.map((t) => new Paragraph({
  numbering: { reference: ref, level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 40, line: 260 },
}));

// figure / table numbers by key (order of appearance), so that text references never go out of date
const FIG_ORDER = ["arch", "design", "flow", "fedcal", "bench", "person", "fusion", "ecg", "fed", "alarm", "streams", "cog", "robust",
  "expl", "edge", "confusion", "curves"];
const TAB_ORDER = ["requirements", "datasets", "bench", "fusion", "ecg", "fed", "forget", "alarm", "older", "cog", "robust", "edge",
  "delivery", "abbr", "hyper", "key"];
const fig = (k) => { const i = FIG_ORDER.indexOf(k); if (i < 0) throw new Error("unknown figure " + k); return `Figure ${i + 1}`; };
const tab = (k) => { const i = TAB_ORDER.indexOf(k); if (i < 0) throw new Error("unknown table " + k); return `Table ${i + 1}`; };
let figNo = 0;
function Fig(file, caption, widthIn = 6.3, key = null) {
  figNo += 1;
  if (key && FIG_ORDER.indexOf(key) + 1 !== figNo) throw new Error(`figure ${key} placed as ${figNo}`);
  const buf = fs.readFileSync(file);
  // PNG width/height from IHDR
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const W = Math.round(widthIn * 96), H = Math.round(W * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: W, height: H },
        altText: { title: `Figure ${figNo}`, description: caption.replace(/\{\+|\+\}/g, "").replace(/\*\*|\^/g, "").slice(0, 200), name: `Figure${figNo}` } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
      children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, font: FONT, size: SMALL }), ...runs(caption, { size: SMALL })] }),
  ];
}

let tabNo = 0;
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function Tab(caption, header, rows, widths, opts = {}) {
  tabNo += 1;
  if (opts.key && TAB_ORDER.indexOf(opts.key) + 1 !== tabNo) throw new Error(`table ${opts.key} placed as ${tabNo}`);
  if (opts.hl) { caption = hl(caption); header = header.map((h) => hl(h)); rows = rows.map((r) => r.map((c) => hl(c))); }
  if (opts.hlRows) rows = rows.map((r, j) => (opts.hlRows.includes(j) ? r.map((c) => hl(c)) : r));
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = TEXT_W / total;
  const W = widths.map((w) => Math.round(w * scale));
  W[W.length - 1] += TEXT_W - W.reduce((a, b) => a + b, 0);
  const cell = (txt, i, isHead, isLast, bold) => new TableCell({
    width: { size: W[i], type: WidthType.DXA },
    borders: { top: isHead ? thin : none, bottom: (isHead || isLast) ? thin : none, left: none, right: none },
    shading: isHead ? { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 30, bottom: 30, left: 60, right: 60 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(txt), { size: TINY, bold: isHead || bold }) })],
  });
  const trs = [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true, false)) })];
  rows.forEach((r, j) => {
    const bold = opts.boldRows && opts.boldRows.includes(j);
    trs.push(new TableRow({ children: r.map((c, i) => cell(c, i, false, j === rows.length - 1, bold)) }));
  });
  const out = [
    new Paragraph({ keepNext: true, spacing: { before: 160, after: 60 }, alignment: AlignmentType.JUSTIFIED,
      children: [new TextRun({ text: `Table ${tabNo}. `, bold: true, font: FONT, size: SMALL }), ...runs(caption, { size: SMALL })] }),
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: W, rows: trs }),
  ];
  if (opts.note) out.push(new Paragraph({ spacing: { before: 40, after: 160 }, children: runs(opts.note, { size: TINY }) }));
  else out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}

function ReferenceList() {
  return order.map((k, i) => new Paragraph({
    tabStops: [{ type: TabStopType.LEFT, position: 400 }],
    indent: { left: 400, hanging: 400 }, spacing: { after: 30 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text: `${i + 1}.\t`, font: FONT, size: SMALL }), new TextRun({ text: refs[k], font: FONT, size: SMALL })],
  }));
}

const disp = (n) => String(n).replace(" (dual-view)", "").replace("MiniRocket-LR", "MiniRocket-lite + LR");
module.exports = { disp, P, H1, H2, H3, Eq, Bullets, Fig, Tab, ReferenceList, runs, hl, fig, tab, HL, FONT, BODY, SMALL, TINY, TEXT_W, PAGE_W, MARGIN,
  figNum: () => figNo, tabNum: () => tabNo, PageBreak, Paragraph, TextRun, AlignmentType };
