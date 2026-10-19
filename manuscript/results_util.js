// Small helpers to read result tables from paper_numbers.json.
module.exports = function (N) {
  const T = (name) => {
    const t = N.tables[name];
    if (!t) throw new Error("missing table " + name);
    return t;
  };
  const row = (name, cond) => {
    const r = T(name).find((x) => Object.entries(cond).every(([k, v]) => x[k] === v));
    if (!r) throw new Error(`no row in ${name} for ${JSON.stringify(cond)}`);
    return r;
  };
  T.__N = N;
  const f = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
  const pct = (x, d = 1) => (x === null || x === undefined ? "–" : (100 * x).toFixed(d) + "%");
  const pv = (p) => (p === null || p === undefined ? "–" : p < 0.001 ? "<0.001" : Number(p).toFixed(3));
  const ci = (r, k, d = 3) => `${f(r[k], d)} (${f(r[k + "_lo"], d)}–${f(r[k + "_hi"], d)})`;
  const best = (rows, k) => rows.reduce((a, b) => (b[k] > a[k] ? b : a));
  return { T, row, f, pct, pv, ci, best };
};
