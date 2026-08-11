// Deck: Composing metasurfaces from measured meta-atoms — method, results,
// the a,b;c,d / D failures, the algorithmic limit, and why the fix is not
// in the aggregation.  Built with pptxgenjs; figures pre-cropped in figs/.
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5

// ---------------------------------------------------------------- palette
const NAVY = "152238";      // dark slides
const INK = "1F2937";       // body text
const GOLD = "B7791F";      // accent (gold resonators)
const GOLD_DK = "8C5A13";
const ICE = "CADCFC";       // light text on navy
const GREY = "6B7280";
const CARD = "F4F6F8";      // light card fill
const CARD_GOLD = "FAF3E3"; // gold-tinted card
const GOOD = "2E7D32";
const WARN = "B45309";
const BAD = "C0392B";
const WHITE = "FFFFFF";

const W = 13.33, M = 0.6; // slide width, margin
const FONT = "Arial";
const SERIF = "Cambria";

// ---------------------------------------------------------------- helpers
let pageNo = 0;
const TOTAL = 16;

function baseSlide(dark = false) {
  const s = pres.addSlide();
  pageNo += 1;
  s.background = { color: dark ? NAVY : WHITE };
  if (!dark) {
    s.addText(
      [
        { text: "Composing metasurfaces from measured meta-atoms", options: { color: GREY } },
        { text: `   ${pageNo} / ${TOTAL}`, options: { color: GOLD, bold: true } },
      ],
      { x: W - 5.4, y: 7.08, w: 4.8, h: 0.3, fontSize: 9, fontFace: FONT, align: "right", margin: 0 }
    );
  }
  return s;
}

function kicker(s, text, dark = false) {
  s.addText(text.toUpperCase(), {
    x: M, y: 0.32, w: W - 2 * M, h: 0.3, fontSize: 12, bold: true,
    color: dark ? ICE : GOLD, charSpacing: 2, fontFace: FONT, margin: 0,
  });
}

function title(s, text, dark = false, size = 30) {
  s.addText(text, {
    x: M, y: 0.6, w: W - 2 * M, h: 0.75, fontSize: size, bold: true,
    color: dark ? WHITE : NAVY, fontFace: FONT, margin: 0,
  });
}

function bullets(s, items, opts) {
  const runs = items.map((it, i) => ({
    text: it.t,
    options: {
      bullet: it.sub ? { code: "2013", indent: 12 } : { indent: 12 },
      indentLevel: it.sub ? 1 : 0,
      bold: !!it.b,
      color: it.c || opts.color || INK,
      breakLine: true,
      paraSpaceAfter: it.gap === undefined ? 8 : it.gap,
    },
  }));
  s.addText(runs, {
    x: opts.x, y: opts.y, w: opts.w, h: opts.h,
    fontSize: opts.fontSize || 13, fontFace: FONT, color: opts.color || INK,
    valign: "top", margin: 0, lineSpacingMultiple: 1.06,
  });
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill }, line: { color: "E3E7EC", width: 0.75 },
    rectRadius: 0.07,
  });
}

function numCircle(s, x, y, n, d = 0.44) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: GOLD }, line: { type: "none" } });
  s.addText(String(n), {
    x, y: y - 0.02, w: d, h: d, align: "center", valign: "middle",
    fontSize: 16, bold: true, color: WHITE, fontFace: FONT, margin: 0,
  });
}

function caption(s, text, x, y, w) {
  s.addText(text, { x, y, w, h: 0.55, fontSize: 9.5, italic: true, color: GREY, fontFace: FONT, margin: 0, valign: "top" });
}

// table row helper
function tRow(cells, o = {}) {
  return cells.map((c, i) => ({
    text: String(c),
    options: {
      fontSize: o.fontSize || 10.5, fontFace: FONT, color: o.colors ? o.colors[i] : (o.color || INK),
      bold: o.bold || false, align: i === 0 ? "left" : "center", valign: "middle",
      fill: { color: o.fill || WHITE },
      margin: 0.04,
    },
  }));
}

// ================================================================ SLIDE 1 — title
{
  const s = baseSlide(true);
  s.addText("T-MATRIX PIPELINE  ·  METHOD, VALIDATION, AND A DIAGNOSED LIMIT", {
    x: M, y: 1.7, w: W - 2 * M, h: 0.35, fontSize: 13, bold: true, color: GOLD, charSpacing: 3, fontFace: FONT, margin: 0,
  });
  s.addText("Composing Metasurfaces from\nMeasured Meta-Atoms", {
    x: M, y: 2.15, w: W - 2 * M, h: 1.9, fontSize: 40, bold: true, color: WHITE, fontFace: FONT, margin: 0, lineSpacingMultiple: 1.05,
  });
  s.addText(
    "Extract each meta-atom's transition matrix from one full-wave simulation, then predict whole\n" +
    "metasurfaces by linear algebra — where that works, where it breaks, and why the break is not fixable\n" +
    "inside the aggregation algorithm.",
    { x: M, y: 4.25, w: 10.6, h: 1.1, fontSize: 14, color: ICE, fontFace: FONT, margin: 0, lineSpacingMultiple: 1.15 }
  );
  s.addShape(pres.ShapeType.line, { x: M, y: 5.9, w: 4.0, h: 0, line: { color: GOLD, width: 1.5 } });
  s.addText(
    "D:\\Claude\\T matrix  ·  ten direct-CST benchmarks  ·  cross-verified against treams (KIT) to 10\u207B\u00B9\u00B2  ·  August 2026",
    { x: M, y: 6.05, w: 11.5, h: 0.4, fontSize: 11, color: "9FB3D9", fontFace: FONT, margin: 0 }
  );
}

// ================================================================ SLIDE 2 — the idea
{
  const s = baseSlide();
  kicker(s, "Method \u00B7 1 of 2");
  title(s, "One full-wave solve per atom — then metasurfaces become linear algebra");

  const cw = 3.85, gap = 0.29, cy = 1.62, ch = 3.55;
  const cards = [
    {
      h: "Extract", sub: "once per meta-atom",
      b: [
        { t: "One CST solve of the isolated atom (PML, plane-wave illumination set)" },
        { t: "Near fields sampled on a sphere, projected onto vector spherical waves (VSWFs)" },
        { t: "Least-squares fit F = T\u00B7A \u2192 the atom's transition matrix T (30\u00D730 at lmax 3)" },
      ],
    },
    {
      h: "Aggregate", sub: "per arrangement",
      b: [
        { t: "Place measured atoms on any periodic cell \u2014 one species or several" },
        { t: "Pair-resolved Ewald lattice sums couple every atom to every other, exactly" },
        { t: "Solve the block Foldy\u2013Lax system for the self-consistent excitations" },
      ],
    },
    {
      h: "Predict", sub: "milliseconds each",
      b: [
        { t: "Project onto Floquet orders \u2192 full complex S-parameters, all diffraction channels" },
        { t: "No full-wave simulation of the composed array is ever run" },
        { t: "64 min of CST per supercell benchmark vs milliseconds per composed design" },
      ],
    },
  ];
  cards.forEach((c, i) => {
    const x = M + i * (cw + gap);
    card(s, x, cy, cw, ch, CARD);
    numCircle(s, x + 0.25, cy + 0.25, i + 1);
    s.addText(c.h, { x: x + 0.85, y: cy + 0.22, w: cw - 1.0, h: 0.35, fontSize: 19, bold: true, color: NAVY, fontFace: FONT, margin: 0 });
    s.addText(c.sub, { x: x + 0.85, y: cy + 0.58, w: cw - 1.0, h: 0.28, fontSize: 10.5, italic: true, color: GOLD_DK, fontFace: FONT, margin: 0 });
    bullets(s, c.b, { x: x + 0.28, y: cy + 1.05, w: cw - 0.55, h: ch - 1.2, fontSize: 11.5 });
  });

  card(s, M, 5.5, W - 2 * M, 1.05, CARD_GOLD);
  s.addText(
    [
      { text: "The promise:  ", options: { bold: true, color: GOLD_DK } },
      { text: "simulate each atom once, then compose arbitrarily many metasurfaces from the measured parts \u2014 " +
              "the repeated cell may hold one meta-atom or several different ones.", options: { color: INK } },
    ],
    { x: M + 0.3, y: 5.66, w: W - 2 * M - 0.6, h: 0.75, fontSize: 13.5, fontFace: FONT, margin: 0, valign: "middle" }
  );
}

// ================================================================ SLIDE 3 — the algebra
{
  const s = baseSlide();
  kicker(s, "Method \u00B7 2 of 2");
  title(s, "The algebra: block-Bloch Foldy\u2013Lax with Ewald lattice sums");

  const eqx = M, eqw = 6.7, eqy = 1.62, eqh = 4.9;
  card(s, eqx, eqy, eqw, eqh, CARD);
  const eqs = [
    ["Atom response (VSWF basis)", "p  =  T a"],
    ["Pair-resolved Bloch lattice sum (Ewald)", "W\u209B\u209C(k\u2225)  =  \u03A3\u2032R  A(\u03C1\u209C \u2212 \u03C1\u209B + R) \u00B7 exp(i k\u2225\u00B7R)"],
    ["Self-consistent block solve, M atoms per cell", "(I \u2212 W T) a  =  a_inc ,     f  =  T a"],
    ["Floquet output map, order G on either side", "E_G  \u221D  (1/k_z,G) \u03A3\u209B exp(\u2212i k_G\u00B7\u03C1\u209B) F\u209B(k\u0302_G)"],
  ];
  eqs.forEach((e, i) => {
    const y = eqy + 0.28 + i * 1.15;
    s.addText(e[0], { x: eqx + 0.35, y, w: eqw - 0.7, h: 0.28, fontSize: 10.5, bold: true, color: GOLD_DK, fontFace: FONT, margin: 0 });
    s.addText(e[1], { x: eqx + 0.35, y: y + 0.3, w: eqw - 0.7, h: 0.5, fontSize: 16, italic: true, color: NAVY, fontFace: SERIF, margin: 0 });
  });

  const bx = eqx + eqw + 0.4, bw = W - M - bx;
  bullets(s, [
    { t: "Basis: vector spherical waves to lmax 3 \u2192 30 modes per atom; a four-atom cell is a dense 120\u00D7120 solve.", gap: 10 },
    { t: "Lattice sums by Ewald summation (via treams); an \u03B7-stability bracket refuses rather than guesses when the split disagrees.", gap: 10 },
    { t: "Conventions measured identical to treams: VSWFs to 7\u00D710\u207B\u00B9\u2076, plane-wave expansion 6\u00D710\u207B\u00B9\u2076, translation operator 5\u00D710\u207B\u00B9\u2075.", gap: 10 },
    { t: "Every propagating diffraction order is returned, power-normalized \u2014 supercells diffract inside the band.", gap: 0 },
  ], { x: bx, y: 1.75, w: bw, h: 4.7, fontSize: 12.5 });
}

// ================================================================ SLIDE 4 — the experiment
{
  const s = baseSlide();
  kicker(s, "The experiment");
  title(s, "Four measured gold meta-atoms, ten blind CST benchmarks");

  s.addImage({ path: "slides/figs/atoms_to_scale.png", x: M, y: 1.75, w: 5.9, h: 5.9 * (605 / 1030) });
  caption(s, "The four spoke-and-wheel resonators, drawn to scale. All are 0.2 \u00B5m-thick gold, extracted in isolation over 10\u201334 THz (25 frequencies, stored at lmax 5).",
    M, 5.45, 5.9);

  const tx = 7.0, tw = W - M - tx;
  s.addTable(
    [
      tRow(["atom", "scale", "ring outer radius", "isolated resonance / scale"], { bold: true, fill: NAVY, color: WHITE, fontSize: 10.5 }).map(c => ({ ...c, options: { ...c.options, color: WHITE, fill: { color: NAVY } } })),
      tRow(["C", "3.25", "2.338 \u00B5m", "3.766"]),
      tRow(["A", "4.00", "2.877 \u00B5m", "3.770"]),
      tRow(["B", "5.00", "3.596 \u00B5m", "3.786"]),
      tRow(["D", "5.50", "3.956 \u00B5m", "3.782"]),
    ],
    { x: tx, y: 1.75, w: tw, colW: [0.8, 0.9, 1.9, 2.13], border: { pt: 0.5, color: "D6DBE1" }, rowH: 0.34 }
  );

  bullets(s, [
    { t: "Layout: 16\u00D716 \u00B5m repeated cell, atoms on the 8 \u00B5m square sub-lattice \u2014 four single-atom lattices + six mixed supercells.", gap: 9 },
    { t: "Every case has an independent direct-CST periodic reference: ten blind benchmarks in all.", gap: 9 },
    { t: "Input imperfections are measured, not assumed: the extracted T's violate passivity by 2\u20135 % and reciprocity by up to ~18 % \u2014 the inputs, not the aggregation, set the error floor of the good cases.", gap: 0 },
  ], { x: tx, y: 3.85, w: tw, h: 2.9, fontSize: 12 });
}

// ================================================================ SLIDE 5 — aggregation exact
{
  const s = baseSlide();
  kicker(s, "Validation \u00B7 the code");
  title(s, "The aggregation itself is exact to round-off");

  const rows = [
    ["M = 1 cell reduces to the one-atom code (coupling, f, S11/S21)", "bit-identical"],
    ["2\u00D72 cell of four identical atoms \u2261 the primitive lattice", "\u2264 8\u00D710\u207B\u00B9\u2076"],
    ["basis-atom relabelling / whole-cell lattice shift", "\u2264 7\u00D710\u207B\u00B9\u2076 / 0"],
    ["checkerboard selection rule: power in odd (n\u2081+n\u2082) orders", "\u2264 6\u00D710\u207B\u00B3\u00B3"],
    ["all T = 0 \u2192 S = S_background exactly", "0"],
    ["independent treams implementation of the whole chain, every cell, complex S", "\u2264 1\u00D710\u207B\u00B9\u00B2"],
  ];
  s.addTable(
    [
      tRow(["algebraic identity / independent implementation", "residual"], { bold: true }).map(c => ({ ...c, options: { ...c.options, color: WHITE, fill: { color: NAVY }, fontSize: 11 } })),
      ...rows.map(r => tRow(r, { fontSize: 11.5 }).map((c, i) => ({ ...c, options: { ...c.options, color: i === 1 ? GOOD : INK, bold: i === 1 } }))),
    ],
    { x: M, y: 1.75, w: 7.6, colW: [5.7, 1.9], border: { pt: 0.5, color: "D6DBE1" }, rowH: 0.42 }
  );

  const cx = 8.5, cw2 = W - M - cx;
  card(s, cx, 1.75, cw2, 3.1, CARD);
  s.addText("What this rules out", { x: cx + 0.3, y: 2.0, w: cw2 - 0.6, h: 0.35, fontSize: 15, bold: true, color: NAVY, fontFace: FONT, margin: 0 });
  bullets(s, [
    { t: "coding errors in the block solve" },
    { t: "lattice-sum convergence" },
    { t: "basis conventions and normalization" },
    { t: "the Floquet output map", gap: 0 },
  ], { x: cx + 0.3, y: 2.45, w: cw2 - 0.6, h: 2.3, fontSize: 12 });

  card(s, M, 5.35, W - 2 * M, 1.15, CARD_GOLD);
  s.addText(
    [
      { text: "Consequence:  ", options: { bold: true, color: GOLD_DK } },
      { text: "whatever error remains against full-wave CST lives in what a truncated single-centre T-matrix can represent \u2014 not in the code that manipulates it.", options: { color: INK } },
    ],
    { x: M + 0.3, y: 5.5, w: W - 2 * M - 0.6, h: 0.85, fontSize: 14, fontFace: FONT, margin: 0, valign: "middle" }
  );
}

// ================================================================ SLIDE 6 — results table
{
  const s = baseSlide();
  kicker(s, "Results \u00B7 ten benchmarks");
  title(s, "Accuracy is ordered by one geometric number, \u03C1 = (a\u1D62 + a\u2C7C)/d");

  const rows = [
    ["C alone", "0.584", "0.00038", "0.017", "good", GOOD],
    ["a,c;c,a", "0.652", "0.00067", "0.019", "good", GOOD],
    ["A alone", "0.719", "0.00107", "0.030", "good", GOOD],
    ["b,c;c,b", "0.742", "0.00171", "0.036", "good", GOOD],
    ["a,b;b,a", "0.809", "0.00367", "0.046", "good", GOOD],
    ["a,d;b,c", "0.854", "0.0118", "0.080", "degrading", WARN],
    ["B alone", "0.899", "0.0039", "0.054", "degrading", WARN],
    ["a,c;d,b", "0.944", "0.1207", "0.244", "broken", BAD],
    ["a,b;c,d", "0.944", "0.1654", "0.308", "broken", BAD],
    ["D alone", "0.989", "0.0351", "0.149", "broken", BAD],
  ];
  s.addTable(
    [
      tRow(["case", "\u03C1 (worst pair)", "MSE of complex S21", "mean |\u0394S21|", "verdict"], { bold: true }).map(c => ({ ...c, options: { ...c.options, color: WHITE, fill: { color: NAVY }, fontSize: 10.5 } })),
      ...rows.map(r => tRow(r.slice(0, 5), { fontSize: 11 }).map((c, i) => ({
        ...c,
        options: { ...c.options, color: i === 4 ? r[5] : INK, bold: i === 4, fill: { color: r[5] === BAD ? "FBEDEB" : WHITE } },
      }))),
    ],
    { x: M, y: 1.7, w: 6.6, colW: [1.3, 1.3, 1.6, 1.25, 1.15], border: { pt: 0.5, color: "D6DBE1" }, rowH: 0.375 }
  );
  caption(s, "MSE = mean |S21_pred \u2212 S21_CST|\u00B2 over the band, on the complex amplitude, 0th order. Seven of ten land at mean |\u0394S21| 0.017\u20130.080 \u2014 the level set by the input T-matrices.",
    M, 6.05, 6.6);

  s.addImage({ path: "slides/figs/mse_vs_rho.png", x: 7.55, y: 1.7, w: 5.15, h: 5.15 * (700 / 870) });
  caption(s, "Same data: MSE vs \u03C1. Above \u03C1 \u2248 0.94 (shaded) the multipole series barely contracts.", 7.55, 6.15, 5.15);
}

// ================================================================ SLIDE 7 — what works
{
  const s = baseSlide();
  kicker(s, "Results \u00B7 what works");
  title(s, "Where \u03C1 is moderate, composition is essentially free");

  bullets(s, [
    { t: "Seven of ten benchmarks: mean |\u0394S21| 0.017\u20130.080 \u2014 within the input T-matrices' own passivity / reciprocity error.", gap: 10 },
    { t: "The supercell step adds nothing: a,b;b,a lands at 0.046, between its two pure lattices (0.030 and 0.054).", gap: 10 },
    { t: "A dark lattice resonance at 16.00 \u00B5m \u2014 invisible to any single-atom model \u2014 was predicted and then confirmed by direct CST: dark-channel power 1.5\u00D710\u207B\u2075 against a 0.987 carrier.", gap: 10 },
    { t: "Its transmission window: predicted 15.57 \u00B5m / |S21| 0.68, measured 15.35 \u00B5m / 0.72.", gap: 10 },
    { t: "Exact selection rule (odd diffraction orders extinguished) confirmed in every cell.", gap: 0 },
  ], { x: M, y: 1.8, w: 6.3, h: 4.6, fontSize: 13 });

  s.addImage({ path: "slides/figs/s21_singles.png", x: 7.35, y: 1.62, w: 5.15, h: 5.15 * (970 / 1060) });
  caption(s, "One atom per 8 \u00B5m cell: prediction (markers) vs direct CST (pale). C, A, B track closely \u2014 D, the largest, already fails. That failure is the subject of the rest of this deck.",
    7.35, 6.55, 5.15);
}

// ================================================================ SLIDE 8 — failure 1: a,b;c,d
{
  const s = baseSlide();
  kicker(s, "Failure 1 \u00B7 the a,b;c,d supercell");
  title(s, "Same atoms, same lattice \u2014 14\u00D7 worse when B and D become neighbours");

  s.addImage({ path: "slides/figs/s21_abcd_vs_cst.png", x: M, y: 1.62, w: 5.35, h: 5.35 * (970 / 1090) });
  caption(s, "a,b;c,d: predicted |S21| (red) vs direct CST (grey). The deepest dip is misplaced by \u22125.51 \u00B5m.", M, 6.55, 5.35);

  const rx = 6.5, rw = W - M - rx;
  const stats = [
    ["MSE 0.1654", "vs 0.0118 for a,d;b,c \u2014 identical atoms, identical lattice, different adjacency", BAD],
    ["\u22125.51 \u00B5m  /  +4.92 \u00B5m", "dip misplacement in a,b;c,d and its \u03C1-twin a,c;d,b \u2014 opposite signs, so no calibratable bias", BAD],
    ["0.409 vs 0.388", "total diffracted power of a,b;c,d, predicted vs measured (5.4 %); the \u03c1-twin a,c;d,b keeps 1.0 % (0.407 vs 0.403) \u2014 the specular channel breaks first", GOOD],
  ];
  stats.forEach((st, i) => {
    const y = 1.7 + i * 1.62;
    card(s, rx, y, rw, 1.45, CARD);
    s.addText(st[0], { x: rx + 0.3, y: y + 0.14, w: rw - 0.6, h: 0.45, fontSize: 21, bold: true, color: st[2], fontFace: FONT, margin: 0 });
    s.addText(st[1], { x: rx + 0.3, y: y + 0.62, w: rw - 0.6, h: 0.75, fontSize: 11.5, color: INK, fontFace: FONT, margin: 0, valign: "top" });
  });
}

// ================================================================ SLIDE 9 — failure 2: D alone
{
  const s = baseSlide();
  kicker(s, "Failure 2 \u00B7 atom D's own lattice");
  title(s, "Atom D alone: the resonance lands 19 % away \u2014 and red-shifted");

  s.addImage({ path: "slides/figs/s21_D_alone.png", x: M, y: 1.7, w: 6.1, h: 6.1 * (690 / 1010) });
  caption(s, "D alone on its 8 \u00B5m lattice: predicted dip 19.1 \u00B5m (repo = treams, indistinguishable) vs 23.5 \u00B5m measured; absorption goes negative.", M, 5.85, 6.1);

  const rx = 7.1, rw = W - M - rx;
  bullets(s, [
    { t: "A, B, C blue-shift from isolated to array resonance by a consistent 1.3\u20132.0 \u00B5m. D red-shifts by +2.74 \u00B5m \u2014 opposite sign.", gap: 8 },
    { t: "Across the parametric sweep the anomaly tracks the closing gap, not the atom:", gap: 6 },
  ], { x: rx, y: 1.72, w: rw, h: 1.7, fontSize: 12 });

  s.addTable(
    [
      tRow(["scale", "gap to neighbour", "CST array dip", "dip / scale"], { bold: true }).map(c => ({ ...c, options: { ...c.options, color: WHITE, fill: { color: NAVY }, fontSize: 10 } })),
      tRow(["4.00", "2.246 \u00B5m", "13.13 \u00B5m", "3.283"]),
      tRow(["4.50", "1.526 \u00B5m", "14.90 \u00B5m", "3.311"]),
      tRow(["5.00", "0.807 \u00B5m", "17.34 \u00B5m", "3.468"]),
      tRow(["5.50  (= D)", "0.088 \u00B5m", "23.51 \u00B5m", "4.275"], { bold: true }).map((c, i) => ({ ...c, options: { ...c.options, color: BAD, bold: true } })),
    ],
    { x: rx, y: 3.45, w: rw, colW: [1.3, 1.65, 1.5, 1.18], border: { pt: 0.5, color: "D6DBE1" }, rowH: 0.34 }
  );
  card(s, rx, 5.5, rw, 1.15, CARD_GOLD);
  s.addText(
    [
      { text: "88 nm apart at scale 5.5:  ", options: { bold: true, color: GOLD_DK } },
      { text: "the classic near-touching capacitive red-shift \u2014 a hybridized gap mode of the pair. Real physics, present in full-wave, absent from the prediction.", options: { color: INK } },
    ],
    { x: rx + 0.28, y: 5.62, w: rw - 0.56, h: 0.95, fontSize: 11.5, fontFace: FONT, margin: 0, valign: "middle" }
  );
}

// ================================================================ SLIDE 10 — suspects cleared
{
  const s = baseSlide();
  kicker(s, "Diagnosis \u00B7 clearing suspects");
  title(s, "The T-matrix is innocent \u2014 and so is the implementation");

  const cw3 = 5.9;
  card(s, M, 1.65, cw3, 2.35, CARD);
  s.addText("The input T-matrix is consistent", { x: M + 0.3, y: 1.82, w: cw3 - 0.6, h: 0.3, fontSize: 14, bold: true, color: NAVY, fontFace: FONT, margin: 0 });
  bullets(s, [
    { t: "Isolated resonance / scale: 3.766, 3.770, 3.786, 3.782 for C, A, B, D \u2014 perfectly linear; D is no outlier." },
    { t: "D is the most dipolar of the four (96.5 % l = 1), so truncating its own multipoles is not the issue.", gap: 0 },
  ], { x: M + 0.3, y: 2.2, w: cw3 - 0.6, h: 1.7, fontSize: 11.5 });

  card(s, M, 4.2, cw3, 2.35, CARD);
  s.addText("The code is consistent \u2014 twice over", { x: M + 0.3, y: 4.37, w: cw3 - 0.6, h: 0.3, fontSize: 14, bold: true, color: NAVY, fontFace: FONT, margin: 0 });
  bullets(s, [
    { t: "treams \u2014 an independent implementation \u2014 reproduces D's wrong answer to 1.2\u00D710\u207B\u00B9\u2075." },
    { t: "It evaluates the same addition theorem at the same lmax, so it inherits the same limit.", gap: 0 },
  ], { x: M + 0.3, y: 4.75, w: cw3 - 0.6, h: 1.7, fontSize: 11.5 });

  const ix = 6.9, iw2 = 5.85;
  s.addImage({ path: "slides/figs/s21_abcd_full.png", x: ix, y: 1.65, w: iw2, h: iw2 * (690 / 1010) });
  caption(s, "a,b;c,d: this repo (blue) and treams (orange) are indistinguishable \u2014 and both sit far from CST (grey).", ix, 5.62, iw2);

  card(s, ix, 6.1, iw2, 0.85, CARD_GOLD);
  s.addText("Cross-code agreement validates the implementation. Only full-wave validates the physics.", {
    x: ix + 0.25, y: 6.18, w: iw2 - 0.5, h: 0.7, fontSize: 12.5, bold: true, italic: true, color: GOLD_DK, fontFace: FONT, margin: 0, valign: "middle",
  });
}

// ================================================================ SLIDE 11 — the cause
{
  const s = baseSlide();
  kicker(s, "Diagnosis \u00B7 the mechanism");
  title(s, "Truncated addition theorem: error contracts by only \u03C1 per order");

  bullets(s, [
    { t: "A T-matrix describes its atom only outside the circumscribing sphere (radius a).", gap: 7 },
    { t: "Re-expanding neighbour j's scattered field about atom i converges like a geometric series in \u03C1 = (a\u1D62 + a\u2C7C)/d \u2014 truncation at lmax leaves an error ~ \u03C1^lmax.", gap: 7 },
  ], { x: M, y: 1.72, w: 6.6, h: 1.55, fontSize: 12.5 });

  s.addTable(
    [
      tRow(["\u03C1", "case", "lmax needed for ~1 %"], { bold: true }).map(c => ({ ...c, options: { ...c.options, color: WHITE, fill: { color: NAVY }, fontSize: 10.5 } })),
      tRow(["0.652", "a,c;c,a", "11"]),
      tRow(["0.809", "a,b;b,a", "22"]),
      tRow(["0.944", "a,b;c,d  (B\u2013D pair)", "80"], {}).map(c => ({ ...c, options: { ...c.options, color: BAD, bold: true } })),
      tRow(["0.989", "D alone  (D\u2013D pair)", "416"], {}).map(c => ({ ...c, options: { ...c.options, color: BAD, bold: true } })),
    ],
    { x: M, y: 3.3, w: 6.0, colW: [1.1, 2.9, 2.0], border: { pt: 0.5, color: "D6DBE1" }, rowH: 0.36 }
  );
  s.addText("Used: lmax 3.", { x: M, y: 5.15, w: 3.0, h: 0.3, fontSize: 11, italic: true, color: GREY, fontFace: FONT, margin: 0 });

  card(s, M, 5.55, 6.6, 1.15, CARD_GOLD);
  s.addText(
    [
      { text: "Silent by construction:  ", options: { bold: true, color: GOLD_DK } },
      { text: "the validity condition a\u1D62 + a\u2C7C < d is satisfied in every case (D\u2013D: 7.912 vs 8.000 \u00B5m). Marginally satisfied is worse than violated \u2014 nothing warns.", options: { color: INK } },
    ],
    { x: M + 0.28, y: 5.68, w: 6.6 - 0.56, h: 0.9, fontSize: 12, fontFace: FONT, margin: 0, valign: "middle" }
  );

  const ix = 7.5, iw3 = 2.6;
  s.addImage({ path: "slides/figs/layout_abcd.png", x: ix, y: 1.75, w: iw3, h: iw3 * (795 / 760) });
  s.addImage({ path: "slides/figs/layout_adbc.png", x: ix + iw3 + 0.15, y: 1.75, w: iw3, h: iw3 * (795 / 760) });
  caption(s, "The knob that matters: a,b;c,d puts B and D adjacent (gap 0.45 \u00B5m, \u03C1 = 0.944); a,d;b,c re-pairs the same four atoms (tightest gap 1.17 \u00B5m, \u03C1 = 0.854) and is 14\u00D7 more accurate.",
    ix, 4.6, 5.35);
}

// ================================================================ SLIDE 12 — why lmax can't save it
{
  const s = baseSlide();
  kicker(s, "Diagnosis \u00B7 the dilemma");
  title(s, "Raising lmax makes it worse: two errors move in opposite directions");

  const cw4 = 5.9;
  card(s, M, 1.65, cw4, 2.85, CARD);
  s.addText("\u2193  truncation error", { x: M + 0.3, y: 1.82, w: cw4 - 0.6, h: 0.35, fontSize: 15, bold: true, color: GOOD, fontFace: FONT, margin: 0 });
  bullets(s, [
    { t: "Falls like \u03C1^lmax \u2014 at \u03C1 = 0.944 each extra order buys a factor 0.944: lmax \u2248 80 for 1 %." },
    { t: "At lmax 3\u20135 the neglected remainder of the B\u2013D coupling is of order 1.", gap: 0 },
  ], { x: M + 0.3, y: 2.25, w: cw4 - 0.6, h: 2.1, fontSize: 11.5 });

  card(s, 6.85, 1.65, cw4, 2.85, CARD);
  s.addText("\u2191  noise amplification", { x: 7.15, y: 1.82, w: cw4 - 0.6, h: 0.35, fontSize: 15, bold: true, color: BAD, fontFace: FONT, margin: 0 });
  bullets(s, [
    { t: "The l = 4, 5 rows of the measured T sit at the extraction noise floor." },
    { t: "The near-field lattice sum amplifies exactly those rows: h\u2097(kp) ~ (2l\u22121)!! / (kp)\u02E1\u207A\u00B9 \u2014 |W| at l = 3 reaches 1025 at \u03BB \u2248 30 \u00B5m vs 1.7 at 8.8 \u00B5m: 227\u00D7 the dipole coupling there.", gap: 0 },
  ], { x: 7.15, y: 2.25, w: cw4 - 0.6, h: 2.1, fontSize: 11.5 });

  card(s, M, 4.85, W - 2 * M, 1.8, CARD_GOLD);
  s.addText("Both measured, not predicted:", { x: M + 0.3, y: 5.0, w: 5.0, h: 0.3, fontSize: 13, bold: true, color: GOLD_DK, fontFace: FONT, margin: 0 });
  bullets(s, [
    { t: "Atom D's reconstruction is worse at lmax 5 than at lmax 3 \u2014 the noise term already dominates before truncation is fixed." },
    { t: "The obvious internal check fails too: the lmax 3\u21924\u21925 spread is largest for the most accurate cell (a,d;b,c), because the spread measures noise amplification, not geometry. There is no usable convergence warning.", gap: 0 },
  ], { x: M + 0.3, y: 5.38, w: W - 2 * M - 0.6, h: 1.2, fontSize: 11.5 });
}

// ================================================================ SLIDE 13 — the limitation stated
{
  const s = baseSlide();
  kicker(s, "The limitation");
  title(s, "The limit is representational, not algorithmic");

  // schematic: two nearly-touching circumscribing spheres
  const cyc = 3.6, r1 = 1.5, r2 = 1.35;
  const c1x = 2.7, c2x = c1x + r1 + r2 + 0.12; // small gap
  s.addShape(pres.ShapeType.ellipse, { x: c1x - r1, y: cyc - r1, w: 2 * r1, h: 2 * r1, fill: { color: "FDF6E9" }, line: { color: GOLD, width: 2, dashType: "dash" } });
  s.addShape(pres.ShapeType.ellipse, { x: c2x - r2, y: cyc - r2, w: 2 * r2, h: 2 * r2, fill: { color: "FBEDEB" }, line: { color: BAD, width: 2, dashType: "dash" } });
  s.addShape(pres.ShapeType.line, { x: c1x, y: cyc, w: c2x - c1x, h: 0, line: { color: GREY, width: 1.25, dashType: "sysDot" } });
  s.addText("B", { x: c1x - 0.3, y: cyc - 0.35, w: 0.6, h: 0.4, align: "center", fontSize: 18, bold: true, color: GOLD_DK, fontFace: FONT, margin: 0 });
  s.addText("D", { x: c2x - 0.3, y: cyc - 0.35, w: 0.6, h: 0.4, align: "center", fontSize: 18, bold: true, color: BAD, fontFace: FONT, margin: 0 });
  s.addText("d = 8 \u00B5m", { x: c1x + 0.4, y: cyc + 0.08, w: 2.2, h: 0.3, fontSize: 11, color: GREY, fontFace: FONT, margin: 0 });
  s.addText("gap 0.45 \u00B5m", {
    x: (c1x + c2x) / 2 + 0.35, y: cyc - r2 - 0.55, w: 1.6, h: 0.55, fontSize: 11, bold: true, color: BAD, fontFace: FONT, margin: 0, align: "center",
  });
  s.addText("each T-matrix is valid only outside its dashed sphere \u2014 the gap region belongs to neither expansion",
    { x: 1.0, y: 5.35, w: 5.6, h: 0.75, fontSize: 10.5, italic: true, color: GREY, fontFace: FONT, margin: 0, align: "center" });

  const rx = 7.2, rw2 = W - M - rx;
  bullets(s, [
    { t: "A single-centre, lmax-3 T-matrix has 30 modes. The field in a 0.45 \u00B5m gap varies on the scale of the gap \u2014 that information does not exist in those 30 numbers.", b: true, gap: 10 },
    { t: "\u03C1 is fixed by geometry: circumscribing radii and centre distance. No solver setting, no lattice-sum method, no basis convention changes it.", gap: 10 },
    { t: "The hybridized gap mode (the measured red-shift) is reachable by the formalism only at lmax \u2248 80\u2013400 \u2014 far beyond what measured T-matrices support.", gap: 0 },
  ], { x: rx, y: 1.9, w: rw2, h: 4.4, fontSize: 13 });
}

// ================================================================ SLIDE 14 — why not aggregation
{
  const s = baseSlide();
  kicker(s, "Why the fix is not in the aggregation");
  title(s, "No modification of the aggregation alone can supply the missing physics");

  card(s, M, 1.65, W - 2 * M, 2.1, CARD);
  s.addText("Proof in one line \u2014 the Neumann sandwich", { x: M + 0.3, y: 1.8, w: 8.0, h: 0.32, fontSize: 14, bold: true, color: NAVY, fontFace: FONT, margin: 0 });
  s.addText("f  =  T a  +  T C T a  +  T C T C T a  +  \u2026", {
    x: M + 0.3, y: 2.18, w: 7.4, h: 0.45, fontSize: 17, italic: true, color: NAVY, fontFace: SERIF, margin: 0,
  });
  bullets(s, [
    { t: "Every lattice operator C is sandwiched between two copies of the stored T. T is zero outside l \u2264 3, so any higher-order content added to C is annihilated on contact \u2014 computing C exactly, at any lmax, changes nothing.", gap: 0 },
  ], { x: M + 0.3, y: 2.72, w: W - 2 * M - 0.6, h: 0.95, fontSize: 12 });

  card(s, M, 4.0, 6.75, 2.65, CARD);
  s.addText("The other obvious fix also dies \u2014 on geometry", { x: M + 0.3, y: 4.15, w: 6.2, h: 0.32, fontSize: 14, bold: true, color: NAVY, fontFace: FONT, margin: 0 });
  s.addText("Merging the near-touching pair into one extracted super-atom makes a sphere that swallows the next neighbour:", { x: M + 0.3, y: 4.5, w: 6.15, h: 0.6, fontSize: 11, color: INK, fontFace: FONT, margin: 0 });
  s.addTable(
    [
      tRow(["merge", "merged radius", "worst \u03C1 vs rest"], { bold: true }).map(c => ({ ...c, options: { ...c.options, color: WHITE, fill: { color: NAVY }, fontSize: 9.5 } })),
      tRow(["B + D", "7.956 \u00B5m", "1.211  \u2014 invalid"], { fontSize: 10 }).map((c, i) => ({ ...c, options: { ...c.options, color: i === 2 ? BAD : INK } })),
      tRow(["A + B", "7.596 \u00B5m", "1.292  \u2014 invalid"], { fontSize: 10 }).map((c, i) => ({ ...c, options: { ...c.options, color: i === 2 ? BAD : INK } })),
      tRow(["C + D", "7.956 \u00B5m", "1.292  \u2014 invalid"], { fontSize: 10 }).map((c, i) => ({ ...c, options: { ...c.options, color: i === 2 ? BAD : INK } })),
    ],
    { x: M + 0.3, y: 5.15, w: 6.1, colW: [1.4, 2.0, 2.7], border: { pt: 0.5, color: "D6DBE1" }, rowH: 0.32 }
  );
  s.addText("(\u03C1 \u2265 1 violates the validity condition outright \u2014 merging only works in sparse cells)", { x: M + 0.3, y: 6.35, w: 6.2, h: 0.3, fontSize: 9.5, italic: true, color: GREY, fontFace: FONT, margin: 0 });

  card(s, 7.6, 4.0, W - M - 7.6, 2.65, CARD_GOLD);
  s.addText("Therefore", { x: 7.9, y: 4.18, w: 4.5, h: 0.32, fontSize: 14, bold: true, color: GOLD_DK, fontFace: FONT, margin: 0 });
  s.addText(
    "The missing gap-scale information must enter upstream of the solve: the representation of each atom \u2014 i.e. the extraction \u2014 has to change, not the aggregation that consumes it.",
    { x: 7.9, y: 4.55, w: W - M - 7.6 - 0.6, h: 1.9, fontSize: 13, bold: true, color: INK, fontFace: FONT, margin: 0, lineSpacingMultiple: 1.15 }
  );
}

// ================================================================ SLIDE 15 — the fix
{
  const s = baseSlide();
  kicker(s, "The path forward");
  title(s, "Multi-centre representation: make \u03C1 an algorithmic choice again");

  bullets(s, [
    { t: "Replace one 4 \u00B5m expansion centre by ~30 sub-centres of radius r\u209B \u2248 0.5 \u00B5m tiling the ring. Since k\u00B7r\u209B \u2248 0.24, each needs only dipoles (lmax 1): ~180 modes per atom, still a trivial dense solve.", gap: 9 },
    { t: "The pair convergence ratio becomes  \u03C1_sub = 2r\u209B / (g + 2r\u209B)  \u2014 set by a chosen radius, no longer by the fixed geometry.", gap: 0 },
  ], { x: M, y: 1.72, w: 6.7, h: 2.3, fontSize: 12.5 });

  s.addTable(
    [
      tRow(["pair", "gap g", "\u03C1 today", "\u03C1_sub at r\u209B = 0.5 \u00B5m"], { bold: true }).map(c => ({ ...c, options: { ...c.options, color: WHITE, fill: { color: NAVY }, fontSize: 10 } })),
      tRow(["B\u2013D", "0.448 \u00B5m", "0.944", "0.691"], {}).map((c, i) => ({ ...c, options: { ...c.options, color: i === 3 ? GOOD : INK, bold: i >= 2 } })),
      tRow(["A\u2013D", "1.167 \u00B5m", "0.854", "0.461"], {}).map((c, i) => ({ ...c, options: { ...c.options, color: i === 3 ? GOOD : INK } })),
      tRow(["D\u2013D", "0.088 \u00B5m", "0.989", "0.919  (graded radii near the gap)"], {}).map((c, i) => ({ ...c, options: { ...c.options, color: i === 3 ? WARN : INK } })),
    ],
    { x: M, y: 4.0, w: 6.7, colW: [0.85, 1.15, 1.1, 3.6], border: { pt: 0.5, color: "D6DBE1" }, rowH: 0.36 }
  );
  caption(s, "B\u2013D at \u03C1_sub = 0.691 would sit among the best-performing cells of the study (\u03C1 \u2248 0.65\u20130.75).", M, 5.6, 6.7);

  const rx = 7.7, rw3 = W - M - rx;
  card(s, rx, 1.72, rw3, 2.2, CARD);
  s.addText("~80 % already built", { x: rx + 0.28, y: 1.88, w: rw3 - 0.56, h: 0.3, fontSize: 13.5, bold: true, color: GOOD, fontFace: FONT, margin: 0 });
  bullets(s, [
    { t: "Block lattice sums take arbitrary per-cell positions (treams shift list)" },
    { t: "Multi-centre cluster operators exist (cluster_T)" },
    { t: "Change needed: dense per-atom blocks instead of diagonal T\u2019s \u2014 a data-model edit", gap: 0 },
  ], { x: rx + 0.28, y: 2.22, w: rw3 - 0.56, h: 1.6, fontSize: 10.5 });

  card(s, rx, 4.12, rw3, 2.2, CARD);
  s.addText("The real cost: extraction", { x: rx + 0.28, y: 4.28, w: rw3 - 0.56, h: 0.3, fontSize: 13.5, bold: true, color: BAD, fontFace: FONT, margin: 0 });
  bullets(s, [
    { t: "\u2265 180 independent excitations per atom" },
    { t: "Plane waves are too smooth \u2014 near-field (dipole) sources must join the illumination set" },
    { t: "A new CST protocol; the existing conditioning machinery is the right design tool", gap: 0 },
  ], { x: rx + 0.28, y: 4.62, w: rw3 - 0.56, h: 1.6, fontSize: 10.5 });
}

// ================================================================ SLIDE 16 — closing
{
  const s = baseSlide(true);
  kicker(s, "Guardrails now \u00B7 takeaways", true);
  title(s, "What to do while the representation is rebuilt", true, 28);

  const cw5 = 3.85, gap5 = 0.29, cy5 = 1.7, ch5 = 2.5;
  const guards = [
    ["Gate on \u03C1", "Warn above 0.85, refuse above 0.95 (with --force). The \u03B7-split already refuses rather than guesses \u2014 extend the policy. Blocked only on storing the circumscribing radius in tmat.h5."],
    ["Design around it", "Never place the two largest atoms adjacent: a,d;b,c vs a,b;c,d is 14\u00D7 better for free \u2014 same atoms, same lattice."],
    ["Denoise, then lmax", "Symmetry-average the noisy l = 4, 5 rows before any lmax raise. Helps the \u03C1 \u2248 0.85 band; cannot reach 0.944."],
  ];
  guards.forEach((g, i) => {
    const x = M + i * (cw5 + gap5);
    s.addShape(pres.ShapeType.roundRect, { x, y: cy5, w: cw5, h: ch5, fill: { color: "1E2F4D" }, line: { color: "2E4368", width: 0.75 }, rectRadius: 0.07 });
    s.addText(g[0], { x: x + 0.28, y: cy5 + 0.2, w: cw5 - 0.56, h: 0.35, fontSize: 15, bold: true, color: GOLD, fontFace: FONT, margin: 0 });
    s.addText(g[1], { x: x + 0.28, y: cy5 + 0.62, w: cw5 - 0.56, h: ch5 - 0.8, fontSize: 11, color: ICE, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.1 });
  });

  s.addShape(pres.ShapeType.line, { x: M, y: 4.65, w: W - 2 * M, h: 0, line: { color: "2E4368", width: 1 } });
  const takes = [
    "Validated: composing metasurfaces from measured atoms is essentially free below \u03C1 \u2248 0.85 \u2014 accuracy is set by the inputs, not the method.",
    "The failures (a,b;c,d, a,c;d,b, atom D) are one diagnosed mechanism \u2014 slow convergence of the truncated addition theorem across sub-\u00B5m gaps \u2014 not noise, and not a bug: treams reproduces them to 10\u207B\u00B9\u2075.",
    "The fix is representational, not algorithmic: the aggregation provably cannot supply the missing gap physics; a multi-centre extraction can.",
  ];
  takes.forEach((t, i) => {
    numCircle(s, M, 4.95 + i * 0.72, i + 1, 0.36);
    s.addText(t, { x: M + 0.55, y: 4.92 + i * 0.72, w: W - 2 * M - 0.6, h: 0.7, fontSize: 12, color: WHITE, fontFace: FONT, margin: 0, valign: "top" });
  });
}

pres.writeFile({ fileName: "slides/tmatrix_composed_metasurfaces.pptx" }).then(() => {
  console.log("written slides/tmatrix_composed_metasurfaces.pptx");
});
