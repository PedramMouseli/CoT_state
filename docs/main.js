// Real examples from the saved rollouts and experiments (Qwen3-8B).
const DATA = {
 "direct": {
  "steps": [
   {
    "obs": "Initial state: you are in /logs.\nalpha.log: locked\nbeta.log: locked\ngamma.log: unlocked\nreport.txt: unlocked\nBegin.",
    "state": {
     "alpha.log": [
      1,
      1
     ],
     "beta.log": [
      0,
      1
     ],
     "gamma.log": [
      1,
      0
     ],
     "report.txt": [
      1,
      0
     ]
    },
    "cmd": "ls"
   },
   {
    "obs": "/logs: alpha.log gamma.log report.txt",
    "state": {
     "alpha.log": [
      1,
      1
     ],
     "beta.log": [
      0,
      1
     ],
     "gamma.log": [
      1,
      0
     ],
     "report.txt": [
      1,
      0
     ]
    },
    "cmd": "stat report.txt"
   },
   {
    "obs": "report.txt: unlocked\n[system] background job TOGGLED the lock on alpha.log",
    "state": {
     "alpha.log": [
      1,
      0
     ],
     "beta.log": [
      0,
      1
     ],
     "gamma.log": [
      1,
      0
     ],
     "report.txt": [
      1,
      0
     ]
    },
    "cmd": "lock report.txt"
   },
   {
    "obs": "ok",
    "state": {
     "alpha.log": [
      1,
      0
     ],
     "beta.log": [
      0,
      1
     ],
     "gamma.log": [
      1,
      0
     ],
     "report.txt": [
      1,
      1
     ]
    },
    "cmd": "mv gamma.log"
   },
   {
    "obs": "ok: moved gamma.log to /archive\n[system] background job TOGGLED the lock on alpha.log",
    "state": {
     "alpha.log": [
      1,
      1
     ],
     "beta.log": [
      0,
      1
     ],
     "gamma.log": [
      0,
      0
     ],
     "report.txt": [
      1,
      1
     ]
    },
    "cmd": "mv alpha.log"
   }
  ],
  "final_obs": "error: alpha.log is locked, cannot move",
  "final_state": {
   "alpha.log": [
    1,
    1
   ],
   "beta.log": [
    0,
    1
   ],
   "gamma.log": [
    0,
    0
   ],
   "report.txt": [
    1,
    1
   ]
  }
 },
 "think": {
  "steps": [
   {
    "obs": "Initial state: you are in /logs.\nalpha.log: locked\nbeta.log: locked\ngamma.log: unlocked\nreport.txt: unlocked\nBegin.",
    "state": {
     "alpha.log": [
      1,
      1
     ],
     "beta.log": [
      0,
      1
     ],
     "gamma.log": [
      1,
      0
     ],
     "report.txt": [
      1,
      0
     ]
    },
    "cmd": "mv gamma.log",
    "think": "So the first command is 'mv gamma.log', then 'lock report.txt', then 'done'."
   },
   {
    "obs": "ok: moved gamma.log to /archive",
    "state": {
     "alpha.log": [
      1,
      1
     ],
     "beta.log": [
      0,
      1
     ],
     "gamma.log": [
      0,
      0
     ],
     "report.txt": [
      1,
      0
     ]
    },
    "cmd": "lock report.txt",
    "think": "So after moving gamma.log, lock report.txt. Then done."
   },
   {
    "obs": "ok\n[system] background job TOGGLED the lock on alpha.log",
    "state": {
     "alpha.log": [
      1,
      0
     ],
     "beta.log": [
      0,
      1
     ],
     "gamma.log": [
      0,
      0
     ],
     "report.txt": [
      1,
      1
     ]
    },
    "cmd": "mv alpha.log",
    "think": "The initial state after moving gamma.log and locking report.txt:\n- alpha.log: unlocked (due to toggle)\n- beta.log: locked\n- report.txt: locked\nSo moving alpha.log is the next step."
   }
  ],
  "final_obs": "ok: moved alpha.log to /archive",
  "final_state": {
   "alpha.log": [
    0,
    0
   ],
   "beta.log": [
    0,
    1
   ],
   "gamma.log": [
    0,
    0
   ],
   "report.txt": [
    1,
    1
   ]
  }
 },
 "fixed": {
  "arms": [
   [
    "No reasoning",
    "",
    "mv gamma.log",
    1
   ],
   [
    "Filler",
    ". . . . . . . . . . . .",
    "mv gamma.log",
    1
   ],
   [
    "Other-history reasoning",
    "So beta.log is now locked, gamma.log is now locked.",
    "stat gamma.log",
    0
   ],
   [
    "State table",
    "alpha.log: unlocked, not in /logs\nbeta.log: unlocked, not in /logs\ngamma.log: locked, in /logs\nreport.txt: unlocked, in /logs",
    "mv gamma.log",
    1
   ],
   [
    "Own reasoning",
    "Gamma.log is locked, so can't move. Then, the next step is to lock report.txt.",
    "lock report.txt",
    0
   ]
  ],
  "history": [
   [
    "Initial state: you are in /archive.\nalpha.log: unlocked\nbeta.log: locked\ngamma.log: locked\nreport.txt: unlocked\nBegin.",
    "cd /logs"
   ],
   [
    "now in /logs",
    "mv alpha.log"
   ],
   [
    "ok: moved alpha.log to /archive\n[system] background job TOGGLED the lock on beta.log",
    "mv beta.log"
   ],
   [
    "ok: moved beta.log to /archive",
    null
   ]
  ]
 },
 "swap": {
  "L": "the remaining files in /logs are alpha.log (locked), gamma.log (locked), and report.txt (locked). … So the task is done.",
  "U": "Gamma.log is unlocked, so I should move that next.",
  "cells": {
   "LL": "done",
   "LU": "mv gamma.log",
   "UL": "done",
   "UU": "mv gamma.log"
  },
  "history": [
   [
    "Initial state: you are in /logs.\nalpha.log: locked\nbeta.log: unlocked\ngamma.log: locked\nreport.txt: unlocked\nBegin.",
    "mv beta.log"
   ],
   [
    "ok: moved beta.log to /archive",
    "lock report.txt"
   ],
   [
    "ok\n[system] background job TOGGLED the lock on beta.log",
    null
   ]
  ]
 }
};

const FILES = ["alpha.log", "beta.log", "gamma.log", "report.txt"];

// ---------- small helpers ----------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Each demo runs as an async script; starting it again cancels the previous run.
function runner(fn) {
  let token = 0;
  return () => {
    const mine = ++token;
    const wait = (ms) => new Promise((res, rej) =>
      setTimeout(() => (mine === token ? res() : rej("stop")), reduced ? 0 : ms));
    fn(wait).catch((e) => { if (e !== "stop") console.error(e); });
  };
}

async function type(node, text, wait, speed = 14) {
  if (reduced) { node.textContent = text; return; }
  node.textContent = "";
  const step = Math.max(1, Math.round(text.length / 90));
  for (let i = 0; i < text.length; i += step) {
    node.textContent = text.slice(0, i + step);
    await wait(speed);
  }
}

function onVisible(node, start) {
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) { io.disconnect(); start(); }
  }, { threshold: 0.35 });
  io.observe(node);
}

function segment(root, group, onChange) {
  const seg = $(`.seg[data-group="${group}"]`, root);
  let value = $("button.on", seg).dataset.v;
  $$("button", seg).forEach((b) => b.addEventListener("click", () => {
    $$("button", seg).forEach((x) => x.classList.toggle("on", x === b));
    value = b.dataset.v;
    onChange();
  }));
  return () => value;
}

const LOCK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>';
const OPEN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 7.5-2"/></svg>';

function termLines(node, text, cls = "obs") {
  text.split("\n").forEach((line) => {
    const c = line.startsWith("[system]") ? "sys" : cls;
    node.appendChild(el("div", "ln " + c, line));
  });
}

function pill(kind, text) { return el("span", "pill " + kind, text); }

// ---------- 1. episode player ----------
(function episode() {
  const root = $("#episode");
  const body = $(".term-body", root);
  const verdict = $(".verdict", root);
  const mode = segment(root, "mode", () => play());

  function drawFiles(state, prev, toggled) {
    $$(".slots", root).forEach((s) => (s.innerHTML = ""));
    FILES.forEach((f) => {
      const [inLogs, locked] = state[f];
      const chip = el("span", "file " + (locked ? "locked" : "unlocked"));
      chip.innerHTML = (locked ? LOCK : OPEN) + "<span>" + f + "</span>";
      if (prev && prev[f][0] !== inLogs) chip.classList.add("arrive");
      if (f === toggled) { chip.classList.add("flash"); chip.appendChild(el("span", "q", "?")); }
      $(`.slots[data-dir="${inLogs}"]`, root).appendChild(chip);
    });
  }

  const toggledIn = (obs) => (obs.match(/TOGGLED the lock on (\S+)/) || [])[1];

  const play = runner(async (wait) => {
    const ep = DATA[mode()];
    body.innerHTML = "";
    verdict.innerHTML = "";
    let prev = null;
    for (const s of ep.steps) {
      termLines(body, s.obs);
      drawFiles(s.state, prev, toggledIn(s.obs));
      prev = s.state;
      body.scrollTop = body.scrollHeight;
      await wait(toggledIn(s.obs) ? 2400 : 1300);
      if (s.think) {
        const t = el("div", "ln think");
        body.appendChild(t);
        await type(t, s.think, wait, 22);
        body.scrollTop = body.scrollHeight;
        await wait(1200);
      }
      const c = el("div", "ln cmd");
      body.appendChild(c);
      await type(c, s.cmd, wait, 55);
      body.scrollTop = body.scrollHeight;
      await wait(1200);
    }
    const bad = ep.final_obs.startsWith("error");
    termLines(body, ep.final_obs, bad ? "err" : "okk");
    drawFiles(ep.final_state, prev, null);
    body.scrollTop = body.scrollHeight;
    await wait(400);
    verdict.appendChild(bad
      ? pill("bad", "Lock violation: alpha.log was toggled twice, so it is locked again")
      : pill("good", "No violation: the reasoning tracks the toggle, so alpha.log is moved while unlocked"));
  });

  $(".replay", root).addEventListener("click", play);
  onVisible(root, play);
})();

// ---------- 2. generating vs retaining ----------
(function conditions() {
  const root = $("#conditions");
  const RATES = {
    q3: { "direct/cmd": 22.8, "think/cmd": 1.9, "think/body": 2.3, "think/full": 1.4, gen: 20.9, ret: 0.5 },
    q35: { "direct/cmd": 15.2, "think/cmd": 4.8, "think/body": 3.7, "think/full": 3.4, gen: 10.4, ret: 1.4 },
  };
  const ORDER = ["direct/cmd", "think/cmd", "think/body", "think/full"];
  const DESC = {
    "direct/cmd": "No reasoning. Earlier turns keep only the command.",
    "think/cmd": "Reasons at every turn. Earlier reasoning and reply text are removed.",
    "think/body": "Reasons at every turn. Earlier reply text is kept (chat-template default).",
    "think/full": "Reasons at every turn. Earlier reasoning is kept as well.",
  };
  const PAST = [
    ["Initial state: you are in /logs. alpha.log: locked …", "mv gamma.log"],
    ["ok: moved gamma.log to /archive", "lock report.txt"],
  ];
  const NOW = "ok  [system] background job TOGGLED the lock on alpha.log";
  const model = segment(root, "model", () => play());
  const prompt = $(".prompt", root);
  const bars = $("#cond-bars", root);

  // build prompt rows once
  const rows = PAST.map(([obs, cmd], i) => {
    const r = el("div", "turnrow");
    r.append(el("span", "t", "Turn " + (i + 1)), el("span", "blk ob", obs), el("span", "blk th", "reasoning …"),
      el("span", "blk rp", "reply text"), el("span", "blk cm", cmd));
    prompt.appendChild(r);
    return r;
  });
  const now = el("div", "turnrow");
  now.append(el("span", "t", "Turn 3"), el("span", "blk ob", NOW), el("span", "blk th live", "reasoning …"), el("span", "blk cm", "CMD: ?"));
  prompt.appendChild(now);

  function show(cond) {
    const think = cond.startsWith("think");
    const keepBody = cond.endsWith("body") || cond.endsWith("full");
    const keepThink = cond.endsWith("full");
    rows.forEach((r) => {
      $(".th", r).classList.toggle("gone", !keepThink);
      $(".rp", r).classList.toggle("gone", !keepBody);
    });
    $(".th", now).classList.toggle("gone", !think);
    const name = $(".cond-name", root);
    name.innerHTML = "";
    name.append(cond, el("small", "", DESC[cond]));
  }

  const play = runner(async (wait) => {
    const R = RATES[model()];
    bars.innerHTML = "";
    $$(".delta", root).forEach((d) => d.classList.remove("show"));
    const fills = {};
    ORDER.forEach((c) => {
      const b = el("div", "bar dim");
      const tr = el("div", "track");
      const f = el("div", "fill" + (c.startsWith("direct") ? " d" : ""));
      tr.appendChild(f);
      b.append(el("span", "nm", c), tr, el("span", "val", ""));
      bars.appendChild(b);
      fills[c] = b;
    });
    await wait(300);
    for (const c of ORDER) {
      show(c);
      Object.values(fills).forEach((b) => b.classList.add("dim"));
      const b = fills[c];
      b.classList.remove("dim");
      await wait(1800);
      $(".fill", b).style.width = (R[c] / 25) * 100 + "%";
      $(".val", b).textContent = R[c].toFixed(1) + "%";
      await wait(3400);
    }
    Object.values(fills).forEach((b) => b.classList.remove("dim"));
    $('[data-k="gen"]', root).textContent = R.gen.toFixed(1) + " points";
    $('[data-k="ret"]', root).textContent = R.ret.toFixed(1) + " points";
    for (const d of $$(".delta", root)) { d.classList.add("show"); await wait(900); }
  });

  $(".replay", root).addEventListener("click", play);
  onVisible(root, play);
})();

// ---------- 3. fixed histories ----------
(function fixed() {
  const root = $("#fixed");
  const hist = $("#fixed-hist", root);
  DATA.fixed.history.forEach(([obs, cmd]) => {
    termLines(hist, obs);
    if (cmd) hist.appendChild(el("div", "ln cmd", cmd));
  });
  const RATE = [
    ["No reasoning", 48.8, "21/43"],
    ["Filler", 46.6, "27/58"],
    ["Other-history reasoning", 82.1, "23/28"],
    ["State table", 31.7, "32/101"],
    ["Own reasoning", 5.8, "3/52"],
  ];
  const bars = $("#fixed-bars", root);
  const arm = $(".slot .arm", root);
  const content = $(".slot .content", root);
  const out = $(".outcome", root);

  const play = runner(async (wait) => {
    bars.innerHTML = "";
    const rows = RATE.map(([name, , n]) => {
      const b = el("div", "bar dim");
      b.style.gridTemplateColumns = "150px 1fr 52px";
      const tr = el("div", "track");
      tr.appendChild(el("div", "fill " + (name === "Own reasoning" ? "g" : "d")));
      const nm = el("span", "nm", name);
      nm.style.fontFamily = "var(--sans)";
      b.append(nm, tr, el("span", "val", ""));
      b.title = n + " attempts violate";
      bars.appendChild(b);
      return b;
    });
    for (let i = 0; i < DATA.fixed.arms.length; i++) {
      const [name, text, cmd, viol] = DATA.fixed.arms[i];
      arm.textContent = name;
      out.innerHTML = "";
      content.classList.toggle("empty", !text);
      if (text) await type(content, text, wait, 24);
      else content.textContent = "(empty)";
      await wait(900);
      out.append(el("span", "c", cmd));
      await wait(500);
      out.append(viol ? pill("bad", "moves a locked file")
        : cmd.startsWith("stat") ? pill("neutral", "checks the file first")
        : pill("good", "leaves gamma.log alone"));
      rows[i].classList.remove("dim");
      $(".fill", rows[i]).style.width = RATE[i][1] + "%";
      $(".val", rows[i]).textContent = RATE[i][1].toFixed(0) + "%";
      await wait(3800);
    }
  });

  $(".replay", root).addEventListener("click", play);
  onVisible(root, play);
})();

// ---------- 4. reasoning swap ----------
(function swap() {
  const root = $("#swapdemo");
  const S = DATA.swap;
  const COUNTS = { q3: { LL: 0, LU: 43, UL: 0, UU: 45, n: 114 }, q35: { LL: 0, LU: 41, UL: 0, UU: 41, n: 104 } };
  const model = segment(root, "model", () => play());

  ["L", "U"].forEach((v) => {
    const t = $(`[data-hist="${v}"]`, root);
    S.history.forEach(([obs, cmd], i) => {
      let o = obs;
      if (i === 0) o = obs.replace("gamma.log: locked", v === "L" ? "gamma.log: locked" : "gamma.log: unlocked");
      o.split("\n").forEach((line) => {
        const d = el("div", "ln " + (line.startsWith("[system]") ? "sys" : "obs"), line);
        if (line.startsWith("gamma.log:")) d.innerHTML = '<span class="hl">' + line + "</span>";
        t.appendChild(d);
      });
      if (cmd) t.appendChild(el("div", "ln cmd", cmd));
    });
  });

  function card(node, src) {
    node.innerHTML = "";
    node.append(el("div", "who", "Reasoning written for " + (src === "L" ? "A (locked)" : "B (unlocked)")), el("div", "", "… " + S[src]));
  }

  function outcome(hist, src) {
    const node = $(`[data-out="${hist}"]`, root);
    const cmd = S.cells[hist + src];
    node.innerHTML = "";
    node.append(el("span", "c", cmd));
    const moves = cmd.startsWith("mv");
    node.append(moves && hist === "L" ? pill("bad", "moves a locked file")
      : moves ? pill("good", "moves the unlocked file")
      : pill("neutral", "does not move gamma.log"));
  }

  const play = runner(async (wait) => {
    const cL = $('[data-pos="L"]', root), cU = $('[data-pos="U"]', root);
    [cL, cU].forEach((c) => { c.style.transition = "none"; c.style.transform = "none"; c.classList.remove("swapped"); });
    $$("[data-out]", root).forEach((n) => (n.innerHTML = ""));
    $$(".counts .n", root).forEach((n) => { n.innerHTML = ""; n.classList.remove("bad"); });
    card(cL, "L"); card(cU, "U");
    await wait(1800);
    outcome("L", "L"); await wait(900); outcome("U", "U");
    await wait(3800);
    $$("[data-out]", root).forEach((n) => (n.innerHTML = ""));
    const a = cL.getBoundingClientRect(), b = cU.getBoundingClientRect();
    const dx = b.left - a.left, dy = b.top - a.top;
    [cL, cU].forEach((c) => (c.style.transition = ""));
    cL.style.transform = `translate(${dx}px, ${dy}px)`;
    cU.style.transform = `translate(${-dx}px, ${-dy}px)`;
    await wait(1200);
    [cL, cU].forEach((c) => { c.style.transition = "none"; c.style.transform = "none"; c.classList.add("swapped"); });
    card(cL, "U"); card(cU, "L");
    await wait(600);
    outcome("L", "U"); await wait(900); outcome("U", "L");
    await wait(2600);
    const C = COUNTS[model()];
    for (const k of ["LL", "LU", "UL", "UU"]) {
      const n = $(`.counts .n[data-c="${k}"]`, root);
      n.innerHTML = "";
      const num = el("span", "", "0");
      n.append(num, el("small", "", "of " + C.n + " pairs"));
      if (k === "LU") n.classList.add("bad");
      for (let v = 0; v <= C[k]; v += Math.max(1, Math.ceil(C[k] / 20))) { num.textContent = v; await wait(40); }
      num.textContent = C[k];
      await wait(500);
    }
  });

  $(".replay", root).addEventListener("click", play);
  onVisible(root, play);
})();

// ---------- 5. decoding versus patching ----------
(function probes() {
  const root = $("#probedemo");
  // Per layer: [decoding AUROC, recovered fraction R] for each direction, and R for the whole final-token activation.
  const LAYERS = {"20": {"full": 0.114, "site_file": [0.969, 0.003], "nat_dm_pool": [0.847, 0.018], "dm_pool": [0.819, 0.111], "rand_1": [0.393, 0.001]}, "22": {"full": 0.308, "site_file": [0.98, 0.011], "nat_dm_pool": [0.878, 0.093], "dm_pool": [0.845, 0.28], "rand_1": [0.514, 0.001]}, "24": {"full": 0.667, "site_file": [0.986, 0.019], "nat_dm_pool": [0.904, 0.482], "dm_pool": [0.898, 0.646], "rand_1": [0.681, 0.001]}, "26": {"full": 0.725, "site_file": [0.986, 0.018], "nat_dm_pool": [0.941, 0.592], "dm_pool": [0.928, 0.742], "rand_1": [0.458, 0.001]}, "28": {"full": 0.792, "site_file": [0.987, 0.01], "nat_dm_pool": [0.938, 0.657], "dm_pool": [0.93, 0.812], "rand_1": [0.573, 0.001]}, "30": {"full": 0.905, "site_file": [0.986, 0.011], "nat_dm_pool": [0.936, 0.739], "dm_pool": [0.931, 0.926], "rand_1": [0.569, 0.001]}, "32": {"full": 0.937, "site_file": [0.985, 0.006], "nat_dm_pool": [0.932, 0.752], "dm_pool": [0.923, 0.949], "rand_1": [0.689, 0.001]}, "34": {"full": 0.981, "site_file": [0.986, 0.012], "nat_dm_pool": [0.93, 0.802], "dm_pool": [0.921, 1.012], "rand_1": [0.882, 0.002]}, "35": {"full": 1.0, "site_file": [0.985, 0.019], "nat_dm_pool": [0.93, 0.837], "dm_pool": [0.918, 1.042], "rand_1": [0.507, -0.001]}};
  const KEYS = Object.keys(LAYERS);
  const DIRS = [
    ["site_file", "Logistic probe", "Trained to decode the lock state on ordinary prompts."],
    ["nat_dm_pool", "Mass-mean probe", "Mean of locked minus mean of unlocked cases on ordinary prompts."],
    ["dm_pool", "Counterfactual mean difference", "Mean difference between the two versions of training pairs."],
    ["rand_1", "Random direction", "Control."],
  ];
  const list = $("#dirs", root);
  const needle = $(".needle", root);
  const fullMark = $("#full-mark", root);
  const current = $("#probe-dir", root);
  const slider = $("#layer", root);
  const layerV = $("#layer-v", root);
  const pos = (r) => (Math.max(0, Math.min(r, 1.1)) / 1.1) * 100 + "%";
  let selected = 0;
  let stopped = false;
  const stop = () => { stopped = true; };

  const rows = DIRS.map(([, name, desc], i) => {
    const r = el("div", "dirrow");
    r.tabIndex = 0;
    r.append(el("div", "nm", name), el("div", "desc", desc));
    [["decodes the state", ""], ["recovered fraction R", " g"]].forEach(([lab, c]) => {
      const b = el("div", "bar");
      const tr = el("div", "track");
      tr.appendChild(el("div", "fill" + c));
      b.append(el("span", "nm", lab), tr, el("span", "val", ""));
      r.appendChild(b);
    });
    const pick = () => { stop(); DIRS.forEach((_, j) => fillRow(j)); select(i); };
    r.addEventListener("click", pick);
    r.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    list.appendChild(r);
    return r;
  });

  const layer = () => LAYERS[KEYS[+slider.value]];

  function fillRow(i) {
    const [auc, r] = layer()[DIRS[i][0]];
    const [dec, mov] = $$(".bar", rows[i]);
    $(".fill", dec).style.width = Math.max(0, (auc - 0.5) / 0.5) * 100 + "%";
    $(".val", dec).textContent = auc.toFixed(2);
    $(".fill", mov).style.width = Math.max(0, Math.min(r, 1)) * 100 + "%";
    $(".val", mov).textContent = r.toFixed(2);
  }

  function select(i) {
    selected = i;
    rows.forEach((r, j) => r.classList.toggle("on", j === i));
    const r = layer()[DIRS[i][0]][1];
    current.textContent = DIRS[i][1];
    needle.style.left = pos(r);
    needle.dataset.v = "R = " + r.toFixed(2);
    needle.dataset.side = r < 0.15 ? "l" : r > 0.95 ? "r" : "";
  }

  function showLayer() {
    layerV.textContent = KEYS[+slider.value] + " of 35";
    fullMark.style.left = pos(layer().full);
  }

  slider.addEventListener("input", () => {
    stop();
    showLayer();
    DIRS.forEach((_, i) => fillRow(i));
    select(selected);
  });

  const play = runner(async (wait) => {
    stopped = false;
    showLayer();
    rows.forEach((r) => $$(".fill", r).forEach((f) => (f.style.width = "0")));
    rows.forEach((r) => $$(".val", r).forEach((v) => (v.textContent = "")));
    rows.forEach((r) => r.classList.remove("on"));
    needle.style.left = "0%";
    needle.dataset.v = "no patch";
    needle.dataset.side = "l";
    current.textContent = "";
    await wait(800);
    for (let i = 0; i < DIRS.length && !stopped; i++) {
      fillRow(i);
      await wait(900);
      if (!stopped) select(i);
      await wait(3200);
    }
    if (!stopped) { DIRS.forEach((_, i) => fillRow(i)); select(0); }
  });

  showLayer();
  $(".replay", root).addEventListener("click", play);
  onVisible(root, play);
})();
