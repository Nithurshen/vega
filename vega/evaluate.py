# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import itertools
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy import stats

from vega import index as index_mod
from vega import neural
from vega import plots
from vega.rank import Engine, config

DATA = os.path.join("data", "trec")
RESULTS = "results"
FIG = os.path.join("report", "figures")
TAB = os.path.join("report", "tables")
DEPTH = 1000
VEGA = "Vega (full)"
NEURAL = "Vega-Neural (full)"
NEURAL_GRID = [dict(zip(("alpha", "inc_view", "exc_view", "elig_w"), g))
	       for g in itertools.product((0.0, 0.3, 0.5, 0.7, 0.85, 1.0),
					  (0.0, 0.1, 0.2, 0.4),
					  (0.0, 0.1, 0.2, 0.4),
					  (0.0, 0.2, 0.5))]
METRICS = ("nDCG@10", "P@10", "P@10 lenient", "RR", "R@1000",
	   "excluded@10", "judged@10", "nDCG@10'", "P@10'")
ZONE_GRID = (0.0, 0.5, 1.0, 2.0, 3.0)
EXC_GRID = (0.0, 0.2, 0.4, 0.6, 1.0, 1.5)
NEG_GRID = (0.0, 0.2, 0.4, 0.6, 1.0)
PEN_RATIOS = (0.05, 0.1)
PRF_GRID = [(d, n, w) for d in (5, 10, 20) for n in (10, 20)
	    for w in (0.5, 1.0)]
SOFT_GRID = (0.1, 0.25, 0.5, 1.0)


def load_topics(year):
	root = ET.parse(os.path.join(DATA, f"topics{year}.xml")).getroot()
	return [{"id": f"{year}-{t.get('number')}", "year": year,
		 "note": " ".join((t.text or "").split())} for t in root]


def load_qrels(year, pos):
	out = {}
	with open(os.path.join(DATA, f"qrels{year}.txt")) as f:
		for line in f:
			topic, _, doc, rel = line.split()
			if doc in pos:
				key = f"{year}-{topic}"
				out.setdefault(key, {})[pos[doc]] = int(rel)
	return out


def metrics(ranked, judged):
	rels = [judged.get(d, 0) for d in ranked[:DEPTH]]
	top = rels[:10] + [0] * (10 - len(rels[:10]))
	disc = 1 / np.log2(np.arange(2, 12))
	ideal = sorted(judged.values(), reverse=True)[:10]
	idcg = sum(g * w for g, w in zip(ideal, disc))
	n_elig = sum(v == 2 for v in judged.values())
	first = next((i for i, r in enumerate(rels) if r == 2), None)
	return {
		"nDCG@10": float(np.dot(top, disc) / idcg) if idcg else 0.0,
		"P@10": sum(r == 2 for r in top) / 10,
		"P@10 lenient": sum(r >= 1 for r in top) / 10,
		"RR": 1 / (first + 1) if first is not None else 0.0,
		"R@1000": sum(r == 2 for r in rels) / n_elig if n_elig else 0,
		"excluded@10": sum(r == 1 for r in top) / 10,
		"judged@10": sum(d in judged for d in ranked[:10]) / 10,
	}


def run(engine, topics, qrels, cfg):
	rows, secs = [], []
	for t in topics:
		start = time.perf_counter()
		res = engine.search(t["note"], dict(cfg, k=DEPTH))
		secs.append(time.perf_counter() - start)
		judged = qrels.get(t["id"], {})
		m = metrics(res["ranked"], judged)
		seen = [d for d in res["ranked"] if d in judged]
		cond = metrics(seen, judged)
		m["nDCG@10'"] = cond["nDCG@10"]
		m["P@10'"] = cond["P@10"]
		m["lexical_ms"] = res["timing"]["lexical"] * 1000
		m["scanned"] = res["scanned"]
		rows.append(m)
	return rows, float(np.mean(secs) * 1000)


def mean(rows, key):
	return float(np.mean([r[key] for r in rows]))


def summary(rows, ms):
	out = {k: mean(rows, k) for k in METRICS}
	out["ms"] = ms
	out["lexical_ms"] = mean(rows, "lexical_ms")
	out["scanned"] = mean(rows, "scanned")
	return out


def score(engine, topics, qrels, cfg):
	rows, _ = run(engine, topics, qrels, cfg)
	return mean(rows, "nDCG@10")


def best(engine, topics, qrels, grid, label):
	scored = [(score(engine, topics, qrels, c), c) for c in grid]
	scored.sort(key=lambda p: -p[0])
	print(f"  {label}: nDCG@10(2021)={scored[0][0]:.4f}", flush=True)
	return scored[0][1], scored


def tune_zones(engine, topics, qrels, base):
	cfg = base
	for _ in range(2):
		for z in range(index_mod.Z):
			grid = []
			for w in ZONE_GRID:
				zw = list(cfg["zone_w"])
				zw[z] = w
				grid.append(dict(cfg, zone_w=tuple(zw)))
			cfg, _ = best(engine, topics, qrels, grid,
				      f"zone {index_mod.ZONES[z]}")
	return cfg


RAW = {"abbrev": False, "negation": False, "family": False}
NOTE_GRID = [{"abbrev": a, "negation": n, "family": f}
	     for a in (False, True) for n in (False, True)
	     for f in (False, True)]


def tune(engine, topics, qrels):
	raw = tune_zones(engine, topics, qrels, config(**RAW))
	prof, _ = best(engine, topics, qrels,
		       [dict(raw, **g) for g in NOTE_GRID], "note processing")
	hard = dict(prof, params="hard")
	soft, _ = best(engine, topics, qrels,
		       [dict(prof, params="soft", soft=s) for s in SOFT_GRID],
		       "soft eligibility")
	params = hard if score(engine, topics, qrels, hard) >= \
		score(engine, topics, qrels, soft) else soft
	prf, _ = best(engine, topics, qrels,
		      [dict(params, prf_docs=d, prf_terms=n, prf_w=w)
		       for d, n, w in PRF_GRID], "conditions feedback")
	exc, _ = best(engine, topics, qrels,
		      [dict(prf, exc_w=w, pen_ratio=r) for w in EXC_GRID
		       for r in PEN_RATIOS], "exclusion penalty")
	full, _ = best(engine, topics, qrels,
		       [dict(exc, neg_w=w) for w in NEG_GRID],
		       "negation penalty")
	out = {"raw": raw, "profile": prof, "params": params, "prf": prf,
	       "exclusion": exc, "full": full, "hard": hard, "soft": soft}
	out.update(penalty_curves(engine, topics, qrels, out))
	return out


def penalty_curves(engine, topics, qrels, tuned):
	full = tuned["full"]
	return {
		"exc_curve": [score(engine, topics, qrels, dict(full, exc_w=w))
			      for w in EXC_GRID],
		"neg_curve": [score(engine, topics, qrels, dict(full, neg_w=w))
			      for w in NEG_GRID],
	}


def curves_only():
	path = os.path.join(RESULTS, "summary.json")
	with open(path) as f:
		out = json.load(f)
	engine = Engine.load()
	pos = {d: i for i, d in enumerate(engine.ix.ids)}
	out["tuned"].update(penalty_curves(engine, load_topics(2021),
					   load_qrels(2021, pos), out["tuned"]))
	with open(path, "w") as f:
		json.dump(out, f, indent=1, default=list)
	figures(out)


def systems(tuned):
	full = tuned["full"]
	return [
		("TF-IDF lnc.ltc (baseline)", config(model="tfidf", **RAW)),
		("BM25 (raw note)", config(model="bm25", **RAW)),
		("BM25F zones (raw note)", tuned["raw"]),
		("+ note processing", tuned["profile"]),
		("+ age/sex filter", tuned["params"]),
		("+ conditions feedback", tuned["prf"]),
		("+ exclusion penalty", tuned["exclusion"]),
		(VEGA, full),
	]


def ablations(tuned):
	full = tuned["full"]
	return [
		(VEGA, full),
		("- note processing", dict(full, **RAW)),
		("- age/sex filter", dict(full, params="off")),
		("- conditions feedback", dict(full, prf_docs=0)),
		("- exclusion penalty", dict(full, exc_w=0.0)),
		("- negation penalty", dict(full, neg_w=0.0)),
		("hard filter instead", dict(full, params="hard")),
		("soft filter instead", dict(full, params="soft",
					     soft=tuned["soft"]["soft"])),
	]


def table_of(engine, topics, qrels, named):
	table, per_topic = {}, {}
	for name, cfg in named:
		rows, ms = run(engine, topics, qrels, cfg)
		table[name] = summary(rows, ms)
		per_topic[name] = [r["nDCG@10"] for r in rows]
		print(f"  {name:28s} nDCG@10={table[name]['nDCG@10']:.4f} "
		      f"P@10={table[name]['P@10']:.4f} "
		      f"excl@10={table[name]['excluded@10']:.4f}", flush=True)
	return table, per_topic


def significance(per_topic, a, b):
	x, y = np.array(per_topic[a]), np.array(per_topic[b])
	t = stats.ttest_rel(y, x)
	w = stats.wilcoxon(y, x) if np.any(y != x) else None
	return {"a": a, "b": b, "delta": float(y.mean() - x.mean()),
		"t_p": float(t.pvalue),
		"wilcoxon_p": float(w.pvalue) if w else 1.0,
		"wins": int((y > x).sum()), "losses": int((y < x).sum())}


def efficiency(engine, topics, qrels, tuned):
	base = tuned["raw"]
	points = []
	for m in (10, 20, 40, 80, 0):
		rows, ms = run(engine, topics, qrels, dict(base, max_terms=m))
		points.append(("all" if m == 0 else f"m={m}",
			       summary(rows, ms)))
	rows, ms = run(engine, topics, qrels, dict(base, champions=True))
	return {"elimination": points, "champions": summary(rows, ms)}


def stemming(engine, topics, qrels, tuned):
	plain = Engine(index_mod.build(do_stem=False), engine.store)
	out, per = {}, {}
	for name, eng in (("Porter stemming", engine), ("no stemming", plain)):
		rows, ms = run(eng, topics, qrels, tuned["full"])
		out[name] = summary(rows, ms)
		out[name]["vocab"] = len(eng.ix.vocab)
		per[name] = [r["nDCG@10"] for r in rows]
	out["test"] = significance(per, "no stemming", "Porter stemming")
	return out


def extraction(engine, topics):
	from vega import patient
	ages = sexes = negs = fams = abbr = 0
	for t in topics:
		p = patient.profile(t["note"])
		ages += p["age"] is not None
		sexes += p["sex"] is not None
		negs += bool(p["neg"])
		fams += bool(p["family"])
		abbr += bool(p["abbrev"])
	n = len(topics)
	return {"topics": n, "age": ages, "sex": sexes, "negation": negs,
		"family": fams, "abbrev": abbr}


def judged_stats(qrels):
	counts = [0, 0, 0]
	for judged in qrels.values():
		for r in judged.values():
			counts[r] += 1
	return {"not_relevant": counts[0], "excluded": counts[1],
		"eligible": counts[2], "topics": len(qrels)}


def shown(table):
	key = [table[VEGA][m] for m in METRICS]
	return {n: v for n, v in table.items()
		if n == VEGA or [v[m] for m in METRICS] != key}


def combined(out):
	table = shown(out["test"])
	per_topic = dict(out["per_topic"])
	focus = VEGA
	if "neural" in out:
		table.update(out["neural"]["test"])
		per_topic.update(out["neural"]["per_topic"])
		focus = NEURAL
	return table, per_topic, focus


def figures(out):
	table, per_topic, focus = combined(out)
	names = list(table)
	plots.emphasis_bars(names, [table[n]["nDCG@10"] for n in names],
			    focus, "Clinical trial retrieval, TREC CT 2022",
			    "nDCG@10 (50 test patients)",
			    os.path.join(FIG, "main.png"))
	plots.paired_bars(names, [table[n]["P@10"] for n in names],
			  [table[n]["excluded@10"] for n in names],
			  os.path.join(FIG, "eligible.png"))
	plots.scatter(per_topic["BM25F zones (raw note)"], per_topic[focus],
		      "nDCG@10, BM25F zones", f"nDCG@10, {focus.split()[0]}",
		      os.path.join(FIG, "pertopic.png"))
	eff = out["efficiency"]
	champ = eff["champions"]
	plots.tradeoff([(n, s["lexical_ms"], s["nDCG@10"])
			for n, s in eff["elimination"]],
		       ("champions", champ["lexical_ms"], champ["nDCG@10"]),
		       os.path.join(FIG, "efficiency.png"))
	tuned = out["tuned"]
	if "exc_curve" in tuned:
		plots.sweep([("exclusion penalty weight", EXC_GRID,
			      tuned["exc_curve"]),
			     ("negation penalty weight", NEG_GRID,
			      tuned["neg_curve"])],
			    os.path.join(FIG, "penalties.png"))


def redraw():
	with open(os.path.join(RESULTS, "summary.json")) as f:
		out = json.load(f)
	figures(out)
	latex(out)


def latex(out):
	os.makedirs(TAB, exist_ok=True)
	cols = ("nDCG@10", "P@10", "P@10 lenient", "RR", "R@1000",
		"excluded@10")
	heads = ("nDCG@10", "P@10", "P@10$_{\\geq1}$", "RR", "R@1000",
		 "Excl@10 $\\downarrow$")
	for key, path in (("test", "results.tex"), ("ablation",
						    "ablation.tex")):
		t = combined(out)[0] if key == "test" else shown(out[key])
		lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}",
			 "\\toprule",
			 "System & " + " & ".join(heads) + " \\\\", "\\midrule"]
		for name, v in t.items():
			cells = []
			for c in cols:
				vals = [x[c] for x in t.values()]
				low = c == "excluded@10"
				good = min(vals) if low else max(vals)
				s = f"{v[c]:.3f}"
				cells.append(f"\\textbf{{{s}}}" if v[c] == good
					     else s)
			if name == "MedCPT alone (topic view)":
				lines.append("\\midrule")
			lines.append(name + " & " + " & ".join(cells) + " \\\\")
		lines += ["\\bottomrule", "\\end{tabular}"]
		write(os.path.join(TAB, path), lines)
	eff = out["efficiency"]
	lines = ["\\begin{tabular}{lrrr}", "\\toprule",
		 "Setting & postings scanned & lexical ms & nDCG@10 \\\\",
		 "\\midrule"]
	rows = [(f"index elimination, {n}", s) for n, s in eff["elimination"]]
	rows.append(("champion lists, r=500", eff["champions"]))
	for n, s in rows:
		lines.append(f"{n} & {s['scanned']:,.0f} & "
			     f"{s['lexical_ms']:.0f} & {s['nDCG@10']:.3f} \\\\")
	lines += ["\\bottomrule", "\\end{tabular}"]
	write(os.path.join(TAB, "efficiency.tex"), lines)


def write(path, lines):
	with open(path, "w") as f:
		f.write("\n".join(lines) + "\n")


def neural_cache(engine, topics, cfg):
	cache = {}
	for t in topics:
		res = engine.search(t["note"], dict(cfg, k=DEPTH, neural=False))
		cands = res["ranked"][:cfg["rerank_k"]]
		ce = engine.neural_scores(t["note"], cands)
		cache[t["id"]] = (res["ranked"], res["final"][cands],
				  res["ok"][cands], ce)
	return cache


def neural_ranking(entry, w):
	ranked, first, ok, ce = entry
	fused = neural.fuse(first, ce, ok, w)
	order = np.argsort(-fused, kind="stable")
	return [ranked[i] for i in order] + ranked[len(first):]


def neural_rows(cache, topics, qrels, w):
	rows = []
	for t in topics:
		judged = qrels.get(t["id"], {})
		ranked = neural_ranking(cache[t["id"]], w)
		m = metrics(ranked, judged)
		cond = metrics([d for d in ranked if d in judged], judged)
		m["nDCG@10'"] = cond["nDCG@10"]
		m["P@10'"] = cond["P@10"]
		rows.append(m)
	return rows


def neural_best(cache, topics, qrels):
	scored = [(mean(neural_rows(cache, topics, qrels, w), "nDCG@10"), w)
		  for w in NEURAL_GRID]
	return max(scored, key=lambda p: p[0])[1]


def neural_cv(cache, topics, qrels, folds=5):
	order = np.random.RandomState(0).permutation(len(topics))
	base = dict(alpha=0.0, inc_view=0.0, exc_view=0.0, elig_w=0.0)
	out = []
	for part in np.array_split(order, folds):
		test = [topics[i] for i in part]
		train = [topics[i] for i in order if i not in set(part)]
		w = neural_best(cache, train, qrels)
		before = neural_rows(cache, test, qrels, base)
		after = neural_rows(cache, test, qrels, w)
		out.append((mean(before, "nDCG@10"), mean(after, "nDCG@10")))
	return np.mean(out, axis=0).tolist()


def neural_systems(w):
	none = dict(alpha=0.0, inc_view=0.0, exc_view=0.0, elig_w=0.0)
	return [
		("MedCPT alone (topic view)", dict(none, alpha=1.0)),
		("Vega + MedCPT topic view", dict(none, alpha=w["alpha"])),
		("+ inclusion view", dict(none, alpha=w["alpha"],
					  inc_view=w["inc_view"])),
		("+ exclusion view", dict(none, alpha=w["alpha"],
					  inc_view=w["inc_view"],
					  exc_view=w["exc_view"])),
		(NEURAL, w),
	]


def neural_stage():
	path = os.path.join(RESULTS, "summary.json")
	with open(path) as f:
		out = json.load(f)
	engine = Engine.load()
	pos = {d: i for i, d in enumerate(engine.ix.ids)}
	dev, test = load_topics(2021), load_topics(2022)
	qrels = {**load_qrels(2021, pos), **load_qrels(2022, pos)}
	cfg = config(**out["tuned"]["full"])
	print("cross-encoder scores", flush=True)
	dcache = neural_cache(engine, dev, cfg)
	tcache = neural_cache(engine, test, cfg)
	w = neural_best(dcache, dev, qrels)
	print(f"  tuned on 2021: {w}", flush=True)
	table, per_topic = {}, {}
	for name, ws in neural_systems(w):
		rows = neural_rows(tcache, test, qrels, ws)
		table[name] = {k: mean(rows, k) for k in METRICS}
		per_topic[name] = [r["nDCG@10"] for r in rows]
		print(f"  {name:28s} nDCG@10={table[name]['nDCG@10']:.4f} "
		      f"P@10={table[name]['P@10']:.4f}", flush=True)
	both = {**out["per_topic"], **per_topic}
	sig = [significance(both, a, NEURAL)
	       for a in ("BM25F zones (raw note)", VEGA)]
	out["neural"] = {"tuned": w, "test": table, "per_topic": per_topic,
			 "significance": sig,
			 "cv": neural_cv(dcache, dev, qrels),
			 "dev": mean(neural_rows(dcache, dev, qrels, w),
				     "nDCG@10")}
	out["tuned"]["neural"] = dict(cfg, neural=True, **w)
	with open(path, "w") as f:
		json.dump(out, f, indent=1, default=list)
	figures(out)
	latex(out)
	print(json.dumps({"cv": out["neural"]["cv"], "significance": sig}))


def stemming_only():
	path = os.path.join(RESULTS, "summary.json")
	with open(path) as f:
		out = json.load(f)
	engine = Engine.load()
	pos = {d: i for i, d in enumerate(engine.ix.ids)}
	test = load_topics(2022)
	out["stemming"] = stemming(engine, test, load_qrels(2022, pos),
				   out["tuned"])
	with open(path, "w") as f:
		json.dump(out, f, indent=1, default=list)
	print(json.dumps(out["stemming"]["test"]))


def main():
	if sys.argv[1:] == ["stem"]:
		return stemming_only()
	if sys.argv[1:] == ["figures"]:
		return redraw()
	if sys.argv[1:] == ["curves"]:
		return curves_only()
	if sys.argv[1:] == ["neural"]:
		return neural_stage()
	os.makedirs(RESULTS, exist_ok=True)
	os.makedirs(FIG, exist_ok=True)
	engine = Engine.load()
	pos = {d: i for i, d in enumerate(engine.ix.ids)}
	dev, test = load_topics(2021), load_topics(2022)
	qrels = {**load_qrels(2021, pos), **load_qrels(2022, pos)}
	print("tuning on 2021", flush=True)
	tuned = tune(engine, dev, qrels)
	print("test on 2022", flush=True)
	table, per_topic = table_of(engine, test, qrels, systems(tuned))
	abl, _ = table_of(engine, test, qrels, ablations(tuned))
	dev_table, _ = table_of(engine, dev, qrels, systems(tuned))
	names = list(shown(table))
	sig = [significance(per_topic, a, VEGA) for a in names[:-1]]
	print("efficiency", flush=True)
	eff = efficiency(engine, test, qrels, tuned)
	out = {"tuned": tuned, "test": table, "ablation": abl,
	       "dev": dev_table, "significance": sig, "efficiency": eff,
	       "extraction": {"2021": extraction(engine, dev),
			      "2022": extraction(engine, test)},
	       "judgements": {"2021": judged_stats(load_qrels(2021, pos)),
			      "2022": judged_stats(load_qrels(2022, pos))},
	       "per_topic": per_topic}
	if "--stem" in sys.argv:
		print("stemming", flush=True)
		out["stemming"] = stemming(engine, test, qrels, tuned)
	with open(os.path.join(RESULTS, "summary.json"), "w") as f:
		json.dump(out, f, indent=1, default=list)
	figures(out)
	latex(out)
	print(json.dumps({"significance": sig}, indent=1))


if __name__ == "__main__":
	main()
