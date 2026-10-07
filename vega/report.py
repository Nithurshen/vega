# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import json
import math
import os
import re


from vega import evaluate
from vega.rank import Engine

TEMPLATE = os.path.join("vega", "report_template.tex")
OUT = os.path.join("report", "main.tex")
VEGA = evaluate.VEGA
BASE = "BM25F zones (raw note)"
LABEL = {2: "eligible", 1: "excluded", 0: "not rel."}


def tex(s):
	s = s.replace("\\", " ")
	for a, b in (("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
		     ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
		     ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
		s = s.replace(a, b)
	return s


def num(n):
	return f"{n:,}".replace(",", "{,}")


def pval(p):
	if p >= 1e-3:
		return f"{p:.3f}"
	e = math.floor(math.log10(p))
	return f"{p / 10 ** e:.1f}\\times 10^{{{e}}}"


def f3(x):
	return f"{x:.3f}"


def pct(a, b):
	return f"{(a / b - 1) * 100:.0f}"


def corpus_values(engine, s):
	ix = engine.ix
	with open(os.path.join("results", "criteria.json")) as f:
		crit = json.load(f)
	j21, j22 = s["judgements"]["2021"], s["judgements"]["2022"]
	ex21, ex22 = s["extraction"]["2021"], s["extraction"]["2022"]
	return {
		"NTRIALS": num(ix.n),
		"NTERMS": num(len(ix.vocab)),
		"NPOST": num(len(ix.post_docs)),
		"NPOS": num(len(ix.pos)),
		"HEADER": f"{crit['header'] / crit['trials'] * 100:.1f}",
		"MOVED": num(crit["moved"]),
		"EXCL_COV": f"{crit['exclusion'] / crit['trials'] * 100:.1f}",
		"J21": num(sum(j21[k] for k in ("eligible", "excluded",
						"not_relevant"))),
		"J22": num(sum(j22[k] for k in ("eligible", "excluded",
						"not_relevant"))),
		"ELIG22": num(j22["eligible"]),
		"EXCL22": num(j22["excluded"]),
		"AGE_FOUND": str(ex21["age"] + ex22["age"]),
		"SEX_FOUND": str(ex21["sex"] + ex22["sex"]),
		"NEG_FOUND": str(ex21["negation"] + ex22["negation"]),
		"FAM_FOUND": str(ex21["family"] + ex22["family"]),
		"ABBR_FOUND": str(ex21["abbrev"] + ex22["abbrev"]),
		"NTOPICS": str(ex21["topics"] + ex22["topics"]),
	}


def result_values(s):
	t = s["test"]
	v, b = t[VEGA], t[BASE]
	tf = t["TF-IDF lnc.ltc (baseline)"]
	out = {
		"NDCG_V": f3(v["nDCG@10"]), "P10_V": f3(v["P@10"]),
		"RR_V": f3(v["RR"]), "EX_V": f3(v["excluded@10"]),
		"NDCGC_V": f3(v["nDCG@10'"]), "JUDGED_V": f3(v["judged@10"]),
		"NDCG_B": f3(b["nDCG@10"]), "P10_B": f3(b["P@10"]),
		"EX_B": f3(b["excluded@10"]), "NDCGC_B": f3(b["nDCG@10'"]),
		"JUDGED_B": f3(b["judged@10"]),
		"NDCG_T": f3(tf["nDCG@10"]), "P10_T": f3(tf["P@10"]),
		"GAIN_NDCG": pct(v["nDCG@10"], b["nDCG@10"]),
		"GAIN_P10": pct(v["P@10"], b["P@10"]),
		"MS_V": f"{v['ms']:.0f}",
	}
	for i, name in enumerate(t):
		out[f"STEP{i}"] = f3(t[name]["nDCG@10"])
		out[f"EXCL{i}"] = f3(t[name]["excluded@10"])
	for i, name in enumerate(s["dev"]):
		out[f"DEV{i}"] = f3(s["dev"][name]["nDCG@10"])
	return out


def tuned_values(s):
	full, soft = s["tuned"]["full"], s["tuned"]["soft"]
	note = [name for name, key in (("abbreviation expansion", "abbrev"),
				       ("negation removal", "negation"),
				       ("family-history removal", "family"))
		if full[key]]
	return {
		"NOTE_CFG": ", ".join(note) or "none",
		"ZONEW": ", ".join(f"{w:g}" for w in full["zone_w"]),
		"PARAMS": full["params"],
		"SOFT": f"{soft['soft']:g}",
		"PRF_D": str(full["prf_docs"]),
		"PRF_N": str(full["prf_terms"]),
		"PRF_W": f"{full['prf_w']:g}",
		"EXC_W": f"{full['exc_w']:g}",
		"NEG_W": f"{full['neg_w']:g}",
		"PEN_R": f"{full['pen_ratio']:g}",
	}


def sig_values(s):
	sig = {x["a"]: x for x in s["significance"]}
	b = sig[BASE]
	t = sig["TF-IDF lnc.ltc (baseline)"]
	return {
		"SIG_B_T": pval(b["t_p"]), "SIG_B_W": pval(b["wilcoxon_p"]),
		"SIG_B_WINS": str(b["wins"]), "SIG_B_LOSS": str(b["losses"]),
		"SIG_B_D": f3(b["delta"]),
		"SIG_T_T": pval(t["t_p"]), "SIG_T_WINS": str(t["wins"]),
		"SIG_T_LOSS": str(t["losses"]),
	}


def ablation_values(s):
	a = s["ablation"]
	full = a[VEGA]["nDCG@10"]
	out = {}
	for name, key in (("- note processing", "AB_NOTE"),
			  ("- age/sex filter", "AB_AGE"),
			  ("- conditions feedback", "AB_PRF"),
			  ("- exclusion penalty", "AB_EXC"),
			  ("- negation penalty", "AB_NEG"),
			  ("soft filter instead", "AB_SOFT"),
			  ("hard filter instead", "AB_HARD")):
		out[key] = f3(a[name]["nDCG@10"])
		out[key + "_D"] = f"{a[name]['nDCG@10'] - full:+.3f}"
	el = dict(s["efficiency"]["elimination"])
	allt, m40 = el["all"], el["m=40"]
	champ = s["efficiency"]["champions"]
	out.update({
		"MS_ALL": f"{allt['lexical_ms']:.0f}",
		"MS_40": f"{m40['lexical_ms']:.0f}",
		"NDCG_40": f3(m40["nDCG@10"]), "NDCG_ALL": f3(allt["nDCG@10"]),
		"SCAN_RATIO": f"{allt['scanned'] / m40['scanned']:.1f}",
		"MS_CH": f"{champ['lexical_ms']:.0f}",
		"NDCG_CH": f3(champ["nDCG@10"]),
	})
	st = s.get("stemming")
	if st:
		p, n = st["Porter stemming"], st["no stemming"]
		shrink = (1 - p["vocab"] / n["vocab"]) * 100
		out.update({
			"STEM_P": f"{st['test']['t_p']:.2f}",
			"STEM_SHRINK": f"{shrink:.0f}",
			"STEM_NDCG": f3(p["nDCG@10"]),
			"NOSTEM_NDCG": f3(n["nDCG@10"]),
		})
	return out


def neural_values(s):
	n = s["neural"]
	t = n["test"]
	v = t[evaluate.NEURAL]
	b, lex = s["test"][BASE], s["test"][VEGA]
	sig = {x["a"]: x for x in n["significance"]}
	sb, sv = sig[BASE], sig[VEGA]
	w = n["tuned"]
	topics = [x["id"] for x in evaluate.load_topics(2022)]
	fail = n["per_topic"][evaluate.NEURAL][topics.index("2022-7")]
	return {
		"N_NDCG": f3(v["nDCG@10"]), "N_P10": f3(v["P@10"]),
		"N_EX": f3(v["excluded@10"]), "N_RR": f3(v["RR"]),
		"N_JUDGED": f3(v["judged@10"]), "N_NDCGC": f3(v["nDCG@10'"]),
		"N_R1000": f3(v["R@1000"]),
		"N_GAIN_B": pct(v["nDCG@10"], b["nDCG@10"]),
		"N_GAIN_V": pct(v["nDCG@10"], lex["nDCG@10"]),
		"N_GAIN_P10": pct(v["P@10"], b["P@10"]),
		"N_SIG_B_T": pval(sb["t_p"]),
		"N_SIG_B_W": pval(sb["wilcoxon_p"]),
		"N_SIG_B_WINS": str(sb["wins"]),
		"N_SIG_B_LOSS": str(sb["losses"]),
		"N_SIG_V_T": pval(sv["t_p"]),
		"N_SIG_V_W": pval(sv["wilcoxon_p"]),
		"N_SIG_V_WINS": str(sv["wins"]),
		"N_SIG_V_LOSS": str(sv["losses"]),
		"N_CV0": f3(n["cv"][0]), "N_CV1": f3(n["cv"][1]),
		"N_DEV": f3(n["dev"]),
		"N_ALPHA": f"{w['alpha']:g}", "N_INC": f"{w['inc_view']:g}",
		"N_EXC": f"{w['exc_view']:g}", "N_ELIG": f"{w['elig_w']:g}",
		"N_ALONE": f3(t["MedCPT alone (topic view)"]["nDCG@10"]),
		"N_TOPIC": f3(t["Vega + MedCPT topic view"]["nDCG@10"]),
		"N_INCV": f3(t["+ inclusion view"]["nDCG@10"]),
		"N_EXCV": f3(t["+ exclusion view"]["nDCG@10"]),
		"N_EXCV_EX": f3(t["+ exclusion view"]["excluded@10"]),
		"N_INCV_EX": f3(t["+ inclusion view"]["excluded@10"]),
		"N_TOPIC_EX": f3(t["Vega + MedCPT topic view"]["excluded@10"]),
		"N_FAIL": f3(fail),
	}


def short(title, n=46):
	return tex(title if len(title) <= n else title[:n - 1] + "...")


def walkthrough(engine, s):
	pos = {d: i for i, d in enumerate(engine.ix.ids)}
	topics = evaluate.load_topics(2022)
	qrels = evaluate.load_qrels(2022, pos)
	full, base = s["tuned"]["neural"], s["tuned"]["raw"]
	best = None
	for t in topics:
		j = qrels.get(t["id"], {})
		a = engine.search(t["note"], dict(base, k=10))["ranked"]
		b = engine.search(t["note"], dict(full, k=10))["ranked"]
		ea = sum(j.get(d, 0) == 2 for d in a)
		eb = sum(j.get(d, 0) == 2 for d in b)
		xa = sum(j.get(d, 0) == 1 for d in a)
		xb = sum(j.get(d, 0) == 1 for d in b)
		gain = (eb - ea) + (xa - xb)
		if len(t["note"]) < 700 and (best is None or gain > best[0]):
			best = (gain, t, a, b, j)
	_, t, a, b, j = best
	res = engine.search(t["note"], dict(full, k=10))
	prof = res["profile"]
	lines = ["\\begin{tabular}{@{}rlll@{}}", "\\toprule",
		 "\\# & BM25F zones (raw note) & Vega-Neural & \\\\",
		 "\\midrule"]
	for i in range(5):
		ta = engine.store.get(a[i])["title"]
		tb = engine.store.get(b[i])["title"]
		la = LABEL.get(j.get(a[i]), "unjudged")
		lb = LABEL.get(j.get(b[i]), "unjudged")
		lines.append(f"{i + 1} & {short(ta)} ({la}) & {short(tb)} "
			     f"({lb}) & \\\\")
	lines += ["\\bottomrule", "\\end{tabular}"]
	neg = ", ".join(tex(prof["surface"].get(x, x))
			for x in list(prof["neg"])[:6]) or "none"
	abbr = ", ".join(f"{tex(k)}$\\to${tex(v)}"
			 for k, v in dict(prof["abbrev"]).items()) or "none"
	return "\n".join([
		"\\begin{sloppypar}",
		f"\\textbf{{Patient {t['id']}}} (test set): "
		f"\\emph{{{tex(t['note'])}}}", "\\end{sloppypar}", "",
		f"Vega's profile: age {prof['age']:g}, sex {prof['sex']}; "
		f"abbreviations {abbr}; negated findings: {neg}.", "",
		"\\begin{center}\\small", "\\resizebox{\\linewidth}{!}{",
		"\n".join(lines), "}", "\\end{center}"])


def main():
	with open(os.path.join(evaluate.RESULTS, "summary.json")) as f:
		s = json.load(f)
	engine = Engine.load()
	values = {}
	for part in (corpus_values(engine, s), result_values(s),
		     tuned_values(s), sig_values(s), ablation_values(s),
		     neural_values(s)):
		values.update(part)
	values["WALK"] = walkthrough(engine, s)
	with open(TEMPLATE) as f:
		body = f.read()
	missing = set(re.findall(r"@@([A-Z0-9_]+)@@", body)) - set(values)
	if missing:
		raise SystemExit(f"missing values: {sorted(missing)}")
	body = re.sub(r"@@([A-Z0-9_]+)@@", lambda m: values[m.group(1)], body)
	with open(OUT, "w") as f:
		f.write(body)
	print(f"wrote {OUT} with {len(values)} values")


if __name__ == "__main__":
	main()
