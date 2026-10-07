# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import html
import json
import os

import pandas as pd
import streamlit as st

from vega import boolean
from vega import evaluate
from vega import index as index_mod
from vega import text
from vega.rank import Engine, config

SUMMARY = os.path.join("results", "summary.json")
FIG = os.path.join("report", "figures")
MODELS = {"BM25F (zones)": "bm25f", "BM25": "bm25", "TF-IDF lnc.ltc": "tfidf"}
PARAMS = {"Hard filter": "hard", "Soft penalty": "soft", "Off": "off"}
LABELS = {2: "✅ eligible", 1: "⛔ excluded", 0: "✖ not relevant"}
COLORS = {"neg": "#e34948", "family": "#8a8984", "cue": "#eb6834"}


@st.cache_resource(show_spinner="Loading the trial index")
def load_engine():
	return Engine.load()


@st.cache_data
def load_benchmark():
	engine = load_engine()
	pos = {d: i for i, d in enumerate(engine.ix.ids)}
	topics = evaluate.load_topics(2022) + evaluate.load_topics(2021)
	qrels = {**evaluate.load_qrels(2021, pos),
		 **evaluate.load_qrels(2022, pos)}
	return topics, qrels


@st.cache_data
def load_summary():
	if not os.path.exists(SUMMARY):
		return None
	with open(SUMMARY) as f:
		return json.load(f)


def tuned():
	s = load_summary()
	if not s:
		return config(params="soft", exc_w=0.2)
	return dict(s["tuned"].get("neural", s["tuned"]["full"]))


def sidebar(engine):
	base = tuned()
	st.sidebar.title("Vega")
	st.sidebar.caption(f"Patient-to-trial matching over {engine.ix.n:,} "
			   "ClinicalTrials.gov studies")
	model = st.sidebar.selectbox("Text model", list(MODELS))
	with st.sidebar.expander("Patient-note processing"):
		abbrev = st.toggle("Expand abbreviations", base["abbrev"])
		negation = st.toggle("Remove negated findings",
				     base["negation"])
		family = st.toggle("Remove family history", base["family"])
	params = st.sidebar.selectbox("Age / sex eligibility", list(PARAMS),
				      index=list(PARAMS.values()).index(
					      base["params"]))
	exc = st.sidebar.slider("Exclusion penalty", 0.0, 1.0,
				float(base["exc_w"]), 0.05)
	neg = st.sidebar.slider("Negation penalty", 0.0, 1.0,
				float(base["neg_w"]), 0.05)
	neural = st.sidebar.toggle("Neural re-ranking (MedCPT)",
				   base.get("neural", False))
	k = st.sidebar.slider("Results", 5, 30, 10, 5)
	with st.sidebar.expander("Efficiency"):
		m = st.number_input("Index elimination: keep top-m terms "
				    "(0 = all)", 0, 300, 0)
		champ = st.toggle("Champion lists (r = 500)", False)
	return dict(base, model=MODELS[model], abbrev=abbrev,
		    negation=negation, family=family,
		    params=PARAMS[params], exc_w=exc, neg_w=neg, k=k,
		    neural=neural,
		    max_terms=int(m), champions=champ)


def pick_topic():
	topics, qrels = load_benchmark()
	names = ["(type your own patient note)"] + [
		f"{t['id']}: {t['note'][:90]}..." for t in topics]
	choice = st.selectbox("Load a TREC patient (2022 = test, 2021 = "
			      "tuning)", names)
	if choice == names[0]:
		return None, {}
	t = topics[names.index(choice) - 1]
	return t, qrels.get(t["id"], {})


def highlight(prof):
	parts = []
	for word, tag in prof.get("tagged", []):
		w = html.escape(word)
		if tag in COLORS:
			style = f"color:{COLORS[tag]}"
			if tag == "neg":
				style += ";text-decoration:line-through"
			parts.append(f"<span style='{style}'>{w}</span>")
		else:
			parts.append(w)
	return " ".join(parts)


def profile_panel(prof):
	c = st.columns(4)
	c[0].metric("age", "unknown" if prof["age"] is None else
		    f"{prof['age']:g}")
	c[1].metric("sex", prof["sex"] or "unknown")
	c[2].metric("negated findings", len(prof["neg"]))
	c[3].metric("family-history terms", len(prof["family"]))
	with st.expander("How the note was read", expanded=False):
		st.markdown(highlight(prof), unsafe_allow_html=True)
		st.caption("red struck-through = negated (NegEx-style), "
			   "grey = family history, orange = negation cue")
		if prof["abbrev"]:
			seen = dict(prof["abbrev"])
			st.dataframe(pd.DataFrame(seen.items(), columns=[
				"abbreviation", "expanded to"]),
				hide_index=True, width="stretch")


def pipeline_panel(engine, res):
	c = st.columns(4)
	c[0].metric("query terms", len(res["kept"]))
	c[1].metric("postings scanned", f"{res['scanned']:,}")
	c[2].metric("trials passing age/sex", f"{int(res['ok'].sum()):,}")
	c[3].metric("ms", f"{sum(res['timing'].values()) * 1000:.0f}")
	with st.expander("Query vector"):
		ix = engine.ix
		st.dataframe(pd.DataFrame([{
			"term": ix.vocab[t], "qtf": q, "df": int(ix.df[t]),
			"bm25 idf": float(ix.bm25_idf[t])}
			for t, q in res["kept"]]), hide_index=True,
			width="stretch")
	if res["expansion"]:
		words = ", ".join(engine.ix.vocab[t]
				  for t, _ in res["expansion"])
		st.caption("Conditions-zone feedback added: " + words)


def topic_metrics(ranked, judged):
	m = evaluate.metrics(ranked, judged)
	c = st.columns(4)
	for col, key in zip(c, ("nDCG@10", "P@10", "excluded@10",
				"judged@10")):
		col.metric(key, f"{m[key]:.2f}")


def age_range(trial):
	hi = trial["max_age"]
	hi = "no max" if hi >= 999 else f"{hi:g}"
	return f"{trial['min_age']:g}–{hi} y, {trial['gender']}"


def show_trial(engine, res, cfg, rank, d, judged):
	trial = engine.store.get(d)
	label = LABELS.get(judged.get(d), "· unjudged") if judged else ""
	ok = "✓" if res["ok"][d] else "✗"
	st.markdown(f"**{rank}. [{trial['id']}](https://clinicaltrials.gov/"
		    f"study/{trial['id']})** {html.escape(trial['title'])}  \n"
		    f"{label} · age/sex {ok} ({age_range(trial)}) · "
		    f"{', '.join(trial['conditions'][:4])}")
	c = st.columns(4)
	if d in res["ce"]:
		ce = res["ce"][d]
		c[0].metric("fused", f"{res['fused'][d]:.3f}")
		c[1].metric("MedCPT: topic", f"{ce[0]:.1f}")
		c[2].metric("MedCPT: inclusion", f"{ce[1]:.1f}")
		c[3].metric("MedCPT: exclusion", f"{ce[2]:.1f}")
	else:
		c[0].metric("final", f"{res['final'][d]:.3f}")
		c[1].metric("text", f"{res['base'][d]:.3f}")
		c[2].metric("exclusion penalty", f"{res['exc'][d]:.3f}")
		c[3].metric("negation penalty", f"{res['neg'][d]:.3f}")
	with st.expander("Why this trial?"):
		why(engine, res, cfg, d, trial)


def why(engine, res, cfg, d, trial):
	rows = engine.explain(d, res, cfg)[:10]
	frame = pd.DataFrame([dict({"term": t, "score": s},
				   **dict(zip(index_mod.ZONES, z)))
			      for t, s, z in rows])
	if len(frame):
		st.dataframe(frame, hide_index=True, width="stretch")
	prof = res["profile"]
	kept = [engine.ix.vocab[t] for t, _ in res["kept"]]
	exc = engine.zone_hits(d, kept, index_mod.ZONES.index("exclusion"),
			       True)
	neg = engine.zone_hits(d, list(prof["neg"]),
			       index_mod.ZONES.index("inclusion"), False)
	if exc:
		st.write("Patient findings in this trial's **exclusion** "
			 "criteria: " + ", ".join(exc))
	if neg:
		st.write("Findings the patient **does not have** that the "
			 "inclusion criteria mention: " + ", ".join(neg))
	st.caption("Inclusion: " + trial["inclusion"][:600])
	st.caption("Exclusion: " + trial["exclusion"][:600])


def tab_match(engine, cfg):
	topic, judged = pick_topic()
	note = st.text_area("Patient note", topic["note"] if topic else "",
			    height=170, key=f"n{topic['id'] if topic else ''}")
	if not note.strip():
		st.info("Paste a patient description or load a TREC patient.")
		return
	with st.spinner("Ranking (neural re-ranking of the top 100 can take "
			"a few seconds)"):
		res = engine.search(note, cfg)
	profile_panel(res["profile"])
	pipeline_panel(engine, res)
	if judged:
		st.subheader("Against the expert judgements")
		topic_metrics(engine.search(note, dict(cfg, k=1000))["ranked"],
			      judged)
	st.subheader("Matching trials")
	for i, d in enumerate(res["ranked"]):
		show_trial(engine, res, cfg, i + 1, d, judged)


def tab_boolean(engine, cfg):
	st.caption('Syntax: AND, OR, NOT, ( ), "phrase", zone:word with zones '
		   + ", ".join(index_mod.ZONES) + "; age:45; sex:female")
	query = st.text_input("Boolean query", 'conditions:"breast cancer" '
			      "AND age:45 AND sex:female NOT "
			      "exclusion:pregnant")
	if not query.strip():
		return
	try:
		docs, trace = boolean.search(engine.ix, query)
	except boolean.QueryError as e:
		st.error(str(e))
		return
	st.dataframe(pd.DataFrame(trace["steps"], columns=[
		"operation", "input", "detail", "result size"]),
		hide_index=True, width="stretch")
	st.write(f"**{len(docs):,} matching trials**, ranked by BM25F on the "
		 "positive terms:")
	kept = [(engine.ix.tid(t), 1) for t in trace["terms"]
		if engine.ix.tid(t) >= 0]
	scores, _ = engine.lexical(kept, cfg)
	for i, d in enumerate(sorted(docs.tolist(),
				     key=lambda d: -scores[d])[:cfg["k"]]):
		trial = engine.store.get(d)
		st.markdown(f"**{i + 1}. {trial['id']}** "
			    f"{html.escape(trial['title'])} "
			    f"({age_range(trial)})")


def tab_index(engine):
	word = st.text_input("Look up a word", "metastases")
	toks = text.tokens(word)
	if not toks:
		return
	term = text.stem(toks[0])
	ix = engine.ix
	t = ix.tid(term)
	st.write(f"normalised `{toks[0]}` → stem `{term}`")
	if t < 0:
		st.warning("Not in the dictionary.")
		return
	docs, tfs = ix.docs(t), ix.tfs(t)
	c = st.columns(3)
	c[0].metric("document frequency", f"{int(ix.df[t]):,}")
	c[1].metric("idf", f"{ix.idf[t]:.3f}")
	c[2].metric("champion list", len(ix.champions(t)))
	zone_df = (tfs > 0).sum(axis=0)
	st.bar_chart(pd.DataFrame({"trials containing it": zone_df},
				  index=index_mod.ZONES))
	rows = []
	for i in range(min(20, len(docs))):
		where = ix.positions(t, i)[:6].tolist()
		rows.append(dict({"trial": ix.ids[int(docs[i])]},
				 **dict(zip(index_mod.ZONES,
					    tfs[i].tolist())),
				 positions=str(where)))
	st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def tab_eval():
	s = load_summary()
	if not s:
		st.info("Run `python -m vega.evaluate` first.")
		return
	st.write("Tuned on the 75 TREC CT 2021 patients; tested on the 50 "
		 "TREC CT 2022 patients.")
	cols = list(evaluate.METRICS)
	for key, title in (("test", "Test systems"), ("ablation",
						    "Ablations")):
		st.subheader(title)
		st.dataframe(pd.DataFrame(s[key]).T[cols].style.format(
			"{:.3f}"), width="stretch")
	for name in ("main", "eligible", "pertopic", "penalties",
		     "efficiency"):
		path = os.path.join(FIG, f"{name}.png")
		if os.path.exists(path):
			st.image(path)


def main():
	st.set_page_config(page_title="Vega", page_icon="✦", layout="wide")
	engine = load_engine()
	cfg = sidebar(engine)
	tabs = st.tabs(["Match trials", "Boolean & phrase", "Index inspector",
			"Evaluation"])
	with tabs[0]:
		tab_match(engine, cfg)
	with tabs[1]:
		tab_boolean(engine, cfg)
	with tabs[2]:
		tab_index(engine)
	with tabs[3]:
		tab_eval()


main()
