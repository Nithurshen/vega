# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import heapq
import os
import time
from collections import Counter

import numpy as np

from vega import index as index_mod
from vega import neural
from vega import patient
from vega import text
from vega import trials

COND = index_mod.ZONES.index("conditions")
INC = index_mod.ZONES.index("inclusion")
EXC = index_mod.ZONES.index("exclusion")
HEAP_MAX = 100

DEFAULT = {
	"model": "bm25f",
	"abbrev": True,
	"negation": True,
	"family": True,
	"zone_w": (2.0, 1.0, 0.5, 2.0, 0.5, 1.0, 0.0),
	"zone_b": (0.5, 0.75, 0.75, 0.5, 0.5, 0.75, 0.75),
	"k1": 1.2,
	"b": 0.75,
	"k3": 8.0,
	"max_terms": 0,
	"champions": False,
	"params": "off",
	"soft": 0.5,
	"exc_w": 0.0,
	"neg_w": 0.0,
	"pen_ratio": 0.0,
	"prf_docs": 0,
	"prf_terms": 10,
	"prf_w": 0.5,
	"neural": False,
	"rerank_k": 100,
	"alpha": 0.5,
	"inc_view": 0.4,
	"exc_view": 0.1,
	"elig_w": 0.2,
	"k": 20,
}


def config(**kw):
	cfg = dict(DEFAULT)
	cfg.update(kw)
	return cfg


def top_k(scores, k, mask=None):
	live = scores > 0 if mask is None else (scores > 0) & mask
	nz = np.flatnonzero(live)
	if k <= HEAP_MAX or len(nz) <= k:
		return heapq.nlargest(k, nz.tolist(), key=scores.__getitem__)
	part = nz[np.argpartition(-scores[nz], k)[:k]]
	return part[np.argsort(-scores[part], kind="stable")].tolist()


def one_hot(z):
	w = [0.0] * index_mod.Z
	w[z] = 1.0
	return tuple(w)


def clinical_ratio(ix):
	owner = np.repeat(np.arange(len(ix.vocab), dtype=np.int32), ix.df)
	hit = ix.post_tf[:, COND] > 0
	cond_df = np.bincount(owner[hit], minlength=len(ix.vocab))
	return (cond_df / np.maximum(ix.df, 1)).astype(np.float32)


class Engine:
	def __init__(self, ix, store=None):
		self.ix = ix
		self.store = store
		self.clinical = clinical_ratio(ix)
		self.cross = None
		self.ce_cache = {}

	@classmethod
	def load(cls, root=os.path.join("data", "trec")):
		ix = index_mod.Index.load(os.path.join(root, "index.pkl"))
		return cls(ix, trials.Store())

	def query(self, note, cfg):
		return patient.profile(note, cfg["abbrev"], cfg["negation"],
				       cfg["family"], self.ix.do_stem)

	def qweight(self, qtf, cfg):
		return (cfg["k3"] + 1) * qtf / (cfg["k3"] + qtf)

	def resolve(self, counts):
		return {self.ix.tid(t): c for t, c in counts.items()
			if self.ix.tid(t) >= 0}

	def feedback(self, base, mask, kept, cfg):
		top = top_k(base, cfg["prf_docs"], mask)
		have = {t for t, _ in kept}
		votes = Counter()
		for d in top:
			trial = self.store.get(d)
			seen = set()
			for item in trial["conditions"]:
				seen.update(text.terms(item, self.ix.do_stem))
			for term in seen:
				t = self.ix.tid(term)
				if t >= 0 and t not in have:
					votes[t] += 1
		ranked = sorted(votes, key=lambda t: -votes[t] *
				float(self.ix.idf[t]))
		return [(t, cfg["prf_w"] * votes[t] / len(top))
			for t in ranked[:cfg["prf_terms"]]] if top else []

	def select(self, terms, cfg):
		ranked = sorted(terms.items(), key=lambda p: -self.qweight(
			p[1], cfg) * self.ix.idf[p[0]])
		cap = cfg["max_terms"]
		return (ranked[:cap], ranked[cap:]) if cap else (ranked, [])

	def postings(self, t, cfg):
		docs, tf = self.ix.docs(t), self.ix.tfs(t)
		if cfg["champions"]:
			keep = np.isin(docs, self.ix.champions(t),
				       assume_unique=True)
			docs, tf = docs[keep], tf[keep]
		return docs, tf

	def term_score(self, t, qtf, docs, tf, cfg, zone_w=None):
		ix = self.ix
		k1 = cfg["k1"]
		if cfg["model"] == "tfidf":
			flat = tf.sum(axis=1, dtype=np.float32)
			return qtf * (1 + np.log10(flat)) / ix.lnc_norm[docs]
		if cfg["model"] == "bm25" and zone_w is None:
			flat = tf.sum(axis=1, dtype=np.float32)
			length = ix.zlen[docs].sum(axis=1)
			norm = k1 * (1 - cfg["b"] + cfg["b"] * length /
				     ix.avg_zlen.sum())
			sat = flat * (k1 + 1) / (flat + norm)
		else:
			zw = np.array(zone_w or cfg["zone_w"], dtype=np.float32)
			zb = np.array(cfg["zone_b"], dtype=np.float32)
			lnorm = 1 - zb + zb * ix.zlen[docs] / ix.avg_zlen
			pseudo = (tf.astype(np.float32) / lnorm) @ zw
			sat = pseudo * (k1 + 1) / (pseudo + k1)
		return ix.bm25_idf[t] * self.qweight(qtf, cfg) * sat

	def ltc(self, terms):
		w = {t: (1 + np.log10(q)) * float(self.ix.idf[t])
		     for t, q in terms}
		norm = np.sqrt(sum(v * v for v in w.values())) or 1.0
		return {t: v / norm for t, v in w.items()}

	def lexical(self, kept, cfg):
		acc = np.zeros(self.ix.n, dtype=np.float32)
		weights = self.ltc(kept) if cfg["model"] == "tfidf" else None
		scanned = 0
		for t, qtf in kept:
			docs, tf = self.postings(t, cfg)
			scanned += len(docs)
			w = weights[t] if weights else qtf
			acc[docs] += self.term_score(t, w, docs, tf, cfg)
		return acc, scanned

	def zone_penalty(self, terms, zone, cfg, skip_condition):
		acc = np.zeros(self.ix.n, dtype=np.float32)
		zw = one_hot(zone)
		for t, qtf in terms.items():
			if self.clinical[t] < cfg["pen_ratio"]:
				continue
			docs, tf = self.ix.docs(t), self.ix.tfs(t)
			hit = tf[:, zone] > 0
			if skip_condition:
				hit &= tf[:, COND] == 0
			if not hit.any():
				continue
			acc[docs[hit]] += self.term_score(t, qtf, docs[hit],
							  tf[hit], cfg, zw)
		return acc

	def eligible(self, prof):
		ix = self.ix
		ok = np.ones(ix.n, dtype=bool)
		if prof["age"] is not None:
			ok &= (ix.min_age <= prof["age"] + 1e-6) & \
			      (prof["age"] <= ix.max_age + 1e-6)
		if prof["sex"] is not None:
			code = index_mod.GENDER[prof["sex"].capitalize()]
			ok &= (ix.gender == 0) | (ix.gender == code)
		return ok

	def search(self, note, cfg):
		cfg = {**DEFAULT, **cfg}
		t0 = time.perf_counter()
		prof = self.query(note, cfg)
		pos = self.resolve(prof["pos"])
		kept, dropped = self.select(pos, cfg)
		base, scanned = self.lexical(kept, cfg)
		ok = self.eligible(prof)
		expansion = []
		if cfg["prf_docs"]:
			gate = ok if cfg["params"] == "hard" else None
			expansion = self.feedback(base, gate, kept, cfg)
			extra, more = self.lexical(expansion, cfg)
			base = base + extra
			scanned += more
		t1 = time.perf_counter()
		top = float(base.max()) or 1.0
		final = base / top
		exc = np.zeros(self.ix.n, dtype=np.float32)
		neg = np.zeros(self.ix.n, dtype=np.float32)
		if cfg["exc_w"] > 0:
			exc = self.zone_penalty(dict(kept), EXC, cfg, True)
			exc = exc / top
			final = final - cfg["exc_w"] * exc
		if cfg["neg_w"] > 0 and prof["neg"]:
			negs = self.resolve(prof["neg"])
			neg = self.zone_penalty(negs, INC, cfg, False) / top
			final = final - cfg["neg_w"] * neg
		mask = base > 0
		if cfg["params"] == "hard":
			mask &= ok
		elif cfg["params"] == "soft":
			final = final - cfg["soft"] * (~ok)
		t2 = time.perf_counter()
		shifted = np.where(mask, final - final[mask].min() + 1e-6
				   if mask.any() else 0, 0)
		depth = max(cfg["k"], cfg["rerank_k"]) if cfg["neural"] \
			else cfg["k"]
		ranked = top_k(shifted, depth)
		t3 = time.perf_counter()
		res = {
			"profile": prof,
			"kept": kept,
			"dropped": dropped,
			"expansion": expansion,
			"ranked": ranked,
			"final": final,
			"base": base / top,
			"exc": exc,
			"neg": neg,
			"ok": ok,
			"scanned": scanned,
			"accumulators": int((base > 0).sum()),
			"timing": {"lexical": t1 - t0, "eligibility": t2 - t1,
				   "heap": t3 - t2},
			"ce": {},
			"fused": {},
		}
		if cfg["neural"]:
			self.rerank(note, res, cfg)
		res["ranked"] = res["ranked"][:cfg["k"]]
		return res

	def neural_scores(self, note, cands):
		if self.cross is None:
			self.cross = neural.CrossEncoder()
		key = hash(note)
		todo = [d for d in cands if (key, d) not in self.ce_cache]
		if todo:
			got = self.cross.trial_scores(
				note, [self.store.get(d) for d in todo])
			for d, row in zip(todo, got):
				self.ce_cache[(key, d)] = row
		return np.array([self.ce_cache[(key, d)] for d in cands])

	def rerank(self, note, res, cfg):
		t0 = time.perf_counter()
		cands = res["ranked"][:cfg["rerank_k"]]
		if not cands:
			return
		ce = self.neural_scores(note, cands)
		fused = neural.fuse(res["final"][cands], ce, res["ok"][cands],
				    cfg)
		order = np.argsort(-fused, kind="stable")
		res["ranked"] = [cands[i] for i in order] + \
			res["ranked"][len(cands):]
		res["ce"] = {d: ce[i] for i, d in enumerate(cands)}
		res["fused"] = {d: float(fused[i]) for i, d in enumerate(cands)}
		res["timing"]["neural"] = time.perf_counter() - t0

	def explain(self, d, res, cfg):
		ix = self.ix
		rows = []
		for t, qtf in res["kept"]:
			docs = ix.docs(t)
			i = int(np.searchsorted(docs, d))
			if i >= len(docs) or docs[i] != d:
				continue
			tf = ix.tfs(t)[i]
			s = float(self.term_score(t, qtf, docs[i:i + 1],
						  ix.tfs(t)[i:i + 1], cfg)[0])
			rows.append((ix.vocab[t], s, tuple(int(x) for x in tf)))
		rows.sort(key=lambda r: -r[1])
		return rows

	def zone_hits(self, d, terms, zone, skip_condition):
		out = []
		for term in terms:
			t = self.ix.tid(term)
			if t < 0:
				continue
			docs = self.ix.docs(t)
			i = int(np.searchsorted(docs, d))
			if i >= len(docs) or docs[i] != d:
				continue
			tf = self.ix.tfs(t)[i]
			if tf[zone] and not (skip_condition and tf[COND]):
				out.append(term)
		return out
