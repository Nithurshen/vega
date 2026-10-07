# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import json
import math
import pickle
from array import array
from multiprocessing import Pool

import numpy as np

from vega import text
from vega import trials

ZONES = ("title", "summary", "description", "conditions", "interventions",
	 "inclusion", "exclusion")
Z = len(ZONES)
ZONE_BASE = 10_000_000
ITEM_GAP = 64
POSITIONAL = (True, True, False, True, True, True, True)
CHAMPIONS = 500
CHAMPION_WEIGHTS = (3.0, 1.0, 0.3, 3.0, 1.0, 1.0, 0.0)
CHUNK = 20000
GENDER = {"All": 0, "Male": 1, "Female": 2}


def zone_of(pos):
	return pos // ZONE_BASE


def zone_texts(doc):
	return [
		[doc["title"], doc["official"]],
		[doc["summary"]],
		[doc["description"]],
		doc["conditions"],
		doc["interventions"],
		[doc["inclusion"]],
		[doc["exclusion"]],
	]


def doc_tokens(doc, do_stem):
	toks = []
	for z, items in enumerate(zone_texts(doc)):
		base = z * ZONE_BASE
		for item in items:
			got = text.analyze(item, do_stem, base)
			toks += got
			if got:
				base = got[-1][1] + ITEM_GAP
	return toks


class Index:
	def __init__(self):
		self.vocab = []
		self.lookup = {}
		self.do_stem = True

	def tid(self, term):
		return self.lookup.get(term, -1)

	def span(self, t):
		return self.off[t], self.off[t + 1]

	def docs(self, t):
		a, b = self.span(t)
		return self.post_docs[a:b]

	def tfs(self, t):
		a, b = self.span(t)
		return self.post_tf[a:b]

	def positions(self, t, i):
		p = self.off[t] + i
		return self.pos[self.pos_off[p]:self.pos_off[p + 1]]

	def champions(self, t):
		return self.champ[self.champ_off[t]:self.champ_off[t + 1]]

	def df_of(self, term):
		t = self.tid(term)
		return 0 if t < 0 else int(self.df[t])

	def save(self, path):
		with open(path, "wb") as f:
			pickle.dump(self.__dict__, f, protocol=5)

	@classmethod
	def load(cls, path):
		ix = cls()
		with open(path, "rb") as f:
			ix.__dict__.update(pickle.load(f))
		return ix


def collect(job):
	start, offsets, do_stem = job
	vocab, lookup = [], {}
	tids, dids = array("i"), array("i")
	tf = [array("B") for _ in range(Z)]
	pos, plen = array("i"), array("i")
	zlen = np.zeros((len(offsets), Z), dtype=np.float32)
	with open(trials.TRIALS, "rb") as f:
		for k, off in enumerate(offsets):
			f.seek(off)
			doc = json.loads(f.readline())
			groups = {}
			for term, p in doc_tokens(doc, do_stem):
				groups.setdefault(term, []).append(p)
				zlen[k, zone_of(p)] += 1
			for term, plist in groups.items():
				t = lookup.get(term)
				if t is None:
					t = lookup[term] = len(vocab)
					vocab.append(term)
				counts = [0] * Z
				for p in plist:
					counts[zone_of(p)] += 1
				tids.append(t)
				dids.append(start + k)
				for z in range(Z):
					tf[z].append(min(counts[z], 255))
				keep = [p for p in plist
					if POSITIONAL[zone_of(p)]]
				pos.extend(keep)
				plen.append(len(keep))
	return (vocab, np.frombuffer(tids, dtype=np.int32),
		np.frombuffer(dids, dtype=np.int32),
		np.stack([np.frombuffer(a, dtype=np.uint8) for a in tf], 1),
		np.frombuffer(pos, dtype=np.int32),
		np.frombuffer(plen, dtype=np.int32), zlen)


def merge_vocab(ix, parts):
	tids = []
	for vocab, local, *_ in parts:
		remap = np.empty(len(vocab), dtype=np.int32)
		for i, term in enumerate(vocab):
			t = ix.lookup.get(term)
			if t is None:
				t = ix.lookup[term] = len(ix.vocab)
				ix.vocab.append(term)
			remap[i] = t
		tids.append(remap[local])
	return np.concatenate(tids)


def finish_postings(ix, tids, parts):
	order = np.argsort(tids, kind="stable")
	ix.post_docs = np.concatenate([p[2] for p in parts])[order]
	ix.post_tf = np.concatenate([p[3] for p in parts])[order]
	ix.df = np.bincount(tids, minlength=len(ix.vocab)).astype(np.int32)
	ix.off = np.zeros(len(ix.vocab) + 1, dtype=np.int64)
	np.cumsum(ix.df, out=ix.off[1:])
	plen = np.concatenate([p[5] for p in parts]).astype(np.int64)
	pos = np.concatenate([p[4] for p in parts])
	starts = np.zeros(len(plen), dtype=np.int64)
	np.cumsum(plen[:-1], out=starts[1:])
	lens = plen[order]
	ix.pos_off = np.zeros(len(lens) + 1, dtype=np.int64)
	np.cumsum(lens, out=ix.pos_off[1:])
	gather = np.repeat(starts[order] - ix.pos_off[:-1], lens)
	gather += np.arange(int(ix.pos_off[-1]), dtype=np.int64)
	ix.pos = pos[gather]


def finish_stats(ix, zlen):
	n = len(zlen)
	ix.n = n
	ix.zlen = zlen
	ix.avg_zlen = np.maximum(zlen.mean(axis=0), 1.0)
	ix.idf = np.log10(n / np.maximum(ix.df, 1)).astype(np.float32)
	ix.bm25_idf = np.log(1 + (n - ix.df + 0.5) / (ix.df + 0.5))
	ix.bm25_idf = ix.bm25_idf.astype(np.float32)
	flat = ix.post_tf.sum(axis=1, dtype=np.float32)
	w = 1 + np.log10(np.maximum(flat, 1))
	ix.lnc_norm = np.sqrt(np.bincount(ix.post_docs, weights=w * w,
					  minlength=n)).astype(np.float32)


def finish_champions(ix):
	weights = np.array(CHAMPION_WEIGHTS, dtype=np.float32)
	impact = ix.post_tf.astype(np.float32) @ weights
	impact /= np.maximum(ix.lnc_norm[ix.post_docs], 1)
	sizes = np.minimum(ix.df, CHAMPIONS).astype(np.int64)
	ix.champ_off = np.zeros(len(ix.vocab) + 1, dtype=np.int64)
	np.cumsum(sizes, out=ix.champ_off[1:])
	ix.champ = np.empty(int(ix.champ_off[-1]), dtype=np.int32)
	small = ix.df <= CHAMPIONS
	owner = np.repeat(np.arange(len(ix.vocab)), ix.df)
	keep = small[owner]
	ix.champ[np.repeat(small, sizes)] = ix.post_docs[keep]
	for t in np.flatnonzero(~small):
		a, b = int(ix.off[t]), int(ix.off[t + 1])
		top = np.argpartition(-impact[a:b], CHAMPIONS)[:CHAMPIONS]
		c = int(ix.champ_off[t])
		ix.champ[c:c + CHAMPIONS] = np.sort(ix.post_docs[a:b][top])


def finish_params(ix, docs_meta):
	ix.ids = [m[0] for m in docs_meta]
	ix.min_age = np.array([m[1] for m in docs_meta], dtype=np.float32)
	ix.max_age = np.array([m[2] for m in docs_meta], dtype=np.float32)
	ix.gender = np.array([GENDER[m[3]] for m in docs_meta], dtype=np.int8)


def read_meta():
	meta = []
	with open(trials.TRIALS) as f:
		for line in f:
			d = json.loads(line)
			meta.append((d["id"], d["min_age"], d["max_age"],
				     d["gender"]))
	return meta


def build(do_stem=True, workers=12):
	with open(trials.OFFSETS) as f:
		offsets = json.load(f)
	jobs = [(i, offsets[i:i + CHUNK], do_stem)
		for i in range(0, len(offsets), CHUNK)]
	with Pool(workers) as pool:
		parts = pool.map(collect, jobs)
	ix = Index()
	ix.do_stem = do_stem
	tids = merge_vocab(ix, parts)
	finish_postings(ix, tids, parts)
	finish_stats(ix, np.concatenate([p[6] for p in parts]))
	del parts
	finish_champions(ix)
	finish_params(ix, read_meta())
	return ix


def skip_intersect(a, b):
	sa = max(1, int(math.sqrt(len(a))))
	sb = max(1, int(math.sqrt(len(b))))
	i = j = cmp = skips = 0
	out = []
	while i < len(a) and j < len(b):
		cmp += 1
		if a[i] == b[j]:
			out.append(int(a[i]))
			i += 1
			j += 1
		elif a[i] < b[j]:
			if i % sa == 0 and i + sa < len(a) and \
			   a[i + sa] <= b[j]:
				i += sa
				skips += 1
			else:
				i += 1
		elif j % sb == 0 and j + sb < len(b) and b[j + sb] <= a[i]:
			j += sb
			skips += 1
		else:
			j += 1
	return np.array(out, dtype=np.int32), cmp, skips


def doc_positions(ix, t, d):
	docs = ix.docs(t)
	i = int(np.searchsorted(docs, d))
	return set(int(p) for p in ix.positions(t, i))


def phrase_in_doc(ix, plan, d, zone=None):
	t0, o0 = plan[0]
	starts = {p - o0 for p in doc_positions(ix, t0, d)}
	if zone is not None:
		starts = {s for s in starts if zone_of(s) == zone}
	for t, off in plan[1:]:
		have = doc_positions(ix, t, d)
		starts = {s for s in starts if s + off in have}
		if not starts:
			return False
	return bool(starts)


def phrase_docs(ix, words, zone=None):
	plan = [(ix.tid(w), off) for w, off in words]
	if not plan or any(t < 0 for t, _ in plan):
		return np.zeros(0, dtype=np.int32)
	plan.sort(key=lambda p: ix.df[p[0]])
	cand = ix.docs(plan[0][0])
	for t, _ in plan[1:]:
		cand = np.intersect1d(cand, ix.docs(t), assume_unique=True)
	hits = [d for d in cand if phrase_in_doc(ix, plan, d, zone)]
	return np.array(hits, dtype=np.int32)
