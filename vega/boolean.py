# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import re

import numpy as np

from vega import index as index_mod
from vega import text

LEX = re.compile(r'\(|\)|[A-Za-z]+:"[^"]*"|"[^"]*"|[^\s()]+')
FIELDS = {z: i for i, z in enumerate(index_mod.ZONES)}
FIELDS.update({"condition": 3, "intervention": 4, "drug": 4})
AGE = re.compile(r"^(\d{1,3}(?:\.\d+)?)$")


class QueryError(Exception):
	pass


class Parser:
	def __init__(self, query, do_stem):
		self.toks = LEX.findall(query)
		self.i = 0
		self.do_stem = do_stem

	def peek(self):
		return self.toks[self.i] if self.i < len(self.toks) else None

	def take(self):
		tok = self.peek()
		if tok is None:
			raise QueryError("unexpected end of query")
		self.i += 1
		return tok

	def parse(self):
		node = self.parse_or()
		if self.peek() is not None:
			raise QueryError(f"unexpected '{self.peek()}'")
		return node

	def parse_or(self):
		kids = [self.parse_and()]
		while self.peek() == "OR":
			self.take()
			kids.append(self.parse_and())
		kids = [k for k in kids if k]
		return kids[0] if len(kids) == 1 else ("or", kids)

	def parse_and(self):
		kids = [self.parse_not()]
		while self.peek() not in (None, ")", "OR"):
			if self.peek() == "AND":
				self.take()
			kids.append(self.parse_not())
		kids = [k for k in kids if k]
		if not kids:
			return None
		return kids[0] if len(kids) == 1 else ("and", kids)

	def parse_not(self):
		if self.peek() == "NOT":
			self.take()
			child = self.parse_not()
			return ("not", child) if child else None
		return self.parse_atom()

	def parse_atom(self):
		tok = self.take()
		if tok == "(":
			node = self.parse_or()
			if self.take() != ")":
				raise QueryError("missing ')'")
			return node
		if tok == ")":
			raise QueryError("unbalanced ')'")
		zone = None
		if ":" in tok:
			field, rest = tok.split(":", 1)
			if field.lower() == "age":
				return self.age(rest)
			if field.lower() == "sex":
				return self.sex(rest)
			if field.lower() in FIELDS:
				zone, tok = FIELDS[field.lower()], rest
		return self.words(tok.strip('"'), zone, tok)

	def age(self, spec):
		m = AGE.match(spec)
		if not m:
			raise QueryError(f"bad age filter '{spec}'")
		return ("age", float(m.group(1)))

	def sex(self, spec):
		code = index_mod.GENDER.get(spec.capitalize())
		if code is None or code == 0:
			raise QueryError("sex must be male or female")
		return ("sex", code, spec.lower())

	def words(self, raw, zone, label):
		toks = text.analyze(raw, self.do_stem)
		if not toks:
			return None
		if len(toks) == 1:
			return ("term", toks[0][0], zone, label)
		first = toks[0][1]
		plan = [(t, p - first) for t, p in toks]
		return ("phrase", plan, zone, label)


def zone_docs(ix, term, zone):
	t = ix.tid(term)
	if t < 0:
		return np.zeros(0, dtype=np.int32)
	docs = ix.docs(t)
	if zone is None:
		return docs
	return docs[ix.tfs(t)[:, zone] > 0]


def run(ix, node, trace, positive):
	kind = node[0]
	if kind == "term":
		docs = zone_docs(ix, node[1], node[2])
		if positive:
			trace["terms"].append(node[1])
		trace["steps"].append(("postings", node[3], node[1], len(docs)))
		return docs
	if kind == "phrase":
		docs = index_mod.phrase_docs(ix, node[1], node[2])
		if positive:
			trace["terms"].extend(t for t, _ in node[1])
		stems = " ".join(t for t, _ in node[1])
		trace["steps"].append(("phrase", node[3], stems, len(docs)))
		return docs
	if kind == "age":
		ok = (ix.min_age <= node[1]) & (node[1] <= ix.max_age)
		docs = np.flatnonzero(ok).astype(np.int32)
		trace["steps"].append(("age", f"{node[1]:g}", "min<=age<=max",
				       len(docs)))
		return docs
	if kind == "sex":
		ok = (ix.gender == 0) | (ix.gender == node[1])
		docs = np.flatnonzero(ok).astype(np.int32)
		trace["steps"].append(("sex", node[2], "All or match",
				       len(docs)))
		return docs
	if kind == "or":
		out = np.zeros(0, dtype=np.int32)
		for k in node[1]:
			out = np.union1d(out, run(ix, k, trace, positive))
		trace["steps"].append(("OR", "", "", len(out)))
		return out
	if kind == "not":
		child = run(ix, node[1], trace, not positive)
		return np.setdiff1d(np.arange(ix.n, dtype=np.int32), child)
	return run_and(ix, node[1], trace, positive)


def run_and(ix, kids, trace, positive):
	pos = [run(ix, k, trace, positive) for k in kids if k[0] != "not"]
	neg = [run(ix, k[1], trace, not positive)
	       for k in kids if k[0] == "not"]
	pos.sort(key=len)
	if pos:
		out = pos[0]
		for p in pos[1:]:
			out, cmp, skips = index_mod.skip_intersect(out.tolist(),
								   p.tolist())
			trace["steps"].append(("AND", f"{cmp} comparisons",
					       f"{skips} skips", len(out)))
	else:
		out = np.arange(ix.n, dtype=np.int32)
	for n in neg:
		out = np.setdiff1d(out, n)
		trace["steps"].append(("AND NOT", "", "", len(out)))
	return out


def search(ix, query):
	node = Parser(query, ix.do_stem).parse()
	trace = {"steps": [], "terms": []}
	if node is None:
		return np.zeros(0, dtype=np.int32), trace
	return run(ix, node, trace, True), trace
