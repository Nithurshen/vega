# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import re
import unicodedata
from functools import lru_cache

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be
because been before being below between both but by can could did do does
doing down during each few for from further had has have having he her
here hers herself him himself his how however i if in into is it its
itself just may me might more most must my myself no nor not now of off
on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these
they this those through thus to too under until up upon us very via was
we were what when where which while who whom why will with within without
would yet you your yours yourself yourselves et al eg ie etc
""".split())

TOKEN = re.compile(r"[a-z0-9]+")


def fold(text):
	text = unicodedata.normalize("NFKD", text)
	text = text.encode("ascii", "ignore").decode("ascii")
	return text.lower()


def tokens(text):
	return TOKEN.findall(fold(text))


def is_cons(w, i):
	c = w[i]
	if c in "aeiou":
		return False
	if c == "y":
		return i == 0 or not is_cons(w, i - 1)
	return True


def measure(s):
	m = 0
	vowel = False
	for i in range(len(s)):
		v = not is_cons(s, i)
		if vowel and not v:
			m += 1
		vowel = v
	return m


def has_vowel(s):
	return any(not is_cons(s, i) for i in range(len(s)))


def double_cons(s):
	return len(s) > 1 and s[-1] == s[-2] and is_cons(s, len(s) - 1)


def cvc(s):
	if len(s) < 3 or s[-1] in "wxy":
		return False
	n = len(s)
	return (is_cons(s, n - 3) and not is_cons(s, n - 2)
		and is_cons(s, n - 1))


def step1a(w):
	if w.endswith("sses") or w.endswith("ies"):
		return w[:-2]
	if w.endswith("ss"):
		return w
	if w.endswith("s"):
		return w[:-1]
	return w


def step1b_fix(s):
	if s.endswith(("at", "bl", "iz")):
		return s + "e"
	if double_cons(s) and s[-1] not in "lsz":
		return s[:-1]
	if measure(s) == 1 and cvc(s):
		return s + "e"
	return s


def step1b(w):
	if w.endswith("eed"):
		return w[:-1] if measure(w[:-3]) > 0 else w
	for suf in ("ed", "ing"):
		if w.endswith(suf) and has_vowel(w[:-len(suf)]):
			return step1b_fix(w[:-len(suf)])
	return w


def step1c(w):
	if w.endswith("y") and has_vowel(w[:-1]):
		return w[:-1] + "i"
	return w


STEP2 = [
	("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
	("anci", "ance"), ("izer", "ize"), ("abli", "able"),
	("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
	("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
	("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
	("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
	("biliti", "ble"),
]

STEP3 = [
	("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
	("ical", "ic"), ("ful", ""), ("ness", ""),
]

STEP4 = [
	"al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
	"ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


def by_length(rules):
	return sorted(rules, key=lambda r: -len(r[0]))


STEP2 = by_length(STEP2)
STEP3 = by_length(STEP3)
STEP4 = sorted(STEP4, key=len, reverse=True)


def replace(w, rules, min_m):
	for suf, rep in rules:
		if w.endswith(suf):
			stem = w[:-len(suf)]
			return stem + rep if measure(stem) > min_m else w
	return w


def step4(w):
	for suf in STEP4:
		if not w.endswith(suf):
			continue
		stem = w[:-len(suf)]
		if measure(stem) <= 1:
			return w
		if suf == "ion" and not stem.endswith(("s", "t")):
			return w
		return stem
	return w


def step5(w):
	if w.endswith("e"):
		stem = w[:-1]
		m = measure(stem)
		if m > 1 or (m == 1 and not cvc(stem)):
			w = stem
	if measure(w) > 1 and double_cons(w) and w.endswith("l"):
		w = w[:-1]
	return w


@lru_cache(maxsize=None)
def stem(w):
	if len(w) <= 2 or not w.isalpha():
		return w
	w = step1c(step1b(step1a(w)))
	w = replace(w, STEP2, 0)
	w = replace(w, STEP3, 0)
	return step5(step4(w))


def analyze(text, do_stem=True, base=0):
	out = []
	for i, tok in enumerate(tokens(text)):
		if tok in STOPWORDS or len(tok) < 2 or tok.isdigit():
			continue
		out.append((stem(tok) if do_stem else tok, base + i))
	return out


def terms(text, do_stem=True):
	return [t for t, _ in analyze(text, do_stem)]


def surface(text_in, do_stem=True):
	words = {}
	for tok in tokens(text_in):
		if tok in STOPWORDS or len(tok) < 2 or tok.isdigit():
			continue
		words.setdefault(stem(tok) if do_stem else tok, tok)
	return words
