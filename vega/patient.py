# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import re
from collections import Counter

from vega import text

ABBREV = {
	"h/o": "history of", "s/p": "status post", "c/o": "complains of",
	"w/": "with", "w/o": "without", "htn": "hypertension",
	"dm": "diabetes mellitus", "t2dm": "type 2 diabetes mellitus",
	"t1dm": "type 1 diabetes mellitus", "dm2": "type 2 diabetes mellitus",
	"chf": "congestive heart failure", "hf": "heart failure",
	"cad": "coronary artery disease", "mi": "myocardial infarction",
	"copd": "chronic obstructive pulmonary disease",
	"sob": "shortness of breath", "doe": "dyspnea on exertion",
	"ckd": "chronic kidney disease", "esrd": "end stage renal disease",
	"aki": "acute kidney injury", "afib": "atrial fibrillation",
	"af": "atrial fibrillation",
	"dvt": "deep vein thrombosis", "uti": "urinary tract infection",
	"uri": "upper respiratory infection", "hiv": "hiv infection",
	"hcv": "hepatitis c", "hbv": "hepatitis b", "tb": "tuberculosis",
	"cva": "stroke", "tia": "transient ischemic attack",
	"sah": "subarachnoid hemorrhage", "ich": "intracerebral hemorrhage",
	"gerd": "gastroesophageal reflux disease",
	"ibd": "inflammatory bowel disease",
	"ibs": "irritable bowel syndrome", "uc": "ulcerative colitis",
	"ra": "rheumatoid arthritis", "sle": "systemic lupus erythematosus",
	"oa": "osteoarthritis", "bph": "benign prostatic hyperplasia",
	"pcos": "polycystic ovary syndrome", "osa": "obstructive sleep apnea",
	"ards": "acute respiratory distress syndrome",
	"nsclc": "non small cell lung cancer", "sclc": "small cell lung cancer",
	"hcc": "hepatocellular carcinoma",
	"cll": "chronic lymphocytic leukemia",
	"aml": "acute myeloid leukemia", "cml": "chronic myeloid leukemia",
	"nhl": "non hodgkin lymphoma", "adhd": "attention deficit "
	"hyperactivity disorder", "ocd": "obsessive compulsive disorder",
	"ptsd": "post traumatic stress disorder", "mdd": "major depressive "
	"disorder", "gad": "generalized anxiety disorder",
	"als": "amyotrophic lateral sclerosis", "tbi": "traumatic brain injury",
	"lvh": "left ventricular hypertrophy", "ef": "ejection fraction",
	"lv": "left ventricular", "rv": "right ventricular",
	"tte": "transthoracic echocardiogram", "ekg": "electrocardiogram",
	"ecg": "electrocardiogram", "cabg": "coronary artery bypass graft",
	"pci": "percutaneous coronary intervention", "le": "lower extremity",
	"rle": "right lower extremity", "lle": "left lower extremity",
	"bmi": "body mass index", "gi": "gastrointestinal",
	"ams": "altered mental status", "loc": "loss of consciousness",
	"n/v": "nausea vomiting", "abd": "abdominal", "hx": "history",
	"dx": "diagnosis", "tx": "treatment", "fx": "fracture",
	"sx": "symptoms", "pmh": "past medical history", "cp": "chest pain",
}

TRIGGERS = [
	("no", "history", "of"), ("no", "evidence", "of"),
	("no", "signs", "of"), ("negative", "for"), ("free", "of"),
	("absence", "of"), ("ruled", "out"), ("rules", "out"), ("no",),
	("not",), ("denies",), ("denied",), ("denying",), ("without",),
	("never",), ("nor",), ("neither",),
]
POST_TRIGGERS = {"negative", "unremarkable", "absent"}
TERMINATORS = {".", ",", ";", ":", "!", "?", "(", ")", "but", "however",
	       "although", "except", "which", "who", "presents",
	       "presented", "reports"}
FAMILY = {"family", "mother", "father", "brother", "sister", "aunt",
	  "uncle", "grandmother", "grandfather", "parents", "sibling",
	  "siblings", "son", "daughter", "cousin"}
SCOPE = 6
FEMALE = {"woman", "female", "girl", "lady", "she", "her", "hers", "mrs",
	  "pregnant", "f"}
MALE = {"man", "male", "boy", "he", "his", "him", "mr", "gentleman", "m"}
LEX = re.compile(r"[a-z0-9]+(?:/[a-z0-9]*)?|[.,;:!?()]")
DEID = re.compile(r"\[\*\*.*?\*\*\]")
AGE_RULES = [
	(re.compile(r"(\d{1,3})[- ]?(?:years?|yrs?)[- ]?old"), 1.0),
	(re.compile(r"(\d{1,3})[- ]?(?:years?|yrs?) (?:man|woman|male|female|"
		    r"boy|girl|patient|lady|gentleman)"), 1.0),
	(re.compile(r"(\d{1,3}) ?(?:yo|y/o|y\.o\.?)\b"), 1.0),
	(re.compile(r"\bage[d]?(?: of)? (\d{1,3})\b"), 1.0),
	(re.compile(r"(\d{1,2})[- ]?months?[- ]?old"), 1 / 12),
	(re.compile(r"(\d{1,2})[- ]?weeks?[- ]?old"), 1 / 52),
	(re.compile(r"(\d{1,2})[- ]?days?[- ]?old"), 1 / 365),
	(re.compile(r"\b(\d{1,3}) ?(?:m|f|male|female|man|woman)\b"), 1.0),
]


SEX_NOUN = re.compile(r"\b(man|woman|male|female|boy|girl|gentleman|lady|"
		      r"m|f)\b")


def find_age(note):
	low = note.lower()
	hits = []
	for rule, scale in AGE_RULES:
		m = rule.search(low)
		if m and int(m.group(1)) < 120:
			age = round(int(m.group(1)) * scale, 2)
			hits.append((m.start(), age))
	return min(hits)[1] if hits else None


def find_sex(note, words):
	m = SEX_NOUN.search(" ".join(words[:15]))
	if m:
		noun = m.group(1)
		return "female" if noun in FEMALE else "male"
	f = sum(w in FEMALE for w in words)
	m = sum(w in MALE for w in words)
	if f == m:
		return None
	return "female" if f > m else "male"


def split_slash(words):
	out = []
	for w in words:
		if "/" in w and w not in ABBREV:
			out.extend(p for p in w.split("/") if p)
		else:
			out.append(w)
	return out


def expand(words, on=True):
	out, used = [], []
	for w in split_slash(words):
		if on and w in ABBREV:
			used.append((w, ABBREV[w]))
			out.extend(ABBREV[w].split())
		else:
			out.append(w)
	return out, used


def trigger_at(words, i):
	for trig in TRIGGERS:
		if tuple(words[i:i + len(trig)]) == trig:
			return len(trig)
	return 0


def clauses(words):
	cur = []
	for w in words:
		if w in TERMINATORS:
			if cur:
				yield cur
			cur = []
		else:
			cur.append(w)
	if cur:
		yield cur


def polarity(clause, negation=True, family=True):
	if family and FAMILY & set(clause):
		return ["family"] * len(clause)
	tags = ["pos"] * len(clause)
	if not negation:
		return tags
	i = 0
	while i < len(clause):
		n = trigger_at(clause, i)
		if n:
			for j in range(i + n, min(len(clause), i + n + SCOPE)):
				tags[j] = "neg"
			for j in range(i, i + n):
				tags[j] = "cue"
			i += n
			continue
		if clause[i] in POST_TRIGGERS:
			for j in range(max(0, i - 3), i):
				tags[j] = "neg"
			tags[i] = "cue"
		i += 1
	return tags


def profile(note, abbrev=True, negation=True, family=True, do_stem=True):
	clean = DEID.sub(" ", note)
	words = LEX.findall(text.fold(clean))
	sex = find_sex(clean, words)
	words, used = expand(words, abbrev)
	tagged = []
	for clause in clauses(words):
		tags = polarity(clause, negation, family)
		tagged.extend(zip(clause, tags))
	groups = {"pos": Counter(), "neg": Counter(), "family": Counter()}
	surface = {}
	for w, tag in tagged:
		if tag == "cue":
			continue
		for term, _ in text.analyze(w.replace("/", " "), do_stem):
			groups[tag][term] += 1
			surface.setdefault(term, w)
	for term in list(groups["neg"]):
		if groups["pos"][term] > groups["neg"][term]:
			del groups["neg"][term]
	return {
		"age": find_age(clean),
		"sex": sex,
		"pos": dict(groups["pos"]),
		"neg": dict(groups["neg"]),
		"family": dict(groups["family"]),
		"abbrev": used,
		"tagged": tagged,
		"surface": surface,
	}
