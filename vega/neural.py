# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import numpy as np

MODEL = "ncbi/MedCPT-Cross-Encoder"
MAX_LEN = 384
BATCH = 50
VIEWS = ("topic", "inclusion", "exclusion")


def views(trial):
	conds = "; ".join(trial["conditions"][:8])
	topic = f"{trial['title']}. Conditions: {conds}. {trial['summary']}"
	return (topic, trial["inclusion"] or trial["title"],
		trial["exclusion"] or "none")


class CrossEncoder:
	def __init__(self):
		import torch
		from transformers import AutoModelForSequenceClassification
		from transformers import AutoTokenizer
		self.torch = torch
		if torch.backends.mps.is_available():
			self.device = "mps"
		elif torch.cuda.is_available():
			self.device = "cuda"
		else:
			self.device = "cpu"
		self.tok = AutoTokenizer.from_pretrained(MODEL)
		auto = AutoModelForSequenceClassification
		self.model = auto.from_pretrained(MODEL).eval().to(self.device)

	def batch(self, query, texts):
		pairs = [[query, t] for t in texts]
		enc = self.tok(pairs, truncation=True, padding=True,
			       return_tensors="pt", max_length=MAX_LEN)
		logits = self.model(**enc.to(self.device)).logits
		return logits.squeeze(1).float().cpu().numpy()

	def score(self, query, texts):
		out = []
		with self.torch.no_grad():
			for i in range(0, len(texts), BATCH):
				chunk = texts[i:i + BATCH]
				out.append(self.batch(query, chunk))
		return np.concatenate(out) if out else np.zeros(0)

	def trial_scores(self, note, trials):
		rows = [views(t) for t in trials]
		cols = [self.score(note, [r[k] for r in rows])
			for k in range(len(VIEWS))]
		return np.stack(cols, axis=1)


def minmax(v):
	lo, hi = float(v.min()), float(v.max())
	return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def fuse(first, ce, ok, cfg):
	a = cfg["alpha"]
	return ((1 - a) * minmax(first) + a * minmax(ce[:, 0]) +
		cfg["inc_view"] * minmax(ce[:, 1]) -
		cfg["exc_view"] * minmax(ce[:, 2]) +
		cfg["elig_w"] * ok.astype(np.float32))
