# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from multiprocessing import Pool

DATA = os.path.join("data", "trec")
TRIALS = os.path.join(DATA, "trials.jsonl")
OFFSETS = os.path.join(DATA, "trials.offsets.json")
EXCLUSION = re.compile(r"exclusion\s+criteria\s*:?", re.I)
INCLUSION = re.compile(r"^\s*inclusion\s+criteria\s*:?", re.I)
BULLET = re.compile(r"\s+-\s+|\s+\*\s+|\s+\d{1,2}[.)]\s+")
NEGATED_ITEM = re.compile(r"\s*(?:no|not|must not|without|absence of|"
			  r"none of|patients? (?:with|who have) no)\b\s*",
			  re.I)
UNITS = {"year": 1.0, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365,
	 "hour": 1 / 8760, "minute": 1 / 525600}
NO_MAX = 999.0


def text_of(root, path):
	node = root.find(path)
	return " ".join(node.text.split()) if node is not None and \
		node.text else ""


def all_of(root, path):
	return [" ".join(n.text.split()) for n in root.findall(path)
		if n.text and n.text.strip()]


def age_years(s, default):
	m = re.match(r"\s*([\d.]+)\s*([a-z]+)", (s or "").lower())
	if not m:
		return default
	unit = m.group(2).rstrip("s")
	return round(float(m.group(1)) * UNITS.get(unit, 1.0), 3)


def split_criteria(criteria):
	m = EXCLUSION.search(criteria)
	inc = criteria[:m.start()] if m else criteria
	exc = criteria[m.end():] if m else ""
	keep, moved = [], []
	for item in BULLET.split(INCLUSION.sub("", inc)):
		cue = NEGATED_ITEM.match(item)
		if cue:
			moved.append(item[cue.end():])
		else:
			keep.append(item)
	exc = " ".join([exc.strip()] + moved).strip()
	return " ".join(keep).strip(), exc


def parse(raw):
	root = ET.fromstring(raw)
	criteria = text_of(root, "eligibility/criteria/textblock")
	inc, exc = split_criteria(criteria)
	gender = text_of(root, "eligibility/gender") or "All"
	return {
		"id": text_of(root, "id_info/nct_id"),
		"title": text_of(root, "brief_title"),
		"official": text_of(root, "official_title"),
		"summary": text_of(root, "brief_summary/textblock"),
		"description": text_of(root, "detailed_description/textblock"),
		"conditions": all_of(root, "condition") +
			all_of(root, "keyword") +
			all_of(root, "condition_browse/mesh_term"),
		"interventions":
			all_of(root, "intervention/intervention_name") +
			all_of(root, "intervention_browse/mesh_term"),
		"inclusion": inc,
		"exclusion": exc,
		"gender": gender if gender in ("Male", "Female") else "All",
		"min_age": age_years(text_of(root, "eligibility/minimum_age"),
				     0.0),
		"max_age": age_years(text_of(root, "eligibility/maximum_age"),
				     NO_MAX),
		"status": text_of(root, "overall_status"),
		"phase": text_of(root, "phase"),
		"type": text_of(root, "study_type"),
	}


def convert(path):
	out = path + ".jsonl"
	n = 0
	with zipfile.ZipFile(path) as z, open(out + ".part", "w") as f:
		for name in z.namelist():
			if not name.endswith(".xml"):
				continue
			try:
				rec = parse(z.read(name))
			except ET.ParseError:
				continue
			if rec["id"]:
				f.write(json.dumps(rec) + "\n")
				n += 1
	os.replace(out + ".part", out)
	return path, n


def merge(parts):
	seen, offsets = set(), []
	with open(TRIALS, "wb") as out:
		for part in parts:
			with open(part, "rb") as f:
				for line in f:
					rid = json.loads(line)["id"]
					if rid in seen:
						continue
					seen.add(rid)
					offsets.append(out.tell())
					out.write(line)
			os.remove(part)
	with open(OFFSETS, "w") as f:
		json.dump(offsets, f)
	return len(offsets)


def load(path=TRIALS):
	with open(path) as f:
		return [json.loads(line) for line in f]


class Store:
	def __init__(self):
		with open(OFFSETS) as f:
			self.offsets = json.load(f)
		self.f = open(TRIALS, "rb")

	def get(self, i):
		self.f.seek(self.offsets[i])
		return json.loads(self.f.readline())


def criteria_counts(path):
	counts = {"trials": 0, "criteria": 0, "header": 0, "moved": 0,
		  "exclusion": 0}
	with zipfile.ZipFile(path) as z:
		for name in z.namelist():
			if not name.endswith(".xml"):
				continue
			try:
				root = ET.fromstring(z.read(name))
			except ET.ParseError:
				continue
			crit = text_of(root, "eligibility/criteria/textblock")
			inc, exc = split_criteria(crit)
			header = EXCLUSION.search(crit)
			counts["trials"] += 1
			counts["criteria"] += bool(crit)
			counts["header"] += bool(header)
			counts["moved"] += bool(exc) and not header
			counts["exclusion"] += bool(exc)
	return counts


def criteria_stats():
	zips = sorted(glob.glob(os.path.join(DATA, "ClinicalTrials.*.zip")))
	total = {}
	with Pool(len(zips)) as pool:
		for c in pool.map(criteria_counts, zips):
			for k, v in c.items():
				total[k] = total.get(k, 0) + v
	with open(os.path.join("results", "criteria.json"), "w") as f:
		json.dump(total, f, indent=1)
	print(total)


def main():
	if sys.argv[1:] == ["stats"]:
		return criteria_stats()
	zips = sorted(glob.glob(os.path.join(DATA, "ClinicalTrials.*.zip")))
	if len(zips) != 5:
		sys.exit("expected 5 corpus zips; run python -m vega.fetch")
	with Pool(len(zips)) as pool:
		for path, n in pool.imap_unordered(convert, zips):
			print(f"{os.path.basename(path)}: {n} trials")
	print(f"merged {merge([z + '.jsonl' for z in zips])} unique trials")


if __name__ == "__main__":
	main()
