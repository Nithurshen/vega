# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import os
import sys
import time
import urllib.parse
import urllib.robotparser

import requests

AGENT = "vega-ir-coursework/1.0"
DELAY = 2.0
OUT = os.path.join("data", "trec")
CORPUS = ("https://www.trec-cds.org/2021_data/"
	  "ClinicalTrials.2021-04-27.part{}.zip")
NIST = "https://trec.nist.gov/data/trials/"
FILES = [NIST + f for f in ("topics2021.xml", "qrels2021.txt",
			    "topics2022.xml", "qrels2022.txt")]
FILES += [CORPUS.format(i) for i in range(1, 6)]


def allowed(session, url, cache):
	parts = urllib.parse.urlsplit(url)
	root = f"{parts.scheme}://{parts.netloc}"
	if root not in cache:
		r = session.get(root + "/robots.txt", timeout=30)
		rp = urllib.robotparser.RobotFileParser()
		rp.parse(r.text.splitlines() if r.status_code == 200 else [])
		cache[root] = rp
	return cache[root].can_fetch(AGENT, url)


def download(session, url):
	path = os.path.join(OUT, url.rsplit("/", 1)[-1])
	if os.path.exists(path):
		print(f"have {path}")
		return
	part = path + ".part"
	with session.get(url, stream=True, timeout=120) as r:
		r.raise_for_status()
		with open(part, "wb") as f:
			for chunk in r.iter_content(1 << 20):
				f.write(chunk)
	os.replace(part, path)
	print(f"got {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def main():
	os.makedirs(OUT, exist_ok=True)
	session = requests.Session()
	session.headers["User-Agent"] = AGENT
	cache = {}
	for url in FILES:
		if not allowed(session, url, cache):
			sys.exit(f"robots.txt disallows {url}")
		download(session, url)
		time.sleep(DELAY)
	print("fetch complete")


if __name__ == "__main__":
	main()
