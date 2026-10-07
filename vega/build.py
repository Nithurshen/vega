# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import os
import time

from vega import index
from vega import trials


def step(name, fn, *args):
	t = time.perf_counter()
	out = fn(*args)
	print(f"{name}: {time.perf_counter() - t:.1f}s", flush=True)
	return out


def main():
	if not os.path.exists(trials.TRIALS):
		step("parse trials", trials.main)
	ix = step("inverted index", index.build)
	print(f"  {ix.n} trials, {len(ix.vocab)} terms, "
	      f"{len(ix.post_docs)} postings, {len(ix.pos)} positions",
	      flush=True)
	step("save", ix.save, os.path.join(trials.DATA, "index.pkl"))
	print("build complete")


if __name__ == "__main__":
	main()
