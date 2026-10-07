# K S Nithurshen, roll no: 2410110157, email: ks622@snu.edu.in

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
MUTED = "#c3c2bc"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def style():
	plt.rcParams.update({
		"figure.facecolor": SURFACE,
		"axes.facecolor": SURFACE,
		"savefig.facecolor": SURFACE,
		"font.size": 9,
		"axes.edgecolor": GRID,
		"axes.linewidth": 0.8,
		"axes.labelcolor": INK2,
		"axes.titlecolor": INK,
		"axes.titlesize": 10,
		"axes.titleweight": "bold",
		"axes.titlelocation": "left",
		"xtick.color": INK2,
		"ytick.color": INK2,
		"xtick.major.size": 0,
		"ytick.major.size": 0,
		"axes.grid": True,
		"grid.color": GRID,
		"grid.linewidth": 0.6,
		"axes.spines.top": False,
		"axes.spines.right": False,
		"legend.frameon": False,
		"legend.fontsize": 8,
		"lines.linewidth": 2,
	})


def save(fig, path):
	fig.tight_layout()
	fig.savefig(path, dpi=220)
	plt.close(fig)


def emphasis_bars(names, values, focus, title, xlabel, path):
	style()
	fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(names) + 0.9))
	y = np.arange(len(names))[::-1]
	colors = [SERIES[0] if n == focus else MUTED for n in names]
	ax.barh(y, values, height=0.62, color=colors, edgecolor=SURFACE,
		linewidth=2)
	ax.set_yticks(y, names)
	ax.grid(axis="y", visible=False)
	for yi, v, n in zip(y, values, names):
		ax.text(v + max(values) * 0.01, yi, f"{v:.3f}", va="center",
			fontsize=8, color=INK if n == focus else INK2)
	ax.set_xlim(0, max(values) * 1.13)
	ax.set_xlabel(xlabel)
	ax.set_title(title)
	save(fig, path)


def merge_points(points):
	out = []
	for label, x, y in points:
		if out and (out[-1][1], out[-1][2]) == (x, y):
			out[-1] = (out[-1][0] + " = " + label, x, y)
		else:
			out.append((label, x, y))
	return out


def tradeoff(points, champ, path):
	points = merge_points(points)
	style()
	fig, ax = plt.subplots(figsize=(3.2, 2.6))
	xs = [p[1] for p in points]
	ys = [p[2] for p in points]
	ax.plot(xs, ys, color=SERIES[0], marker="o", markersize=5)
	for i, (label, x, y) in enumerate(points):
		last = i == len(points) - 1
		ax.annotate(label, (x, y), textcoords="offset points",
			    xytext=(-6, 6) if last else (4, 6),
			    ha="right" if last else "left", fontsize=7,
			    color=INK2)
	ax.annotate("index elimination", (xs[-1], ys[-1]),
		    textcoords="offset points", xytext=(0, -16), ha="right",
		    fontsize=7.5, color=SERIES[0])
	ax.plot([champ[1]], [champ[2]], color=SERIES[1], marker="s",
		markersize=6, linestyle="none")
	ax.annotate("champion lists (r=500)", (champ[1], champ[2]),
		    textcoords="offset points", xytext=(7, -3), fontsize=7.5,
		    color=INK2)
	pad = (max(ys) - min(ys + [champ[2]])) * 0.12
	ax.set_ylim(min(ys + [champ[2]]) - pad, max(ys) + pad * 1.5)
	ax.set_xlim(0, max(xs) * 1.12)
	ax.set_xlabel("ms per query (lexical stage)")
	ax.set_ylabel("nDCG@10")
	ax.set_title("Effectiveness vs cost")
	save(fig, path)


def sweep(series, path):
	style()
	fig, ax = plt.subplots(figsize=(3.2, 3.0))
	for i, (name, xs, ys) in enumerate(series):
		ax.plot(xs, ys, color=SERIES[i], marker="o", markersize=4,
			label=name)
	ax.set_xlabel("penalty weight")
	ax.set_ylabel("nDCG@10 (2021 topics)")
	ax.set_title("Eligibility penalties")
	ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22))
	save(fig, path)


def paired_bars(names, eligible, excluded, path):
	style()
	fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 0.9))
	y = np.arange(len(names))[::-1]
	h = 0.36
	ax.barh(y + h / 2, eligible, height=h, color=SERIES[0],
		edgecolor=SURFACE, linewidth=2,
		label="eligible trials in top 10 (P@10)")
	ax.barh(y - h / 2, excluded, height=h, color=SERIES[1],
		edgecolor=SURFACE, linewidth=2,
		label="trials the patient is excluded from (Excl@10)")
	ax.set_yticks(y, names)
	ax.grid(axis="y", visible=False)
	top = max(max(eligible), max(excluded))
	for yi, a, b in zip(y, eligible, excluded):
		ax.text(a + top * 0.01, yi + h / 2, f"{a:.3f}", va="center",
			fontsize=7.5, color=INK2)
		ax.text(b + top * 0.01, yi - h / 2, f"{b:.3f}", va="center",
			fontsize=7.5, color=INK2)
	ax.set_xlim(0, top * 1.15)
	ax.set_xlabel("fraction of the top 10 (2022 test patients)")
	ax.set_title("More eligible trials, fewer excluded ones")
	ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=1)
	save(fig, path)


def scatter(a, b, xlabel, ylabel, path):
	style()
	fig, ax = plt.subplots(figsize=(3.2, 2.9))
	ax.scatter(a, b, s=12, color=SERIES[0], alpha=0.55, linewidths=0)
	ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8)
	top = max(max(a), max(b)) * 1.05
	ax.set_xlim(0, top)
	ax.set_ylim(0, top)
	wins = int(sum(y > x for x, y in zip(a, b)))
	loss = int(sum(y < x for x, y in zip(a, b)))
	ax.text(0.97, 0.04, f"above line: {wins} patients\n"
		f"below line: {loss} patients", transform=ax.transAxes,
		va="bottom", ha="right", fontsize=7.5, color=INK2)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title("Per-patient nDCG@10")
	save(fig, path)
