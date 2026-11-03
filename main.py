"""
Optimasi Rute Pengantaran Dokumen Antar Gedung di UNTAD menggunakan TSP
-----------------------------------------------------------------------
Metode  : Brute Force (eksak) + pembanding Nearest Neighbor (heuristik)
Data    : jarak antar fakultas (meter), dari pengukuran Google Earth
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. DATA JARAK (meter)
# ------------------------------------------------------------------
nodes = ["Teknik", "Peternakan", "FISIP", "Kedokteran", "FAHUM", "Kehutanan"]

jarak = {
    ("Teknik", "Peternakan"): 908.16,
    ("Teknik", "FISIP"): 1134.53,
    ("Teknik", "Kedokteran"): 449.69,
    ("Teknik", "FAHUM"): 1311.51,
    ("Teknik", "Kehutanan"): 1264.76,
    ("Peternakan", "FISIP"): 1195.19,
    ("Peternakan", "Kedokteran"): 372.33,
    ("Peternakan", "FAHUM"): 1055.24,
    ("Peternakan", "Kehutanan"): 393.61,
    ("FISIP", "Kedokteran"): 1217.56,
    ("FISIP", "FAHUM"): 630.08,
    ("FISIP", "Kehutanan"): 875.39,
    ("Kedokteran", "FAHUM"): 1097.36,
    ("Kedokteran", "Kehutanan"): 729.96,
    ("FAHUM", "Kehutanan"): 743.91,
}

n = len(nodes)
idx = {name: i for i, name in enumerate(nodes)}
D = np.zeros((n, n))
for (a, b), d in jarak.items():
    D[idx[a], idx[b]] = D[idx[b], idx[a]] = d

# ------------------------------------------------------------------
# 2. FUNGSI TSP
# ------------------------------------------------------------------
def panjang_rute(rute):
    """Total jarak rute tertutup (kembali ke titik awal)."""
    return sum(D[rute[i], rute[(i + 1) % len(rute)]] for i in range(len(rute)))


def tsp_brute_force(start=0):
    """Coba semua permutasi (n-1)! -> solusi optimal pasti."""
    terbaik, jarak_min = None, float("inf")
    for perm in itertools.permutations([i for i in range(n) if i != start]):
        rute = (start,) + perm
        d = panjang_rute(rute)
        if d < jarak_min:
            terbaik, jarak_min = rute, d
    return list(terbaik), jarak_min


def tsp_nearest_neighbor(start=0):
    """Heuristik: selalu ke gedung terdekat yang belum dikunjungi."""
    rute, sisa = [start], set(range(n)) - {start}
    while sisa:
        berikut = min(sisa, key=lambda j: D[rute[-1], j])
        rute.append(berikut)
        sisa.remove(berikut)
    return rute, panjang_rute(rute)


def cetak(judul, rute, total):
    nama = " -> ".join(nodes[i] for i in rute) + " -> " + nodes[rute[0]]
    print(f"{judul}\n  Rute  : {nama}\n  Total : {total:,.2f} m ({total/1000:.2f} km)\n")


# ------------------------------------------------------------------
# 3. JALANKAN
# ------------------------------------------------------------------
START = idx["Teknik"]  # titik awal/akhir, ganti sesuai kebutuhan

rute_opt, d_opt = tsp_brute_force(START)
rute_nn, d_nn = tsp_nearest_neighbor(START)

cetak("RUTE OPTIMAL (Brute Force)", rute_opt, d_opt)
cetak("RUTE Nearest Neighbor", rute_nn, d_nn)
print(f"Selisih NN vs Optimal: {d_nn - d_opt:,.2f} m")

# ------------------------------------------------------------------
# 4. PETA SKEMATIK
# Koordinat tiap gedung diperkirakan dari matriks jarak (Classical MDS),
# jadi posisi relatifnya mengikuti jarak sebenarnya.
# ------------------------------------------------------------------
def classical_mds(D, dim=2):
    m = D.shape[0]
    J = np.eye(m) - np.ones((m, m)) / m
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:dim]
    return vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))

XY = classical_mds(D)

# Orientasikan peta supaya sesuai posisi di foto satelit:
# Teknik di kiri, Kedokteran di bawah, FISIP di atas, Kehutanan di kanan.
jangkar = {"Teknik": (0.0, 0.3), "Kedokteran": (0.55, 0.0),
           "FISIP": (0.45, 1.0), "Kehutanan": (1.0, 0.6)}
ids = [idx[k] for k in jangkar]
A = XY[ids]; B = np.array(list(jangkar.values()))
muA, muB = A.mean(0), B.mean(0)
U, _, Vt = np.linalg.svd((A - muA).T @ (B - muB))   # rotasi/refleksi terbaik
XY = (XY - muA) @ (U @ Vt) * 1.0 + muB

fig, ax = plt.subplots(figsize=(10, 8))

# semua jalur (abu-abu tipis) + label jarak
for i in range(n):
    for j in range(i + 1, n):
        ax.plot(*zip(XY[i], XY[j]), color="lightgray", lw=0.8, zorder=1)

# rute optimal
urut = rute_opt + [rute_opt[0]]
for k in range(len(urut) - 1):
    a, b = urut[k], urut[k + 1]
    ax.annotate("", xy=XY[b], xytext=XY[a],
                arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.5,
                                shrinkA=14, shrinkB=14), zorder=3)
    mid = (XY[a] + XY[b]) / 2
    ax.text(*mid, f"{D[a, b]:,.2f} m", fontsize=9, color="crimson",
            ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="crimson", lw=0.8))

# titik gedung
for i, nama in enumerate(nodes):
    warna = "gold" if i == START else "steelblue"
    ax.scatter(*XY[i], s=700, color=warna, edgecolor="black", zorder=4)
    ax.text(XY[i][0], XY[i][1], str(urut.index(i) + 1), ha="center",
            va="center", fontweight="bold", color="black", zorder=6)
    ax.text(XY[i][0], XY[i][1] - 0.06 * np.ptp(XY[:, 1]), nama, ha="center", fontsize=11,
            fontweight="bold", zorder=6)

ax.set_title("Rute Optimal Pengantaran Dokumen Antar Fakultas UNTAD (TSP)\n"
             f"Total jarak: {d_opt:,.2f} m  |  Titik awal: {nodes[START]}",
             fontsize=13)
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/peta_rute_tsp_untad.png", dpi=200)
print("Peta disimpan: peta_rute_tsp_untad.png")