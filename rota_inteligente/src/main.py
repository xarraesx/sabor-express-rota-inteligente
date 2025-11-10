import json, math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Utils ----------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

def build_graph(df):
    nodes = df["id"].tolist()
    coords = {row["id"]: (row["lat"], row["lon"]) for _, row in df.iterrows()}
    adj = {n: {} for n in nodes}
    for i, n1 in enumerate(nodes):
        lat1, lon1 = coords[n1]
        for j, n2 in enumerate(nodes):
            if i == j: 
                continue
            lat2, lon2 = coords[n2]
            w = haversine_km(lat1, lon1, lat2, lon2)
            adj[n1][n2] = w
    return adj, coords

import heapq
def astar(adj, coords, start, goal):
    def h(n):
        lat1, lon1 = coords[n]
        lat2, lon2 = coords[goal]
        return haversine_km(lat1, lon1, lat2, lon2)
    open_set = [(h(start), 0.0, start, None)]
    came = {}
    g = {start: 0.0}
    visited = set()
    while open_set:
        f, gscore, node, parent = heapq.heappop(open_set)
        if node in visited: 
            continue
        came[node] = parent
        visited.add(node)
        if node == goal:
            path = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path, gscore
        for nb, w in adj[node].items():
            tentative = gscore + w
            if nb in visited:
                continue
            if nb not in g or tentative < g[nb]:
                g[nb] = tentative
                heapq.heappush(open_set, (tentative + h(nb), tentative, nb, node))
    return [start], 0.0

def kmeans(points_xy, k=3, max_iter=100, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points_xy), size=k, replace=False)
    centroids = points_xy[idx].copy()
    for _ in range(max_iter):
        dists = np.linalg.norm(points_xy[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([points_xy[labels==i].mean(axis=0) for i in range(k)])
        if np.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids
    return labels, centroids

def greedy_order(origin_id, ids, adj):
    order = [origin_id]
    remaining = set(ids)
    cur = origin_id
    while remaining:
        nxt = min(remaining, key=lambda n: adj[cur][n])
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order

def run():
    base_dir = os.path.dirname(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
    data_path = os.path.join(base_dir, "data", "points.csv")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    adj, coords = build_graph(df)

    # Cluster only delivery points
    pts_array = np.array([[row["lat"], row["lon"]] for _, row in df.iloc[1:].iterrows()])
    labels, _ = kmeans(pts_array, k=3, seed=7)
    cluster_map = {df.iloc[i+1]["id"]: int(labels[i]) for i in range(len(labels))}

    routes = {}
    metrics = {"clusters": {}, "total_km": 0.0}
    for c in sorted(set(labels)):
        ids = [pid for pid, lab in cluster_map.items() if lab == c]
        order = greedy_order("R0_Pizzaria", ids, adj)
        km = 0.0
        segments = []
        for a, b in zip(order, order[1:]):
            path, dist = astar(adj, coords, a, b)
            km += dist
            segments.append({"from": a, "to": b, "path": path, "km": dist})
        routes[c] = {"order": order, "segments": segments, "km": km}
        metrics["clusters"][str(c)] = {"stops": ids, "km": km}
        metrics["total_km"] += km

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Plot
    plt.figure(figsize=(7,7))
    for _, row in df.iterrows():
        plt.scatter(row["lon"], row["lat"])
        plt.text(row["lon"]+0.0005, row["lat"]+0.0003, row["id"])
    for c, info in routes.items():
        xs, ys = [], []
        for node in info["order"]:
            lat, lon = coords[node]
            xs.append(lon); ys.append(lat)
        plt.plot(xs, ys, marker='o', label=f"Cluster {c}")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Rotas por Cluster (A* + Greedy) — Santana/SP")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "rotas_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("OK - métricas salvas em outputs/metrics.json e gráfico em outputs/rotas_clusters.png")

if __name__ == "__main__":
    run()
