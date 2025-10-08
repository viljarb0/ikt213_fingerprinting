import os
import matplotlib.pyplot as plt
import cv2

from utils import list_images
from pipeline import match_orb_score, match_sift_score


def save_match_viz(query_path, db_best_path, method, out_folder):

    os.makedirs(out_folder, exist_ok=True)

    img_q = cv2.imread(query_path, 0)
    img_d = cv2.imread(db_best_path, 0)

    if method == "orb":
        det, norm = cv2.ORB_create(1000), cv2.NORM_HAMMING
    else:
        det, norm = cv2.SIFT_create(), cv2.NORM_L2

    kp1, des1 = det.detectAndCompute(img_q, None)
    kp2, des2 = det.detectAndCompute(img_d, None)
    if des1 is None or des2 is None:
        return

    bf = cv2.BFMatcher(norm)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    good = sorted(good, key=lambda m: m.distance)[:60]

    viz = cv2.drawMatches(img_q, kp1, img_d, kp2, good, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    out_name = f"{method}__{os.path.basename(query_path)}__{os.path.basename(db_best_path)}.png"
    cv2.imwrite(os.path.join(out_folder, out_name), viz)


def best_matches(db_dir, query_dir, method="orb"):
    db_images = list_images(db_dir)
    q_images  = list_images(query_dir)
    results = {}

    for q in q_images:
        qname = os.path.basename(q)

        best_db = None
        best_score = -1
        total_time_ms = 0.0

        for d in db_images:
            if method == "orb":
                score, t_ms = match_orb_score(q, d)
            else:
                score, t_ms = match_sift_score(q, d)
            total_time_ms += t_ms
            if score > best_score:
                best_score = score
                best_db = os.path.basename(d)

        results[qname] = (best_db, best_score, total_time_ms)
        print(f"[{method.upper()}] {qname}  ->  {best_db}   (score={best_score}, time={total_time_ms:.1f} ms)")

        dataset_label = "fingerprints" if "fingerprints" in db_dir.lower() else "uia"
        out_dir = os.path.join("output", dataset_label, method)
        save_match_viz(q, os.path.join(db_dir, best_db), method, out_dir)

    return results


def plot_query_scores(results, title):
    queries = list(results.keys())
    scores  = [results[q][1] for q in queries]
    plt.figure(figsize=(10,5))
    plt.bar(queries, scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("good matches higherscore = better")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_comparison_scores(orb_results, sift_results, title):
    queries = list(orb_results.keys())

    orb_scores = [orb_results[q][1] for q in queries]
    sift_scores = [sift_results[q][1] for q in queries]

    x = range(len(queries))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], orb_scores, width, label="ORB", color="orange")
    plt.bar([i + width/2 for i in x], sift_scores, width, label="SIFT", color="blue")

    plt.xticks(x, queries, rotation=45, ha="right")
    plt.ylabel("Good Matches (Higher = Better)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_comparison_times(orb_results, sift_results, title):
    queries = list(orb_results.keys())

    orb_times = [orb_results[q][2] for q in queries]
    sift_times = [sift_results[q][2] for q in queries]

    x = range(len(queries))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], orb_times, width, label="ORB", color="orange")
    plt.bar([i + width/2 for i in x], sift_times, width, label="SIFT", color="blue")

    plt.xticks(x, queries, rotation=45, ha="right")
    plt.ylabel("Total Processing Time (ms)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ROOT = os.path.join(os.path.dirname(__file__), "data")

    fp_db = os.path.join(ROOT, "fingerprints", "db")
    fp_q  = os.path.join(ROOT, "fingerprints", "query")

    print("\nFINGERPRINTS ORB")
    fp_orb = best_matches(fp_db, fp_q, method="orb")
    plot_query_scores(fp_orb, "Fingerprint ORB scores for each query")

    print("\nFINGERPRINTS SIFT")
    fp_sift = best_matches(fp_db, fp_q, method="sift")
    plot_query_scores(fp_sift, "Fingerprints SIFT scores for each query")

    uia_db = os.path.join(ROOT, "uia", "db")
    uia_q  = os.path.join(ROOT, "uia", "query")

    print("\nUIA ORB")
    uia_orb = best_matches(uia_db, uia_q, method="orb")
    plot_query_scores(uia_orb, "UiA ORB scores for each query")

    print("\nUIA SIFT")
    uia_sift = best_matches(uia_db, uia_q, method="sift")
    plot_query_scores(uia_sift, "UiA SIFT scores for each query")
    
    plot_comparison_scores(fp_orb, fp_sift, "Fingerprint ORB vs SIFT Comparison")
    plot_comparison_scores(uia_orb, uia_sift, "UiA ORB vs SIFT Comparison")

    plot_comparison_times(fp_orb, fp_sift, "Fingerprint ORB vs SIFT Processing Time")
    plot_comparison_times(uia_orb, uia_sift, "UiA ORB vs SIFT Processing Time")