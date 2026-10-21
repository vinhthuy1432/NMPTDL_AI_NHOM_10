"""Giao diện Streamlit - Nhóm 10 - Stage 2 (dự đoán rating_2026 phim IMDb Top 250).

Chạy:  streamlit run app_stage2.py
Cần cùng thư mục: stage2_models.joblib (tạo ở mục 11 của notebook).
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="IMDb Top 250 - Dự đoán rating_2026", layout="wide")


@st.cache_resource
def load_artifact(path="stage2_models.joblib"):
    return joblib.load(path)


# Khoảng giá trị hợp lý cho từng thuộc tính: (min, max, bước, có thể bỏ trống)
FEATURE_SPEC = {
    "year":         (1920, 2026, 1,        False, "Năm phát hành"),
    "rank_v1":      (1, 250, 1,            True,  "Thứ hạng ở bảng cũ"),
    "rating_v1":    (8.0, 9.5, 0.1,        True,  "Điểm IMDb ở bảng cũ"),
    "votes_v1":     (0, 3_000_000, 10_000, True,  "Lượt vote ở bảng cũ"),
    "runtime_v1":   (40, 400, 1,           True,  "Thời lượng ở bảng cũ (phút)"),
    "votes_2026":   (0, 3_500_000, 10_000, False, "Lượt vote năm 2026"),
    "runtime_2026": (40, 400, 1,           False, "Thời lượng năm 2026 (phút)"),
    "metascore":    (0, 100, 1,            True,  "Metascore (giới phê bình)"),
    "budget":       (0, 400_000_000, 1_000_000, True, "Ngân sách (USD)"),
    "box_office":   (0, 3_000_000_000, 10_000_000, True, "Doanh thu (USD)"),
}


def build_inputs(art):
    """Vẽ widget nhập liệu; trả về DataFrame 1 dòng. Ô 'không biết' -> NaN (pipeline tự điền median)."""
    defaults = art["feature_defaults"]
    values = {}
    cols = st.columns(2)
    for i, name in enumerate(art["feature_cols"]):
        lo, hi, step, optional, label = FEATURE_SPEC.get(
            name, (0.0, float(max(1, defaults.get(name, 1)) * 5), 1.0, True, name))
        default = float(defaults.get(name, lo))
        default = min(max(default, lo), hi)
        is_float = isinstance(step, float)
        with cols[i % 2]:
            unknown = st.checkbox(f"Không biết: {label}", key=f"na_{name}") if optional else False
            if unknown:
                values[name] = np.nan
            else:
                v = st.slider(label, float(lo), float(hi), default, float(step), key=name) if is_float \
                    else st.slider(label, int(lo), int(hi), int(default), int(step), key=name)
                values[name] = float(v)
    return pd.DataFrame([values], columns=art["feature_cols"])


def predict_all(art, X):
    reg = float(art["regression_model"].predict(X)[0])
    tier = str(art["classification_model"].predict(X)[0])
    Z = art["cluster_scaler"].transform(art["cluster_imputer"].transform(X))
    cluster = int(art["kmeans_pca"].predict(art["pca"].transform(Z))[0])
    return reg, tier, cluster


def tab_predict(art):
    st.subheader("Nhập thuộc tính của phim")
    X = build_inputs(art)
    if st.button("Dự đoán", type="primary"):
        reg, tier, cluster = predict_all(art, X)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Rating dự đoán ({art['regression_model_name']})", f"{reg:.2f}")
        c2.metric(f"Nhóm rating ({art['classification_model_name']})", tier)
        c3.metric("Cụm K-Means", f"Cụm {cluster}")
        if X["rating_v1"].isna().any() if "rating_v1" in X else False:
            st.warning("Bạn chưa nhập rating_v1: đây là biến ảnh hưởng mạnh nhất, "
                       "nên kết quả chỉ mang tính tham khảo.")
        st.caption("Dự đoán chỉ mang tính tham khảo, không phải đánh giá chắc chắn.")


def tab_viz(art):
    viz = art.get("viz")
    pca = art["pca"]
    st.subheader("PCA - phương sai tích lũy")
    if viz and "cum_var" in viz:
        cum = np.asarray(viz["cum_var"])
    else:
        cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(cum) + 1), cum, marker="o")
    for t in (0.85, 0.90, 0.95):
        ax.axhline(t, linestyle="--", alpha=.5)
    ax.axvline(pca.n_components_, color="gray", linestyle=":")
    ax.set_xlabel("Số thành phần chính"); ax.set_ylabel("Phương sai tích lũy")
    st.pyplot(fig)

    st.subheader("K-Means trên không gian PCA")
    if viz and "X_pca_2d" in viz:
        xy = np.asarray(viz["X_pca_2d"])
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sc = ax2.scatter(xy[:, 0], xy[:, 1], c=viz["labels"], cmap="viridis", alpha=.75)
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
        fig2.colorbar(sc, label="Cluster")
        st.pyplot(fig2)
    else:
        st.info("Chưa có dữ liệu vẽ cụm. Chạy cell 'viz' trong stage2_extra_cells.py "
                "trước khi lưu joblib để hiển thị biểu đồ này.")


def main():
    st.title("Dự đoán rating_2026 phim IMDb Top 250")
    try:
        art = load_artifact()
    except FileNotFoundError:
        st.error("Không tìm thấy stage2_models.joblib. Chạy mục 11 của notebook rồi đặt file cùng thư mục.")
        return
    t1, t2 = st.tabs(["Dự đoán", "Biểu đồ PCA / K-Means"])
    with t1:
        tab_predict(art)
    with t2:
        tab_viz(art)
    st.sidebar.markdown(f"**Quy tắc nhóm rating:** {art.get('rating_tier_rule', '')}")


if __name__ == "__main__":
    main()
