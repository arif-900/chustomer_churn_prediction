import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_fscore_support

st.set_page_config(page_title="Churn Prediction", layout="wide")


DATA_PATH = os.path.join(os.path.dirname(__file__), "Customer_churn_prediction.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def show_overview(df):
    st.header("Dataset preview")
    st.dataframe(df.head(200))
    st.markdown(f"**Rows:** {df.shape[0]} — **Columns:** {df.shape[1]}")


def show_target_distribution(df):
    st.header("Target: churn")
    counts = df["churn"].value_counts().sort_index()
    pct = counts / counts.sum() * 100
    summary = pd.DataFrame({"count": counts, "percent": pct})
    st.write(summary)
    fig = px.pie(names=summary.index.astype(str), values=summary["count"], title="Churn (0=no, 1=yes)")
    st.plotly_chart(fig, width="stretch")


def show_country_gender(df):
    st.header("Categorical breakdowns")
    try:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df, names="country", title="Country distribution")
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.histogram(df, x="gender", color="gender", title="Gender count")
            st.plotly_chart(fig, width="stretch")

        st.subheader("Churn by category")
        c3, c4 = st.columns(2)
        with c3:
            ct = pd.crosstab(df["country"], df["churn"])
            fig = px.bar(ct, barmode="stack", title="Churn by Country")
            st.plotly_chart(fig, width="stretch")
        with c4:
            ct2 = pd.crosstab(df["gender"], df["churn"])
            fig = px.bar(ct2, barmode="stack", title="Churn by Gender")
            st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error(f"Error rendering categorical plots. Try refreshing or using Desktop view.")


def show_numeric_histograms(df):
    st.header("Numeric distributions")
    import pandas.api.types as ptypes
    numeric = [c for c in df.columns if ptypes.is_numeric_dtype(df[c]) and c not in ["customer_id"]]
    sel = st.multiselect("Pick numeric columns to plot", numeric, default=numeric[:2])
    # chart type: histogram or pie (binned)
    chart_type = st.radio("Chart type", ["Histogram", "Pie (binned)"], horizontal=True)
    # allow sampling to speed up plotting (default 2000 for mobile)
    sample_n = st.slider("Sample rows for numeric plots (0 = use all)", 0, 50000, 2000)
    if sample_n and sample_n < len(df):
        dff = df.sample(sample_n, random_state=42)
    else:
        dff = df

    for col in sel:
        if chart_type == "Histogram":
            try:
                fig = px.histogram(dff, x=col, nbins=50, title=col)
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.warning(f"Could not render histogram for {col}. Try: 1) Fewer samples, 2) Pie chart mode")
        else:
            # bin numeric values and show pie of bin counts
            ser = dff[col].dropna()
            if ser.empty:
                st.write(f"No data to plot for {col}")
                continue
            # limit bins to sample size
            max_bins = min(50, max(2, len(ser) // 10))
            bins = st.slider(f"Bins for {col}", 2, max_bins, min(8, max_bins), key=f"bins_{col}")
            try:
                binned = pd.cut(ser, bins=bins)
                counts = binned.value_counts().sort_index()
                labels = [f"{interval.left:.2f}–{interval.right:.2f}" for interval in counts.index]
                fig = px.pie(values=counts.values, names=labels, title=f"{col} (binned)")
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.warning(f"Could not bin {col} with {bins} bins. Try fewer bins or increase sample size.")


def show_correlation(df):
    st.header("Correlation matrix")
    if st.checkbox("Show correlation heatmap (may be slow on mobile)", value=False):
        corr = df.select_dtypes(include=["number"]).corr()
        if corr.empty:
            st.write("No numeric columns to compute correlation.")
            return
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
        st.plotly_chart(fig, width="stretch")
    else:
        st.write("Click the checkbox above to load the correlation matrix.")


def train_and_show(df, auto_run: bool = False):
    st.header("Train simple model")
    # prepare features
    df2 = df.copy()
    df2 = pd.get_dummies(df2, columns=["country", "gender"], drop_first=True)
    X = df2.drop(columns=["customer_id", "churn"], errors="ignore")
    y = df2["churn"]

    default_features = list(X.columns)
    features = st.multiselect("Features", default_features, default=default_features)

    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
    reg_strength = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)

    do_train = False
    if auto_run:
        do_train = True
    else:
        do_train = st.button("Train")

    if do_train:
        X_train, X_test, y_train, y_test = train_test_split(
            X[features], y, test_size=test_size, random_state=42
        )
        model = LogisticRegression(C=reg_strength, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("ROC AUC", f"{auc:.4f}")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        fig_roc = px.area(roc_df, x="fpr", y="tpr", title="ROC curve", labels={"fpr": "False Positive Rate", "tpr": "True Positive Rate"})
        fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
        st.plotly_chart(fig_roc, width="stretch")

        # Confusion matrix heatmap
        cm = confusion_matrix(y_test, preds)
        cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
        st.subheader("Confusion matrix")
        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", title="Confusion matrix")
        st.plotly_chart(fig_cm, width="stretch")

        # Precision / recall / fscore per class
        p, r, f, s = precision_recall_fscore_support(y_test, preds, labels=[0, 1])
        pr_df = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=["class_0", "class_1"])
        st.subheader("Per-class metrics")
        st.dataframe(pr_df)

        # Feature coefficients
        coef = pd.Series(model.coef_[0], index=features).sort_values(ascending=False)
        st.subheader("Feature coefficients")
        st.bar_chart(coef)


def main():
    st.title("Customer Churn — Interactive Dashboard")
    st.write("A simple Streamlit dashboard for EDA and quick modeling on the churn dataset.")
    
    st.sidebar.markdown("### 📱 Mobile Tip")
    st.sidebar.write("For faster loading on mobile, reduce sample size in numeric plots and skip the correlation matrix.")

    df = load_data()

    if st.sidebar.checkbox("Show raw dataset", value=False):
        show_overview(df)

    show_target_distribution(df)
    show_country_gender(df)
    show_numeric_histograms(df)
    show_correlation(df)

    auto_train = st.sidebar.checkbox("Auto-train default model", value=False)
    train_and_show(df, auto_run=auto_train)


if __name__ == "__main__":
    main()
