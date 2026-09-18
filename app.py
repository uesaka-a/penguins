import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# -------------------------
# 基本設定
# -------------------------

st.set_page_config(
    page_title="ペンギン分類チャレンジ",
    page_icon="🐧",
    layout="centered"
)

st.title("🐧 ペンギン分類チャレンジ")

st.write(
    """
    ペンギンの種類を分類するには，
    どの特徴量を使えばよいでしょうか？

    使いたい特徴量を選んで，
    **「分類してみる」**ボタンを押してください．
    """
)


# -------------------------
# データ読み込み
# -------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("penguins.csv")

    # 今回使用する列だけに限定
    columns = [
        "species",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    df = df[columns]

    # 授業用なので欠損行はあらかじめ除外
    df = df.dropna()

    return df


df = load_data()


# -------------------------
# 表示名
# -------------------------

feature_names = {
    "くちばしの長さ": "bill_length_mm",
    "くちばしの深さ": "bill_depth_mm",
    "翼の長さ": "flipper_length_mm",
    "体重": "body_mass_g"
}


# -------------------------
# 特徴量選択
# -------------------------

st.subheader("① 特徴量を選んでください")

selected_labels = st.multiselect(
    "使いたい特徴量",
    list(feature_names.keys()),
    placeholder="特徴量を選択"
)

st.caption(
    "まずは1個だけ選んで試してみましょう．"
)


# -------------------------
# 分類
# -------------------------

st.subheader("② 分類してみましょう")

if st.button(
    "🐧 この特徴量で分類する",
    type="primary",
    use_container_width=True
):

    if len(selected_labels) == 0:

        st.warning(
            "特徴量を1個以上選んでください．"
        )

    else:

        selected_features = [
            feature_names[label]
            for label in selected_labels
        ]

        X = df[selected_features]
        y = df["species"]

        # 全員が同じ結果になるように固定
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

        # モデルは固定
        model = DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        correct = int(
            (prediction == y_test).sum()
        )

        total = len(y_test)

        # -------------------------
        # 結果表示
        # -------------------------

        st.success("分類が完了しました！")

        st.subheader("🎯 結果")

        st.metric(
            "正解率",
            f"{accuracy * 100:.1f}%"
        )

        st.write(
            f"**{total}羽中 {correct}羽**を"
            "正しく分類できました．"
        )

        st.write("使用した特徴量：")

        for label in selected_labels:
            st.write(f"・{label}")

        st.divider()

        if len(selected_labels) == 1:

            st.info(
                "💡 他の特徴量を1個だけ選んで，"
                "結果を比べてみましょう．"
            )

        elif len(selected_labels) == 2:

            st.info(
                "💡 別の2つの組み合わせでは"
                "どうなるでしょうか？"
            )

        else:

            st.info(
                "💡 特徴量を減らしても，"
                "同じくらい分類できるでしょうか？"
            )


# -------------------------
# ミッション
# -------------------------

st.divider()

st.subheader("🏆 今日のミッション")

st.write(
    """
    **MISSION 1**

    特徴量を**1個だけ**使って，
    最も高い正解率になる特徴量を探してください．

    **MISSION 2**

    特徴量を**2個だけ**使って，
    最も高い正解率になる組み合わせを探してください．

    **MISSION 3**

    4個すべての特徴量を使った場合と
    比較してください．
    """
)

st.caption(
    "特徴量は，多ければ多いほどよいのでしょうか？"
)