import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

from utils.config import (
    DATA_PATH,
    TEXT_COL,
    LABEL_COL,
    SEED
)


# =========================================================
# Load Dataset
# =========================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


# =========================================================
# Remove Exact Duplicate Rows
# =========================================================

def remove_exact_duplicates(df):

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            TEXT_COL,
            LABEL_COL,
            "source_file"
        ]
    ).reset_index(drop=True)

    after = len(df)

    removed = before - after

    print("Exact Duplicate Removal")
    print("-" * 70)

    print(f"Before : {before:,}")
    print(f"After  : {after:,}")
    print(f"Removed: {removed:,}")

    print()

    return df


# =========================================================
# Create Text Groups
# =========================================================

def create_text_groups(df):

    """
    동일한 text는 하나의 group으로 묶는다.

    목적:
        동일 text가 train / validation / test에
        동시에 존재하는 data leakage 방지
    """

    df = df.copy()

    df["text_group"] = (
        pd.factorize(
            df[TEXT_COL]
        )[0]
    )

    return df


# =========================================================
# Create Group Labels
# =========================================================

def create_group_labels(df):

    """
    StratifiedGroupKFold는 각 group에 대해
    하나의 대표 label이 필요하다.

    일반적으로:
        동일 text의 모든 label이 같으면 해당 label 사용

    label conflict가 존재하면:
        해당 group에서 가장 많이 등장하는 label을
        stratification용 대표 label로 사용한다.

    주의:
        이 대표 label은 실제 데이터의 label을 변경하는 것이 아니다.
        오직 split의 class balance를 맞추기 위한 용도다.
    """

    group_labels = (
        df
        .groupby("text_group")[LABEL_COL]
        .agg(
            lambda x: x.value_counts().idxmax()
        )
    )

    return group_labels


# =========================================================
# Stratified Group Split
# =========================================================

def split_data(
    df,
    seed=SEED
):

    """
    동일 text group이 서로 다른 split에 들어가지 않도록
    StratifiedGroupKFold를 사용한다.

    1차:
        10-fold 중 7개 → Train
        3개 → Temporary

    2차:
        Temporary를 2개로 분할
        1개 → Validation
        1개 → Test

    결과:
        약 70 / 15 / 15
    """

    df = df.copy()

    group_labels = create_group_labels(df)

    group_ids = group_labels.index.to_numpy()
    group_y = group_labels.to_numpy()

    # -----------------------------------------------------
    # 1st Split
    # -----------------------------------------------------

    sgkf_10 = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=seed
    )

    folds = list(
        sgkf_10.split(
            group_ids,
            group_y,
            groups=group_ids
        )
    )

    train_group_idx = []
    temp_group_idx = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):

        if fold_idx < 7:
            train_group_idx.extend(test_idx)
        else:
            temp_group_idx.extend(test_idx)

    train_groups = set(
        group_ids[train_group_idx]
    )

    temp_groups = set(
        group_ids[temp_group_idx]
    )

    # -----------------------------------------------------
    # 2nd Split
    # -----------------------------------------------------

    temp_group_ids = group_ids[temp_group_idx]
    temp_group_y = group_y[temp_group_idx]

    sgkf_2 = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=seed
    )

    val_idx, test_idx = next(
        sgkf_2.split(
            temp_group_ids,
            temp_group_y,
            groups=temp_group_ids
        )
    )

    val_groups = set(
        temp_group_ids[val_idx]
    )

    test_groups = set(
        temp_group_ids[test_idx]
    )

    # -----------------------------------------------------
    # Safety Check
    # -----------------------------------------------------

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    # -----------------------------------------------------
    # Assign Split
    # -----------------------------------------------------

    df["split"] = "unknown"

    df.loc[
        df["text_group"].isin(train_groups),
        "split"
    ] = "train"

    df.loc[
        df["text_group"].isin(val_groups),
        "split"
    ] = "validation"

    df.loc[
        df["text_group"].isin(test_groups),
        "split"
    ] = "test"

    # -----------------------------------------------------
    # Final Check
    # -----------------------------------------------------

    assert (
        df["split"] != "unknown"
    ).all()

    return df


# =========================================================
# Get Split Data
# =========================================================

def get_split_data(df):

    train_df = (
        df[
            df["split"] == "train"
        ]
        .reset_index(drop=True)
    )

    val_df = (
        df[
            df["split"] == "validation"
        ]
        .reset_index(drop=True)
    )

    test_df = (
        df[
            df["split"] == "test"
        ]
        .reset_index(drop=True)
    )

    return (
        train_df,
        val_df,
        test_df
    )