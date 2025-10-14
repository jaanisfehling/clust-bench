from typing import Dict, List, Optional

import pandas as pd


def one_hot_encode_feature(
    df: pd.DataFrame, feature_to_encode: str, weight: float = 1
) -> pd.DataFrame:
    dummies = pd.get_dummies(
        df[feature_to_encode], dtype="int32", prefix=feature_to_encode
    )
    dummies *= weight
    result_df = pd.concat([df, dummies], axis=1)
    return result_df.drop(columns=feature_to_encode)

