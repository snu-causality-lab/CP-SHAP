"""Decision rules on the semi-synthetic clinical task (Table 2 and Section 6.3)."""

import numpy as np
import pandas as pd

def load_phi_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) > 0 and str(df.columns[0]).lower().startswith("unnamed"):
        df = df.drop(columns=df.columns[0])
    return df

def align_feature_columns(phi_total, phi_local, phi_global):
    common = sorted(set(phi_total.columns) & set(phi_local.columns) & set(phi_global.columns))
    if len(common) == 0:
        raise ValueError("No common feature columns across phi_total/local/global.")
    return phi_total[common], phi_local[common], phi_global[common], common

def build_scores(phi_total, phi_local, phi_global_row, rule_name: str) -> pd.DataFrame:
    """
    phi_global_row: pd.Series (index=features) OR (1 x d) DataFrame
    """
    if isinstance(phi_global_row, pd.DataFrame):
        if len(phi_global_row) != 1:
            raise ValueError("phi_global is expected to have exactly 1 row in this setting.")
        g = phi_global_row.iloc[0]
    else:
        g = phi_global_row  # Series

    if rule_name == "total":
        return phi_total.copy()
    if rule_name == "local":
        return phi_local.copy()
    if rule_name == "mix_05_05":
        return 0.5 * phi_local + 0.5 * g  # broadcast
    if rule_name == "mix_08_02":
        return 0.8 * phi_local + 0.2 * g  # broadcast
    raise ValueError(f"Unknown rule_name: {rule_name}")

def decide_retest_and_count(scores_row: pd.Series, K: int):
    neg = scores_row[scores_row > 0]
    if len(neg) == 0:
        return False, 0, []
    ordered_idx = neg.abs().sort_values(ascending=False).index
    m = min(K, len(ordered_idx))
    return True, m, list(ordered_idx[:m])

def utility(retest: bool, high_risk: bool, m: int, c: float, fn_penalty: float, tp_reward: float, fp_reward: float, tn_reward: float) -> float:
    if retest and high_risk:        # TP
        return tp_reward - c * m
    if retest and (not high_risk):  # FP
        return fp_reward - c * m
    if (not retest) and high_risk:  # FN
        return -float(fn_penalty)
    return tn_reward                       # TN

def evaluate_rules(
    data_csv_path: str,
    phi_total_path: str,
    phi_local_path: str,
    phi_global_path: str,
    y_col: str = "y",
    tau: float = 18.0,
    K: int = 3,
    c: float = 0.1,
    fn_penalty: float = 4.0,
    tp_reward: float = 2.0, 
    fp_reward: float = 0.0, 
    tn_reward: float = 1.0
):
    df = pd.read_csv(data_csv_path)
    high = (df[y_col].to_numpy() >= tau)
    n = len(df)

    phi_total = load_phi_df(phi_total_path)
    phi_local = load_phi_df(phi_local_path)
    phi_global = load_phi_df(phi_global_path)

    if len(phi_global) != 1:
        raise ValueError(f"Expected phi_global to have 1 row, but got {len(phi_global)} rows.")

    phi_total, phi_local, phi_global, common = align_feature_columns(phi_total, phi_local, phi_global)

    if len(phi_total) != n or len(phi_local) != n:
        raise ValueError(f"Row mismatch: data={n}, total={len(phi_total)}, local={len(phi_local)}")

    rules = ["total", "local", "mix_05_05", "mix_08_02"]
    rows = []

    g_row = phi_global.iloc[0]  # Series (features -> global value)

    for r in rules:
        scores = build_scores(phi_total, phi_local, g_row, r)

        TP=FP=TN=FN=0
        util = np.empty(n, dtype=float)
        retest_flag = np.zeros(n, dtype=bool)
        m_used = np.zeros(n, dtype=int)

        for i in range(n):
            retest, m, _ = decide_retest_and_count(scores.iloc[i], K=K)
            retest_flag[i] = retest
            m_used[i] = m
            util[i] = utility(retest, high[i], m, c=c, fn_penalty=fn_penalty,tn_reward=tn_reward, tp_reward=tp_reward,fp_reward=fp_reward)

            if retest and high[i]: TP += 1
            elif retest and (not high[i]): FP += 1
            elif (not retest) and high[i]: FN += 1
            else: TN += 1

        avg_tests_if_retest = float(m_used[retest_flag].mean()) if retest_flag.any() else 0.0

        rows.append({
            "rule": r,
            "avg_utility": float(util.mean()),
            "total_utility": float(util.sum()),
            "retest_rate": float(retest_flag.mean()),
            "avg_tests_if_retest": avg_tests_if_retest,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        })

    return pd.DataFrame(rows).sort_values("avg_utility", ascending=False)



