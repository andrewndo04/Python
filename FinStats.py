import pandas as pd
from statsmodels.formula.api import ols

df_raw = pd.read_excel("data_coursework1_Q1.xls")


# SP500: market price/index level
# IBM: stock price
# 1-m Tbill: risk-free rate (in % per month)
prices = df_raw[["SP500", "IBM", "1-month Tbill"]].copy()

# Make them numeric and drop non-numeric header rows (adjusted/closed/price)
for col in ["SP500", "IBM", "1-month Tbill"]:
    prices[col] = pd.to_numeric(prices[col], errors="coerce")

prices = prices.dropna(subset=["SP500", "IBM", "1-month Tbill"]).reset_index(drop=True)

# returns
# Simple monthly returns from prices
prices["r_M"] = prices["SP500"].pct_change()      # market
prices["r_i"] = prices["IBM"].pct_change()        # stock

# convert T-bill rate % to decimal monthly rate
prices["r_f"] = prices["1-month Tbill"] / 100.0

# Drop first row and any NaNs
df = prices.dropna(subset=["r_i", "r_M", "r_f"]).reset_index(drop=True)

print("First 5 rows of returns:")
print(df[["r_i", "r_M", "r_f"]].head(), "\n")

# excess returns & 2 regressors

df["Excess_Return_i"] = df["r_i"] - df["r_f"]
df["Excess_Return_M"] = df["r_M"] - df["r_f"]

# Indicator for up vs down markets based on market excess return
D_t = (df["Excess_Return_M"] > 0).astype(int)
df["X1_Up_Market"] = D_t * df["Excess_Return_M"]          # β1
df["X2_Down_Market"] = (1 - D_t) * df["Excess_Return_M"]  # β2
df["X3_Squared"] = df["Excess_Return_M"] ** 2             # β3

print("Preview of regression data:")
print(df[["Excess_Return_i", "Excess_Return_M",
          "X1_Up_Market", "X2_Down_Market", "X3_Squared"]].head(), "\n")

# regression analysis
print("\n=== Model (1): Standard CAPM (OLS) ===")
capm_model = ols("Excess_Return_i ~ Excess_Return_M", data=df).fit()
print(capm_model.summary())

print("\n=== Model (2): Extended Asymmetric Model (OLS) ===")
extended_model = ols(
    "Excess_Return_i ~ X1_Up_Market + X2_Down_Market + X3_Squared",
    data=df
).fit()
print(extended_model.summary())

# hyphotesis test
# F-Test: H0: β1 = β2
print("\n=== F-Test for H_0: beta_1 = beta_2 ===")
f_test_result = extended_model.f_test("X1_Up_Market = X2_Down_Market")
F_stat = float(f_test_result.fvalue)
F_pval = float(f_test_result.pvalue)

print(f"F-statistic: {F_stat:.4f}")
print(f"P-value: {F_pval:.4f}")

if F_pval < 0.05:
    print("Decision: Reject H_0. Up- and down-market betas differ significantly.")
else:
    print("Decision: Fail to reject H_0. No significant difference in betas.")

# t-test: H0: α = 0 in Model (1)
print("\n=== t-Test for H_0: alpha = 0 (Model 1) ===")
t_stat = capm_model.tvalues["Intercept"]
p_val = capm_model.pvalues["Intercept"]

print(f"t-statistic (alpha=0): {t_stat:.4f}")
print(f"P-value (alpha=0): {p_val:.4f}")

if p_val < 0.05:
    print("Decision: Reject H_0. Alpha is statistically different from zero.")
else:
    print("Decision: Fail to reject H_0. Alpha is not statistically different from zero.")
