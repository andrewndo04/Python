import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

# read the excel data
df_raw = pd.read_excel("data_coursework1_Q1.xls")

# SP500: market price/index level
# IBM: stock price
# 1-m Tbill: risk-free rate (given data is multiplied by 100 and in month)
prices = df_raw[["SP500", "IBM", "1-month Tbill"]].copy()

# numeric and drop non-numeric header rows
for col in ["SP500", "IBM", "1-month Tbill"]:
    prices[col] = pd.to_numeric(prices[col], errors="coerce")   # drop non-numeric into NaN

prices = prices.dropna(subset=["SP500", "IBM", "1-month Tbill"]).reset_index(drop=True)

# simple monthly returns
prices["r_M"] = prices["SP500"].pct_change()      # market
prices["r_IBM"] = prices["IBM"].pct_change()      # stock

# T-bill given is multiplied by 100 and not expressed in yearly basis
prices["r_f"] = prices["1-month Tbill"] / 100.0

# Drop first row and any NaNs
df = prices.dropna(subset=["r_IBM", "r_M", "r_f"]).reset_index(drop=True)

#check
print("First 5 rows of returns:")
print(df[["r_IBM", "r_M", "r_f"]].head(), "\n")

# excess returns & 2 regressors
df["Excess_Return_IBM"] = df["r_IBM"] - df["r_f"]
df["Excess_Return_M"] = df["r_M"] - df["r_f"]

# indicator for up vs down markets based on market excess return
D_t = (df["Excess_Return_M"] > 0).astype(int)
df["X1_Up_M"] = D_t * df["Excess_Return_M"]          # β1
df["X2_Down_M"] = (1 - D_t) * df["Excess_Return_M"]  # β2
df["X3_Squared"] = df["Excess_Return_M"] ** 2        # β3

print("Preview of regression data:")
print(df[["Excess_Return_IBM", "Excess_Return_M",
          "X1_Up_M", "X2_Down_M", "X3_Squared"]].head(), "\n")

# regression analysis
print("\n Model 1: Standard CAPM (OLS)")
# dependent v = Ri,t     independent v = RM,t
capm = ols("Excess_Return_IBM ~ Excess_Return_M", data=df).fit()
print(capm.summary())

print("\n Model 2: Extended Asymmetric Model (OLS)")
# 3 variables β (1,2,3)
extended_model = ols(
    "Excess_Return_IBM ~ X1_Up_M + X2_Down_M + X3_Squared",
    data=df).fit()
print(extended_model.summary())

# hyphotesis test
# F-Test: H0: β1 = β2 in model 2
print("\n F-Test for H_0: beta_1 = beta_2")
f_test_result = extended_model.f_test("X1_Up_M = X2_Down_M")
F_stat = float(f_test_result.fvalue)
F_pval = float(f_test_result.pvalue)

print(f"F-statistic: {F_stat:.4f}")
print(f"P-value: {F_pval:.4f}")

if F_pval < 0.05:
    print("Decision: Reject H_0. Up and down market betas significantly different from each other.")
else:
    print("Decision: Fail to reject H_0. No significant difference in betas.")

# t-test: H0: α = 0 in model 1
print("\n t-Test for H_0: alpha = 0")
t_stat = capm.tvalues["Intercept"]
p_val = capm.pvalues["Intercept"]

print(f"t-statistic (alpha=0): {t_stat:.4f}")
print(f"P-value (alpha=0): {p_val:.4f}")

if p_val < 0.05:
    print("Decision: Reject H_0. Alpha is statistically different from zero.")
else:
    print("Decision: Fail to reject H_0. Alpha is not statistically different from zero.")
