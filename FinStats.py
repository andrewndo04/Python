import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# read the excel data
df_raw = pd.read_excel("data_coursework1_Q1.xls")

# SP500: market price/index level
# IBM: stock price
# 1-m Tbill: risk-free rate (given data is multiplied by 100 and in month)
prices = df_raw[["SP500", "IBM", "1-month Tbill"]].copy()

prices["SP500"] = pd.to_numeric(prices["SP500"], errors="coerce")
prices["IBM"] = pd.to_numeric(prices["IBM"], errors="coerce")
prices["1-month Tbill"] = pd.to_numeric(prices["1-month Tbill"], errors="coerce")
prices = prices.dropna(subset=["SP500", "IBM", "1-month Tbill"]).reset_index(drop=True)

# simple monthly returns
prices["r_M"] = prices["SP500"].pct_change()      # market
prices["r_IBM"] = prices["IBM"].pct_change()      # stock

# T-bill given is multiplied by 100 and not expressed in yearly basis
prices["r_f"] = prices["1-month Tbill"] / 100.0

# drop first row and any NaNs
df = prices.dropna(subset=["r_IBM", "r_M", "r_f"]).reset_index(drop=True)

# excess returns & 2 regressors
df["Excess_R_IBM"] = df["r_IBM"] - df["r_f"]
df["Excess_R_M"] = df["r_M"] - df["r_f"]

# indicator: up vs down markets based on market excess return
D_t = (df["Excess_R_M"] > 0).astype(int)

# beta 1,2,3
df["X1_U_M"] = D_t * df["Excess_R_M"]          
df["X2_D_M"] = (1 - D_t) * df["Excess_R_M"] 
df["X3_Squared"] = df["Excess_R_M"] ** 2       

print("Regression data:")
print(df[["Excess_R_IBM", "Excess_R_M", "X1_U_M", "X2_D_M", 
          "X3_Squared"]].head(), "\n")

# regression analysis
print("\n Model 1: Standard CAPM (OLS)")
# dependent v = Ri,t     independent v = RM,t
capm = ols("Excess_R_IBM ~ Excess_R_M", data=df).fit()
print(capm.summary())

print("\n Model 2: Extended Asymmetric Model (OLS)")
# 3 variables β (1,2,3)
extended_model = ols("Excess_R_IBM ~ X1_U_M + X2_D_M + X3_Squared", 
                     data=df).fit()
print(extended_model.summary())

# hyphotesis test
# F-Test: H0: β1 = β2 in model 2
print("\n F-Test for H_0: beta_1 = beta_2")
f_test_result = extended_model.f_test("X1_U_M = X2_D_M")
F_stat = float(f_test_result.fvalue)
F_pval = float(f_test_result.pvalue)

print(f"F-statistic: {F_stat:.5f}")
print(f"P-value: {F_pval:.5f}")

if F_pval < 0.05:
    print("Decision: We reject H_0. The up and down market betas are significantly different under the observed data.")
else:
    print("Decision: We fail to reject H_0. No significant difference in betas under the observed data.")

# t-test: H0: α = 0 in model 1
print("\n t-Test for H_0: alpha = 0")
t_stat = capm.tvalues["Intercept"]
p_val = capm.pvalues["Intercept"]

print(f"t-statistic: {t_stat:.5f}")
print(f"P-value: {p_val:.5f}")

if p_val < 0.05:
    print("Decision: We reject H_0. Alpha is different from zero at 5% significance level.")
else:
    print("Decision: Fail to reject H_0. Alpha is not statistically different from zero at 5% significance level.")
