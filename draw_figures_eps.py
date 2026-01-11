import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. The Data
# data_str = """
# eps=0.0050, -eps*ln(eps)=0.0265 -> Value Error=0.04114
# eps=0.0099, -eps*ln(eps)=0.0456 -> Value Error=0.05867
# eps=0.0147, -eps*ln(eps)=0.0622 -> Value Error=0.08176
# eps=0.0196, -eps*ln(eps)=0.0771 -> Value Error=0.11046
# eps=0.0245, -eps*ln(eps)=0.0909 -> Value Error=0.12013
# eps=0.0294, -eps*ln(eps)=0.1036 -> Value Error=0.14249
# eps=0.0342, -eps*ln(eps)=0.1156 -> Value Error=0.17236
# eps=0.0391, -eps*ln(eps)=0.1268 -> Value Error=0.16186
# eps=0.0440, -eps*ln(eps)=0.1374 -> Value Error=0.19549
# eps=0.0489, -eps*ln(eps)=0.1475 -> Value Error=0.20522
# eps=0.0537, -eps*ln(eps)=0.1571 -> Value Error=0.22417
# eps=0.0586, -eps*ln(eps)=0.1663 -> Value Error=0.24161
# eps=0.0635, -eps*ln(eps)=0.1751 -> Value Error=0.23979
# eps=0.0684, -eps*ln(eps)=0.1834 -> Value Error=0.28054
# eps=0.0733, -eps*ln(eps)=0.1915 -> Value Error=0.29203
# eps=0.0781, -eps*ln(eps)=0.1992 -> Value Error=0.29668
# eps=0.0830, -eps*ln(eps)=0.2066 -> Value Error=0.30376
# eps=0.0879, -eps*ln(eps)=0.2137 -> Value Error=0.30762
# eps=0.0927, -eps*ln(eps)=0.2205 -> Value Error=0.33422
# eps=0.0976, -eps*ln(eps)=0.2271 -> Value Error=0.33347
# eps=0.1025, -eps*ln(eps)=0.2335 -> Value Error=0.35941
# eps=0.1074, -eps*ln(eps)=0.2396 -> Value Error=0.34778
# eps=0.1123, -eps*ln(eps)=0.2455 -> Value Error=0.39437
# eps=0.1171, -eps*ln(eps)=0.2512 -> Value Error=0.38344
# eps=0.1220, -eps*ln(eps)=0.2567 -> Value Error=0.40553
# eps=0.1269, -eps*ln(eps)=0.2619 -> Value Error=0.41987
# eps=0.1318, -eps*ln(eps)=0.2670 -> Value Error=0.41003
# eps=0.1366, -eps*ln(eps)=0.2720 -> Value Error=0.46675
# eps=0.1415, -eps*ln(eps)=0.2767 -> Value Error=0.42762
# eps=0.1464, -eps*ln(eps)=0.2813 -> Value Error=0.41991
# eps=0.1512, -eps*ln(eps)=0.2857 -> Value Error=0.48306
# eps=0.1561, -eps*ln(eps)=0.2899 -> Value Error=0.46062
# eps=0.1610, -eps*ln(eps)=0.2940 -> Value Error=0.46908
# eps=0.1659, -eps*ln(eps)=0.2980 -> Value Error=0.47813
# eps=0.1708, -eps*ln(eps)=0.3018 -> Value Error=0.46534
# eps=0.1756, -eps*ln(eps)=0.3055 -> Value Error=0.51337
# eps=0.1805, -eps*ln(eps)=0.3090 -> Value Error=0.53918
# eps=0.1854, -eps*ln(eps)=0.3124 -> Value Error=0.51103
# eps=0.1903, -eps*ln(eps)=0.3157 -> Value Error=0.58223
# eps=0.1951, -eps*ln(eps)=0.3189 -> Value Error=0.52079
# eps=0.2000, -eps*ln(eps)=0.3219 -> Value Error=0.53762
# """

data_str = """
eps=0.0050, -eps*ln(eps)=0.0265 -> Value Error=0.03423
eps=0.0099, -eps*ln(eps)=0.0456 -> Value Error=0.06219
eps=0.0147, -eps*ln(eps)=0.0622 -> Value Error=0.07863
eps=0.0196, -eps*ln(eps)=0.0771 -> Value Error=0.10270
eps=0.0245, -eps*ln(eps)=0.0909 -> Value Error=0.12985
eps=0.0294, -eps*ln(eps)=0.1036 -> Value Error=0.13547
eps=0.0342, -eps*ln(eps)=0.1156 -> Value Error=0.15020
eps=0.0391, -eps*ln(eps)=0.1268 -> Value Error=0.16470
eps=0.0440, -eps*ln(eps)=0.1374 -> Value Error=0.18202
eps=0.0489, -eps*ln(eps)=0.1475 -> Value Error=0.20638
eps=0.0537, -eps*ln(eps)=0.1571 -> Value Error=0.21399
eps=0.0586, -eps*ln(eps)=0.1663 -> Value Error=0.23753
eps=0.0635, -eps*ln(eps)=0.1751 -> Value Error=0.24892
eps=0.0684, -eps*ln(eps)=0.1834 -> Value Error=0.26737
eps=0.0733, -eps*ln(eps)=0.1915 -> Value Error=0.28304
eps=0.0781, -eps*ln(eps)=0.1992 -> Value Error=0.28941
eps=0.0830, -eps*ln(eps)=0.2066 -> Value Error=0.30145
eps=0.0879, -eps*ln(eps)=0.2137 -> Value Error=0.33095
eps=0.0927, -eps*ln(eps)=0.2205 -> Value Error=0.32439
eps=0.0976, -eps*ln(eps)=0.2271 -> Value Error=0.32495
eps=0.1025, -eps*ln(eps)=0.2335 -> Value Error=0.36315
eps=0.1074, -eps*ln(eps)=0.2396 -> Value Error=0.38994
eps=0.1123, -eps*ln(eps)=0.2455 -> Value Error=0.37984
eps=0.1171, -eps*ln(eps)=0.2512 -> Value Error=0.39444
eps=0.1220, -eps*ln(eps)=0.2567 -> Value Error=0.40077
eps=0.1269, -eps*ln(eps)=0.2619 -> Value Error=0.42121
eps=0.1318, -eps*ln(eps)=0.2670 -> Value Error=0.41613
eps=0.1366, -eps*ln(eps)=0.2720 -> Value Error=0.44444
eps=0.1415, -eps*ln(eps)=0.2767 -> Value Error=0.43765
eps=0.1464, -eps*ln(eps)=0.2813 -> Value Error=0.44278
eps=0.1512, -eps*ln(eps)=0.2857 -> Value Error=0.42833
eps=0.1561, -eps*ln(eps)=0.2899 -> Value Error=0.44808
eps=0.1610, -eps*ln(eps)=0.2940 -> Value Error=0.46045
eps=0.1659, -eps*ln(eps)=0.2980 -> Value Error=0.47261
eps=0.1708, -eps*ln(eps)=0.3018 -> Value Error=0.49987
eps=0.1756, -eps*ln(eps)=0.3055 -> Value Error=0.52204
eps=0.1805, -eps*ln(eps)=0.3090 -> Value Error=0.55095
eps=0.1854, -eps*ln(eps)=0.3124 -> Value Error=0.50419
eps=0.1903, -eps*ln(eps)=0.3157 -> Value Error=0.48686
eps=0.1951, -eps*ln(eps)=0.3189 -> Value Error=0.56965
eps=0.2000, -eps*ln(eps)=0.3219 -> Value Error=0.54224
"""


# Parsing
pattern = r"eps=([\d\.]+), -eps\*ln\(eps\)=([\d\.]+) -> Value Error=([\d\.]+)"
matches = re.findall(pattern, data_str)

if not matches:
    print("Error: No data found matching the pattern.")
    exit(1)

data = []
for match in matches:
    data.append({
        'eps': float(match[0]),
        '-eps*ln(eps)': float(match[1]),
        'Value_error': float(match[2])
    })
df = pd.DataFrame(data)

# 2. Regression
# Model 1 (Right Panel): Value_error vs eps
X1 = df[['eps']]
y = df['Value_error']
model1 = LinearRegression().fit(X1, y)
y_pred1 = model1.predict(X1)
rmse1 = np.sqrt(mean_squared_error(y, y_pred1))
m1 = model1.coef_[0]
c1 = model1.intercept_

# Model 2 (Left Panel): Value_error vs -eps*ln(eps)
X2 = df[['-eps*ln(eps)']]
model2 = LinearRegression().fit(X2, y)
y_pred2 = model2.predict(X2)
rmse2 = np.sqrt(mean_squared_error(y, y_pred2))
m2 = model2.coef_[0]
c2 = model2.intercept_

# 3. Plotting with Object-Oriented Interface (fig, axes)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left Panel: Value_error vs -eps*ln(eps) ---
axes[0].scatter(df['-eps*ln(eps)'], y, color='blue', label='Simulation Data')
axes[0].plot(
    df['-eps*ln(eps)'], 
    y_pred2, 
    'r--', 
    label=f'Fit: y={m2:.5f}x  {c2:.5f}\nRMSE: {rmse2:.5f}'
)
axes[0].set_xlabel(r'$- \epsilon \ln(\epsilon)$')
axes[0].set_ylabel('Value Error')
axes[0].set_title(r'Value Error vs $- \epsilon \ln(\epsilon)$')
axes[0].legend(loc='upper left')

# --- Right Panel: Value_error vs eps ---
axes[1].scatter(df['eps'], y, color='green', label='Simulation Data')
axes[1].plot(
    df['eps'], 
    y_pred1, 
    'r--', 
    label=f'Fit: y={m1:.5f}x + {c1:.5f}\nRMSE: {rmse1:.5f}'
)
axes[1].set_xlabel(r'$\epsilon$')
axes[1].set_ylabel('Value Error')
axes[1].set_title(r'Value Error vs $\epsilon$')
axes[1].legend(loc='upper left')

plt.tight_layout()
plt.show()
# plt.savefig('figure_eps.png')
# print("Figure saved to figure_eps.png")