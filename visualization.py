import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


three_dist_flag = True

df_in_5 = pd.read_csv('1in5.csv', header=None)
df_in_10 = pd.read_csv('1in10.csv', header=None)
if three_dist_flag:
    df_in_20 = pd.read_csv('1in20.csv', header=None)


norm_vec_in_5 = df_in_5.loc[1]/.2
norm_vec_in_10 = df_in_10.loc[1]/.1
if three_dist_flag:
    norm_vec_in_20 = df_in_20.loc[1]/.05

# mean
print("mean for i in 5 case: ", np.mean(norm_vec_in_5))
print("mean for i in 10 case: ", np.mean(norm_vec_in_10))
if three_dist_flag:
    print("mean for i in 20 case: ", np.mean(norm_vec_in_20))

# median
print("median for i in 5 case: ", np.median(norm_vec_in_5))
print("median for i in 10 case: ", np.median(norm_vec_in_10))
if three_dist_flag:
    print("median for i in 20 case: ", np.median(norm_vec_in_20))

# standard deviation
print("standard deviation for i in 5 case: ", np.std(norm_vec_in_5))
print("standard deviation for i in 10 case: ", np.std(norm_vec_in_10))
if three_dist_flag:
    print("standard deviation for i in 20 case: ", np.std(norm_vec_in_20))

# maximum
print("max for i in 5 case: ", max(norm_vec_in_5))
print("max for i in 10 case: ", max(norm_vec_in_10))
if three_dist_flag:
    print("max for i in 20 case: ", max(norm_vec_in_20))

# number above 1
print("num. above 1 for i in 5 case: ", sum(1*(norm_vec_in_5 > 1)))
print("num. above 1 for i in 10 case: ", sum(1*(norm_vec_in_10 > 1)))
if three_dist_flag:
    print("num. above 1 for i in 20 case: ", sum(1*(norm_vec_in_20 > 1)))


if three_dist_flag:
    df = pd.DataFrame([np.array(norm_vec_in_5), np.array(norm_vec_in_10), np.array(norm_vec_in_20)]).T
else:
    df = pd.DataFrame([np.array(norm_vec_in_5), np.array(norm_vec_in_10)]).T
# df = pd.DataFrame([np.array(norm_vec_in_5)]).T


if three_dist_flag:
    df.columns = ['1 in 5', '1 in 10', '1 in 20']
else:
    df.columns = ['1 in 5', '1 in 10']
# df.columns = ['1 in 5']

plt.ylim(-.43, 17)
# plt.ylim(-.43, 20.5)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax = sns.violinplot(data=df, inner=None, cut=0, linewidth=.0, palette="pastel")

ax = sns.boxplot(data=df, color='black', palette='pastel', linewidth=.8, fliersize=2, width=0.07, boxprops={'zorder': 1}, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"white", "markersize":"5"})

ax.set_xlabel("Scenario",size = 20,fontname='Times New Roman')  
ax.set_ylabel("Normalized Performance",size = 20,fontname='Times New Roman')  


plt.savefig('figure.pdf', bbox_inches='tight')
plt.show()