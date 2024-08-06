import matplotlib.pyplot as plt
import pandas as pd

k = 3

times_base = pd.read_csv(f'times_record_base_{k}.csv', index_col=0)
times_v1 = pd.read_csv(f'times_record_v1_{k}.csv', index_col=0)
times_v2 = pd.read_csv(f'times_record_v2_{k}.csv', index_col=0)

times_base = times_base.values.tolist()
times_v1 = times_v1.values.tolist()
times_v2 = times_v2.values.tolist()

ks = range(len(times_base))

fig, ax = plt.subplots() # 创建图实例
plt.title(f"Time records of threshold {k}")
ax.plot(ks, times_base, label='base')
ax.plot(ks, times_v1, label='v1')
ax.plot(ks, times_v2, label='v2')
plt.xlabel("k")
plt.ylabel("time/s")
plt.grid()
ax.legend()
plt.savefig(f'time_{k}.jpg')
plt.show()