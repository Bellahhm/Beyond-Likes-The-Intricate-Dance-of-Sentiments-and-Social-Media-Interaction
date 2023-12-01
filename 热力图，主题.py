# # coding=UTF-8
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
#
# # 读取Excel表格
# data = pd.read_excel('C:/Users/lenovo/Desktop/主题词及推测+标签 - 副本.xlsx', index_col=0)
#
# # 对每年的数据进行归一化处理
# scaler = MinMaxScaler()
# data_normalized = pd.DataFrame(scaler.fit_transform(data.T), columns=data.index, index=data.columns)
#
# import matplotlib as mpl
# # 绘制热力图
# mpl.rcParams['axes.titleweight'] = 'bold'  # 设置标题为粗体字
#
# sns.heatmap(data_normalized, cmap='PuBu', annot=False, fmt='.2%', vmin=0, vmax=1)
# plt.title('Theme Categories Scores')
# plt.xlabel('Years')
# plt.ylabel('Categories')
# plt.subplots_adjust(left=0.25, right=0.95, bottom=0.2, top=0.9)
#
# plt.show()


# coding=UTF-8
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取Excel表格
data = pd.read_excel('C:/Users/lenovo/Desktop/twittwe/主题词及推测+标签 - 副本.xlsx', index_col=0)

# 调整横纵坐标
data = data.transpose()


fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data, cmap='PuBu', annot=False, fmt='d', vmin=0, vmax=5)
ax.set_xlabel('Years', fontsize=10)
ax.set_ylabel('Categories', fontsize=10)
ax.set_title('Thematic variation heatmap', fontsize=14, fontweight='bold')
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.9)

filename = f"Changes of Theme Categories.pdf"
filepath = f"{filename}"

# 保存矢量图
plt.savefig(filepath, format='pdf', dpi=1200, bbox_inches='tight')

plt.show()


