import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from langdetect import detect
from langdetect import detect_langs

# 读取Excel数据
df = pd.read_excel('D:/pythonProject2/twitter/excel/新建 XLSX 工作表.xlsx')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

stop_words = set(stopwords.words('english'))  # 英文停用词列表
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: re.sub(r'http\S+', '', x))  # 去除URL
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: re.sub(r'@[\w_]+', '', x))  # 去除@符号
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: re.sub(r'\b(rt|im|cz|na)\b', '', x))  # 去除指定词汇
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: re.sub(r'[^\w\s]|_', '', x))  # 去除标点符号和下划线
df['Embedded_text'] = df['Embedded_text'].apply(
    lambda x: " ".join([word for word in x.split() if word.lower() not in stop_words]))  # 去除停用词
df['Embedded_text'] = df['Embedded_text'].apply(lambda x: x.lower())  # 将文本转换为小写字母
df['Embedded_text'] = df['Embedded_text'].apply(
    lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.lower() not in stop_words]))

# 定义TF-IDF向量化器
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

# 导入字体库
import matplotlib.font_manager as fm

# 循环计算每一年的前五个主题，并添加进度条
for year in tqdm(range(2010, 2024)):
    # 筛选出当年的微博文本数据
    year_data = df[df['Timestamp'].dt.year == year]
    if len(year_data) > 0:
        # 将微博文本进行向量化
        tfidf = tfidf_vectorizer.fit_transform(year_data['Embedded_text'])
        # 使用NMF进行分解
        nmf_model = NMF(n_components=5, init='nndsvd')
        nmf = nmf_model.fit_transform(tfidf)
        # 获取每个主题的名称和权重
        feature_names = tfidf_vectorizer.get_feature_names_out()
        weights = nmf_model.components_
        # 输出每个主题的前10个关键词
        print(f"Year {year}:")
        for i, topic_weights in enumerate(weights):
            top_keyword_indexes = topic_weights.argsort()[::-1][:10]
            top_keywords = [feature_names[i] for i in top_keyword_indexes]
            print(f"Topic {i + 1}: {', '.join(top_keywords)}")

# ###seaborn画图
# import matplotlib.pyplot as plt
# import networkx as nx
# import seaborn as sns
# from tqdm import tqdm
#
# # Define color palette
# colors = sns.color_palette('hls', n_colors=5)
#
# # Define output directory
# output_dir = 'output'
#
# # Loop through each year and compute the top five topics for that year
# for year in tqdm(range(2010, 2024)):
#     # Filter the data for the current year
#     year_data = df[df['Timestamp'].dt.year == year]
#     if len(year_data) > 0:
#         # Vectorize the text data using TF-IDF
#         tfidf = tfidf_vectorizer.fit_transform(year_data['Embedded_text'])
#         # Perform NMF factorization
#         nmf_model = NMF(n_components=5, init='nndsvd')
#         nmf = nmf_model.fit_transform(tfidf)
#         # Get the names and weights of each topic
#         feature_names = tfidf_vectorizer.get_feature_names_out()
#         weights = nmf_model.components_
#         # Create a co-occurrence network
#         G = nx.Graph()
#
#         # Add nodes to the graph
#         for i, topic_weights in enumerate(weights):
#             top_keyword_indexes = topic_weights.argsort()[::-1][:10]
#             top_keywords = [feature_names[i] for i in top_keyword_indexes]
#             for keyword in top_keywords:
#                 # Check if the node name is a string
#                 if isinstance(keyword, str):
#                     G.add_node(keyword, color=colors[i])
#
#         # Add edges to the graph
#         for i, topic_weights in enumerate(weights):
#             top_keyword_indexes = topic_weights.argsort()[::-1][:10]
#             top_keywords = [feature_names[i] for i in top_keyword_indexes]
#             for j in range(len(top_keywords)):
#                 for k in range(j + 1, len(top_keywords)):
#                     G.add_edge(top_keywords[j], top_keywords[k])
#
#         # Draw the network graph
#         plt.figure(figsize=(10, 8))
#         pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)
#
#         # 设置中文字体为黑体
#         font_chinese = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=9)
#         # 设置其他语言字体为Times New Roman
#         font_others = fm.FontProperties(fname='C:/Windows/Fonts/times.ttf', size=9)
#
#         for node in G.nodes():
#             # Check if the node name is a string and if it contains only ASCII characters
#             if isinstance(node, str) and node.strip():
#                 # Check if the node contains Chinese characters
#                 if any('\u4e00' <= c <= '\u9fff' for c in node):
#                     nx.draw_networkx_labels(G, pos, {node: node}, font_family=font_chinese.get_name(),
#                                             font_size=font_chinese.get_size())
#                 else:
#                     nx.draw_networkx_labels(G, pos, {node: node}, font_family=font_others.get_name(),
#                                             font_size=font_others.get_size())
#             else:
#                 nx.draw_networkx_labels(G, pos, {node: node}, font_family=font_others.get_name(),
#                                         font_size=font_others.get_size())
#
#         node_colors = [G.nodes[n]['color'] for n in G.nodes()]
#         nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, alpha=0.7)
#         nx.draw_networkx_edges(G, pos, edge_color='gray')
#
#         plt.title(f"Co-occurrence Network {year}", fontdict={"family": "Times New Roman", "size": 16})
#
#         plt.axis('off')
#         sns.despine()
#
#         # 修改保存路径和文件名
#         filename = f"co_occurrence_network_{year}.pdf"
#         filepath = f"{output_dir}/{filename}"
#
#         # 保存矢量图
#         plt.savefig(filepath, format='pdf', dpi=1200, bbox_inches='tight')
#
#         plt.show()

#
# #4年8主题
###seaborn画图
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from tqdm import tqdm

# Define color palette
colors = sns.color_palette('hls', n_colors=6)

# Define output directory
output_dir = 'output'

# Loop through each year and compute the top five topics for that year
for year in tqdm(range(2010, 2023, 3)):
    # Filter the data for the current year
    year_data = df[df['Timestamp'].dt.year == year]
    if len(year_data) > 0:
        # Vectorize the text data using TF-IDF
        tfidf = tfidf_vectorizer.fit_transform(year_data['Embedded_text'])
        # Perform NMF factorization
        nmf_model = NMF(n_components=6, init='nndsvd')
        nmf = nmf_model.fit_transform(tfidf)
        # Get the names and weights of each topic
        feature_names = tfidf_vectorizer.get_feature_names_out()
        weights = nmf_model.components_
        # Create a co-occurrence network
        G = nx.Graph()

        # Add nodes to the graph
        for i, topic_weights in enumerate(weights):
            top_keyword_indexes = topic_weights.argsort()[::-1][:10]
            top_keywords = [feature_names[i] for i in top_keyword_indexes]
            for keyword in top_keywords:
                # Check if the node name is a string
                if isinstance(keyword, str):
                    G.add_node(keyword, color=colors[i])

        # Add edges to the graph
        for i, topic_weights in enumerate(weights):
            top_keyword_indexes = topic_weights.argsort()[::-1][:10]
            top_keywords = [feature_names[i] for i in top_keyword_indexes]
            for j in range(len(top_keywords)):
                for k in range(j + 1, len(top_keywords)):
                    G.add_edge(top_keywords[j], top_keywords[k])

        # Draw the network graph
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)

        # 设置中文字体为黑体
        font_chinese = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=9)
        # 设置其他语言字体为Times New Roman
        font_others = fm.FontProperties(fname='C:/Windows/Fonts/times.ttf', size=9)

        for node in G.nodes():
            # Check if the node name is a string and if it contains only ASCII characters
            if isinstance(node, str) and node.strip():
                # Check if the node contains Chinese characters
                if any('\u4e00' <= c <= '\u9fff' for c in node):
                    nx.draw_networkx_labels(G, pos, {node: node}, font_family=font_chinese.get_name(),
                                            font_size=font_chinese.get_size())
                else:
                    nx.draw_networkx_labels(G, pos, {node: node}, font_family=font_others.get_name(),
                                            font_size=font_others.get_size())
            else:
                nx.draw_networkx_labels(G, pos, {node: node}, font_family=font_others.get_name(),
                                        font_size=font_others.get_size())

        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, alpha=0.7)
        nx.draw_networkx_edges(G, pos, edge_color='gray')

        plt.title(f"Co-occurrence Network {year}-{year+2}", fontdict={"family": "Times New Roman", "size": 16})

        plt.axis('off')
        sns.despine()

        # 修改保存路径和文件名
        # filename = f"co_occurrence_network_{year}-{year+2}.pdf"
        # filepath = f"{output_dir}/{filename}"
        #
        # # 保存矢量图
        # plt.savefig(filepath, format='pdf', dpi=1200, bbox_inches='tight')

        plt.show()

# import networkx as nx
# import seaborn as sns
#
# # 定义颜色映射
# colors = sns.color_palette('hls', n_colors=8)
#
# # 循环计算每四年的前八个主题，并添加进度条
# for year in tqdm(range(2010, 2024, 4)):
#     # 筛选出当年的微博文本数据
#     year_data = df[(df['Timestamp'].dt.year >= year) & (df['Timestamp'].dt.year < year+4)]
#     if len(year_data) > 0:
#         # 将微博文本进行向量化
#         tfidf = tfidf_vectorizer.fit_transform(year_data['Embedded_text'])
#         # 使用NMF进行分解
#         nmf_model = NMF(n_components=8, init='nndsvd')
#         nmf = nmf_model.fit_transform(tfidf)
#         # 获取每个主题的名称和权重
#         feature_names = tfidf_vectorizer.get_feature_names_out()
#         weights = nmf_model.components_
#         # 创建共现图
#         G = nx.Graph()
#
#         # 添加节点
#         for i, topic_weights in enumerate(weights):
#             top_keyword_indexes = topic_weights.argsort()[::-1][:10]
#             top_keywords = [feature_names[i] for i in top_keyword_indexes]
#             for keyword in top_keywords:
#                 G.add_node(keyword, color=colors[i])
#
#         # 添加边
#         for i, topic_weights in enumerate(weights):
#             top_keyword_indexes = topic_weights.argsort()[::-1][:10]
#             top_keywords = [feature_names[i] for i in top_keyword_indexes]
#             for j in range(len(top_keywords)):
#                 for k in range(j + 1, len(top_keywords)):
#                     G.add_edge(top_keywords[j], top_keywords[k])
#
#         # 绘制网络图
#
#         plt.figure(figsize=(10, 8))
#         pos = nx.spring_layout(G,k=0.4)
#         node_colors = [G.nodes[n]['color'] for n in G.nodes()]
#         nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, alpha=0.7)
#         nx.draw_networkx_edges(G, pos, edge_color='gray')
#         labels = {node: node for node in G.nodes()}
#         nx.draw_networkx_labels(G, pos, labels, font_size=9, font_family='Times New Roman', font_weight='bold')
#         plt.title(f"Co-occurrence Network {year}-{year+3}", fontsize=16, fontname='Times New Roman', fontweight='bold')
#         plt.axis('off')
#         sns.despine()
#         plt.show()

#
# from pyvis.network import Network
# import seaborn as sns
# import colorsys
#
# # 定义彩虹色颜色映射
# num_colors = 5
# HSV_tuples = [(x * 1.0 / num_colors, 0.7, 0.7) for x in range(num_colors)]
# colors = list(map(lambda x: '#' + ''.join([format(int(c * 255), '02x') for c in colorsys.hsv_to_rgb(*x)]), HSV_tuples))
#
# # 循环计算每一年的前五个主题，并添加进度条
# for year in tqdm(range(2010, 2024)):
#     # 筛选出当年的微博文本数据
#     year_data = df[df['Timestamp'].dt.year == year]
#     if len(year_data) > 0:
#         # 将微博文本进行向量化
#         tfidf = tfidf_vectorizer.fit_transform(year_data['Embedded_text'])
#         # 使用NMF进行分解
#         nmf_model = NMF(n_components=5, init='nndsvd')
#         nmf = nmf_model.fit_transform(tfidf)
#         # 获取每个主题的名称和权重
#         feature_names = tfidf_vectorizer.get_feature_names_out()
#         weights = nmf_model.components_
#         plt.rcParams['font.sans-serif'] = ['SimHei']
#
#         # 创建共现图
#         # 创建共现图
#         G = Network(height="750px", width="100%", bgcolor="#696969", font_color="white", notebook=True)
#         G.set_options("""
#             var options = {
#               "nodes": {
#                 "borderWidth": 2,
#                 "color": {
#                   "highlight": {
#                     "background": "rgba(255,255,255,0.5)"
#                   }
#                 },
#                 "font": {
#                   "size": 18,
#                   "face": "Times New Roman",
#                   "bold": true
#                 },
#                 "scaling": {
#                   "label": {
#                     "min": 14,
#                     "max": 24
#                   }
#                 }
#               },
#               "edges": {
#                 "color": {
#                   "inherit": true
#                 },
#                 "smooth": false
#               },
#               "physics": {
#                 "barnesHut": {
#                   "springLength": 200
#                 },
#                 "minVelocity": 0.1,
#                 "solver": "barnesHut",
#                 "timestep": 1e-4,
#                 "adaptiveTimestep": true,
#                 "stabilization": {
#                   "enabled": true,
#                   "iterations": 10000,
#                   "fit": true
#                 }
#               }
#             }
#         """)
#
#         # 添加节点
#         for i, topic_weights in enumerate(weights):
#             top_keyword_indexes = topic_weights.argsort()[::-1][:10]
#             top_keywords = [feature_names[i] for i in top_keyword_indexes]
#             for keyword in top_keywords:
#                 G.add_node(keyword, color=colors[i])
#
#         # 添加边
#         for i, topic_weights in enumerate(weights):
#             top_keyword_indexes = topic_weights.argsort()[::-1][:10]
#             top_keywords = [feature_names[i] for i in top_keyword_indexes]
#             for j in range(len(top_keywords)):
#                 for k in range(j + 1, len(top_keywords)):
#                     G.add_edge(top_keywords[j], top_keywords[k])
#
#         # 显示共现图
#         G.show(f"Co-occurrence Network {year}.html")
