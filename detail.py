# Nhập các thư viện cần thiết
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time 
import json

# Đọc dữ liệu từ file CSV
df = pd.read_csv('./data/netflix_titles.csv')

# Chuyển đổi các cột "director, listed_in, cast, và country" thành các cột chứa danh sách thực sự
# Hàm strip được áp dụng cho các phần tử
# Nếu giá trị là NaN, cột mới chứa một danh sách trống []
df['directors'] = df['director'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
df['categories'] = df['listed_in'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
df['actors'] = df['cast'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
df['countries'] = df['country'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])

# Import các thư viện
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans

# Xây dựng ma trận TF-IDF với mô tả
start_time = time.time()
text_content = df['description']
vector = TfidfVectorizer(max_df=0.4,
                         min_df=1,
                         stop_words='english',
                         lowercase=True,
                         use_idf=True,
                         norm=u'l2',
                         smooth_idf=True
                        )

tfidf = vector.fit_transform(text_content)

# Clustering Kmeans
k = 200
kmeans = MiniBatchKMeans(n_clusters=k)
kmeans.fit(tfidf)
centers = kmeans.cluster_centers_.argsort()[:,::-1]
terms = vector.get_feature_names_out()

request_transform = vector.transform(df['description'])
# Thêm cột mới 'cluster' dựa trên mô tả
df['cluster'] = kmeans.predict(request_transform) 
df['cluster'].value_counts().head()

# Tìm các bộ phim tương tự
def find_similar(tfidf_matrix, index, top_n=5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [index for index in related_docs_indices][0:top_n]  

# Tạo đồ thị
G = nx.Graph(label="MOVIE")
start_time = time.time()
for i, rowi in df.iterrows():       
    G.add_node(rowi['title'], key=rowi['show_id'], label="MOVIE", mtype=rowi['type'], rating=rowi['rating'])
    for element in rowi['actors']:
        G.add_node(element, label="PERSON")
        G.add_edge(rowi['title'], element, label="ACTED_IN")
    for element in rowi['categories']:
        G.add_node(element, label="CAT")
        G.add_edge(rowi['title'], element, label="CAT_IN")
    for element in rowi['directors']:
        G.add_node(element, label="PERSON")
        G.add_edge(rowi['title'], element, label="DIRECTED")
    for element in rowi['countries']:
        G.add_node(element, label="COU")
        G.add_edge(rowi['title'], element, label="COU_IN")
    
    indices = find_similar(tfidf, i, top_n=5)
    snode = "Sim("+rowi['title'][:15].strip()+")"        
    G.add_node(snode, label="SIMILAR")
    G.add_edge(rowi['title'], snode, label="SIMILARITY")
    for element in indices:
        G.add_edge(snode, df['title'].loc[element], label="SIMILARITY")
