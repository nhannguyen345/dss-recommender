# import librairies
from flask import jsonify
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time 
import json

df = pd.read_csv('./data/netflix_titles.csv')

# convert columns "director, listed_in, cast and country" in columns that contain a real list
# the strip function is applied on the elements
# if the value is NaN, the new column contains a empty list []
df['directors'] = df['director'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
df['categories'] = df['listed_in'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
df['actors'] = df['cast'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
df['countries'] = df['country'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])

#Import Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans

# Build the tfidf matrix with the descriptions
start_time = time.time()
text_content = df['description']
vector = TfidfVectorizer(max_df=0.4,         # drop words that occur in more than X percent of documents
                             min_df=1,      # only use words that appear at least X times
                             stop_words='english', # remove stop words
                             lowercase=True, # Convert everything to lower case 
                             use_idf=True,   # Use idf
                             norm=u'l2',     # Normalization
                             smooth_idf=True # Prevents divide-by-zero errors
                            )

tfidf = vector.fit_transform(text_content)

# Clustering  Kmeans
k = 200
kmeans = MiniBatchKMeans(n_clusters = k)
kmeans.fit(tfidf)
centers = kmeans.cluster_centers_.argsort()[:,::-1]
terms = vector.get_feature_names_out()

request_transform = vector.transform(df['description'])
# new column cluster based on the description
df['cluster'] = kmeans.predict(request_transform) 
df['cluster'].value_counts().head()

# Find similar : get the top_n movies with description similar to the target description 
def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [index for index in related_docs_indices][0:top_n]  

G = nx.Graph(label="MOVIE")
start_time = time.time()
for i, rowi in df.iterrows():       
    G.add_node(rowi['title'],key=rowi['show_id'],label="MOVIE",mtype=rowi['type'],rating=rowi['rating'])
#    G.add_node(rowi['cluster'],label="CLUSTER")
#    G.add_edge(rowi['title'], rowi['cluster'], label="DESCRIPTION")
    for element in rowi['actors']:
        G.add_node(element,label="PERSON")
        G.add_edge(rowi['title'], element, label="ACTED_IN")
    for element in rowi['categories']:
        G.add_node(element,label="CAT")
        G.add_edge(rowi['title'], element, label="CAT_IN")
    for element in rowi['directors']:
        G.add_node(element,label="PERSON")
        G.add_edge(rowi['title'], element, label="DIRECTED")
    for element in rowi['countries']:
        G.add_node(element,label="COU")
        G.add_edge(rowi['title'], element, label="COU_IN")
    
    indices = find_similar(tfidf, i, top_n = 5)
    snode="Sim("+rowi['title'][:15].strip()+")"        
    G.add_node(snode,label="SIMILAR")
    G.add_edge(rowi['title'], snode, label="SIMILARITY")
    for element in indices:
        G.add_edge(snode, df['title'].loc[element], label="SIMILARITY")


def get_recommendation(root):
    commons_dict = {}
    for e in G.neighbors(root):
        for e2 in G.neighbors(e):
            if e2 == root:
                continue
            if G.nodes[e2]['label'] == "MOVIE":
                commons = commons_dict.get(e2)
                if commons is None:
                    commons_dict.update({e2: [e]})
                else:
                    commons.append(e)
                    commons_dict.update({e2: commons})

    movies = []
    weight = []
    for key, values in commons_dict.items():
        w = 0.0
        for e in values:
            w = w + 1 / math.log(G.degree(e))
        movies.append(key)
        weight.append(w)

    result = pd.Series(data=np.array(weight), index=movies)
    result.sort_values(inplace=True, ascending=False)

    # Chỉ lấy 10 kết quả có trọng số cao nhất bỏ qua kết quả đầu tiên vì nó là bộ phim được chọn
    top_10_results = result.iloc[1:11]

    return json.dumps(top_10_results.to_dict())





def get_title_by_show_id(show_id):
    # Tìm dòng trong DataFrame có show_id tương ứng
    movie_row = df[df['show_id'] == show_id]

    # Kiểm tra xem có tồn tại dòng nào hay không
    if not movie_row.empty:
        # Lấy giá trị title từ dòng tìm được
        title = movie_row['title'].values[0]
        return title
    else:
        return "Không tìm thấy bộ phim với show_id này."
    

def titleToFullInfo(title):
    # Kiểm tra xem tiêu đề có trong DataFrame không
    row = df[df['title'] == title]


    if not row.empty:
        # Chuyển đổi dữ liệu thành định dạng mong muốn
        result = {
           "show_id": int(row['show_id'].values[0]),
            "type": row['type'].values[0],
            "title": row['title'].values[0],
            "director": row['director'].values[0] if not pd.isna(row['director'].values[0]) else "",
            "cast": row['cast'].values[0].encode('ascii', 'ignore').decode('ascii') if not pd.isna(row['cast'].values[0]) else "",
            "country": row['country'].values[0] if not pd.isna(row['country'].values[0]) else "",
            "date_added": row['date_added'].values[0] if not pd.isna(row['date_added'].values[0]) else "",
            "release_year": int(row['release_year'].values[0]) if not pd.isna(row['release_year'].values[0]) else 0,
            "rating": row['rating'].values[0] if not pd.isna(row['rating'].values[0]) else "",
            "duration": row['duration'].values[0] if not pd.isna(row['duration'].values[0]) else "",
            "listed_in": row['listed_in'].values[0] if not pd.isna(row['listed_in'].values[0]) else "",
            "description": row['description'].values[0].encode('ascii', 'ignore').decode('ascii') if not pd.isna(row['description'].values[0]) else ""
        }
        return result
    else:
        return None

    

def convert_to_desired_format(input):
    data = json.loads(input)
    # Khởi tạo danh sách kết quả
    results = []

    # Lặp qua từng phần tử trong output
    for movie, weight in data.items():
        # Chuyển đổi thông tin từ data
        movie_info = titleToFullInfo(movie)

        if movie_info is not None:
            # Thêm trường weight vào thông tin đầy đủ
            movie_info["weight"] = weight
            movie_info["backdrop_path"]="https://nocodebuilding.com/wp-content/uploads/2021/07/xem-phim-titanic-thuyet-minh.jpg"
            movie_info["poster_path"]="https://upload.wikimedia.org/wikipedia/vi/a/ab/Titanic_3D_poster_Vietnam.jpg"
            # Thêm vào danh sách kết quả
            results.append(movie_info)

    # Tạo đối tượng kết quả cuối cùng
    final_result = {"page":1,"results": results}
    return final_result








