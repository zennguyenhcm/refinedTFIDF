#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pyvi
from pyvi import ViTokenizer
import numpy as np
import pandas as pd
import stop_words
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import json


# # Hàm sử dụng để chuẩn hoá chuỗi

# In[12]:


def standarlize_duplicate_token(tokens):
    length = len(tokens)
    for i in range(0, length - 2):
        for j in range(i + 1, length - 1):
            if tokens[i].lower() == tokens[j].lower():
                tokens[j] = tokens[i]
    return tokens


def remove_links(text):
    text = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '',
                  text, flags=re.MULTILINE)
    text = re.sub(r'[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '', text,
                  flags=re.MULTILINE)
    return text


def remove_img(text):
    text = re.sub(r'img_[0-9a-fA-F-]+', '', text, flags=re.MULTILINE)
    return text


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# # Hàm tách từ

# In[13]:


def vi_term_tokenize(text):
    tokens = []
    text = remove_html_tags(text)

    terms = ViTokenizer.tokenize(text)
    for term in terms.split(" "):
        if (term.lower() not in stop_words.STOP_WORDS):
            if ("_" in term) or (term.isalpha() == True) and (len(term) >= 3):
                tokens.append(term)
    tokens = standarlize_duplicate_token(tokens)
    return tokens


# # Lấy dữ liệu trực tuyến từ trang cafebiz.vn

# In[14]:


import requests
import os
from datetime import date, timedelta, datetime
import datetime as dt
from bs4 import BeautifulSoup


# In[15]:


def unix_time_to_str(timestamp):
    return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')


# In[26]:


def get_link_like_share_count(link):
    url = "https://sharefb.cnnd.vn/?urls=" + link
    payload = {}
    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Origin': 'https://cafebiz.vn'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    return json.loads(response.text)


# In[27]:


def get_link_view(news_id):
    url = "https://cmsanalytics.admicro.vn/cms/cafebiz.vn/item/report?news_ids=[" + news_id + "]&cmskey=97492f372c754b059190066f7a75e093&fbclid=IwAR3MEPSL4ubz594m3YwFZU706BIY9vS4RD_AUsFwdK6mTyq4FEhi9-FTDkY"
    payload = {}
    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Origin': 'https://cmsanalytics.admicro.vn/'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    return json.loads(response.text)


# In[28]:


def get_link_content(link):
    try:
        request = requests.get(link)
    except:
        pass

    request_html = BeautifulSoup(request.content, "html.parser")

    description = request_html.find("h2", attrs={"class": "sapo"}).get_text().strip()

    content = [x.get_text().strip() for x in request_html.find("div", attrs={"class": "detail-content"}).find_all('p')]

    tags = [x.text.strip() for x in request_html.find("span", {"class": "tags-item"}).find_all('a')]

    return {
        "description": description,
        "content": " ".join(content),
        "tags": ";".join(tags)
    }


# In[29]:


def get_100_news_post():
    url = 'https://nspapi.aiservice.vn/request/client?guid=1509518141984318107&domain=CafeBiz&boxid=4&url=http://cafebiz.vn&numnews=100'
    response = requests.get(url)
    return json.loads(response.text.replace('CafeBiz_Box_4=', ''))


# In[30]:


_file_open_mod = 'w'
_header = True

_data_file_path = os.path.join(os.curdir, 'data.csv')
print("Đang lấy dữ liệu ...")

news_data = pd.DataFrame(
    columns=['id', 'url', 'title', 'description', 'content', 'tags', 'crawl_date', 'public_date', 'page_view',
             'fb_like', 'fb_share', 'fb_comment', 'fb_total'])

news_posts = get_100_news_post()

for post in news_posts['listnews']:
    fb = get_link_like_share_count(post['url'])
    view = get_link_view(post['id'])
    content = get_link_content(post['url'])
    news_data.loc[len(news_data)] = [post['id'], post['url'], post['title'], content['description'], content['content'],
                                     content['tags'], datetime.now().strftime("%d-%m-%Y  %H:%M:%S"),
                                     unix_time_to_str(post['publishDate']),
                                     view[0]['total_view']['view_pc'] + view[0]['total_view']['view_mob'],
                                     fb[0]['like_count'], fb[0]['share_count'], fb[0]['comment_count'],
                                     fb[0]['total_count']];
print("Ghi tập tin ...")
with open(_data_file_path, mode=_file_open_mod, encoding="utf-8") as data_file:
    news_data.to_csv(data_file, header=_header, index=False, line_terminator='\n')
print("Done cralw !")

# # Đọc dữ liệu đã lấy về vào Frame dữ liệu

# In[31]:


data = pd.read_csv('data.csv')

# In[32]:


filter = data.content.str.contains("Tết")
data[filter]


# In[33]:


def compute_tf(word_dict, l):
    tf = {}
    sum_nk = len(l)
    for word, count in word_dict.items():
        tf[word] = count / sum_nk
    return tf


# In[34]:


import string
from math import log


def compute_idf(docs_vector):
    n = len(docs_vector)
    idf = dict.fromkeys(docs_vector[0].keys(), 0)
    for document in docs_vector:
        for word, count in document.items():
            if count > 0:
                idf[word] += 1

    for word, v in idf.items():
        idf[word] = log(n / 1 + float(v))
    return idf


# In[35]:


def compute_tf_idf(tf, idf):
    tf_idf = dict.fromkeys(tf.keys(), 0)
    for word, v in tf.items():
        tf_idf[word] = v * idf[word]
    return tf_idf


# # Tính độ lệch chuẩn theo paper

# In[41]:


num_of_document = len(data)
_CONTENT_HEADER = 'description'
_PUBLIC_DATE_HEADER = 'public_date'

copus = data[_CONTENT_HEADER].head(num_of_document)

all_document = copus.values.tolist()

cv = CountVectorizer(all_document, tokenizer=vi_term_tokenize, lowercase=False)
word_count_vector = cv.fit_transform(all_document)

idf_list = []

data['time'] = pd.to_datetime(data[_PUBLIC_DATE_HEADER], format="%Y-%m-%d %H:%M:%S")

input_data = data.head(num_of_document).set_index('time')

for _, document_group in input_data.groupby(pd.Grouper(freq='2D')):

    document_group_word_dict = []
    all_features = cv.get_feature_names()

    document_group_content = document_group[_CONTENT_HEADER].values.tolist()
    for document in document_group_content:
        document_group_word_dict.append(dict.fromkeys(all_features, 0))

    count = 0
    for document_token_vector in document_group_word_dict:
        document_tokens = vi_term_tokenize(document_group_content[count])

        for token in document_tokens:
            if token in document_token_vector:
                document_token_vector[token] += 1
        count = count + 1

    idf = compute_idf(document_group_word_dict)
    idf_list.append(idf)
print("Số nhóm : ", len(idf_list))
ajust_time = {}

for token in cv.get_feature_names():
    idf_vals = []
    for idf_dict in idf_list:
        idf_vals.append(idf_dict[token])

    SD = np.std(idf_vals, dtype=np.float64, ddof=0)  # N - 0,

    mean = np.mean(idf_vals)

    above = [x for x in idf_vals if x > mean]

    SD_Above = np.std(above, dtype=np.float64, ddof=0)

    below = [x for x in idf_vals if x < mean]

    SD_Below = np.std(below, dtype=np.float64, ddof=0)

    ajust_time[token] = SD  # (SD_Above + 0.001) / (SD_Below + 0.001)

# # In ra danh sách từ khoá sau khi sắp xếp để kiểm tra

# In[42]:


import operator

sorted_x = sorted(ajust_time.items(), key=operator.itemgetter(1), reverse=True)
for key, value in sorted_x:
    print(key, '=>', value)

# In[ ]:


# In[ ]:



