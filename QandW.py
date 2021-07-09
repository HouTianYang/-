import json
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from queue import PriorityQueue
from collections import defaultdict
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
# 读取文本，将文本中的问题与答案分别存入question和answer列表中
def read_file(file_path):
    with open(file_path) as f:
        datas = json.load(f)
        questions = []
        answers = []
        for data in datas['data']:
            for p in data['paragraphs']:
                for qa in p['qas']:
                    questions.append(qa['question'])
                    try:
                        answers.append(qa['answers'][0]['text'])
                    except IndexError:
                        questions.pop()
        assert len(questions) == len(answers)
        print(len(questions))
        print(questions[:10])
        print(answers[:10])
        return questions,answers
# 理解数据，将数据可视化
def data_comprehension(questions):
    word_counter = Counter()
    for text in questions:
        word_counter.update(text.strip(' .!?').split(' '))
    different_word = len(word_counter.keys())
    all_word = sum(word_counter.values())
    print('不同词汇出现个数：',different_word)
    print('所有词数：',all_word)

    word_value_sort = sorted(word_counter.values(),reverse=True)
    plt.plot(word_value_sort[:40])
    plt.show()
    dict1 = dict(zip(word_counter.values(),word_counter.keys()))
    print([[dict1[v],v]for v in word_value_sort[:40]])
# 文本预处理
# 停用词过滤、大写变小写、stemming
def word_pre_process(text):
    # 载入nltk停用词库，对停用词库进行操作
    stop_words_nltk = set(stopwords.words('english'))
    stop_words_nltk -= {'who', 'where', 'why', 'when', 'hoe', 'which'}
    # 去符号
    stop_words_nltk.update(['\'s', '``', '\'\''])
    stemming_fun = PorterStemmer()
    seg = list()
    # nltk做分词
    for word in word_tokenize(text):
        word = stemming_fun.stem(word.lower())
        word = '#number' if word.isdigit() else word
        if len(word)>1 and word not in stop_words_nltk:
            seg.append(word)
    return seg
def the_final_pre_process(questions):
    words_counter = Counter()
    questions_seg = []
    for text in questions:
        seg = word_pre_process(text)
        questions_seg.append(seg)
        words_counter.update(seg)
    value_sort = sorted(words_counter.values(),reverse=True)
    # 计算99%词频,用于滤除低频词
    min_tf = value_sort[int(math.exp(0.99*math.log(len(words_counter))))]
    for cur in range(len(questions_seg)):
        questions_seg[cur] = [word for word in questions_seg[cur] if words_counter[word]>min_tf]
    print('np.array(questions_seg).shape',np.array(questions_seg).shape)
    return questions_seg
# 矩阵稀疏度
def sparsity_ratio(X):
    return 1.0 - X.nnz / float(X.shape[0] * X.shape[1])
# 返回最佳的几个答案
def top5results(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
    2. 计算跟每个库里的问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    q_vector = vectorizer.transform([' '.join(word_pre_process(input_q))])
    # 计算余弦相似度，tf-idf默认使用l2范数；矩阵乘法
    sim = (X * q_vector.T).toarray()
    # 使用优先队列找出top5, 最先出来的是相似度小的
    pq = PriorityQueue()
    for cur in range(sim.shape[0]):
        pq.put((sim[cur][0], cur))
#         print(pq.queue)
        if len(pq.queue) > 5:
            pq.get()

    pq_rank = sorted(pq.queue, reverse=True, key=lambda x:x[0])
    top_idxs = [x[1] for x in pq_rank]  # top_idxs存放相似度最高的（存在qlist里的）问题的下表
                                        # hint: 利用priority queue来找出top results. 思考为什么可以这么做？

    return [answers[i] for i in top_idxs]
# 倒排表，降低计算量。首先构建倒排表，然后计算相似度
def daopaibiao(questions_seg):
    inverted_idx = defaultdict(set)  # 制定一个一个简单的倒排表
    for cur in range(len(questions_seg)):
        for word in questions_seg[cur]:
            inverted_idx[word].add(cur)
    return inverted_idx
def best_result_daopaibiao(input_q,inverted_idx):
    seg = word_pre_process(input_q)
    candidates = set()
    for word in seg:
        candidates = candidates|inverted_idx[word]
    candidates = list(candidates)
    q_vector = vectorizer.transform([' '.join(seg)])
    # 计算余弦相似度
    sim = (X[candidates] * q_vector.T).toarray()
    # 找到最优答案
    pq = PriorityQueue()
    for cur in range(sim.shape[0]):
        pq.put((sim[cur][0], candidates[cur]))
        if len(pq.queue) > 5:
            pq.get()
    pq_rank = sorted(pq.queue, reverse=True, key=lambda x: x[0])
    # print([x[0] for x in pq_rank])
    top_idxs = [x[1] for x in pq_rank]  # top_idxs存放相似度最高的（存在qlist里的）问题的下表

    return [answers[top_idxs[0]]]
# 基于词向量的文本表示
def doc2vec(seg):
    vector = np.zeros((1, 100))
    size = len(seg)
    for word in seg:
        try:
            vector += model.wv[word]
        except KeyError:
            size -= 1

    return vector / size


def best_answer_word2vec(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate
    2. 对于用户的输入 input_q，转换成句子向量
    3. 计算跟每个库里的问题之间的相似度
    4. 找出相似度最高的top5问题的答案
    """
    # 用词向量后用词形还原更合理，此处就不做变更了
    seg = word_pre_process(input_q)
    # 直接用上边建好的倒排表
    candidates = set()
    for word in seg:
        # 取所有包含任意一个词的文档的并集
        candidates = candidates | inverted_idx[word]
    candidates = list(candidates)

    q_vector = doc2vec(seg)
    # 计算问题向量的l2范数
    qnorm2 = np.linalg.norm(q_vector, axis=1, keepdims=True)
    q_vector = q_vector / qnorm2
    # 计算余弦相似度，前边已经l2规范化过，所以直接相乘

    sim = (X[candidates] @ q_vector.T)

    # 使用优先队列找出top5
    pq = PriorityQueue()
    for cur in range(sim.shape[0]):
        pq.put((sim[cur][0], candidates[cur]))
        if len(pq.queue) > 5:
            pq.get()

    pq_rank = sorted(pq.queue, reverse=True, key=lambda x: x[0])
    print([x[0] for x in pq_rank])
    top_idxs = [x[1] for x in pq_rank]  # top_idxs存放相似度最高的（存在qlist里的）问题的下表

    return [answers[top_idxs[0]] ]


if __name__ == "__main__":
    file_path = '../data/train-v2.0.json'
    # 读取数据
    questions,answers = read_file(file_path)
    # 词汇统计，可视化、作图
    data_comprehension(questions)
    # 预处理
    questions_seg = the_final_pre_process(questions)
    # 文本表示    tf-idf表示法表示文本
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([' '.join(seg) for seg in questions_seg])
    print(X.shape)
    print("input sparsity ratio:", sparsity_ratio(X))
    # 常规的问答检索，在所有问答上检索
    # TODO: 编写几个测试用例，并输出结果
    # print(top5results("When was the People's Republic of China founded?"))
    # print(top5results("What government blocked aid after Cyclone Nargis?"))
    # print(top5results("Which government stopped aid after Hurricane Nargis?"))
    # 倒排表优化
    # inverted_idx = daopaibiao(questions_seg)
    # print('the question is:Which airport was shut down?')
    # print(best_result_daopaibiao("Which airport was shut down?",inverted_idx))  # 在问题库中存在，经过对比，返回的首结果正确
    # print('the question is:Which airport is closed?')
    # print(best_result_daopaibiao("Which airport is closed?",inverted_idx))
    # print('the question is:What government blocked aid after Cyclone Nargis?')
    # print(best_result_daopaibiao("What government blocked aid after Cyclone Nargis?",inverted_idx))
    # print('the question is:Which government stopped aid after Hurricane Nargis?')
    # print(best_result_daopaibiao("Which government stopped aid after Hurricane Nargis?",inverted_idx))
    # 词向量表示
    _ = glove2word2vec('../data/glove.6B.100d.txt', '../data/glove2word2vec.6B.100d.txt')
    model = KeyedVectors.load_word2vec_format('../data/glove2word2vec.6B.100d.txt')
    X = np.zeros((len(questions_seg), 100))
    for cur in range(X.shape[0]):
        X[cur] = doc2vec(questions_seg[cur])
    inverted_idx = daopaibiao(questions_seg)
    # 计算X每一行的l2范数
    Xnorm2 = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / Xnorm2
    print(best_answer_word2vec("Which airport was shut down?"))  # 在问题库中存在，经过对比，返回的首结果正确
    print(best_answer_word2vec("Which airport is closed?"))
    print(best_answer_word2vec("What government blocked aid after Cyclone Nargis?"))  # 在问题库中存在，经过对比，返回的首结果正确
    print(best_answer_word2vec("Which government stopped aid after Hurricane Nargis?"))
