#! /usr/bin/python
# -*-encoding:utf-8-*-
from gensim.models import word2vec
import os, sys, io
import gensim


#
# @brief 对初始语料进行分词处理，作为训练模型的语料
# @param filename 语料输入的文件名
#
def cut_file_content_text(filename):
    import jieba

    try:
        fi = io.open(filename, 'r', encoding='utf-8')
    except BaseException as e:  # 因BaseException是所有错误的基类，用它可以获得所有错误类型
        print(Exception, ":", e)  # 追踪错误详细信息
        raise BaseException(e)

    text = fi.read()  # 获取文本内容
    new_text = jieba.cut(text, cut_all=False)  # 精确模式
    str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')  # 去掉标点符号

    cut_file = filename + '_cut.txt'
    fo = io.open(cut_file, 'w', encoding='utf-8')
    fo.write(str_out)

    return cut_file


#
# @brief 训练模型
# @param model_file_name为训练语料的路径
# @param save_model为保存模型名
#
def model_train(train_file_name, save_model_file):
    # 模型训练，生成词向量
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    model = gensim.models.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5
    model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_file + ".bin", binary=True)  # 以二进制类型保存模型以便重用


#
# @brief 主函数入口
#
if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')

    source_file_name = u'xiaoshuo.txt'
    cut_file = cut_file_content_text(source_file_name)  # 须注意文件必须先另存为utf-8编码格式

    save_model_name = u'xiaoshuo.model'
    if not os.path.exists(save_model_name):  # 判断文件是否存在
        # 进行模型训练
        model_train(cut_file, save_model_name)
    else:
        print(u'此训练模型已经存在，不用再次训练')

    # 加载已训练好的模型
    model_1 = word2vec.Word2Vec.load(save_model_name)

    # 查看词典内容
    for key, val in model_1.wv.vocab.items():
        print '%s=>%s' % (key, val)

    # 查询某一单词对应的向量
    print(model_1[u'白坛主'])

    # 应用1： 计算两个词的相似度/相关程度
    y1 = model_1.similarity(u'赵敏', u'韦一笑')
    print(u"赵敏和韦一笑的相似度为：%f" % y1)
    print("-------------------------------\n")

    # 应用2：计算某个词的相关词列表
    y2 = model_1.most_similar(u"张三丰", topn=10)  # 10个最相关的
    print(u"和张三丰最相关的词有：\n")
    for item in y2:
        print "%s:%f" % (item[0], item[1])
    print("-------------------------------\n")

    # 应用3：查找异类词
    word_list = [u'陈友谅', u'方位', u'郭姑娘', u'张三丰']
    y3 = model_1.doesnt_match(word_list)
    print(u"在[%s]中的异类词为：%s" % (' '.join(word_list), y3))
    print("-------------------------------\n")
