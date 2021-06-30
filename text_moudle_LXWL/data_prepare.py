import pandas as pd
from function import Index

table = pd.read_excel('predict_check_data.xlsx')
#category_set = list(set(table['categoryCodeName'].tolist()))
#len(list(category_set))
f = open('/home/fyw/PycharmProjects/xifenlei/venv/textcnn/text-classification-cnn-master/data/name2category/name2category.val.txt','w',encoding='utf-8')
g = open('/home/fyw/PycharmProjects/xifenlei/venv/textcnn/text-classification-cnn-master/data/name2category/name2category.train.txt','w',encoding='utf-8')
h = open('/home/fyw/PycharmProjects/xifenlei/venv/textcnn/text-classification-cnn-master/data/name2category/name2category.test.txt','w',encoding='utf-8')
m = 0
len_table = len(table)
index = Index()
for category,name in zip(table['name'],table['skuname']):
    name = str(name).replace('\t',' ')
    category = str(category)
    if '错误子类' not in category:
        if m%13 == 1:
            f.write(format(str(category)+'\t'+str(name)+'\n'))
        elif m%13 in [2,3]:
            h.write(format(str(category)+'\t'+str(name)+'\n'))
        else:
            g.write(format(str(category)+'\t'+str(name)+'\n'))
    m += 1
    s = m/len_table
    print(index(m, len_table-1),end = '%')

