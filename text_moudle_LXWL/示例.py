from run_cnn import name2subcategory

a = name2subcategory()
while True:
    #name = input('请输入产品名称:')
    name_list = ['世达（star）SB224F 4号学生青少年儿童成人用球手缝耐磨防水比赛训练足球']
    category = a.namelyst_predict(name_list)
    print(category[0])
