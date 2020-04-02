from entity.label import LabesLoader

# 通过加载老数据，进行新格式数据生成并保存
LabesLoader('../data/I/label.txt', '../data/I', '../cut/I').save(split=True, fill=False)
LabesLoader('../data/II/label.txt', '../data/II', '../cut/II').save(split=True, fill=False)
LabesLoader('../data/I/label.txt', '../data/I', '../non/I').unique().save(split=False, fill=True)
LabesLoader('../data/II/label.txt', '../data/II', '../non/II').unique().save(split=False, fill=True)
