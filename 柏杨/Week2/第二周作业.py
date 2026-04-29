import torch 
import torch.nn as nn
import torch.optim as optim   #这是优化器
import numpy as np
import matplotlib.pyplot as plt

#因为上面这几个都是很常用的库。所以不管三七二十一请打上再说。
#torch张量计算、自动求导。
#numpy生成模拟数据
#torch.nn神经网络层、损失函数（？）
#一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。
#这是一个神经网络，每个神经网络要有一个单独的类。
class trainningdemo(nn.Module):
    #第一件事写初始化函数
    def __init__(self,inputsize1,hiddensize1,hiddensize2):
        super(trainningdemo,self).__init__()
        self.linear1 = nn.Linear(inputsize1,hiddensize1)
        self.linear2 = nn.Linear(hiddensize1,hiddensize2)
        self.loss = nn.functional.cross_entropy   #这样就可以规定用交叉熵当loss函数
    #第二件事规定前面两个线性层的组合顺序。注意对位。
    def forward(self,x,y=None):
        x = self.linear1(x)
        x = torch.relu(x)
        #先输入一个x，然后激活一下（使其非线性）
        y_pred = self.linear2(x)
        #然后处理第一层的非线性之后得到的结果，作为第二层的输出。
        if y is not None:
            return self.loss(y_pred,y)   #如果y不为空那就是有答案，则算损失
        else :
            return torch.softmax(y_pred,dim=1)   #如果y为空那就返回预测值
        #老师说的也可以不这样写是啥意思？不管了
#反正一个神经网络的类就这么定义完了，所以下面的其他功能要顶格，作为其他独立的组件。
#forward就是我告诉你你拿到我的数据之后应该干点什么事儿？

#生成一个样本
def create_elem():    
    x = np.random.random(5)   #就假设这是一个5维向量
    max_index = np.argmax(x)   #然后argmax可以用来返回这个向量里面最大的元素下标。
    return x,max_index       #返回向量、标签
    
#生成一堆样本，建立数据集
def create_dataset(tutal_num):   #需要样本总数
    X=[]
    Y=[]    #这可能是两个大矩阵,错误的，这是两个大列表。
    for i in range(tutal_num):
        x,y = create_elem()
        X.append(x)
        Y.append(y)     #为什么x要转成浮点型？，y要转成整型（因为Y是标签）
        #还有tensor是张量。虽然不知道为啥叫张量。我初中年级主任就叫张亮and他现在当校长了。emmmm这个tensor是啥东西。
        #张量是一个多维数组，可以是标量、向量、矩阵或更高维度的数据结构。
    return torch.FloatTensor(X), torch.LongTensor(Y)

#好的应该到了这一步我们已经有数据集了？？？能不能输出一下。
#n = 10
#print(build_dataset(n))
#okok这一步为止没问题！！烧个香~欸嘿，生成训练数据之后，就该开始处理数据了吧？

#这个用在每一轮训练结束之后测试结果。
def 狗(model):    #不管了，我这个模型就叫狗。
    #pytorch有个平菇模式，比较严格，我们用它做测试。
    model.eval()
    max_num = 100
    X,Y = create_dataset(max_num)    #我们就是创建了一个100规模的数据集。所以凭啥不大写啊！就大写！
    correct,wrong= 0,0    #初始化分类正确和错误的数量都是0
    with torch.no_grad():
        y_pred = model(X)      #把 X 塞进待测试的模型，让它预测，输出存进 y_pred
        for y_p,y_t in zip(y_pred,Y):   #zip让每一个预测值和真实值都结对
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print(f"预测正确个数：{correct}，正确率{correct/(wrong + correct):.2f}")
    return correct/(correct+wrong)
#ok这是一个完整的预测过程…

#主程序
#在主函数里面设置训练轮数，规模，样本总数等超参数
def main():
    epoch = 25   #轮数：30
    batch_size = 20    #一批样本20个
    train_sum = 800  #总共1000个样本
    input_size = 5    #5维向量
    hidden1= 10
    hidden2 = 5
    lr = 0.05   #设置步长为0.01    
    model = trainningdemo(input_size,hidden1,hidden2)
    #好啊，他adam我也adam！
    optim = torch.optim.Adam(model.parameters(),lr =lr)   #选择优化器就行了。和选择loss函数差不多。
    log= []    #这是自己给自己建了一个日志
    train_X,train_Y = create_dataset(train_sum)   #开始生成数据集了
    #开始做吧
    for e in range(epoch):   #开始定义每一轮
        model.train()    #不评估了，训练吧
        record_loss = []   #记录每一轮的损失函数值
        for i in range(train_sum//batch_size):   #开始规定分几批训练
            #可以直接用下标来切分每一次训练哪部分数据
            x_train = train_X[i*batch_size:(i+1)*batch_size]
            y_train = train_Y[i*batch_size:(i+1)*batch_size]
            loss = model(x_train,y_train)   #正向训练
            loss.backward()   #反向传播
            optim.step()     #更新权重
            optim.zero_grad()  #每一批训练完了都要归零权重要不然累加会很恶心
            record_loss.append(loss.item())
            #loss是return的MultiClassficationModel值，
            #它最后return的是loss或是预测值。
            #因为有Y.append(y)所以标签也就是答案不为空，
            #所以return的是一个loss值，它是tensor，但为了运算，
            #要.item一下把它变成普通的数，加进记录单轮的record_loss本本里。
        #np.mean是求平均数用的。
        loss_avg = np.mean(record_loss)   #求30批的平均损失
        print(f"第{epoch}轮训练平均loss：{loss_avg}")    #打个印吧
        point = 狗(model=model)    #测试本轮得分，就是模拟考试啦！那个15%
        log.append([point,float(loss_avg)])    #记在log里面。
    #好的现在25轮过去了，你训练完了。    
    torch.save(model.state_dict(), "model.pt")    #存个档吧。
    #torch.save()和后面的torch.load()是对应的
    #打印你存的log，再重复一次log是用来记每一轮的得分和损失率的列表。
    print(log)
    # 画图代码，这是很好的，我要了。    
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return
#好的main函数也写完了。
#main是什么，是学习和测验的过程。但是最后还要上考场。
#现在我们来规定考试流程
def predict_exam(modelpath,examvecs):    #这里面放的俩参数是啥我没搞懂。
    #ok第一个是之前存的模型权重文件在哪。
    inputsize1,hiddensize1,hiddensize2=5,10,5    
    #我是高贵的二层神经网络，所以我需要规定5x10矩阵和10x5矩阵最后输出还是回到了5维
    model = trainningdemo(inputsize1,hiddensize1,hiddensize2)   #规定用trainningdemo解题
    model.load_state_dict(torch.load(modelpath))   #把训练好的权重读出来，准备开考
    print(model.state_dict())    #看看成功读出来了没有。
    model.eval()   
    with torch.no_grad():
        output= model.forward(torch.FloatTensor(examvecs))   #把向量表转换成真正的张量然后用forward运算输出结果
        for i,y in zip(examvecs,output):   #打包每一个向量和对应的输出值
            print(f"向量：{i},预测类别：{torch.argmax(y)},预测值：{y}")   #同样的forward函数，因为考试没有答案，所以输出预测值，不算损失了。


#最后真实地跑一跑吧
if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.4963533, 0.5524256, 0.95758807, 0.65520434, 0.84890681],
                [0.48797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.49349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict_exam("model.pt",test_vec)
