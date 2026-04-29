import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random


chars = "阿巴阿巴测试数据火锅烧烤麻辣烫吃喝玩乐我好喜欢李晓波啊我要和她在一起！" 
vocab = {'<PAD>': 0, '你': 1} 
for c in chars:
    if c not in vocab:
        vocab[c] = len(vocab)

# 生成一个样本（造句：5个字，包含1个你）
def create_elem():
    other_chars = random.choices(list(vocab.keys())[2:], k=4)
    # 随机决定“你”出现的位置 (0, 1, 2, 3, 4)
    index = random.randint(0, 4)
    # 把“你”塞进刚才随机的位置
    other_chars.insert(index, '你')
    text = "".join(other_chars)
    
    # 查字典，把汉字变成ID列表
    x = [vocab[c] for c in text]
    # index刚好就是我们要预测的类别（第几位就是第几类）
    return x, index 

# 生成数据集
def create_dataset(total_num):
    X = []
    Y = []
    for i in range(total_num):
        x, y = create_elem()
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)

class trainningdemo(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super(trainningdemo, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        # x 进入时的 shape: (batch_size, 5)
        x = self.embedding(x)     # 变成 (batch_size, 5, embed_dim)
        
        # rnn_out 是每一步的输出，hidden 是读完一整句话后的最终记忆
        rnn_out, hidden = self.rnn(x) 
        
        x = hidden.squeeze(0)     # 提取最后一步记忆：shape (batch_size, hidden_size)
        y_pred = self.linear(x)   # 打分预测：shape (batch_size, 5)
        
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred # 预测时不加softmax也行，因为后面直接用argmax取最大值

def 狗(model):
    model.eval()
    X, Y = create_dataset(100) # 模拟考试发100张卷子
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(X)
        for y_p, y_t in zip(y_pred, Y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print(f"模拟考试 -> 正确个数：{correct}，正确率：{correct/(wrong + correct):.2f}")
    return correct / (correct + wrong)

def main():
    epoch = 15      
    batch_size = 20
    train_sum = 1000
    
    # 新的超参数！
    vocab_size = len(vocab)
    embed_dim = 16   
    hidden_size = 32 # RNN容量
    num_classes = 5  # 因为位置是0到4，共5个类别
    lr = 0.005       

    model = trainningdemo(vocab_size, embed_dim, hidden_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    
    train_X, train_Y = create_dataset(train_sum)

    for e in range(epoch):
        model.train()
        record_loss = []
        for i in range(train_sum // batch_size):
            x_train = train_X[i*batch_size:(i+1)*batch_size]
            y_train = train_Y[i*batch_size:(i+1)*batch_size]
            loss = model(x_train, y_train)
            loss.backward()
            optim.step()
            optim.zero_grad()
            record_loss.append(loss.item())

        loss_avg = np.mean(record_loss)
        # 强迫症修复：从第1轮开始打印
        print(f"第{e+1}轮训练平均loss：{loss_avg:.4f}") 
        point = 狗(model=model)
        log.append([point, float(loss_avg)])

    # 存盘
    torch.save(model.state_dict(), "model_rnn.pt")
    #画图狂魔又开始画图了
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

def predict_exam(modelpath, test_texts):
    vocab_size, embed_dim, hidden_size, num_classes = len(vocab), 16, 32, 5
    model = trainningdemo(vocab_size, embed_dim, hidden_size, num_classes)
    model.load_state_dict(torch.load(modelpath, weights_only=True))
    model.eval()
    
    # 把汉字变成ID
    encoded_texts = [[vocab.get(c, 0) for c in text] for text in test_texts]
    X = torch.LongTensor(encoded_texts)
    
    with torch.no_grad():
        output = model(X)
        for i, (text, y) in enumerate(zip(test_texts, output)):
            print(f"原句：{text} -> 预测'你'在第几位：{torch.argmax(y).item()}")

if __name__ == "__main__":
    main()
    
    # 最后真实地跑一跑
    print("\n======= 最终测试 ========")
    test_vec = [
        "你爱不爱我", 
        "我你烤麻辣", 
        "测试你数据", 
        "吃喝玩你乐", 
        "巴阿巴阿你",
        "你是笨蛋吗"  
    ]
    predict_exam("model_rnn.pt", test_vec)
