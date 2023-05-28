import math
import socket
import threading
import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
# import wandb


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 123456,
    'ratio': 0.8,
    'epochs': 5000,
    'batch_size': 256,
    'lr': 1e-5,
    'save_path': 'model.ckpt',
    'save_path2': 'model2_local.ckpt',
    'early_stop': 50,
    'best_loss': math.inf
}
# wandb.init(project='tianqi',name='test1',config=config)


class MyDataset(Dataset):

    def __init__(self, x, y):
        if y is None:
            self.y = None
        else:
            self.y = torch.FloatTensor(y)

        self.x = torch.FloatTensor(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]


class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            # nn.Linear(input_dim, 256),
            # nn.Dropout(0.2),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Linear(256, 1024),
            # nn.Dropout(0.2),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.Dropout(0.2),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Linear(256, output_dim)
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Dropout(0.2),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.Dropout(0.2),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.2),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self,x):
        y = self.net(x)
        return y


class ServerThreading(threading.Thread):
    # words = text2vec.load_lexicon()
    def __init__(self,clientsocket,recvsize=1024*1024,encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        pass

    def run(self):
        print("开启线程.....")
        try:
            #接受数据
            msg = ''
            while True:
                # 读取recvsize个字节
                rec = self._socket.recv(self._recvsize)
                # 解码
                msg += rec.decode(self._encoding)
                # 文本接受是否完毕，因为python socket不能自己判断接收数据是否完毕，
                # 所以需要自定义协议标志数据接受完毕
                if msg.strip().endswith('over'):
                    msg=msg[:-4]
                    break
            # 解析json格式的数据 json转dict格式 如果是list的json 会转list的dict
            result = json.loads(msg)
            result = [result]*7
            data = pd.DataFrame(columns = result[0].keys())
            for info in result:
                index = info.keys()
                data2 = info.values()
                row = pd.Series(data=data2, index=index)
                data = pd.concat([data.T,row],axis=1, ignore_index=True).T
            # 调用神经网络模型处理请求
            # data = data.sort_values(by="creattime")
            # data = data.drop(columns=["id","creattime"])
            data = torch.FloatTensor(data[["weather","temperature","winddirection","windpower","humidity"]].values)
            data = torch.flatten(data)
            data = torch.unsqueeze(data,0)
            pred = test2("./model2_layer.ckpt",data)
            weather = pred[:,:11].softmax(dim=1).argmax(dim=1)
            temperature = pred[:,11]
            winddirection = pred[:,12:20].softmax(dim=1).argmax(dim=1)
            windpower = pred[:,20]
            humidity = pred[:,21]
            sendmsg = "{{\"weather\":{:5.1f},\"temperature\":{:5.1f},\"winddirection\":{:5.1f},\"windpower\":{:5.1f},\"humidity\":{:5.1f}}}".format(weather[0],temperature[0],winddirection[0],windpower[0],humidity[0])
            self._socket.send(("%s"%sendmsg).encode(self._encoding))
            pass
        except Exception as identifier:
            self._socket.send("".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close()
        print("任务结束.....")
        pass

    def __del__(self):

        pass


# path:文件路径 ratio:训练比例 seed:随机分割种子
def DataProcessSplit(path, ratio, seed):
    data = pd.read_csv(path)
    train_rows = int(len(data)*ratio)
    valid_rows = len(data)-train_rows
    x = data.iloc[:, :-5].values
    y = data.iloc[:, -5:].values
    dataset = MyDataset(x, y)
    train_set, valid_set = random_split(dataset,[train_rows,valid_rows],torch.Generator().manual_seed(seed))
    return train_set, valid_set


def test1(path, data):
    model = MyModel(35,5)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    pred = model(data)
    return pred
    # print("pred:",pred,end="\n")


def test2(path, data):
    model = MyModel(35,22)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    pred = model(data)
    return pred

def testFromJava():
    serversocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # 获取本地主机名称
    host = socket.gethostname()
    # 设置一个端口
    port = 10086
    # 将套接字与本地主机和端口绑定
    serversocket.bind((host,port))
    # 设置监听最大连接数
    serversocket.listen(5)
    # 获取本地服务器的连接信息
    myaddr = serversocket.getsockname()
    print("服务器地址:%s"%str(myaddr))
    # 循环等待接受客户端信息
    while True:
        # 获取一个客户端连接
        clientsocket,addr = serversocket.accept()
        print("连接地址:%s" % str(addr))
        try:
            t = ServerThreading(clientsocket)#为每一个请求开启一个处理线程
            t.start()
        except Exception as identifier:
            print(identifier)
    serversocket.close()


def train1(model, train_loader, valid_loader, config):
    best_loss = math.inf
    early_stop = 0
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.RAdam(model.parameters(), lr=config['lr'], weight_decay= 0.001)
    for epoch in range(config['epochs']):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loader)
        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix({'loss': loss.detach().item()})
            loss_record.append(loss.detach().item())
            # wandb.log({"train/loss":loss.item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        tqdm.write("epoch:"+str(epoch)+" train_loss:"+str(mean_train_loss))
        model.eval()
        loss_record = []
        valid_pbar = tqdm(valid_loader)
        for x, y in valid_pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = loss_fn(pred, y)
            loss_record.append(loss.item())
            valid_pbar.set_postfix({'loss': loss.item()})
            # wandb.log({"valid/loss":loss.item()})

        mean_valid_loss = sum(loss_record)/len(loss_record)
        tqdm.write("epoch:"+str(epoch)+" valid_loss:"+str(mean_valid_loss))
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            early_stop = 0
        else:
            early_stop += 1

        if early_stop >= config['early_stop']:
            print("不再优化")
            return

# weather 11 winddriection 8
def train2(model, train_loader, valid_loader, config):
    best_loss = math.inf
    early_stop = 0
    loss_fn1 = torch.nn.MSELoss(reduction='mean')
    loss_fn2 = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=config['lr'], weight_decay= 0.001)
    for epoch in range(config['epochs']):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loader)
        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss1 = loss_fn1(pred[:,11], y[:,1])+loss_fn1(pred[:,20:],y[:,3:])
            loss2 = loss_fn2(pred[:,:11],y[:,0].long())+loss_fn2(pred[:,12:20],y[:,2].long())
            loss = loss1+100*loss2
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix({'loss': loss.detach().item()})
            loss_record.append(loss.detach().item())
            # wandb.log({"train/loss":loss.item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        tqdm.write("epoch:"+str(epoch)+" train_loss:"+str(mean_train_loss))
        model.eval()
        loss_record = []
        valid_pbar = tqdm(valid_loader)
        for x, y in valid_pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss1 = loss_fn1(pred[:,11], y[:,1])+loss_fn1(pred[:,20:],y[:,3:])
                loss2 = loss_fn2(pred[:,:11],y[:,0].long())+loss_fn2(pred[:,12:20],y[:,2].long())
                loss = loss1+loss2
            loss_record.append(loss.item())
            valid_pbar.set_postfix({'loss': loss.item()})
            # wandb.log({"valid/loss":loss.item()})

        mean_valid_loss = sum(loss_record)/len(loss_record)
        tqdm.write("epoch:"+str(epoch)+" valid_loss:"+str(mean_valid_loss))
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path2'])
            early_stop = 0
        else:
            early_stop += 1

        if early_stop >= config['early_stop']:
            print("不再优化")
            return

# weather(11)	temperature	winddirection(8)	windpower	humidity
if __name__== '__main__':
    # train_set, valid_set = DataProcessSplit("./new2.csv", config['ratio'], config['seed'])
    # train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    # valid_loader = DataLoader(valid_set, batch_size=config['batch_size'])
    # model = MyModel(35, 5)
    # train1(model,train_loader,valid_loader,config)
    # data = pd.read_csv("./new2.csv")
    # x = data.iloc[1125:1200,:-5].values
    # y = data.iloc[1125:1200,-5:].values
    # test1("./model.ckpt",torch.FloatTensor(x))
    # print("y",y,end="\n")
    # testFromJava()


    # train_set, valid_set = DataProcessSplit("./new2.csv", config['ratio'], config['seed'])
    # train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    # valid_loader = DataLoader(valid_set, batch_size=config['batch_size'])
    # model = MyModel(35, 22)
    # train2(model,train_loader,valid_loader,config)

    # data = pd.read_csv("./new2.csv")
    # x = data.iloc[:100,:-5].values
    # y = data.iloc[:100,-5:].values
    # pred = test2("./model2_layer.ckpt",torch.FloatTensor(x))
    # weather = pred[:,:11].softmax(dim=1).argmax(dim=1)
    # temperature = pred[:,11]
    # winddirection = pred[:,12:20].softmax(dim=1).argmax(dim=1)
    # windpower = pred[:,20]
    # humidity = pred[:,21]
    # print("y",y,end="\n")

    testFromJava()













#
#
# a = MyDataset("./new.csv")
# loader = DataLoader(a)
# for x,y in loader:
#     print(x,"\n------------\n",y)


