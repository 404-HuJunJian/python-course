import json
import socket
import threading

import openai
openai.api_key = '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'
messages = []

messages.append({'role': 'user', 'content': "你认识坤坤吗"})
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=messages,
)
content = response.choices[0].message['content']
print(content)

class GPTThreading(threading.Thread):
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

            messages.append({'role':'user','content':result})
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
            )
            content = response.choices[0].message['content']
            messages.append({'role':'assistant','content':content})
            self._socket.send(("%s"%content).encode(self._encoding))
        except Exception as identifier:
            self._socket.send("".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close()
        print("任务结束.....")
        pass

def messageFromJava():
    serversocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # 获取本地主机名称
    host = socket.gethostname()
    # 设置一个端口
    port = 10087
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
            t = GPTThreading(clientsocket)#为每一个请求开启一个处理线程
            t.start()
        except Exception as identifier:
            print(identifier)
    serversocket.close()

if __name__=='__main__':
    messageFromJava()


#上面的GPT-3.5-turbo是使用的GPT3.5模型 可以替换为其他模型如：
# gpt-4
# gpt-4-0314
# gpt-4-32k
# gpt-4-32k-0314
# system	设置chatgpt的角色。
# user	    消息是 给chatgpt提交的我们的问题。
# assistant	消息 是chatgpt给返回的消息。
