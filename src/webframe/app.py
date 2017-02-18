#coding:utf-8
'''
需要安装web.py

Requirements：
    web.py library , if you don't have this library , please run " sudo pip install web.py " firstly
'''
import os
import web
import hashlib
from lxml import etree
import time
import WeixinInterface as weixin

'''
定义URL的结构

define the structure of urls
'''
urls = (
    '/', 'index'
)
class index:
    def __init__(self):
        self.app_root = os.path.dirname(__file__)
        self.templates_root = os.path.join(self.app_root, 'templates')
        self.render = web.template.render(self.templates_root)

    '''
        get方式为验证微信登录用的
    '''
    def GET(self):
        print ' weixin'
        # 获取输入参数
        data = web.input()
        signature = data.signature
        timestamp = data.timestamp
        nonce = data.nonce
        echostr = data.echostr
        # 自己的token
        token = "yangyanxing"  # 这里改写你在微信公众平台里输入的token
        # 字典序排序
        list = [token, timestamp, nonce]
        list.sort()
        sha1 = hashlib.sha1()
        map(sha1.update, list)
        hashcode = sha1.hexdigest()
        # sha1加密算法
        # 如果是来自微信的请求，则回复echostr
        if hashcode == signature:
            return echostr


    def POST(self):
        str_xml = web.data()  # 获得post来的数据
        xml = etree.fromstring(str_xml)  # 进行XML解析
        content = xml.find("Content").text  # 获得用户所输入的内容
        msgType = xml.find("MsgType").text
        fromUser = xml.find("FromUserName").text
        toUser = xml.find("ToUserName").text
        return self.render.reply_text(fromUser, toUser, int(time.time()), u"我现在还在开发中，还没有什么功能，您刚才说的是：" + content)


# class index:
#     def GET(self):
#         return 'Hello World Get'


if __name__ == '__main__':
    '''
    用于列举这些url的application

    the 'app' here is used to handle the urls
    '''
    app = web.application(urls, globals())
    app.run()