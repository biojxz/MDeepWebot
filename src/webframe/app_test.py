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
import urllib2
import json

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

    def xiaohuangji(self,ask):
        ask = ask.encode('UTF-8')
        enask = urllib2.quote(ask)
        baseurl = r'http://www.simsimi.com/func/req?msg='
        url = baseurl + enask + '&lc=ch&ft=0.0'
        resp = urllib2.urlopen(url)
        reson = resp.read()

        return reson

    def GET(self):
        print ' weixin'
        data = web.input()
        content = data.content
        return self.xiaohuangji(content)

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