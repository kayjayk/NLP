#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import sys

url = 'https://brunch.co.kr/@kakao-it/51'
# 게시물 1개 수집하는 함수
def brunch_to_html(url):
    # 웹페이지 수집
    soup = BeautifulSoup(urlopen(url), 'html.parser')
    articles = soup.findAll(re.compile(r'(p|div|h[1-6]|blockquote)'),     {'class':re.compile(r'(wrap_item item_type_text|wrap_item item_type_bullet|wrap_img_float)')})
    articles[9].text
    ## 제목 추출
    title = soup.find('h1', {'class':'cover_title'}).text
    ## 소제목 추출
    sub_title = soup.find('p', {'class':'cover_sub_title'}).text
    # 결과물의 body 생성
    body = '<h1>' + title + '</h1><br><h3>' + sub_title + '</h3><br>'
    # 본문에서 텍스트와 그림 링크를 추출하여 body 생성
    for line in articles:
        if (line.text != '\xa0'):# & (line.text != ''):
            if str(line).__contains__('div'):
                try:
                    img_link = re.search(r'http.*', line.find('img')['src']).group(0)
                except:
                    img_link = ''
                try:
                    caption = line.find('span', {'class':'text_caption'}).text
                except:
                    caption = ''
                body += '<center><img src="' + img_link + '"><br>' + caption + '</center><br>'
            elif str(line).__contains__('h2'):
                body += '<br><h2>' + line.text + '</h2><br>'
            elif str(line).__contains__('h3'):
                body += '<h3>' + line.text + '</h3>'
            elif str(line).__contains__('h4'):
                body += '<h4>' + line.text + '</h4>'
            elif str(line).__contains__('h5'):
                body += '<h5>' + line.text + '</h5>'
            elif str(line).__contains__('h6'):
                body += '<h6>' + line.text + '</h6>'    
            elif str(line).__contains__('blockquote'):
                if str(line).__contains__('type1'):
                    body += '<br><center><h3><font color="grey">' + line.text + '</font></h3></center><br>'
                elif str(line).__contains__('type2'):
                    line_text = re.sub(r"(<blockquote[^>]*>|</blockquote>)", '', str(line)).                    replace('</br>', '<br>')
                    body += '<br><div style="border-color:#acacac;border-width:0 0 0 2px;border-style:solid;padding:1px 0 0 12px;color:#666;line-height:18pt;text-align:left}"><font color="grey">' + line_text + '</font></div><br>'
                elif str(line).__contains__('type3'):
                    line_text = re.sub(r"(<blockquote[^>]*>|</blockquote>)", '', str(line)).                    replace('</br>', '<br>')
                    body += '<br><div style="border:1px solid #d7d7d7;text-align:left;padding:21px 25px 20px;color:#666;line-height:18pt}"><font color="grey">' + line_text + '</font></div><br>'
                else:
                    body += '<br><font color="grey">' + line.text + '</font><br>'
            elif str(line).__contains__('bullet'):
                body += '<li>' + line.text + '</li>'
            else:
                line_text = re.sub(r'(<blockquote[^>]*>|</blockquote>|<p[^>]*>|</p>)', '', str(line)).                    replace('<br/>', '<br>').replace('<b></b>', '')
                if line_text != '<br>':
                    #print(line_text)
                    body += '<br>' + line_text + '<br>'
            
    # 전체 html 파일 생성
    pad_front = '<html><head><style>' + '</style></head><body>'
    pad_rear = '</body></html>'
    html = pad_front + body + pad_rear

    return(html)

# 매거진의 글 수집하는 함수
def magazine_scrap(url):
    base_url = 'https://brunch.co.kr/'
    url = 'https://brunch.co.kr/magazine/kakaoaireport'
    soup = BeautifulSoup(urlopen(url), 'html.parser')
    soup
    soup.findAll('div')[2]
    articles = soup.findAll('li', {'class':re.compile(r'magazine_article.*')})
    print(articles)
    soup.findAll('div', {'class':'magazine_articles'})
    articles_url = [x.find('a')['href'] for x in articles]

    for i, article in enumerate(articles_url):
        print(article)
        docs =brunch_to_html(base_url + article)
        # html로 저장
        with open('/Volumes/RAMDisk/brunch_%2d.html'%i, 'w') as f:
            f.write(docs)        



docs = brunch_to_html(sys.argv[1])

with open('/Volumes/RAMDisk/brunch.html', 'w') as f:
    f.write(docs)

