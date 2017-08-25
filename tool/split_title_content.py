#encoding=UTF-8
from bs4 import BeautifulSoup

with open("C:/Users/xulei/zhiziyun/autoTitle/Data/news_sohusite_xml.dat","r",encoding="gb18030") as fread,open("C:/Users/xulei/zhiziyun/autoTitle/Data/news_sohusite_title.txt","w",encoding="UTF-8") as fwrite_title, open("C:/Users/xulei/zhiziyun/autoTitle/Data/news_sohusite_content.txt","w",encoding="UTF-8") as fwrite_content:
    soup = BeautifulSoup(fread,"lxml")

    titles = soup.findAll('contenttitle')
    for title in titles:
        if title.string is None:
            title.string = "<UNK>"
        fwrite_title.write(title.string + '\n')
    fwrite_title.flush()
    
    contents = soup.findAll("content")
    for content in contents:
        if content.string is None:
            content.string = "<UNK>"
        fwrite_content.write(content.string + '\n')
    fwrite_content.flush()
fread.close()
fwrite_title.close()
fwrite_content.close()
