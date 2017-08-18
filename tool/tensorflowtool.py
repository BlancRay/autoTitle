#encoding=utf-8
titdir="C:/Users/xulei/zhiziyun/workspace/Data/testtitle.txt"
contdir="C:/Users/xulei/zhiziyun/workspace/Data/testcontent.txt"
tokendir="C:/Users/xulei/zhiziyun/workspace/Data/test.txt"
with open(titdir,"r",encoding="UTF-8") as freadtitle , open(contdir,"r",encoding="UTF-8") as freadcontent ,open(tokendir,"w",encoding="UTF-8") as fwrite:
    for (titleline,contentline) in zip(freadtitle.readlines(),freadcontent.readlines()):
        titleline = titleline.strip('\n')
        contentline = contentline.strip('\n')
        sents = contentline.split("。")
        ss=""
        for s in sents:
            if s!="":
                ss = ss + s + " </s> <s>"
        fwrite.write("abstract=<d> <p> <s>" + titleline + " </s> </p> </d>	article=<d> <p> <s> " + ss + " </s> </p> </d>	publisher=AFP \n")
fwrite.flush
freadtitle.close
freadcontent.close
fwrite.close