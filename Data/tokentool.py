#encoding=utf-8
import jieba
fread,fwrite = open("C:/Users/xulei/zhiziyun/workspace/Data/net_title","r",encoding="UTF-8"),open("C:/Users/xulei/zhiziyun/workspace/Data/net_title_token.txt","w",encoding="UTF-8")
lines = fread.readlines()
for line in lines:
    seg_list = list(jieba.cut(line,cut_all=False))
    #print(" ".join(seg_list))
    tmp = " ".join(seg_list)
    fwrite.write(tmp+"\n")
    fwrite.flush
fread.close
fwrite.close
