#encoding=utf-8
import re

#dir
datadir = "C:/Users/xulei/zhiziyun/autotitle/Data/"
titdir = datadir+"testtitle.txt"
contdir = datadir+"testcontent.txt"
tokendir = datadir+"test.txt"
titvocab_dic_dir = datadir+"testtitle_dic"
contvocab_dic_dir = datadir+"testcontent_dic"
vocab_dic_dir = datadir + "vocab_dic"

#vocab
titvocab_dic = {'<UNK>':1,'<s>':1,'</s>':1,'<PAD>':1}
contvocab_dic = {'<UNK>':1,'<s>':1,'</s>':1,'<PAD>':1}
vocab_dic = {'<UNK>':1,'<s>':1,'</s>':1,'<PAD>':1}
#symbol
NON_CHINESE_PATTERN = u'[^\u4e00-\u9fa5]'

with open(titdir,"r",encoding="UTF-8") as freadtitle , open(contdir,"r",encoding="UTF-8") as freadcontent ,open(tokendir,"w",encoding="UTF-8") as fwrite:
    for (titleline,contentline) in zip(freadtitle.readlines(),freadcontent.readlines()):
        #deal title file
        titleline = titleline.strip('\n')
        titleline = re.sub(NON_CHINESE_PATTERN,"",titleline)
        fwrite.write("abstract=<d> <p> <s> ")
        for titleword in titleline:
            fwrite.write(titleword + " ")
            if titleword in titvocab_dic:
                titvocab_dic[titleword] += 1
            else:
                titvocab_dic[titleword] = 1
            if titleword in vocab_dic:
                vocab_dic[titleword] += 1
            else:
                vocab_dic[titleword] = 1
        fwrite.write("</s> </p> </d>\tarticle=<d> <p> ")

        #deal content file
        contentline = contentline.strip('\n')
        sents = contentline.split("。")
        nb_sents = 0
        for s in sents:
            if s=='':
                continue
            fwrite.write("<s> ")
            s = re.sub(NON_CHINESE_PATTERN,"",s)
            for contentword in s:
                fwrite.write(contentword + " ")
                if contentword in contvocab_dic:
                    contvocab_dic[contentword] += 1
                else:
                    contvocab_dic[contentword] = 1
                if contentword in vocab_dic:
                    vocab_dic[contentword] += 1
                else:
                    vocab_dic[contentword] = 1
            fwrite.write("</s> ")
            nb_sents+=1
            if nb_sents == 5:
                break
        fwrite.write("</p> </d>	publisher=AFP \n")
        fwrite.flush
freadtitle.close
freadcontent.close
fwrite.close

with open(titvocab_dic_dir,'w',encoding="UTF-8") as tit_dic_write , open(contvocab_dic_dir,'w',encoding="UTF-8") as cont_dic_write , open(vocab_dic_dir,'w',encoding="UTF-8") as dictmerged_write:
    for k,v in contvocab_dic.items():
        cont_dic_write.write(k + " " + str(v) + "\n")
    cont_dic_write.close()
    for k,v in titvocab_dic.items():
        tit_dic_write.write(k + " " + str(v) + "\n")
    tit_dic_write.close()
    for k,v in vocab_dic.items():
        dictmerged_write.write(k + " " + str(v) + "\n")
    dictmerged_write.close()
