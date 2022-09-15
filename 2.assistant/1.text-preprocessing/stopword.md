# 停用词应用和企业数据

## **什么是停用词**

在汉语中，有一类没有多少有实际意义的词语，比如组词“的”，连词“以及”、副词“甚至”，语气词“吧”，被称为停用词。

一个句子去掉这些停用词，并不影响理解。

所以，进行自然语言处理时，我们一般将停用词过滤掉。

## **行业停用词表汇总**

这里列举一些业界公开的中文停用词表。

这些词表文件中每一行存储一个停用词，行数就是停用词个数。

当然，根据自己任务的需求完全可以自我定制停用词表。


| 词表名 | 词表文件 |
| - | - |
| 中文停用词表                   | cn\_stopwords.txt    |
| 哈工大停用词表                 | hit\_stopwords.txt   |
| 百度停用词表                   | baidu\_stopwords.txt |
| 四川大学机器智能实验室停用词库 | scu\_stopwords.txt   |

[中文停用词表库](https://github.com/goto456/stopwords)

## **去除停用词**

那么如何去除文本中的停用词呢？这里我们马上开始Pyhton代码实战。

**Step1. 加载停用词表**

```python
# 这里我们以百度的停用词为例
def load_stopwords(path='data/baidu_stopwords.txt'):    
		with open(path, 'r', encoding='utf-8') as f:        
				lines = f.readlines()# 取出所有行    
		return [line.strip() for line in lines]# 把每行的换行符去掉，追加到新的列表中
stopwords = load_stopwords()
```

**Step2. 移除句子中的停用词**

```python
# 这里我们用两个分词后的句子来演示
documents = [['他', '脾气',' 不好', '，', '不是', '瞪', '别人', '就是', '说', '粗口', '，', '甚至', '打', '人'],
             ['这', '本书', '让', '我', '懂得', '了', '人生', '的', '法则', '，', '以及', '生活', '的', '残酷'],
            ]
def remove_stopwords(documents, stopwords):
    cleaned_documents = []
    for tokens in documents:
        temp = []
        for token in tokens:
            if token not in stopwords:
                temp.append(token)
        cleaned_documents.append(temp)
    return cleaned_documents
cleaned_documents = remove_stopwords(documents, stopwords)
print(cleaned_documents)
```

**打印结果：**

```
[['脾气', ' 不好', '，', '瞪', '别人', '说', '粗口', '，', '人'],
['本书', '懂得', '人生', '法则', '，', '生活', '残酷']]
```

可以看到句子中无效的词被去除，能表达句子主要内容的词被保留。

## **思考**

**我们总是删除停用词吗？他们总是对我们没用吗？**

答案是否定的！

我们并不能总是删除停用词。停用词的删除在很大程度上取决于我们正在执行的任务和我们想要实现的目标。例如，如果我们正在训练一个可以执行情绪分析任务的模型，我们可能不会删除停用词。

**影评**：“The movie was not good at all.”

**删除停用词后的文字**：“movie good”

我们可以清楚地看到这部电影的评论是负面的。然而，在去掉停用词之后，评论变得积极起来，这不是现实。因此，删除停用词在这里可能是有问题的。