# Sentiment-Analysis
基于Python的简易情感分析模型
- 运行时间会稍微长一点，大概3分钟
- Sentimental_analysis.py 是采用朴素贝叶斯调节标记参数的方法，训练：测试=3：1，准确率大约为77-78%，最高能单极性达到79%
- NaiveBayesClassifier.py 是nltk库里最朴素的算法，代码比较短，前期起步用，准确率大约为74-75%
- record10.txt和record25.txt是我调节参数用Sentiment_analysis_Further.py导出的数据
用Draw_3D.py绘图成Figure_1.png进行可视化，采用准确率最高的参数
- test.py是输出测试集662组测试数据的极性值（1/-1）
- 导入data用相对目录 维持现状即可
- 开发集数据太散了，导致自测准确率不高，不知测试集如何，不要嫌弃太低的正确率！

## 谢谢车教授和助教们！！！！
Power By Gzn
