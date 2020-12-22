<!--
 * @Author: Wang Huzhen
 * @Version: 1.0
 * @FilePath: \AI_minist\README.md
 * @Email: 2327253081@qq.com
 * @Date: 2020-12-22 18:47:38
-->
# AI_minist
TensorFlow2.3手写识别(dnn/cnn)+AutoEncoder

# 环境配置

Tensorflow 2.3.2

python 3.6.0+

# 运行

重新训练神经网络前删除掉保存在data/model中已经训练好的模型

python minist.py

只进行预测输出，在保证模型已经训练好后，注释掉训练模型的代码块，使用model.predict()可以对单个数据进行预测，使用model.predictclass()可以对多个数据集进行预测分类。

# 实验一(dnn)

## Result

![minist](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%B8%80(dnn)/minist.png)

![model](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%B8%80(dnn)/model.PNG)

<img src="https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%B8%80(dnn)/history.PNG" alt="history"  />

![accura](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%B8%80(dnn)/accura.png)

![loss](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%B8%80(dnn)/loss.png)

<img src="https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%B8%80(dnn)/res1.png" alt="res1" style="zoom:50%;" />

# 实验二(cnn)

## Result

![minist_cnn](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%8C(cnn)/minist_cnn.png)

![model](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%8C(cnn)/model.png)

![history](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%8C(cnn)/history.png)

![accuracy](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%8C(cnn)/accuracy.png)

![loss](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%8C(cnn)/loss.png)

<img src="https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%BA%8C(cnn)/res.png" alt="res" style="zoom:50%;" />

# 实验三(AE)

## Result

![model](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%B8%89(AE)/model.PNG/model.PNG)

![res](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%B8%89(AE)/model.PNG/res.png)

![res2](https://github.com/wanghuzhen/AI_minist/blob/main/%E5%AE%9E%E9%AA%8C%E4%B8%89(AE)/model.PNG/res2.png)