使用c++ opencv 中dnn 模块加载pb 文件并进行推理

python `cpp_test.py` 包含模型构建，训练、转为pb、以及对pb进行测试。

`cv_test.py` 使用Python的opencv 库调用dnn测试pb 模型

|    文件名     |   类别   |      概率    |
|---------------|--------|----------------|
|file : 1.png   | cat : 7 | score : 1.00  |
|file : 10.png  | cat : 9 | score : 1.00  |
|file : 100.png | cat : 9 | score : 1.00  | 
|file : 101.png | cat : 6 | score : 1.00  |
|file : 102.png | cat : 0 | score : 1.00  |
|file : 103.png | cat : 5 | score : 1.00  | 
|file : 104.png | cat : 4 | score : 1.00  |
|file : 105.png | cat : 9 | score : 1.00  |
|file : 106.png | cat : 9 | score : 1.00  |
