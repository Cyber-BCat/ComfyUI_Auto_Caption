注意：

1.安装依赖requirements.txt(注意：transformers 版本不能太低, windows使用则需要安装windows 版本的相关依赖)

      直接点击:install_req.bat 安装依赖

2.运行自动下载模型(推荐手动下载)

  (1).https://huggingface.co/google/siglip-so400m-patch14-384 放到clip/siglip-so400m-patch14-384

![1](![alt text](image-2.png))


  (2).推荐下载 https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit  （如果你有A100 可以考虑下载meta-llama/Meta-Llama-3.1-8B）放到LLM/Meta-Llama-3.1-8B-bnb-4bit
  
![2](![alt text](image-1.png))


  (3).必须手动下载:https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/tree/main/wpkklhc6   放到Auto_Caption 下

 ![3](![alt text](image.png))


基于反推:

![反推](.\workflow\Tag result.png)

flux运行效果:

![flux](.\workflow\show flux example.png)



