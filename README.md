# [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](https://arxiv.org/abs/2407.05000)

- [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](#lora-ga-low-rank-adaptation-with-gradient-approximation)
  - [Overview](#overview)
  - [tips](#tips)
  - [⚡️ quick start](#️-quick-start)
    - [What exactly does the above code do?](#what-exactly-does-the-above-code-do)
  - [citation](#citation)

## Overview

we introduce a novel initialization method, LoRA-GA (Low Rank Adaptation with Gradient Approximation), which aligns the gradients of low-rank matrix product with those of full fine-tuning at the first step. Our extensive experiments demonstrate that LoRA-GA achieves a convergence rate comparable to that of full fine-tuning (hence being significantly faster than vanilla LoRA as well as various recent improvements) while simultaneously attaining comparable or even better performance.
![](./resource/pic/fig-1.png)
(Left) Training loss curves of Llama 2-7B on MetaMathQA to training steps. LoRA-GA
converges as quickly as full fine-tuning and outperforms LoRA. (Right) Initialization procedures
used in LoRA and LoRA-GA. The key difference is that LoRA-GA initializes adapters using the
eigenvectors of the gradient matrix, as opposed to random initialization with a scaling factor.

## tips

The reproduce directory is only used to reproduce the paper, and is not recommended
It is recommended for you to use LoRA-GA through peft(The usage is in the [quick start](#⚡️-quick-start) below.)

## ⚡️ quick start

1. install peft
   ```bash
   git clone https://github.com/Outsider565/LoRA-GA.git
   cd LoRA-GA
   git submodule init
   git submodule update peft
   cd peft
   pip install -e .
   ```
2. use LoRA-GA in peft
   ```python
   from peft import LoraGAConfig
   from peft.utils.lora_ga_utils import estimate_gradient, LoraGAContext
   peft_config = LoraGAConfig()
   # model should be float(such as bf16, fp32) model
   named_grad = estimate_gradient(
       model=model,
       dataloader=dataloader,
       accelerator=accelerator,
       quant_flag=False,
   )
   """
   if you want to use quantized model:
       you can set quant_flag=Ture to Reduce GPU memory usage during gradient estimation.
       then re-get your quantizied model here.
   """
   with LoraGAContext(model=model, named_grad=named_grad):
       model = get_peft_model(model=model, peft_config=peft_config)
   ```

Detailed usage(such as quantizaion model, detailed usage of function and class) see [Detailed usage](./doc/detail.md)

### What exactly does the above code do?

1. LoraGAConfig is subclass of LoraConfig. LoraGAConfig will set peft_type to PeftType.LORAGA and init_lora_weights = "lora_ga".

2. estimate_gradient will use the data in the dataloader for forward and backward propagation, and return a dictionary named_grad. The key of this dictionary belongs to the submodule name of the model, and the value is the gradient of the weight W of the corresponding module.

3. LoraGAContext will attach named_grad to model as an attribute of model. named_grad will pass named_grad to LoraGAModel which is a subclass of LoraModel. After using named_grad to initialize LoraGAModel(LoraModel), LoraGAModel frees it.

after you get_peft_model, you can use your peft model as lora model to finetune

## citation

```
@misc{wang2024loragalowrankadaptationgradient,
      title={LoRA-GA: Low-Rank Adaptation with Gradient Approximation},
      author={Shaowen Wang and Linxi Yu and Jian Li},
      year={2024},
      eprint={2407.05000},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.05000},
}
```
