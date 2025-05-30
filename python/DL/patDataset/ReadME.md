<div style="width: 100%;
            text-align:center;" > 
    <div style="width: 100%; height: 100px;"></div>
    <h1 style = "font-size: 80px;"> patDataset </h1>
    <span>
        <b>模式匹配Dataset<b>		
    </span>
</div>
<div style="page-break-after: always;"></div>

# 目录

[TOC]

<div style="page-break-after: always;"></div>

# 1| 使用事项

　　需将数据按照一个规则的命名格式放入同一个文件夹内，如：

```
CT体数据：
data_dir/
	L001_0_input.npy
	L001_1_input.npy
	...
	L030_122_input.npy
	...
	L001_0_output.npy
	L001_1_output.npy
	...
	L030_122_output.npy
	...
```

使用说明：

构造函数参数：

* data_dir：数据存放的路径

* is_train：当前数据用于的阶段

* regex_pattern = [r'.*']：模式匹配的规则，必须为列表，每个元素为一个匹配规则

* process_fun = None：数据的处理函数
  * process_fun函数需要外部实现，其默认参数第一个为本次数据的路径，第二个参数表示数据用于的阶段。
  * process_fun的返回值应是一个数据或容器，不可为多个。
* args = None: 其他参数

> [!tip]
>
> 　　当每次需要提取n个类型的数据时，如typeA - typeN，则regex_pattern列表中应有n个匹配规则，用于分别匹配搜索typeA - typeN类型的数据路径。此时，process_fun函数所接收到的路径，为第index批数据typeA [index - 1] ~ typeN [index - 1]的路径，以一维numpy数组的方式接收。用户将在process_fun中对数据进行详细处理。

# 2| 实例：

```python
CT体数据：
data_dir/
	L001_0_input.npy
	L001_1_input.npy
	L002_0_input.npy
	L002_1_input.npy
	L001_0_output.npy
	L001_1_output.npy
	L002_0_output.npy
	L002_1_output.npy

task_configs = {
	"use_device": "..."
}

def my_process_fun(data_paths, is_train, etc_params, args):
    # 有两个规则，则data_paths为[规则1第index数据的路径，规则2第index数据的路径]
    input_data  = np.load(data_paths[0])	# 读取相应数据
    output_data = np.load(data_paths[1])	# 读取相应数据
    
    if is_train: ... # 训练阶段的数据预处理
    else:        ... # 测试阶段的数据预处理
    
    # 返回为一个字典
    return {
        'a': torch.from_numpy(input_data).float().to(args["use_devise"]),
    	'b': torch.from_numpy(output_data).float().to(args["use_devise"])
    }

dataset = patDataset(
	data_dir      = "data_dir",
	is_train      = "True",
	regex_pattern = [
		r"(L001|L002)_[a-ZA-Z0-9]_input.npy",	# 获取规则符合的数据
		r"(L001|L002)_[a-ZA-Z0-9]_output.npy"   # 获取规则符合的数据
	],
    process_fun   = my_process_fun,
	args          = task_configs
)
```

