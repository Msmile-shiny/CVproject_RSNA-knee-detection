# 执行剩余窗口计划：新增资产筛选

已重新检索公开数据集并归档文件清单、小型配置和推理源码，见experiments/assets1015/audit.json。未下载模型大权重、未执行第三方源码、未启动GPU。5个Python文件静态语法检查通过，不代表运行正确。

## 优先对象：nartaa/rsna-knee-s6fall-weights

公开可访问，存在完整配套脚本nartaa/rsna-knee-s6fall-solo。清单记录单检查点E_prior_s42_fall.pt，SHA 0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12。作者报告单独Public0.950，非本方实测。

训练配置：4263例、JEV blend报告软标签、320输入/288裁剪、64训练窗口、16epochs、seed42、无EMA；作者声明gold58未训练、无OAI训练数据。相比现有R7的384/352、不同标签及训练配方，有互补假设，但仍是ConvNeXt-small，同源架构不能当独立误差证据。它并非旧R6四折的原样复制，也不能仅凭分辨率差异宣称改善。

源码实际是Python script而非Notebook；其main包含A/B阶段、parity及blend，不能只按solo标题假设调用流程。下一执行门禁：逐项审计MODE、A/B阶段、s6_extras.bin及code_manifest，确认实际输出为哪个成员及其失败规则；下载小型额外代码包审计并固定哈希，验证离线依赖和受控融合接口，再决定单个候选。不要直接把该脚本整段拼入R7造成重复Raptor。

## 暂缓对象

- dkim54/rsna-knee-models-v11c公开可访问：20个约100MB权重、model/dataset/infer_lib代码和cp310/311/312解码wheel，models.json确认20个条目。资产更重，当前按用户检索未找到配套公开knee kernel，需补训练来源、完整调用及预算，不立即启动20模型堆叠。
- bliverrigpdoge/rsna-knee-kcd25leg仍403，停止尝试访问，不绕过权限。

## 未完成项

最终提交选择仍未核实，等待用户登录页面的选择标记截图；不将0.956保护版本当成已经选中。官方最终截止为北京时间10月23日07:59，报名/合并为10月16日07:59，详见deadline_plan_20261015.md。本轮未更改最终选择，未设置自动跟进，后续不会在无新请求时自动运行。
