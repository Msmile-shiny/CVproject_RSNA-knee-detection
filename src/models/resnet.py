"""ResNet 系列 backbone (ResNet50 / ResNet101).

阶段一: 2D 切片级多标签分类
阶段二: 2.5D 三平面 (作为共享 backbone)
阶段三: ResNet3D-18 (轻量 3D)
"""