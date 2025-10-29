main:

- 除了task(以及resume, noload) 外, 所有的参数通过config读取

- 应当输入目标config相对于./config/文件夹的目录


split_data:
- 从一坨数据中随机分割出val集和train集，并生成yolo格式的yaml文件
- 给定images, labels, output, val集占比(默认为0.2), names。随机分割
- 可以启用软链
- 所有地址为相对datasets的地址

labels_reclassify:
修改标签中的class

- `-m` 映射，支持[10:3, 5:7]， 10:3,5:7， {10:3,5:7}
- `-l` labels 文件夹相对于datasets路径
- `-y` 可选：指定 data.yaml 路径（若不指定脚本会在 labels 的父目录和祖父目录尝试自动定位）
- `-n` 可选：用一个 names 列表覆盖 yaml中的names 如：-n B1 B2 ...