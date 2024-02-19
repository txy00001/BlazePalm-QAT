
import os
import json
import cv2

count = 1

# 显示带有矩阵框的图片
def show(path, points):
    global count
    image = cv2.imread(path)
    cv2.rectangle(image, points[0], points[1], (0, 255, 0), 1, 4)
    cv2.namedWindow("AlanWang")
    cv2.imshow('AlanWang', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    count += 1

# 这边修改你数据集的路径，建议备份一下，因为是直接修改原来的json
for filename in os.walk('/home/cmm/PycharmProjects/datas/handpose_datasets_v2'):
    for name in filename[2]:
        # json_path = os.path.join(filename[0], name)
        json_path = '/home/cmm/PycharmProjects/datas/handpose_datasets_v2/2021-02-01_14-19-34_17725.json'
        new_dict = {}
        # 处理json文件，图片不处理
        if json_path.endswith(".json"):
            with open(json_path, mode="r", ) as f:
                data = json.load(f)
        else:
            continue
        # 构造新json
        new_dict["maker"] = data["maker"]
        new_dict["date"] = data["date"]
        new_dict["info"] = []
        # 遍历每一只手，提取几个关键点
        for i in data["info"]:
            s = {"pts": {}}
            s["pts"]["0"] = i["pts"]["0"]
            s["pts"]["1"] = i["pts"]["1"]
            s["pts"]["2"] = i["pts"]["2"]
            s["pts"]["5"] = i["pts"]["5"]
            s["pts"]["9"] = i["pts"]["9"]
            s["pts"]["13"] = i["pts"]["13"]
            s["pts"]["17"] = i["pts"]["17"]
            # 获取上面几个点的x\y坐标
            x = [i["pts"]["0"]["x"], i["pts"]["1"]["x"], i["pts"]["2"]["x"], i["pts"]["5"]["x"], i["pts"]["9"]["x"],
                 i["pts"]["13"]["x"], i["pts"]["17"]["x"]]
            y = [i["pts"]["0"]["y"], i["pts"]["1"]["y"], i["pts"]["2"]["y"], i["pts"]["5"]["y"], i["pts"]["9"]["y"],
                 i["pts"]["13"]["y"], i["pts"]["17"]["y"]]
            # 拿到最大最小值
            x_min = min(x)
            x_max = max(x)
            y_min = min(y)
            y_max = max(y)
            # 构建bbox 左上右下
            s["bbox"] = [x_min, y_min, x_max, y_max]
            new_dict["info"].append(s)
            # 可以一边跑一边显示，注释掉可以跑得快
            show(json_path.replace(".json", ".jpg"), [[x_min, y_min], [x_max, y_max]])
        # 直接覆盖原来的json文件
        # with open(json_path, mode="w", ) as f:
        #     json.dump(new_dict, f, indent=4)