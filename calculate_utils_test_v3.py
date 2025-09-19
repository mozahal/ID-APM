import math

import camera_configs_left_middle
import camera_configs_left_right
def get_depth(xl,xr,w):

    d = math.fabs(xl - xr * camera_configs_left_right.fl / camera_configs_left_right.fr)
    z = abs(camera_configs_left_right.baseline*camera_configs_left_right.fl/d)

    return z/10
def get_disparity(distance):
    disparity=camera_configs_left_middle.baseline*camera_configs_left_middle.fl/(distance*10)
    # print('disparity',disparity)
    return disparity
def getxr(xl,distance):
    xr = (xl - get_disparity(distance))*(camera_configs_left_middle.fr/camera_configs_left_middle.fl)
    # xr = xr*camera_configs_left_ray.fr/camera_configs_left_ray.fl
    return xr
def get_3d_coordinate(u, v, d):
    '''
    计算目标物体在左相机坐标系下的三维坐标
    u: 目标物体在左图像中的水平像素坐标
    v: 目标物体在左图像中的竖直像素坐标
    d: 左相机与目标物体之间的距离
    K: 左相机内部参数矩阵
    return: 目标物体在相机坐标系下的三维坐标
    '''
    K=camera_configs_left_right.left_camera_matrix
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    z = 1.0
    #
    Z=d
    Y=y*Z/z
    X=x*Z/z
    # print('coordination({},{},{})'.format(X,Y,Z))
    return [X,Y,Z]
def get_angle(m1,m2):#输入两个坐标，返回角度
    tan=math.fabs(m1[2]-m2[2])/math.fabs(math.sqrt((m1[0]-m2[0])*(m1[0]-m2[0])+(m1[1]-m2[1])*(m1[1]-m2[1])))
    print(tan)
    angle=math.atan(tan)
    angle=math.degrees(angle)
    # print('angle:',angle)
    return angle
def read_yolo_label_file(file_path):

    with open(file_path, 'r') as f:
        lines = f.readlines()

    bounding_boxes = []
    for line in lines:
        values = line.strip().split(' ')
        class_id = int(values[0])
        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])


        bounding_boxes.append([class_id,x_center,y_center,width,height])

    return bounding_boxes

def euclidean_distance(point1, point2):
    """
    计算两个点之间的欧几里得距离
    """
    x1, y1 = point1[1], point1[2]
    x2, y2 = point2[1], point2[2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def distance_diff_percentage(dist1, dist2):
    """
    计算两个距离之间的差异百分比
    """
    if dist1 == 0 or dist2 == 0:
        return float('inf')
    return abs(dist1 - dist2) / min(dist1, dist2)

import numpy as np

def find_matching_points(list1, list2, threshold=0.25):
    # print(list1)
    # assert False
    # Calculate distances between all pairs of points within each list
    distances1 = np.zeros((len(list1), len(list1),2))
    distances2 = np.zeros((len(list2), len(list2),2))
    for i, point1 in enumerate(list1):
        for j, point2 in enumerate(list1):
            if j < i:
                continue
            if i != j:  # avoid matching with itself
                # distances1[i, j] = [abs(point1[1] - point2[1]),abs(point1[2] - point2[2])]
                distances1[i, j] = [abs(point1[1] - point2[1]), point1[2] - point2[2]]
    for i, point1 in enumerate(list2):
        for j, point2 in enumerate(list2):
            if j < i:
                continue
            if i != j:  # avoid matching with itself
                distances2[i, j] = [abs(point1[1] - point2[1]), point1[2] - point2[2]]
                # distances2[i, j] = [abs(point1[1] - point2[1]), abs(point1[2] - point2[2])]


    matches=[]
    for i1 in range(len(list1)):
        for j1 in range(i1+1,len(list1)):
            # if i1 == j1:
            #     continue
            for i2 in range(len(list2)):
                if list1[i1][0] != list2[i2][0]:
                    continue
                for j2 in range(i2+1,len(list2)):
                    if list1[j1][0] != list2[j2][0]:
                        continue
                    if distances1[i1, j1][1] * distances2[i2, j2][1] < 0:
                        continue
                    if get_area_diff(list1[i1],list2[i2])<2*threshold or get_area_diff(list1[j1],list2[j2])<2*threshold:

                        if  get_xy_diff((list1[i1],list1[j1]),(list2[i2],list2[j2]))<threshold:
                            matches.append([[i1,j1],[i2,j2],np.min([distances1[i1, j1][0], distances2[i2, j2][0]])])
                    # if get_area_diff(list1[i1],list2[i2])*get_xy_diff((list1[i1],list1[j1]),(list2[i2],list2[j2]))<threshold or get_area_diff(list1[j1],list2[j2])*get_xy_diff((list1[i1],list1[j1]),(list2[i2],list2[j2]))<threshold:
                    #
                    #     matches.append([[i1,j1],[i2,j2],np.min([distances1[i1, j1], distances2[i2, j2]])])
    matches.sort(key=lambda x: x[2], reverse=True)
    # print(matches)
    if len(matches)==0:
        return []
    # print(matches[0])
    # Return the indices of the matching points with the largest distance
    return matches[0]

def label_map(label,label_map,threshold=0.75):
    area=label[3]*label[4]
    area_map=label_map[3]*label_map[4]

    if area_map!=0 and area_map<threshold*area:
        if label_map[1]>0.5:
            label_map[1]=label_map[1]-label_map[3]/2+label[3]/2
            label_map[3]=label[3]
        else:
            label_map[1] = label_map[1] + label_map[3] / 2 - label[3] / 2
            label_map[3] = label[3]
    return label_map


#通过分布情况进行匹配，建议输入的list为根据类别和x值进降序配列的数组
def match_detections(list1, list2 ,area_threshold=0.25):#输入检测框的list，和面积比的阈值
    ret=False
    if len(list1) == 1:
        if len(list2) == 1:
            min_x1 = min(x[1] for x in list1)
            min_x2 = min(x[1] for x in list2)
            max_x1 = 2 * min_x1
            max_x2 = 2 * min_x2
            min_y1 = min(x[2] for x in list1)
            min_y2 = min(x[2] for x in list2)
            max_y1 = 2 * min_y1
            max_y2 = 2 * min_y2
        else:
            min_x2, max_x2 = min(x[1] for x in list2), max(x[1] for x in list2)
            min_x1 = min(x[1] for x in list1)
            max_x1 = min_x1 + (max_x2 - min_x2)

            min_y2, max_y2 = min(x[2] for x in list2), max(x[2] for x in list2)
            min_y1 = min(x[2] for x in list1)
            max_y1 = min_y1 + (max_y2 - min_y2)

    elif len(list2) == 1:
        min_x1, max_x1 = min(x[1] for x in list1), max(x[1] for x in list1)
        min_x2 = min(x[1] for x in list2)
        max_x2 = min_x2 + (max_x1 - min_x1)

        min_y1, max_y1 = min(x[2] for x in list1), max(x[2] for x in list1)
        min_y2 = min(x[2] for x in list2)
        max_y2 = min_y2 + (max_y1 - min_y1)

    else:
        result = find_matching_points(list1, list2, area_threshold)
        # print('result:',result)
        # if len(result) == 0:
        #     return []
        # left, right = result[0], result[1]
        #
        # min_x1, max_x1 = list1[left[0]][1], list1[left[1]][1]
        # min_x2, max_x2 = list2[right[0]][1], list2[right[1]][1]
        #
        # min_y1, max_y1 = list1[left[0]][2], list1[left[1]][2]
        # min_y2, max_y2 = list2[right[0]][2], list2[right[1]][2]
        #
        # list1 = list1[left[0]:left[1] + 1]
        # list2 = list2[right[0]:right[1] + 1]
        # ret = True
        if len(result)==0:
            min_x1, max_x1 = min(x[1] for x in list1), max(x[1] for x in list1)
            min_y1, max_y1 = min(x[2] for x in list1), max(x[2] for x in list1)
            min_x2, max_x2 = min(x[1] for x in list2), max(x[1] for x in list2)
            min_y2, max_y2 = min(x[2] for x in list2), max(x[2] for x in list2)

        else:
            left,right=result[0],result[1]
            print('matched result(l_and_r):',left,right)

            min_x1, max_x1=list1[left[0]][1],list1[left[1]][1]
            min_x2, max_x2=list2[right[0]][1],list2[right[1]][1]

            min_y1, max_y1 = list1[left[0]][2], list1[left[1]][2]
            min_y2, max_y2 = list2[right[0]][2], list2[right[1]][2]

            # list1 = list1[left[0]:left[1] + 1]
            # list2 = list2[right[0]:right[1] + 1]
            # ret=True
            if max_x1 - min_x1==0 or max_y1 - min_y1==0 or max_x2 - min_x2==0 or max_y2 - min_y2==0:
                min_x1, max_x1 = list1[left[0]][1]-list1[left[0]][3]/2, list1[left[1]][1]+list1[left[1]][3]/2
                min_x2, max_x2 = list2[right[0]][1]-list2[right[0]][3]/2, list2[right[1]][1]+list2[right[1]][3]/2

                min_y1, max_y1 = list1[left[0]][2]-list1[left[0]][4]/2, list1[left[1]][2]+list1[left[1]][4]/2
                min_y2, max_y2 = list2[right[0]][2]-list2[right[0]][4]/2, list2[right[1]][2]+list2[right[1]][4]/2
    if max_y1==min_y1:
        min_y1,max_y1=0,1
    elif max_x1 == min_x1:
        min_x1,max_x1=0,1
    if max_y2==min_y2:
        min_y2,max_y2=0,1
    elif max_x2 == min_x2:
        min_x2,max_x2=0,1
    list1_norm = [(x[0], (x[1] - min_x1) / (max_x1 - min_x1), (x[2] - min_y1) / (max_y1 - min_y1), x[3], x[4]) for x in
                  list1]
    list2_norm = [(x[0], (x[1] - min_x2) / (max_x2 - min_x2), (x[2] - min_y2) / (max_y2 - min_y2), x[3], x[4]) for x in
                  list2]

    # Step 4: Match detections based on class and x-coordinate proximity
    matches = []
    # print(list1_norm,list2_norm)
    for i, det1 in enumerate(list1_norm):
        best_match = None
        best_dist = float('inf')
        for j, det2 in enumerate(list2_norm):
            if det1[0] != det2[0]:  # class mismatch, skip
                continue
            cx1, cy1 = det1[1], det1[2]
            cx2, cy2 = det2[1], det2[2]
            # dist = (((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5)*get_x_area_diff(det1,det2)

            if (1-get_area_diff(det1, det2))>0:
                if det1[0]==0:
                    dist = (((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5) * (1 + get_x_area_diff(det1, det2))
                else:
                    dist = (((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5)* (2 + get_x_area_diff(det1, det2))
                if dist < best_dist:
                    best_match = j
                    best_dist = dist
        if best_match is not None:
            # area1 = det1[3] * det1[4]
            # area2 = list2_norm[best_match][3] * list2_norm[best_match][4]
            area_diff = get_area_diff(det1,list2_norm[best_match])
            # print(i, best_match,area_diff,area_threshold)
            if area_diff < area_threshold:
                matches.append((i, best_match,area_diff))
    matches=remove_error_matches(matches)
    # print(matches)
    # if ret:
    #     matches = [[x+left[0], y+right[0],dist] for x, y,dist in matches]
    return matches

#注意！！！！！！！返回的是y，x而不是x，y
def find(x,y,map):
    # print(x,y,map)
    distances = np.linalg.norm(map - np.array([x, y]), axis=2)
    # 找到距离最小的索引值
    min_index = np.unravel_index(np.argmin(distances), map.shape[:2])
    return min_index

def get_LL_label(label,map1,map2,x_gap=0,y_gap=0):
    result=[]
    temp_result=[]
    for cls, x, y, w, h in label:
        temp_x=int(x * map1.shape[1])
        temp_y=int(y * map1.shape[0])
        x_,y_=map1[temp_y,temp_x][0]-x_gap,map1[temp_y,temp_x][1]-y_gap
        temp_result.append([cls, x_, y_, w, h])
    for cls, x, y, w, h in temp_result:
        y_,x_=find(x,y,map2)
        x_,y_=x_/map2.shape[1],y_/map2.shape[0]
        w_,h_=w * map1.shape[1] / map2.shape[1],h*map1.shape[0] / map2.shape[0]
        result.append([cls,x_,y_,w_,h_])
    return result

def get_LL_label(label,map1,map2,x_gap=0,y_gap=0):
    result=[]
    temp_result=[]
    for cls, x, y, w, h in label:
        temp_x=int(x * map1.shape[1])
        temp_y=int(y * map1.shape[0])
        x_,y_=map1[temp_y,temp_x][0]-x_gap,map1[temp_y,temp_x][1]-y_gap
        temp_result.append([cls, x_, y_, w, h])
    for cls, x, y, w, h in temp_result:
        y_,x_=find(x,y,map2)
        x_,y_=x_/map2.shape[1],y_/map2.shape[0]
        w_,h_=w * map1.shape[1] / map2.shape[1],h*map1.shape[0] / map2.shape[0]
        result.append([cls,x_,y_,w_,h_])
    return result

def get_LL_label_v2(label,map1,map2,x_gap=0,y_gap=0):
    result=[]
    temp_result=[]
    for cls, x, y, w, h in label:
        temp_x1,temp_x2=max(0,int((x-w/2) * map1.shape[1])),min(map1.shape[1]-1,int((x+w/2) * map1.shape[1]))
        temp_y1,temp_y2=max(0,int((y-h/2) * map1.shape[0])),min(map1.shape[0]-1,int((y+h/2) * map1.shape[0]))
        x1,y1=map1[temp_y1,temp_x1][0]-x_gap,map1[temp_y1,temp_x1][1]-y_gap
        x2,y2=map1[temp_y2,temp_x2][0]-x_gap,map1[temp_y2,temp_x2][1]-y_gap
        x_,y_,w_,h_=(x1+x2)/2,(y1+y2)/2,(x2-x1),(y2-y1)
        temp_result.append([cls, x_,y_,w_,h_])
    for cls, x, y, w, h in temp_result:
        x1, y1=x-w/2,y-h/2
        x2, y2=x+w/2,y+h/2
        y1_,x1_=find(x1,y1,map2)
        y2_,x2_=find(x2,y2,map2)
        x_, y_, w_, h_ = (x1_ + x2_) / 2/map2.shape[1], (y1_ + y2_) / 2/map2.shape[0], (x2_ - x1_)/map2.shape[1], (y2_ - y1_)/map2.shape[0]
        # y_,x_=find(x,y,map2)
        # x_,y_=x_/map2.shape[1],y_/map2.shape[0]
        # w_,h_=w * map1.shape[1] / map2.shape[1],h*map1.shape[0] / map2.shape[0]
        if w_==0 or h_==0:
            continue
        result.append([cls,x_,y_,w_,h_])
    return result
def get_LL_label_v3(label,map1,map2,x_gap=0,y_gap=0):#使用原生图像上的标签框
    result=[]
    temp_result=[]
    for cls, x, y, w, h in label:
        x, y, w, h=x* map1.shape[1]-x_gap, y*map1.shape[0]-y_gap, w* map1.shape[1], h*map1.shape[0]
        temp_result.append([cls, x,y,w,h])
    for cls, x, y, w, h in temp_result:
        x1, y1=x-w/2,y-h/2
        x2, y2=x+w/2,y+h/2
        y1_,x1_=find(x1,y1,map2)
        y2_,x2_=find(x2,y2,map2)
        x_, y_, w_, h_ = (x1_ + x2_) / 2/map2.shape[1], (y1_ + y2_) / 2/map2.shape[0], (x2_ - x1_)/map2.shape[1], (y2_ - y1_)/map2.shape[0]
        temp=[cls,x_,y_,w_,h_]
        temp=label_map([cls,x/map2.shape[1],y/map2.shape[0],w/map2.shape[1],h/map2.shape[0]],temp)
        # y_,x_=find(x,y,map2)
        # x_,y_=x_/map2.shape[1],y_/map2.shape[0]
        # w_,h_=w * map1.shape[1] / map2.shape[1],h*map1.shape[0] / map2.shape[0]
        if w_==0 or h_==0:
            continue
        result.append(temp)
    return result

def get_LL_label_v4(label,map1,map2,x_gap=0,y_gap=0):#优化了其中信息框中因图像边缘遮挡带来的损失
    result=[]
    temp_result=[]
    for cls, x, y, w, h in label:
        temp_x1,temp_x2=max(0,int((x-w/2) * map1.shape[1])),min(map1.shape[1]-1,int((x+w/2) * map1.shape[1]))
        temp_y1,temp_y2=max(0,int((y-h/2) * map1.shape[0])),min(map1.shape[0]-1,int((y+h/2) * map1.shape[0]))
        x1,y1=map1[temp_y1,temp_x1][0]-x_gap,map1[temp_y1,temp_x1][1]-y_gap
        x2,y2=map1[temp_y2,temp_x2][0]-x_gap,map1[temp_y2,temp_x2][1]-y_gap
        x_,y_,w_,h_=(x1+x2)/2,(y1+y2)/2,(x2-x1),(y2-y1)
        temp_result.append([cls, x_,y_,w_,h_,w* map1.shape[1],h* map1.shape[0]])

    for cls, x, y, w, h,origin_w,origin_h in temp_result:
        x1, y1=x-w/2,y-h/2
        x2, y2=x+w/2,y+h/2
        y1_,x1_=find(x1,y1,map2)
        y2_,x2_=find(x2,y2,map2)
        x_, y_, w_, h_ = (x1_ + x2_) / 2/map2.shape[1], (y1_ + y2_) / 2/map2.shape[0], (x2_ - x1_)/map2.shape[1], (y2_ - y1_)/map2.shape[0]
        w_size,h_size=x2_ - x1_,y2-y1
        iou=min(w_size,origin_w)*min(h_size,origin_h)/(max(w_size,origin_w)*max(h_size,origin_h))
        if iou<0.8:
            if x<map1.shape[1]/2:
                if y<map1.shape[0]/2:
                    x1_,y1_=x2_-origin_w,y2_-origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map2.shape[1], (y1_ + y2_) / 2 / map2.shape[0], (x2_ - x1_) / map2.shape[1], (y2_ - y1_) / map2.shape[0]
                else:
                    y1_,x2_=find(x2,y1,map2)
                    x1_,y2_=x2_-origin_w,y1_+origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map2.shape[1], (y1_ + y2_) / 2 / map2.shape[0], (x2_ - x1_) / map2.shape[1], (y2_ - y1_) / map2.shape[0]
            else:
                if y < map1.shape[0] / 2:
                    y2_,x1_=find(x1,y2,map2)
                    x2_,y1_=x1_+origin_w,y2_-origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map2.shape[1], (y1_ + y2_) / 2 / map2.shape[0], (x2_ - x1_) / map2.shape[1], (y2_ - y1_) / map2.shape[0]
                else:
                    x2_,y2_=x1_+origin_w,y1_+origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map2.shape[1], (y1_ + y2_) / 2 / map2.shape[0], (x2_ - x1_) / map2.shape[1], (y2_ - y1_) / map2.shape[0]

        # y_,x_=find(x,y,map2)
        # x_,y_=x_/map2.shape[1],y_/map2.shape[0]
        # w_,h_=w * map1.shape[1] / map2.shape[1],h*map1.shape[0] / map2.shape[0]
        if w_==0 or h_==0:
            continue
        result.append([cls,x_,y_,w_,h_])
    return result


def get_LL_label_v5(label,map1,map2,x_gap=0,y_gap=0):#相比v4算法没有改进，只是更改了变换方式，从原图标签变换为了校正图像标签
    left_label=[]
    left_label_=[]
    matches_LL=[]
    i=-1
    j=-1
    for cls, x, y, w, h in label:

        origin_w, origin_h=w*map1.shape[1],h*map1.shape[0]

        x1, y1=max(0,int((x-w/2) * map1.shape[1])),max(0,int((y-h/2) * map1.shape[0]))
        x2, y2=min(map1.shape[1]-1,int((x+w/2) * map1.shape[1])),min(map1.shape[0]-1,int((y+h/2) * map1.shape[0]))
        y1_, x1_ = find(x1, y1, map1)
        y2_, x2_ = find(x2, y2, map1)
        x_, y_, w_, h_ = (x1_ + x2_) / 2 / map1.shape[1], (y1_ + y2_) / 2 / map1.shape[0], (x2_ - x1_) / map1.shape[
            1], (y2_ - y1_) / map1.shape[0]
        if w_==0 or h_==0:
            continue
        w_size, h_size = x2_ - x1_, y2 - y1
        iou = min(w_size, origin_w) * min(h_size, origin_h) / (max(w_size, origin_w) * max(h_size, origin_h))
        if iou < 0.8:
            if x < map1.shape[1] / 2:
                if y < map1.shape[0] / 2:
                    x1_, y1_ = x2_ - origin_w, y2_ - origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map1.shape[1], (y1_ + y2_) / 2 / map1.shape[0], (x2_ - x1_) / \
                                     map1.shape[1], (y2_ - y1_) / map1.shape[0]
                else:
                    y1_, x2_ = find(x2, y1, map1)
                    x1_, y2_ = x2_ - origin_w, y1_ + origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map1.shape[1], (y1_ + y2_) / 2 / map1.shape[0], (x2_ - x1_) / \
                                     map1.shape[1], (y2_ - y1_) / map1.shape[0]
            else:
                if y < map1.shape[0] / 2:
                    y2_, x1_ = find(x1, y2, map1)
                    x2_, y1_ = x1_ + origin_w, y2_ - origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map1.shape[1], (y1_ + y2_) / 2 / map1.shape[0], (x2_ - x1_) / \
                                     map1.shape[1], (y2_ - y1_) / map1.shape[0]
                else:
                    x2_, y2_ = x1_ + origin_w, y1_ + origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map1.shape[1], (y1_ + y2_) / 2 / map1.shape[0], (x2_ - x1_) / \
                                     map1.shape[1], (y2_ - y1_) / map1.shape[0]
        left_label.append([cls, x_, y_, w_, h_])
        i=i+1

        x1, y1 = max(0, int((x - w / 2) * map1.shape[1]))-x_gap, max(0, int((y - h / 2) * map1.shape[0]))-y_gap
        x2, y2 = min(map1.shape[1] - 1, int((x + w / 2) * map1.shape[1]))-x_gap, min(map1.shape[0] - 1,
                                                                               int((y + h / 2) * map1.shape[0]))-y_gap

        y1_,x1_=find(x1,y1,map2)
        y2_,x2_=find(x2,y2,map2)
        x_, y_, w_, h_ = (x1_ + x2_) / 2/map2.shape[1], (y1_ + y2_) / 2/map2.shape[0], (x2_ - x1_)/map2.shape[1], (y2_ - y1_)/map2.shape[0]
        if w_==0 or h_==0:
            continue
        w_size,h_size=x2_ - x1_,y2-y1
        iou=min(w_size,origin_w)*min(h_size,origin_h)/(max(w_size,origin_w)*max(h_size,origin_h))
        if iou<0.8:
            if x<map1.shape[1]/2:
                if y<map1.shape[0]/2:
                    x1_,y1_=x2_-origin_w,y2_-origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map2.shape[1], (y1_ + y2_) / 2 / map2.shape[0], (x2_ - x1_) / map2.shape[1], (y2_ - y1_) / map2.shape[0]
                else:
                    y1_,x2_=find(x2,y1,map2)
                    x1_,y2_=x2_-origin_w,y1_+origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map2.shape[1], (y1_ + y2_) / 2 / map2.shape[0], (x2_ - x1_) / map2.shape[1], (y2_ - y1_) / map2.shape[0]
            else:
                if y < map1.shape[0] / 2:
                    y2_,x1_=find(x1,y2,map2)
                    x2_,y1_=x1_+origin_w,y2_-origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map2.shape[1], (y1_ + y2_) / 2 / map2.shape[0], (x2_ - x1_) / map2.shape[1], (y2_ - y1_) / map2.shape[0]
                else:
                    x2_,y2_=x1_+origin_w,y1_+origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map2.shape[1], (y1_ + y2_) / 2 / map2.shape[0], (x2_ - x1_) / map2.shape[1], (y2_ - y1_) / map2.shape[0]

        # y_,x_=find(x,y,map2)
        # x_,y_=x_/map2.shape[1],y_/map2.shape[0]
        # w_,h_=w * map1.shape[1] / map2.shape[1],h*map1.shape[0] / map2.shape[0]

        left_label_.append([cls,x_,y_,w_,h_])
        j = j + 1
        matches_LL.append((i,j,0))

    return left_label,left_label_,matches_LL

def get_R_rectified(label,map1):#使用原生图像上的标签框
    result=[]
    for cls, x, y, w, h in label:

        origin_w, origin_h = w * map1.shape[1], h * map1.shape[0]

        x1, y1 = max(0, int((x - w / 2) * map1.shape[1])), max(0, int((y - h / 2) * map1.shape[0]))
        x2, y2 = min(map1.shape[1] - 1, int((x + w / 2) * map1.shape[1])), min(map1.shape[0] - 1,
                                                                               int((y + h / 2) * map1.shape[0]))
        y1_, x1_ = find(x1, y1, map1)
        y2_, x2_ = find(x2, y2, map1)
        x_, y_, w_, h_ = (x1_ + x2_) / 2 / map1.shape[1], (y1_ + y2_) / 2 / map1.shape[0], (x2_ - x1_) / map1.shape[
            1], (y2_ - y1_) / map1.shape[0]
        if w_ == 0 or h_ == 0:
            continue
        w_size, h_size = x2_ - x1_, y2 - y1
        iou = min(w_size, origin_w) * min(h_size, origin_h) / (max(w_size, origin_w) * max(h_size, origin_h))
        if iou < 0.8:
            if x < map1.shape[1] / 2:
                if y < map1.shape[0] / 2:
                    x1_, y1_ = x2_ - origin_w, y2_ - origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map1.shape[1], (y1_ + y2_) / 2 / map1.shape[0], (x2_ - x1_) / \
                                     map1.shape[1], (y2_ - y1_) / map1.shape[0]
                else:
                    y1_, x2_ = find(x2, y1, map1)
                    x1_, y2_ = x2_ - origin_w, y1_ + origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map1.shape[1], (y1_ + y2_) / 2 / map1.shape[0], (x2_ - x1_) / \
                                     map1.shape[1], (y2_ - y1_) / map1.shape[0]
            else:
                if y < map1.shape[0] / 2:
                    y2_, x1_ = find(x1, y2, map1)
                    x2_, y1_ = x1_ + origin_w, y2_ - origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map1.shape[1], (y1_ + y2_) / 2 / map1.shape[0], (x2_ - x1_) / \
                                     map1.shape[1], (y2_ - y1_) / map1.shape[0]
                else:
                    x2_, y2_ = x1_ + origin_w, y1_ + origin_h
                    x_, y_, w_, h_ = (x1_ + x2_) / 2 / map1.shape[1], (y1_ + y2_) / 2 / map1.shape[0], (x2_ - x1_) / \
                                     map1.shape[1], (y2_ - y1_) / map1.shape[0]
        result.append([cls, x_, y_, w_, h_])
    return result

#计算x一致时的iou
def get_x_area_diff(list1,list2):

    list2=[list2[0],list1[1],list2[2],list2[3],list2[4]]

    area_diff=1-calculate_iou(list1,list2)
    return area_diff

#计算面积的Iou
def get_area_diff(list1,list2):
    area=min(list1[3],list2[3])*min(list1[4],list2[4])
    area1 = list1[3] * list1[4]
    area2 = list2[3] * list2[4]
    area_diff = 1-area / (area1+area2-area)
    return area_diff

def get_xy_diff(list1xy,list2xy):
    # w1=min(abs(list1xy[0][1]-list1xy[1][1]),abs((list1xy[0][1]-list1xy[0][3]/2)-(list1xy[1][1]+list1xy[1][3]/2)),
    #        abs((list1xy[1][1]-list1xy[1][3]/2)-(list1xy[0][1]+list1xy[0][3]/2)))
    # h1=min(abs(list1xy[0][2]-list1xy[1][2]),abs((list1xy[0][2]-list1xy[0][4]/2)-(list1xy[1][2]+list1xy[1][4]/2)),
    #    abs((list1xy[1][2]-list1xy[1][4]/2)-(list1xy[0][2]+list1xy[0][4]/2)))
    # w2 = min(abs(list2xy[0][1] - list2xy[1][1]),
    #          abs((list2xy[0][1] - list2xy[0][3] / 2) - (list2xy[1][1] + list2xy[1][3] / 2)),
    #          abs((list2xy[1][1] - list2xy[1][3] / 2) - (list2xy[0][1] + list2xy[0][3] / 2)))
    # h2 = min(abs(list2xy[0][2] - list2xy[1][2]),
    #          abs((list2xy[0][2] - list2xy[0][4] / 2) - (list2xy[1][2] + list2xy[1][4] / 2)),
    #          abs((list2xy[1][2] - list2xy[1][4] / 2) - (list2xy[0][2] + list2xy[0][4] / 2)))

    w1=abs(list1xy[0][1]-list1xy[1][1])
    h1=abs(list1xy[0][2]-list1xy[1][2])

    w2 = abs(list2xy[0][1] - list2xy[1][1])
    h2 = abs(list2xy[0][2] - list2xy[1][2])

    area = min(w1,w2) * min(h1,h2)
    area1 = w1*h1
    area2 = w2*h2
    # print(area1,area2,area)
    if area1==0 or area2==0 or area==0:
        # print(list1xy,list2xy)
        w1 = abs(list1xy[0][1] - list1xy[1][1]-list1xy[0][3]/2+list1xy[1][3]/2)
        h1 = abs(list1xy[0][2] - list1xy[1][2]-list1xy[0][4]/2+list1xy[1][4]/2)

        w2 = abs(list2xy[0][1] - list2xy[1][1]-list2xy[0][3]/2+list2xy[1][3]/2)
        h2 = abs(list2xy[0][2] - list2xy[1][2]-list2xy[0][4]/2+list2xy[1][4]/2)

        area = min(w1, w2) * min(h1, h2)
        area1 = w1 * h1
        area2 = w2 * h2
    area_diff = 1 - area / (area1 + area2 - area)
    return area_diff

#删除多对多的组合，并保留面积占比最小值
def remove_error_matches(matches):
    unique_matches = {}
    for match in matches:
        i, j, area_diff = match
        unique_matches.setdefault(i, match)
        if area_diff < unique_matches[i][2]:
            unique_matches[i] = match
    matches = list(unique_matches.values())
    unique_matches = {}
    for match in matches:
        i, j, area_diff = match
        unique_matches.setdefault(j, match)
        if area_diff < unique_matches[j][2]:
            unique_matches[j] = match
    return list(unique_matches.values())

#iou计算，输入的是两组检测框
def calculate_iou(box1, box2):

    c1,x1, y1, w1, h1 = box1
    c2,x2, y2, w2, h2 = box2
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2

    # Calculate the intersection area
    intersection_xmin = max(x1_min, x2_min)
    intersection_xmax = min(x1_max, x2_max)
    intersection_ymin = max(y1_min, y2_min)
    intersection_ymax = min(y1_max, y2_max)
    intersection_area = max(0, intersection_xmax - intersection_xmin) * max(0, intersection_ymax - intersection_ymin)

    # Calculate the union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou

if __name__=='__main__':
    m1=[4.796756184945723,11.48006904978894,45.25471130874779]
    m2=[6.569695638719702,11.479524786115496,45.461434170472614]
    print(get_angle(m1, m2))