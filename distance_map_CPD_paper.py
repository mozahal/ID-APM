import argparse
import os

import cv2

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from flask import Flask, request, jsonify
import base64
import camera_configs_left_middle
import camera_configs_left_right
from calculate_utils_test_v3 import *
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import csv
import time
from ctypes import *
import threading

import numpy as np

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def create_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    weights, imgsz = opt.weights, opt.img_size
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsize = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    # Second-stage classifier
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsize, imgsize).to(device).type_as(next(model.parameters())))
    return model


def detect(model, img0, device):  # 输入图片，返回[x1,y1,x2,y2,预测的物体，置信度]的数组
    # 进行模型配置和启动
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')

    parser.add_argument('--view-img', action='store_true', help='display results', default=False)
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    half = device.type != 'cpu'
    stride = int(model.stride.max())  # model stride
    imgsize = check_img_size(opt.img_size, s=stride)  # check img_size
    # 开始处理图片
    img = letterbox(img0, imgsize, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    # 开始进行识别
    im0s = img0
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=opt.augment)[0]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    result = []  # 输出
    for i, det in enumerate(pred):
        im0 = im0s
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                temp = []
                xyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                # if names[int(cls)]!='eye':
                #     continue
                temp.append(int(cls))
                temp.append(float((xyxy[0] + xyxy[2]) / 2 / img0.shape[1]))
                temp.append(float((xyxy[1] + xyxy[3]) / 2 / img0.shape[0]))
                temp.append(float((xyxy[2] - xyxy[0]) / img0.shape[1]))
                temp.append(float((xyxy[3] - xyxy[1]) / img0.shape[0]))
                # ray_rectified.append(float(conf))
                result.append(temp)


    # result.sort(key=lambda x: (x[0], x[1]))#根据类别和x值进降序配列
    result.sort(key=lambda x: x[1])  # 根据x值进降序配列
    return result, img0

def cv_rectangle(img0, labels,color=(0, 255, 0)):
    for lable in labels:
        n,x,y,w,h=lable
        x1, y1 = int((x - w / 2) * img0.shape[1]), int((y - h / 2) * img0.shape[0])
        x2, y2 = int((x + w / 2) * img0.shape[1]), int((y + h / 2) * img0.shape[0])
        # cv2.rectangle(img0,(x1, y1), (x2, y2), color, 2)
    return img0





def distance_match(left,ray,right,ray_label,iou_threshold=0.5):
    ret=True

    y_gap = int((camera_configs_left_right.size[1] - camera_configs_left_middle.size[1]) / 2)
    x_gap = int((camera_configs_left_right.size[0] - camera_configs_left_middle.size[0]) / 2)

    left_ = left[y_gap:int(camera_configs_left_right.size[1] - y_gap),
            x_gap:int(camera_configs_left_right.size[0] - x_gap)]

    img1_rectified = cv2.remap(left, camera_configs_left_right.left_map1, camera_configs_left_right.left_map2,
                               cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(right, camera_configs_left_right.right_map1, camera_configs_left_right.right_map2,
                               cv2.INTER_LINEAR)
    img11_rectified = cv2.remap(left_, camera_configs_left_middle.left_map1, camera_configs_left_middle.left_map2,
                                cv2.INTER_LINEAR)
    left_label, img_L = detect(model, img1_rectified, device)
    right_label, img_R = detect(model, img2_rectified, device)
    # result_L_, img_L_ = detect(model, img11_rectified, device)


    left_label, left_label_, matches_LL=get_LL_label_v5(left_label,camera_configs_left_right.left_map1,
                                     camera_configs_left_middle.left_map1,x_gap,y_gap)
    right_label=get_R_rectified(right_label,camera_configs_left_right.right_map1)

    # img_L=cv_rectangle(img1_rectified,left_label)
    img_L_=cv_rectangle(img11_rectified,left_label_)
    # img_R=cv_rectangle(img2_rectified,right_label)



    img_M=cv_rectangle(ray,ray_label,color=(0,0,255))
    # img_M=ray
    if left_label==[] or right_label==[]  :
        ret=False
    if ret:

        import cpd_update as custom_cpd
        temp_left_label= [row[1:3] for row in left_label]
        temp_right_label = [row[1:3] for row in right_label]
        matches_LR = custom_cpd.match_points(np.array(temp_left_label), np.array(temp_right_label))

        temp_left_label_ = [row[1:3] for row in left_label_]
        temp_ray_label = [row[1:3] for row in ray_label]
        matches_L_M = custom_cpd.match_points(np.array(temp_left_label_), np.array(temp_ray_label))
        # matches_LR = match_detections(left_label, right_label, area_threshold)
        # # matches_LL=match_detections(result_L,result_L_,area_threshold=area_threshold*camera_configs_left_right.size[0]
        # #                             *camera_configs_left_right.size[1]/camera_configs_left_middle.size[0]/camera_configs_left_middle.size[1])
        #
        # matches_L_M = match_detections(left_label_, ray_label, area_threshold=0.6)
        L_in_M = [row[0] for row in matches_L_M]

        L_in_L_ = [row[0] for row in matches_LL]
        match_area = []
        deviation = []

        if len(matches_LR) == 0 or len(matches_LL) == 0 or len(matches_L_M) == 0:
            ret = False

        if ret:

            # 匹配框匹配遗失计算
            # print('LR_match:',len(matches_LR))
            i, j = 0, 0
            for match in matches_LR:
                l, r = match
                if not l in L_in_L_:
                    continue
                else:
                    i = i + 1
                    l_ = matches_LL[L_in_L_.index(l)][1]
                if not l_ in L_in_M:
                    continue
                else:
                    m = matches_L_M[L_in_M.index(l_)][1]
                    j = j + 1
            if i * j == 0:
                ret = False

            if ret:
                global temp_i, temp_n
                temp_i = temp_i + len(matches_LR)
                temp_n = temp_n + min(len(left_label), len(right_label))

                print('{}\ttemp_i:{},temp_n:{}'.format(len(matches_LR), temp_i, temp_n))
                print('LL_overage:', i, '一级遗留:', i / len(matches_LR))
                print('L_Moverage:', j, '二级遗留:', j / i)
                print('最终遗留：', j / len(matches_LR))

                for i, match in enumerate(matches_LR):
                    cls, x, y, w, h = left_label[match[0]]

                    left_x1, left_y1 = int((x - w / 2) * img_L.shape[1]), int((y - h / 2) * img_L.shape[0])

                    cls, x, y, w, h = right_label[match[1]]
                    right_x1, right_y1 = int((x - w / 2) * img_R.shape[1]), int((y - h / 2) * img_R.shape[0])
                    cv2.putText(img_L, text=str(i) + 'cls:' + str(cls), org=(left_x1, left_y1 + 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                                thickness=5)
                    cv2.putText(img_R, text=str(i), org=(right_x1, right_y1 + 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                                thickness=5)

                iou = []
                for match in matches_LR:
                    l, r = match
                    if not l in L_in_L_:
                        continue
                    else:
                        l_ = matches_LL[L_in_L_.index(l)][1]

                    distance = get_depth(left_label[l][1] * img_L.shape[1], right_label[r][1] * img_R.shape[1],
                                         img_L.shape[1]) * distance_wight + distance_bias
                    cls = left_label_[l_][0]
                    x = getxr(left_label_[l_][1] * img_L_.shape[1], distance) / img_M.shape[1]

                    y, w, h = left_label_[l_][2:]

                    x1, y1 = int((x - w / 2) * img_M.shape[1]), int((y - h / 2) * img_M.shape[0])
                    x2, y2 = int((x + w / 2) * img_M.shape[1]), int((y + h / 2) * img_M.shape[0])
                    # cv2.rectangle(img_M, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    # 欧氏距离计算

                    fit_x1, fit_x2 = int(
                        x * img_M.shape[1] * map_weight_x + map_bias_x - (w / 2) * img_M.shape[1]), int(
                        x * img_M.shape[1] * map_weight_x + map_bias_x + (w / 2) * img_M.shape[1])

                    fit_y1, fit_y2 = int(
                        y * img_M.shape[0] * map_weight_y + map_bias_y - (h / 2) * img_M.shape[0]), int(
                        y * img_M.shape[0] * map_weight_y + map_bias_y + (h / 2) * img_M.shape[0])

                    # cv2.rectangle(img_M, (fit_x1, fit_y1), (fit_x2, fit_y2), (0, 255, 0), 2)
                    # 计算iou,辅助计算miou
                    temp = []
                    m = None
                    for tempindex, lable in enumerate(ray_label):
                        temp_iou = calculate_iou(lable, [left_label_[l_][0], ((fit_x1 + fit_x2) / 2) / img_M.shape[1],
                                                         ((fit_y1 + fit_y2) / 2) / img_M.shape[0], w, h])
                        # print('temp_iou:',temp_iou)
                        if temp_iou > iou_threshold:
                            temp.append([lable[0], temp_iou, tempindex])


                    # for tempindex, lable in enumerate(ray_label):
                    #     temp_distance = ((lable[1]-((fit_x1 + fit_x2) / 2) / img_M.shape[1])**2+(lable[2]-((fit_y1 + fit_y2) / 2) / img_M.shape[0])**2)**(1/2)
                    #     # print('temp_iou:',temp_iou)
                    #     if temp_distance > iou_threshold:
                    #         temp.append([lable[0], temp_distance, tempindex])
                    temp.sort(key=lambda x: x[1], reverse=False)
                    if len(temp) > 0:
                        iou.append(temp[0][0:2])
                        m = temp[0][2]
                    else:
                        temp=[]
                        for tempindex, lable in enumerate(ray_label):
                            # print('label:',lable)
                            distance_=((lable[1]* img_M.shape[1]-((fit_x1 + fit_x2) / 2) )**2+
                                       (lable[2]* img_M.shape[0]-((fit_y1 + fit_y2) / 2) )**2)**(1/2)
                            # print('temp_iou:',temp_iou)
                            if distance_<80:
                                temp.append([lable[0], distance_, tempindex])
                        temp.sort(key=lambda x: x[1], reverse=False)
                        if len(temp) > 0:
                            m = temp[0][2]
                        else:
                            continue

                    '''  
                    if l_ not in L_in_M:
                        continue
                    m = matches_L_M[L_in_M.index(l_)][1]
                    '''
                    # if l_ not in L_in_M:
                    #     continue
                    # m = matches_L_M[L_in_M.index(l_)][1]

                    print((ray_label[m][1] - x) * img_M.shape[1], '欧氏距离：', (
                            (ray_label[m][1] * img_M.shape[1] - x * img_M.shape[1]) ** 2 + (
                            ray_label[m][2] * img_M.shape[0] - y * img_M.shape[0]) ** 2) ** 0.5, 'aim coordinate:',
                          ray_label[m][1] * img_M.shape[1], ray_label[m][2] * img_M.shape[0])

                    deviation.append([ray_label[m][0], (ray_label[m][1] - x) * img_M.shape[1],
                                      (ray_label[m][2] - y) * img_M.shape[0],
                                      calculate_iou(ray_label[m], [cls, x, y, w, h]), x * img_M.shape[1],
                                      ray_label[m][1] * img_M.shape[1],
                                      y * img_M.shape[0], ray_label[m][2] * img_M.shape[0],
                                      x * img_M.shape[1] * map_weight_x + map_bias_x,
                                      y * img_M.shape[0] * map_weight_y + map_bias_y])
                    match_area.append([cls, x, y, w, h])
                    # cv2.rectangle(img_M, (x1, y1), (x2, y2), (0, 0, 0), 1)

    if ret == False:
        return ret, None, img_M, None, img_L, img_R, img_L_, None
    return ret, match_area, img_M, deviation, img_L, img_R, img_L_, iou




if __name__ == '__main__':
    model = create_model()
    device = torch.device("cuda:0")
    #初始化
    temp_i = 0
    temp_n = 0

    number_of_cls=0
    number_of_cls_l = 0
    number_of_cls_r = 0


    path='G:/my_code_project/triple_camera_plus/data_for_paper/mouse'
    distance_wight=1
    distance_bias = 0

    map_weight_x = 0.9968
    map_bias_x = -24.9107

    map_weight_y = 1.0036
    map_bias_y = 4.5612

    map_weight_x = 0.9857
    map_bias_x = -18.6324

    map_weight_y = 0.9724
    map_bias_y = 24.9146

    map_weight_x = 0.9709
    map_bias_x = -2.1490

    map_weight_y = 0.9651
    map_bias_y = 28.1599


    map_weight_x = 0.9616
    map_bias_x = 6.5459

    map_weight_y = 0.9616
    map_bias_y = 30.3178


    area_threshold=0.8#面积比阈值过滤
    iou_threshold=0.0#iou阈值


    iou=[]
    x=[]
    RMSE_=[]
    cls_number = [776,898]
    path_l=path+'/left/'
    path_m=path+'/ray_rectified/'
    path_r=path+'/right/'


    path_m_txt=path+'/label/ray/'
    path_l_txt =path+ '/label/left/'
    path_r_txt =path+ '/label/right/'

    path_result_save=os.path.dirname(os.path.abspath(__file__))

    history = open(r'./deviation.txt', 'w', encoding='utf-8')
    history.write('class\tdeviation x\tdeviation y\tIou\tx_map\tx_aim\ty_map\ty_aim\tx_fit\ty_fit\tdeviation x\tdeviation y\n')

    csvfilex = open('error_x.csv', 'w', newline='')
    writerx = csv.writer(csvfilex)
    writerx.writerow(['x','aim_x'])

    csvfiley = open('error_y.csv', 'w', newline='')
    writery = csv.writer(csvfiley)
    writery.writerow(['y', 'aim_y'])

    iou_remember = open(r'./iou.txt', 'w', encoding='utf-8')

    txt_Paths=os.listdir(path_m_txt)

    for txt in txt_Paths:
        print(txt)
        yolo_txt = path_m_txt + txt

        left = cv2.imread(path_l + txt.replace('ray', 'three_left').replace('txt', 'jpg'))
        middle = cv2.imread(path_m + txt.replace('txt', 'jpg'))
        right = cv2.imread(path_r + txt.replace('ray', 'three_right').replace('txt', 'jpg'))

        yolo_label = read_yolo_label_file(yolo_txt)
        # yolo_label.sort(key=lambda x: (x[0], x[1]))
        yolo_label.sort(key=lambda x: x[1])
        ret, match_area, img_M, deviation, img_L, img_R, img_L_, temp_iou = distance_match(left, middle, right,
                                                                                           yolo_label, iou_threshold)
        cv2.imwrite(path_result_save + '/result/map/' + txt.replace('txt', 'jpg'), img_M)
        cv2.imwrite(path_result_save + '/result/left/left' + txt.replace('txt', 'jpg').replace('ray', ''), img_L)
        cv2.imwrite(path_result_save + '/result/right/right' + txt.replace('txt', 'jpg').replace('ray', ''), img_R)
        cv2.imwrite(path_result_save + '/result/left_/left_' + txt.replace('txt', 'jpg').replace('ray', ''), img_L_)
        print('number of cls:',number_of_cls)
        print('number of cls_l:', number_of_cls_l)
        print('number of cls_r:', number_of_cls_r)

        if ret==False:
            continue
        iou=iou+temp_iou
        print('number of iou:',len(iou))
        # cv2.imwrite(path_result_save+'/result/map/'+txt.replace('txt','jpg'),img_M)
        # cv2.imwrite(path_result_save + '/result/left/left' + txt.replace('txt', 'jpg').replace('ray',''), img_L)
        # cv2.imwrite(path_result_save + '/result/right/right' + txt.replace('txt', 'jpg').replace('ray',''), img_R)
        # cv2.imwrite(path_result_save + '/result/left_/left_' + txt.replace('txt', 'jpg').replace('ray',''), img_L_)
        with open(path_result_save+'/result_txt/map/'+txt, 'w') as f:
            for result in match_area:
                f.write(
                    '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(result[0], result[1], result[2], result[3], result[4]) + '\n')
        # with open(path_result_save+'/result_txt/yolov5/'+txt, 'w') as f:
        #     for result in yolo_M:
        #         f.write(
        #             '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(result[0], result[1], result[2], result[3], result[4]) + '\n')
        for i in deviation:
            history.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[5]-i[8],i[7]-i[9]))
            writerx.writerow(i[4:6])
            writery.writerow(i[6:8])
            RMSE_.append([i[0],(i[5]-i[8])*(i[5]-i[8])+(i[7]-i[9])*(i[7]-i[9])])
            x.append(i[1])
    print('sum_RMSE:',sum([row[1] for row in RMSE_]))
    if len(RMSE_) > 0:

        classes = set([row[0] for row in RMSE_])
        for cls in classes:
            class_RMSE = [x for x in RMSE_ if x[0] == cls]
            if len(class_RMSE) > 0:
                m = (sum([row[1] for row in class_RMSE]) / len(class_RMSE))**0.5
                print('class {}:{}'.format(cls, m))
                iou_remember.write('class {} RMSE:{}\n'.format(cls, m))
        m = (sum([row[1] for row in RMSE_]) / len(RMSE_))**0.5
        iou_remember.write('all of RMSE:{}\n'.format(m))
    if len(iou) > 0:
        iou_remember.write('拟合前的iou：\n')
        print('拟合前的iou：')
        classes = set([row[0] for row in iou])
        for cls in classes:
            class_iou = [x for x in iou if x[0] == cls]
            if len(class_iou) > 0:
                moiu = sum([row[1] for row in class_iou]) / len(class_iou)
                print('class {}:{}'.format(cls, moiu))
                iou_remember.write('class {}:{} {}\n'.format(cls, moiu,100*len(class_iou)/cls_number[cls]))
        moiu = sum([row[1] for row in iou]) / len(iou)
        print('mIou:', moiu)
        iou_remember.write('mIou:{}\n'.format(moiu))
        print('匹配框数量',len(iou))
        iou_remember.write('匹配框数量{} {}\n'.format(len(iou),100*len(iou)/(cls_number[0]+cls_number[1])))
        iou_remember.write('综合： {}\n'.format(moiu* 100 * len(iou) / (cls_number[0] + cls_number[1])))
    if len(x) > 0:
        x_ = sum(x) / len(x)
        print('average deviation x:', x_)
