import re
import dlib
import cv2
import numpy as np
import glob

# 读取示例图像
filename_src = "I2G/man.jpeg"
filename_dst = "I2G/man2.jpeg"

img_src = cv2.imread(filename_src)
img_dst = cv2.imread(filename_dst)
img_src = cv2.resize(img_src,(256,256),interpolation= cv2.INTER_AREA)
img_dst= cv2.resize(img_dst,(256,256),interpolation= cv2.INTER_AREA)

img_dst_warped = np.copy(img_dst) # 目标脸的备份

idx_fig = 1

points_src = np.load('I2G/man.npy')
points_src = points_src.astype('int')
points_src = points_src.tolist() # ndarray 不支持插入

points_dst = np.load('I2G/man2.npy')
points_dst = points_dst.astype('int')
points_dst = points_dst.tolist() 

#print(points_dst)
# convex 包含的点集
hull_pt_src = []
hull_pt_dst = []

hull_pt_indices = cv2.convexHull(np.array(points_dst),
                                 returnPoints = False)
hull_pt_indices = hull_pt_indices.flatten()  # 凸包的下标

# print(hull_pt_indices)
for idx_pt in hull_pt_indices:
    hull_pt_src.append(points_src[idx_pt])
    hull_pt_dst.append(points_dst[idx_pt])

def draw_delaunay(img,shape):
    rect = (0, 0, 256,256)
    subdiv = cv2.Subdiv2D(rect)
    for r in shape:
        subdiv.insert(r)

    white = (255,255,255)
    trangleList = subdiv.getTriangleList()
    for t in trangleList:
        pt1 = (int(t[0]),int(t[1]))
        pt2 = (int(t[2]),int(t[3]))
        pt3 = (int(t[4]),int(t[5]))
        cv2.line(img,pt1,pt2,(255,255,255),1)
        cv2.line(img,pt2,pt3,(255,255,255),1)
        cv2.line(img,pt3,pt1,(255,255,255),1)
        cv2.imshow('I',img)           
        cv2.waitKey(100)

    cv2.waitKey(0)

#draw_delaunay(img_src,points_src)
#draw_delaunay(img_dst,points_dst)

def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def cal_delaunay_tri(rect, points):
    """计算狄洛尼三角剖分"""
    
    subdiv = cv2.Subdiv2D(rect)

    # 逐点插入
    for pt in points:
        # subdiv.insert 输入类型：tuple
        subdiv.insert(tuple(pt))

    lst_tri = subdiv.getTriangleList()

    # 狄洛尼三角网格顶点索引
    lst_delaunay_tri_pt_indices = []

    for tri in lst_tri:
        lst_tri_pts = [(tri[0], tri[1])]
        lst_tri_pts.append((tri[2], tri[3]))
        lst_tri_pts.append((tri[4], tri[5]))
        
        # 查询三角网格顶点索引
        lst_pt_indices = []
        for tri_pt in lst_tri_pts:

            for idx_pt in range(len(points)):

                if (abs(tri_pt[0] - points[idx_pt][0]) < 1) and \
                    (abs(tri_pt[1] - points[idx_pt][1]) < 1):
                    lst_pt_indices.append(idx_pt)

        lst_delaunay_tri_pt_indices.append(lst_pt_indices)

    return lst_delaunay_tri_pt_indices

rect = (0,0,256,256)
lst_delaunay_tri_pt_indices = cal_delaunay_tri(rect, hull_pt_dst) # 凸包三角剖分对应的顶点下标
# print(lst_delaunay_tri_pt_indices) 
#----------------------------------------------------------------------
def warp_affine(img_src, tri_src, tri_dst, size):
    """仿射"""

    # 仿射矩阵
    mat_warp = cv2.getAffineTransform(np.float32(tri_src), np.float32(tri_dst))

    # 仿射变换
    img_dst = cv2.warpAffine(img_src, mat_warp, (size[0], size[1]), None,
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

    return img_dst


#----------------------------------------------------------------------
def warp_tri(img_src, img_dst, tri_src, tri_dst, alpha=1) :
    """仿射三角剖分，源图像到目标图像"""

    # 三角区域框
    rect_src = cv2.boundingRect(np.array(tri_src))
    rect_dst = cv2.boundingRect(np.array(tri_dst))
    
    # 三角形顶点相对于三角区域框的偏移
    tri_src_to_rect = [(item[0] - rect_src[0], item[1] - rect_src[1])
                       for item in tri_src]
    tri_dst_to_rect = [(item[0] - rect_dst[0], item[1] - rect_dst[1])
                       for item in tri_dst]
    
    # 蒙板
    mask = np.zeros((rect_dst[3], rect_dst[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.array(tri_dst_to_rect), (1, 1, 1), 16, 0)
    
    # 截取三角区域框中的源图像
    img_src_rect = img_src[rect_src[1] : rect_src[1] + rect_src[3],
                           rect_src[0] : rect_src[0] + rect_src[2]]
    
    size = (rect_dst[2], rect_dst[3])
    
    # 三角区域框仿射
    img_src_rect_warpped = warp_affine(img_src_rect, tri_src_to_rect, tri_dst_to_rect, size)
    
    # 蒙板 * 透明度
    mask *= alpha
    # 目标图像 = 目标图像 * (1 - 蒙板) + 源图像 * 蒙板
    img_dst[rect_dst[1] : rect_dst[1] + rect_dst[3],
            rect_dst[0] : rect_dst[0] + rect_dst[2]] = \
        img_dst[rect_dst[1] : rect_dst[1] + rect_dst[3],
                rect_dst[0] : rect_dst[0] + rect_dst[2]] * (1 - mask) + \
        img_src_rect_warpped * mask
    
# 狄洛尼三角剖分仿射
for tri_pt_indices in lst_delaunay_tri_pt_indices:
    
    # 源图像、目标图像三角顶点坐标
    tri_src = [hull_pt_src[tri_pt_indices[idx]] for idx in range(3)]
    tri_dst = [hull_pt_dst[tri_pt_indices[idx]] for idx in range(3)]

    warp_tri(img_src, img_dst_warped, tri_src, tri_dst, 1)

# 和谐化
mask = np.zeros(img_dst.shape, dtype=img_dst.dtype)
cv2.fillConvexPoly(mask, np.array(hull_pt_dst), (255, 255, 255)) #蒙版

rect = cv2.boundingRect(np.float32([hull_pt_dst]))

center = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2) # 包围矩形的中心

img_dst_src_grad = cv2.seamlessClone(img_dst_warped, img_dst,
                                     mask, center, cv2.NORMAL_CLONE)
img_dst_mix_grad = cv2.seamlessClone(img_dst_warped, img_dst,
                                     mask, center, cv2.MIXED_CLONE)
img_dst_mono_grad = cv2.seamlessClone(img_dst_warped, img_dst,
                                     mask, center, cv2.MONOCHROME_TRANSFER)

#plot_imgs(idx_fig, [img_src, img_dst, img_dst_warped, 
# img_dst_src_grad, img_dst_mix_grad, img_dst_mono_grad])

cv2.imshow('I',np.hstack([img_dst, img_dst_warped, img_dst_src_grad, img_dst_mix_grad, img_dst_mono_grad]))
cv2.waitKey(0)
