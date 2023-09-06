import cv2
import numpy as np

image = cv2.imread('1.jpg')
filtered_image = cv2.blur(image, (3, 3))
# cv2.imshow('filtered_image', filtered_image)

# Gamma值
gamma = 0.7

# 将像素值范围映射到0-1之间
normalized_image = filtered_image / 255.0

# Gamma校正
gamma_corrected = np.power(normalized_image, gamma)

# 将像素值重新映射回0-255之间
gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
cv2.imshow('gamma_corrected', gamma_corrected)
# cv2.imwrite('gamma.png',gamma_corrected)

# 将图片转换为HSV颜色空间
hsv_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)

# def on_trackbar_change(pos):
#     # 读取滑动条的当前位置
#     hue_min = cv2.getTrackbarPos('Hue Min', 'Color Selector')
#     hue_max = cv2.getTrackbarPos('Hue Max', 'Color Selector')
#     saturation_min = cv2.getTrackbarPos('Saturation Min', 'Color Selector')
#     saturation_max = cv2.getTrackbarPos('Saturation Max', 'Color Selector')
#     value_min = cv2.getTrackbarPos('Value Min', 'Color Selector')
#     value_max = cv2.getTrackbarPos('Value Max', 'Color Selector')

#     # 创建颜色阈值范围
#     lower_color = np.array([hue_min, saturation_min, value_min], dtype=np.uint8)
#     upper_color = np.array([hue_max, saturation_max, value_max], dtype=np.uint8)

#     # 应用颜色阈值，获取二值图像
#     mask = cv2.inRange(hsv_image, lower_color, upper_color)

#     # 对二值图像进行形态学操作，以去除噪声或连接区域
#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#     # 将二值图像与原始图像进行按位与操作，得到分割结果
#     seg_image = cv2.bitwise_and(gamma_corrected, gamma_corrected, mask=mask)

#     # 显示原始图像和分割结果
#     cv2.imshow('Segmentation Result', seg_image)

# # 创建一个窗口
# cv2.namedWindow('Color Selector')

# # 创建滑动条用于选择颜色阈值
# cv2.createTrackbar('Hue Min', 'Color Selector', 0, 179, on_trackbar_change)
# cv2.createTrackbar('Hue Max', 'Color Selector', 179, 179, on_trackbar_change)
# cv2.createTrackbar('Saturation Min', 'Color Selector', 0, 255, on_trackbar_change)
# cv2.createTrackbar('Saturation Max', 'Color Selector', 255, 255, on_trackbar_change)
# cv2.createTrackbar('Value Min', 'Color Selector', 0, 255, on_trackbar_change)
# cv2.createTrackbar('Value Max', 'Color Selector', 255, 255, on_trackbar_change)

# # 调用一次回调函数，确保初始效果可见
# on_trackbar_change(0)

# 创建颜色阈值范围
lower_color = np.array([0,0,211])
upper_color = np.array([25,100,225])

# 应用颜色阈值，获取二值图像
mask = cv2.inRange(hsv_image, lower_color, upper_color)

# 对二值图像进行形态学操作，以去除噪声或连接区域
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 将二值图像与原始图像进行按位与操作，得到分割结果
seg_image = cv2.bitwise_and(gamma_corrected, gamma_corrected, mask=mask)
# cv2.imshow('Segmentation Result', seg_image)

# 将图像转换为灰度图像
gray_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)

# # 对灰度图像进行二值化处理
# threshold_value = 127  # 阈值
# max_value = 255  # 最大值
# _, binary_image = cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY)

# # 显示灰度图像和二值化图像
# cv2.imshow("Gray Image", gray_image)
# cv2.imshow("Binary Image", binary_image)

edges = cv2.Canny(gray_image, 50, 150)
cv2.imshow("edges", edges)

image_height, image_width = image.shape[0],image.shape[1]
# 轮廓检测
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours[0]),len(contours[1]))

# 按每个contour的长度从大到小排序,防止有多个contours的情况
contours= sorted(contours, key=len, reverse=True)

left_contour = contours[0]
right_contour = contours[1]

# 计算左侧轮廓的中心点
left_moment = cv2.moments(left_contour)
left_center_x = int(left_moment['m10'] / left_moment['m00'])
left_center_y = int(left_moment['m01'] / left_moment['m00'])
left_center = (left_center_x, left_center_y)

# 计算右侧轮廓的中心点
right_moment = cv2.moments(right_contour)
right_center_x = int(right_moment['m10'] / right_moment['m00'])
right_center_y = int(right_moment['m01'] / right_moment['m00'])
right_center = (right_center_x, right_center_y)

# 在原始图像上绘制质心点
point_color = (0, 0, 255)  # 红色点
point_radius = 1
point_thickness = -1  # 实心圆

# 在原始图像上分别绘制两个质心点
cv2.circle(image, left_center, point_radius, point_color, point_thickness)
cv2.circle(image, right_center, point_radius, point_color, point_thickness)

# 计算质心连线的斜率
slope = (right_center[1] - left_center[1]) / (right_center[0] - left_center[0])
if slope == 0:
    # 计算中间点的平均位置
    middle_point = ((left_center[0] + right_center[0]) // 2, (left_center[1] + right_center[1]) // 2)

    # 在图像上绘制分割线
    line_start = (middle_point[0], middle_point[1] - 10)
    line_end = (middle_point[0], middle_point[1] + 10)
else:
    # 计算质心连线的中点
    midpoint = ((left_center[0] + right_center[0]) // 2, (left_center[1] + right_center[1]) // 2)

    # 计算质心连线的垂直斜率
    perpendicular_slope = -1 / slope

    # 计算垂直平分线的斜率
    line_slope = perpendicular_slope

    # 在图像上绘制分割线
    line_start = (midpoint[0] - 10, int(midpoint[1] - 10 * line_slope))
    line_end = (midpoint[0] + 10, int(midpoint[1] + 10 * line_slope))

line_thickness = 1
line_color = (255, 255, 0)  # 蓝色线条

# 创建一个空白图像用于绘制分割线
line_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# 在空白图像上绘制直线
cv2.line(line_image, line_start, line_end, line_color, line_thickness)

# 将绘制好的直线与原始图像进行叠加
result_image = cv2.addWeighted(image, 1, line_image, 0.5, 0)
# 显示结果图像
cv2.imshow("Result", result_image)

# 等待键盘输入，按下任意键退出
cv2.waitKey(0)

# 关闭窗口
cv2.destroyAllWindows()