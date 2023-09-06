import cv2
import numpy as np

# 读取图像并进行预处理
image = cv2.imread('1.PNG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
image_height, image_width = image.shape[0],image.shape[1]
# 轮廓检测
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours),hierarchy)

left_contour = contours[1]
right_contour = contours[0]

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
    line_start = (middle_point[0], 0)
    line_end = (middle_point[0], image_height)
else:
    # 计算质心连线的中点
    midpoint = ((left_center[0] + right_center[0]) // 2, (left_center[1] + right_center[1]) // 2)

    # 计算质心连线的垂直斜率
    perpendicular_slope = -1 / slope

    # 计算垂直平分线的斜率
    line_slope = perpendicular_slope

    # 在图像上绘制分割线
    line_start = (midpoint[0] - 10, int(midpoint[1] - 10 * line_slope))
    line_end = (midpoint[0] + 100, int(midpoint[1] + 100 * line_slope))
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
cv2.waitKey(0)
cv2.destroyAllWindows()