import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# 日志打印开关
print_switch = False

# loadJsonMatrix函数，用于加载JSON文件中的4✖️4矩阵
def load_json_matrix(jsonFile, paths) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.float64)
    if not os.path.exists(jsonFile):
        print(f"file {jsonFile} not exist")
        return None

    with open(jsonFile, 'r') as f:
        data = json.load(f)
        for path in paths:
            if path not in data:
                print(f"key {path} not exist in {jsonFile}")
                return None
            data = data[path]

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i < len(data) and j < len(data[i]):
                    matrix[i, j] = data[i][j]
                else:
                    print(f"load {jsonFile} data error")
                    return None
    
    return matrix

# gnss到lidar的转换矩阵
def gnss2lidar(clip_path) -> np.ndarray:
    paths = ["gnss-to-lidar-top", "param", "sensor_calib", "data"]
    gnss2lidar = load_json_matrix(os.path.join(clip_path, "calib_extract", "calib_gnss_to_lidar_top.json"), paths)
    if gnss2lidar.shape != (4, 4):
        print("load calib_gnss_to_lidar_top data error")
        return None

    if gnss2lidar[0, 0] > gnss2lidar[0, 1]:
        data = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        gnss2lidar = np.dot(data, gnss2lidar)

    return gnss2lidar

# lidar到车的转换矩阵
def lidar2car(clip_path) -> np.ndarray:
    paths = ["lidar-top-to-car", "param", "sensor_calib", "data"]
    lidar2car = load_json_matrix(os.path.join(clip_path, "calib_extract", "calib_lidar_top_to_car.json"), paths)
    if lidar2car.shape != (4, 4):
        print("load calib_lidar_top_to_car data error")
        return None

    return lidar2car

# gnss到车的转换矩阵
def gnss2car(clip_path) -> np.ndarray:
    gnss2lidar = gnss2lidar(clip_path)
    lidar2car = lidar2car(clip_path)
    gnss2car = np.dot(lidar2car, gnss2lidar)
    if print_switch:
        print("gnss2lidar:")
        print(gnss2lidar)
        print("lidar2car:")
        print(lidar2car)
        print("gnss2car:")
        print(gnss2car)
    return gnss2car


# WGS-84定义的常数，用于CGCS2000系统（与WGS-84非常接近）
a = 6378137.0  # 长半轴（单位：米）
b = 6356752.3142
#f = (a - b) / a
f = 1 / 298.257223563  # 扁率  CGCS2000系统
#f = 1 / 298.257223565  # 扁率  WGS-84
e2 = 2*f - f**2  # 第一偏心率的平方
 
pi = 3.14159265359
# gps经纬度转换到ecef
def llh2ecef(lat, lon, h):
    """将地理坐标（经度、纬度、高程）转换为ECEF坐标系"""
    lat = np.radians(lat)
    lon = np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + h) * np.sin(lat)
    return x, y, z

# enu转换到ecef
'''
功能： enu坐标转化到ecef坐标
输入：
等待转换的ENU坐标   坐标 xEast, yNorth, zUp
GPS第一帧原点      坐标 lat0, lon0, h0
输出：
ecef  坐标 x, y, z
'''
def enu2ecef(east, north, up, lat_ref, lon_ref, h_ref):
 
    # 1 参考GNSS点 转化到ecef
    # 定义参考点的CGCS2000坐标（经度, 纬度, 高度）
    #lon_ref, lat_ref, h_ref = 116.391, 39.907, 50.0  # 示例参考点
    ref_ecef = llh2ecef(lat_ref,lon_ref,h_ref)
 
    ecef_x_ref = ref_ecef[0]
    ecef_y_ref=ref_ecef[1]
    ecef_z_ref=ref_ecef[2]
 
    # 2 等待转换的enu点变换到到ecef坐标系下相对位移
    # 将参考点的地理坐标转换为弧度
    lat_ref = np.radians(lat_ref)
    lon_ref = np.radians(lon_ref)
 
    # ECEF到ENU的旋转矩阵
    R = np.array([
        [-np.sin(lon_ref), np.cos(lon_ref), 0],
        [-np.sin(lat_ref)*np.cos(lon_ref), -np.sin(lat_ref)*np.sin(lon_ref), np.cos(lat_ref)],
        [np.cos(lat_ref)*np.cos(lon_ref), np.cos(lat_ref)*np.sin(lon_ref), np.sin(lat_ref)]
    ])
 
    # 将ENU坐标转换为ECEF坐标
    # 定义ENU坐标（East, North, Up）
    #east, north, up = 100, 200, 30  # 示例ENU坐标
    enu_vector = np.array([east, north, up])
    ecef_vector = R.T @ enu_vector  # 使用矩阵转置（R转置=R逆，ENU到ECEF）进行旋转
 
    # 将ECEF坐标添加到参考点的ECEF坐标
    x = ecef_x_ref + ecef_vector[0]
    y = ecef_y_ref + ecef_vector[1]
    z = ecef_z_ref + ecef_vector[2]
 
    return x,y,z

# ecef转换到enu
'''
功能： ecef坐标转化到enu坐标
输入：
等待转换的ecef坐标   坐标 x, y, z
GPS第一帧原点      坐标 lat0, lon0, h0
输出：
enu  坐标 xEast, yNorth, zUp
'''
def ecef2enu(x, y, z, lat_ref, lon_ref, h_ref):
    # 1 参考GNSS点 转化到ecef
    # 定义参考点的CGCS2000坐标（经度, 纬度, 高度）
    #lon_ref, lat_ref, h_ref = 116.391, 39.907, 50.0  # 示例参考点
    ref_ecef = llh2ecef(lat_ref,lon_ref,h_ref)
 
    ecef_x_ref = ref_ecef[0]
    ecef_y_ref=ref_ecef[1]
    ecef_z_ref=ref_ecef[2]
 
    # 2 等待转换的enu点变换到到ecef坐标系下相对位移
    # 将参考点的地理坐标转换为弧度
    lat_ref = np.radians(lat_ref)
    lon_ref = np.radians(lon_ref)
 
    # ECEF到ENU的旋转矩阵
    R = np.array([
        [-np.sin(lon_ref), np.cos(lon_ref), 0],
        [-np.sin(lat_ref)*np.cos(lon_ref), -np.sin(lat_ref)*np.sin(lon_ref), np.cos(lat_ref)],
        [np.cos(lat_ref)*np.cos(lon_ref), np.cos(lat_ref)*np.sin(lon_ref), np.sin(lat_ref)]
    ])
 
    # 将ECEF坐标转换为ENU坐标
    # 定义ECEF坐标
    #x, y, z = 1.0e7, 4.0e6, 3.0e6  # 示例ECEF坐标
    ecef_vector = np.array([x - ecef_x_ref, y - ecef_y_ref, z - ecef_z_ref])
    enu_vector = R @ ecef_vector
 
    return enu_vector

def gnss2enu(phi, theta, psi) :
    return compute_dcm(phi, theta, psi)

# 方向余弦矩阵（Direction Cosine Matrix， DCM）
def compute_dcm(phi, theta, psi):
    # 绕Z轴旋转
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi), np.cos(phi), 0],
                   [0, 0, 1]])
    # 绕Y轴旋转
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    # 绕X轴旋转
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(psi), -np.sin(psi)],
                   [0, np.sin(psi), np.cos(psi)]])
    
    # 计算方向余弦矩阵
    DCM = Rz @ Ry @ Rx
    return DCM

def compute_dcm2(phi, theta, psi):
    r = R.from_euler('ZYX', [phi, theta, psi], degrees=False)
    matrix = r.as_matrix()
    return matrix

def compute_quaternion(phi, theta, psi):
    r = R.from_euler('ZYX', [phi, theta, psi], degrees=False)
    quaternion = r.as_quat()
    return quaternion

# # 示例：计算给定欧拉角的方向余弦矩阵
# # pitch,yaw,roll = 0.10000000149011612,0.20000000298023224,0.30000001192092896
# phi = 0.20000000298023224  # 绕Z轴旋转（yaw）
# theta =0.10000000149011612  # 绕Y轴旋转（pitch）
# psi = 0.30000001192092896  # 绕X轴旋转（roll）

# # 欧拉角计算方向余弦矩阵(NEU)
# r = R.from_euler('ZYX', [phi, theta, psi], degrees=False)
# matrix = r.as_matrix()
# print(matrix)
# # 方向余弦矩阵转换为欧拉角
# euler = r.as_euler('ZYX', degrees=False)
# print(euler)
# # 方向余弦矩阵转换为四元数
# quaternion = r.as_quat()
# print(quaternion)
# # 四元数转换为方向余弦矩阵
# r = R.from_quat(quaternion)
# matrix = r.as_matrix()
# print(matrix)
# # 欧拉角转换为四元数
# r = R.from_euler('ZYX', [phi, theta, psi], degrees=False)
# quaternion = r.as_quat()
# print(quaternion)
# # 四元数转换为欧拉角
# r = R.from_quat(quaternion)
# euler = r.as_euler('ZYX', degrees=False)
# print(euler)
# # quaterntion.x,quaterntion.y,quaterntion.z,quaterntion.w = 0.14357218146324158,0.06407134979963303,0.09115754812955856,0.9833474159240723

# 捷联惯导到ECEF
def gnss2ecef(east, north, up, lat_ref, lon_ref, h_ref, phi, theta, psi):
    enu2Ecef = enu2ecef(east, north, up, lat_ref, lon_ref, h_ref)
    gnss2Enu = gnss2enu(phi, theta, psi)
    gnss2Ecef = np.dot(enu2Ecef, gnss2Enu)
    return gnss2Ecef

# ECEF到捷联惯导
def ecef2gnss(east, north, up, lat_ref, lon_ref, h_ref, phi, theta, psi):
    gnss2Ecef = gnss2ecef(east, north, up, lat_ref, lon_ref, h_ref, phi, theta, psi)
    return np.linalg.inv(gnss2Ecef)


# ECEF到自车（car）
def ecef2car(params_file_path, east, north, up, lat_ref, lon_ref, h_ref, phi, theta, psi):
    gnss2Car = gnss2car(params_file_path)
    ecef2Gnss = ecef2gnss(east, north, up, lat_ref, lon_ref, h_ref, phi, theta, psi)
    ecef2car = np.dot(gnss2Car, ecef2Gnss)
    return ecef2car

# ECEf到自车（car）
def ecef2car_with_gnss2Car(gnss2Car, east, north, up, lat_ref, lon_ref, h_ref, phi, theta, psi):
    ecef2Gnss = ecef2gnss(east, north, up, lat_ref, lon_ref, h_ref, phi, theta, psi)
    ecef2car = np.dot(gnss2Car, ecef2Gnss)
    return ecef2car

# 经纬度高程到自车（car）
def llh2car(gnss2Car, lng, lat, height, lat_ref, lon_ref, h_ref, phi, theta, psi):
    x, y, z = llh2ecef(lat, lng, height)
    east, north, up = ecef2enu(x, y, z, lat_ref, lon_ref, h_ref)
    ecef2Gnss = ecef2gnss(east, north, up, lat_ref, lon_ref, h_ref, phi, theta, psi)
    ecef2car = np.dot(gnss2Car, ecef2Gnss)
    return ecef2car
