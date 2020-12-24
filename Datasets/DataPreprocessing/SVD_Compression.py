import cv2
import numpy as np
import os


def methods(sigma, u, v, K):  # svd方法的核心（奇异值，左奇异矩阵，右奇异矩阵，奇异值个数）
    uk = u[:, 0:K]  # 构建左奇异矩阵

    sigmak = np.diag(sigma[0:K])  # 构建奇异值
    vk = v[0:K, ]  # 构建右奇异矩阵
    a = np.dot(np.dot(uk, sigmak), vk)  # 返回奇异值压缩后的矩阵
    a[a < 0] = 0  # 异常数值处理
    a[a > 255] = 255  # 异常数值处理
    return np.rint(a).astype('uint8')  # 整数化


def SVDCompression(img_path, save_dir, K):  # 奇异值个数
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # BGR通道顺序
    u_r, sigma_r, v_r = np.linalg.svd(img[:, :, 2])  # 进行svd分解(返回一个元组)
    u_g, sigma_g, v_g = np.linalg.svd(img[:, :, 1])  # 进行svd分解(返回一个元组)
    u_b, sigma_b, v_b = np.linalg.svd(img[:, :, 0])  # 进行svd分解(返回一个元组)

    R = methods(sigma_r, u_r, v_r, K)
    G = methods(sigma_g, u_g, v_g, K)
    B = methods(sigma_b, u_b, v_b, K)
    new_img = np.stack((B, G, R), 2)  # BGR顺序
    cv2.imwrite(os.path.join(save_dir, img_path.split("/")[-1]), new_img)


if __name__ == '__main__':
    SVDCompression("./0002-image04733.jpg", "./result", 15)
