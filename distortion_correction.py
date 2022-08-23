import cv2
import numpy as np


def undistort(frame):

    # cameraParams.IntrinsicMatrix
    k = np.array([    
        [1573.01060319569, 0, 0],
        [1.07182734755906, 1573.94527138598, 0],
        [1311.08454512324, 751.715630691956, 1]
    ])

    K = k.T

    # cameraParams.RadialDistortion -0.345179915419884	0.101735998044953
    k1, k2 = -0.345179915419884,	0.101735998044953

    # cameraParams.TangentialDistortion 7.43477036082965e-05	0.000456238468165776
    p1, p2 = 7.43477036082965e-05,	0.000456238468165776

    # 畸变系数
    D = np.array([
        k1, k2, p1, p2, 0
    ])
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)


if __name__ == '__main__':
    img_35 = cv2.imread('cam35.png')
    cv2.imwrite('img_35_undistort.png', undistort(img_35))
    
    img_37 = cv2.imread('cam37.png')
    cv2.imwrite('img_37_undistort.png', undistort(img_37))
    
    # cv2.imshow('img_35', undistort(img_35))
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()