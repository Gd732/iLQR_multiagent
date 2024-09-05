import numpy as np
import ctypes

if __name__ == "__main__":
    lib = ctypes.CDLL('./compute_B_matrix.so')

    # 定义函数参数和返回类型
    lib.compute_B_matrix.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_double, 
                                    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, 
                                    ctypes.c_double, ctypes.c_double, ctypes.c_int]

    # 定义返回类型为 None (void)
    lib.compute_B_matrix.restype = None
    horizon = 10
    B = np.zeros((12, 4, horizon), dtype=np.float64)

    # 调用 C++ 函数
    lib.compute_B_matrix(B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, horizon)

    # 打印结果
    print(B.shape)