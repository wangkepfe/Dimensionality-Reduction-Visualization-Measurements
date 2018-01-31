import numpy as np
import sys
import matplotlib.pyplot as plt

def main():
    #file1 = sys.argv[1]
    #file2 = sys.argv[2]
    #file3 = sys.argv[3]

    file_input = "fashion-mnist_train5000.txt"
    file_trimap = "triMap_result.txt"
    file_tsne = "tsne_result.txt"
    file_largevis = "largeVis_result.txt"
    file_res_trimap = "twct_res_trimap.txt"
    file_res_tsne = "twct_res_tsne.txt"
    file_res_largevis = "twct_res_largevis.txt"

    x = np.loadtxt(file_input)
    y1 = np.loadtxt(file_trimap)
    y2 = np.loadtxt(file_tsne)
    y3 = np.loadtxt(file_largevis)    

    #trimap
    res = np.array([])
    print("trimap")
    for i in range(1, 50):
        print('k =', i)
        ans = trustworthiness_continuity(x, y1, i)
        if i == 1:
            res = ans
        else:
            res = np.vstack([res, ans])
        m1_1 = ans[0]
        m2_1 = ans[1]
    np.savetxt(file_res_trimap, res)

    #tsne
    res = np.array([])
    for i in range(1, 50):
        print('k = %d', i)
        ans = trustworthiness_continuity(x, y2, i)
        if i == 1:
            res = ans
        else:
            res = np.vstack([res, ans])
        m1_2 = ans[0]
        m2_2 = ans[1]
    np.savetxt(file_res_tsne, res)
    
    #largevis
    print("largevis")
    res = np.array([])
    for i in range(1, 50):
        print('k = %d', i)
        ans = trustworthiness_continuity(x, y3, i)
        if i == 1:
            res = ans
        else:
            res = np.vstack([res, ans])
        m1_3 = ans[0]
        m2_3 = ans[1] 
    np.savetxt(file_res_largevis, res)

    f = plt.figure(1)
    plt.title('m1')
    plt.xlabel('k')
    plt.ylabel('m1 value')
    plt.plot(list(range(1,50)), m1_1, color="red", label="TriMap")
    plt.plot(list(range(1,50)), m1_2, color="blue", label="t-SNE")
    plt.plot(list(range(1,50)), m1_3, color="green", label="LargeVis")
    plt.grid(True)
    plt.legend()
    f.show()

    g = plt.figure(1)
    plt.title('m1')
    plt.xlabel('k')
    plt.ylabel('m1 value')
    plt.plot(list(range(1,50)), m2_1, color="red", label="TriMap")
    plt.plot(list(range(1,50)), m2_2, color="blue", label="t-SNE")
    plt.plot(list(range(1,50)), m2_3, color="green", label="LargeVis")
    plt.grid(True)
    plt.legend()
    g.show()
    

def trustworthiness_continuity(x, y, k = 50):
    
    n, dim = x.shape
    m1 = 0.0
    m2 = 0.0
    A = 2.0 / (n * k * (2 * n + 3 * k - 1))

    for i in range(n):
        d_x = np.sum(np.square(x - x[i]), axis = -1)
        d_y = np.sum(np.square(y - y[i]), axis = -1)

        r_x = sorted(range(n), key = lambda j: d_x[j])
        r_y = sorted(range(n), key = lambda j: d_y[j])
        r_x = np.array(r_x)
        r_y = np.array(r_y)

        for j in range(1, k):
            t = np.nonzero(r_x == r_y[j])
            if int(t[0]) > k:
                m1 += int(t[0]) - k

            t = np.nonzero(r_y == r_x[j])
            if int(t[0]) > k:
                m2 += int(t[0]) - k

        if i % 1000 == 0:
            print('step %d/%d, m1 temp: %f, m2 temp: %f' % (i, n, m1, m2))

    m1 = 1 - A * m1
    m2 = 1 - A * m2



    print('m1 = %f' % m1)
    print('m2 = %f' % m2)
    print('trustworthiness_continuity mean = %f' % ((m1 + m2) / 2))

    return np.asarray([m1, m2, (m1 + m2) / 2])

if __name__ == '__main__':
    main()
