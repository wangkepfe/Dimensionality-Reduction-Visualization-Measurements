import numpy as np
import sys

def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]

    x = np.loadtxt(file1)
    y = np.loadtxt(file2)

    res = precision_recall(x, y)

    np.savetxt(file3, res)

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = np.sum(P)
    H = np.sum(P * (beta * D + np.log(sumP))) / sumP
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

def precision_recall(x, y, perplexity = 30.0):
    n, dim = x.shape
    x -= np.min(x)
    x /= np.max(x)
    x -= np.mean(x, axis=0)
    y -= np.min(y)
    y /= np.max(y)
    y -= np.mean(y, axis=0)

    p = x2p(x);
    q = x2p(y);

    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)

    recall = np.sum(p * np.log(p / q));
    precision = np.sum(q * np.log(q / p));

    print('precision loss = %f' % precision)
    print('recall loss = %f' % recall)
    print('precision-recall mean = %f' % ((recall + precision) / 2))

    return np.asarray([precision, recall, (recall + precision) / 2])

if __name__ == '__main__':
    main()
