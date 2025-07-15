import numpy as np
from scipy.stats import truncnorm

def multiplyGauss(m1, s1, m2, s2):
    # computes the Gaussian distribution N(m,s) being proportional to N(m1,s1)*N(m2,s2)
    #
    # Input:
    # m1, s1: mean and variance of first Gaussian 
    # m2, s2: mean and variance of second Gaussian
    #
    # Output:
    # m, s: mean and variance of the product Gaussian
    s = 1/(1/s1 + 1/s2)
    m = (m1/s1 + m2/s2) * s
    return m, s

def divideGauss(m1, s1, m2, s2):
    # computes the Gaussian distribution N(m,s) being proportional to N(m1,s1)/N(m2,s2)
    #
    # Input:
    # m1, s1: mean and variance of the numerator Gaussian
    # m2, s2: mean and variance of the denominator Gaussian
    #
    # Output:
    # m, s: mean and variance of the quotient Gaussian
    m, s = multiplyGauss(m1, s1, m2, -s2)
    return m, s

def truncGaussMM(a, b, m0, s0):
    # computes the mean and variance of a truncated Gaussian distribution
    #
    # Input:
    # a, b: The interval [a, b] on which the Gaussian is being truncated 
    # m0, s0: mean and variance of the Gaussian which is to be truncated
    #
    # Output:
    # m, s: mean and variance of the truncated Gaussian
    # scale interval with mean and variance
    a_scaled, b_scaled = (a - m0) / np.sqrt(s0), (b - m0) / np.sqrt(s0)
    m = truncnorm.mean(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    s = truncnorm.var(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    return m, s


def main():
    m_s1 = 0  # The mean of the priors p(s1) = p(s2)
    m_s2 = 0
    s_s1 = 1
    s_s2 = 1
    sv = 2  # The variance of p(t|s1,s2)
    y0 = 1  # The measurement


    # Message mu4 from node s1 to factor f_t,s1,s2
    mu4_m = m_s1  # mean of message
    mu4_s = s_s1  # variance of message

    # Message mu6 from node s2 to factor f_t,s1,s2
    mu6_m = m_s2  # mean of message
    mu6_s = s_s2  # variance of message

    # Message mu7 from factor f_t,s1,s2 to node t
    mu7_m = mu4_m - mu6_m
    mu7_s = mu4_s + mu6_s + sv

    # Do moment matching of the marginal of t
    if y0 == 1:
        a, b = 0, np.Inf
    else:
        a, b = np.NINF, 0

    pt_m, pt_s = truncGaussMM(a, b, mu7_m, mu7_s)

    mu8_m, mu8_s = divideGauss(pt_m, pt_s, mu7_m, mu7_s)

    # Compute the message from f_t,s1,s2 to s1
    mu9_m = mu8_m + m_s2
    mu9_s = mu8_s + s_s2 + sv

    # Compute the marginal of s1
    ps1_m, ps1_s = multiplyGauss(mu4_m, mu4_s, mu9_m, mu9_s)

    # Compute the message from f_t,s1,s2 to s2
    mu10_m = -mu8_m -m_s1
    mu10_s = mu8_s + s_s1 + sv

    # Compute the marginal of s2
    ps2_m, ps2_s = multiplyGauss(mu6_m, mu6_s, mu10_m, mu10_s)

if __name__=='__main__':
    main()