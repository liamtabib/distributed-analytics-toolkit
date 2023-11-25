import numpy as np
from scipy.stats import truncnorm
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

def multiplyGauss(m1, s1, m2, s2):

    s = 1/(1/s1 + 1/s2)
    m = (m1/s1 + m2/s2) * s
    return m, s

def divideGauss(m1, s1, m2, s2):

    m, s = multiplyGauss(m1, s1, m2, -s2)
    return m, s

def truncGaussMM(a, b, m0, s0):

    a_scaled, b_scaled = (a - m0) / np.sqrt(s0), (b - m0) / np.sqrt(s0)
    m = truncnorm.mean(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    s = truncnorm.var(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
    return m, s

def message_passing_inference():
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

    return ps1_m, ps1_s, ps2_m, ps2_s


def gibbs_sampling(num_samples):

    # init values
    mu_s1, sigma_s1, mu_s2, sigma_s2 = 0, 1, 0, 1

    #define variance of t_given_S to be fixed
    Sigma_t_given_S = sigma_s1 + sigma_s2


    #Sample s1,s2 from the joint P(S1,S2 | t,y)
    def S_sample_given_t(t):

        mu_s = np.array([mu_s1, mu_s2]) 
        A = np.array([1, -1])
        Sigma_S = np.array([[sigma_s1, 0], [0, sigma_s2]])

        Sigma_S_given_t = inv(inv(Sigma_S) + np.outer(A, A) / Sigma_t_given_S)
        Mu_S_given_t = Sigma_S_given_t.dot ( inv(Sigma_S).dot(mu_s) + np.transpose(A) * (t/Sigma_t_given_S) )

        S_sample = np.random.multivariate_normal(Mu_S_given_t, Sigma_S_given_t, 1)

        #Return numpy array with 2 elements, S1,S2
        return S_sample
    
    #Sample t from P(t | s1,s2,y)
    def t_sample_given_S_Y(mean): #Y truncates t s.t t>0, whilst S shifts the mode of t s.t. mode(t) = 0 if s1-s2 is negative, otherwise mode(t) = s1-s2
        return stats.truncnorm.rvs((0 - mean) / Sigma_t_given_S, (np.inf - mean) / Sigma_t_given_S, loc=mean, scale=np.sqrt(Sigma_t_given_S))

    s1_samples = []
    s2_samples = []

    #initial value of t
    t = mu_s1 - mu_s2
    #Run gibbs sampler
    for i in range(num_samples):
        S = S_sample_given_t(t)
        t_mean = S[0][0] - S[0][1]
        t = t_sample_given_S_Y(t_mean)
        s1_samples.append(S[0][0])
        s2_samples.append(S[0][1])

    return s1_samples, s2_samples


def main():

    num_samples = 100000

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    start = time.time()

    s1_samples, s2_samples = gibbs_sampling(num_samples)
    end = time.time()
    print(f'time to draw {num_samples} samples: {end-start}')

    s1_mean = np.mean(s1_samples)
    s1_variance = np.var(s1_samples)
    s2_mean = np.mean(s2_samples)
    s2_variance = np.var(s2_samples)

    message_passing_approx = message_passing_inference()
    mp_s1_mean= message_passing_approx[0]
    mp_s1_var = message_passing_approx[1]
    mp_s2_mean = message_passing_approx[2]
    mp_s2_var = message_passing_approx[3]

    # Plot histogram for S1
    ax[0].hist(s1_samples, bins=30, color='brown', edgecolor='black', alpha=0.7, density=True, label=f"Histogram of $S_1$, n={num_samples}")  
    x = np.linspace(s1_mean - 3*np.sqrt(s1_variance), s1_mean + 3*np.sqrt(s1_variance), 100)
    ax[0].plot(x, stats.norm.pdf(x, s1_mean, s1_variance), color='blue', label=f"Gibbs approx: $\mu_{{S_1}} = {s1_mean:.2f}, \ \sigma_{{S_1}}^2 = {s1_variance:.2f}$")  
    ax[0].plot(x, stats.norm.pdf(x, mp_s1_mean, mp_s1_var), color='black', label=f"Message passing approx: $\mu_{{S_1}} = {mp_s1_mean:.2f}, \ \sigma_{{S_1}}^2 = {mp_s1_var:.2f}$")  
    ax[0].set_title("Distribution of $S_1$")
    ax[0].set_ylabel("P($S_1$)")
    ax[0].legend(loc='upper right')  # Added legend for S1

    # Plot histogram for S2
    ax[1].hist(s2_samples, bins=30, color='brown', edgecolor='black', alpha=0.7, density=True, label=f"Histogram of $S_2$, n={num_samples}")  
    x = np.linspace(s2_mean - 3*np.sqrt(s2_variance), s2_mean + 3*np.sqrt(s2_variance), 100)
    ax[1].plot(x, stats.norm.pdf(x, s2_mean, s2_variance), color='blue', label=f"Gibbs approx: $\mu_{{S_2}} = {s2_mean:.2f}, \ \sigma_{{S_2}}^2 = {s2_variance:.2f}$")  
    ax[1].plot(x, stats.norm.pdf(x, mp_s2_mean, mp_s2_var), color='black', label=f"Message passing approx: $\mu_{{S_2}} = {mp_s2_mean:.2f}, \ \sigma_{{S_2}}^2 = {mp_s2_var:.2f}$")  
    ax[1].set_title("Distribution of $S_2$")
    ax[1].set_ylabel("P($S_2$)")
    ax[1].legend(loc='upper right')  # Added legend for S2

    plt.tight_layout()
    plt.savefig('comparison.png',dpi=400)

if __name__ == "__main__":
    main()