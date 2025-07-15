from numpy.linalg import inv
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


def gibbs_sampling(outcome, num_samples,
                   mu_s1,
        sigma_s1,
        mu_s2,
        sigma_s2,
        covariance):

    #define variance of t_given_S to be fixed
    Sigma_t_given_S = sigma_s1 + sigma_s2 - 2*covariance


    #Sample s1,s2 from the joint P(S1,S2 | t,y)
    def S_sample_given_t(t):


        mu_s = np.array([mu_s1, mu_s2]) 
        A = np.array([1, -1])
        Sigma_S = np.array([[sigma_s1, covariance], [covariance, sigma_s2]])

        Sigma_S_given_t = inv(inv(Sigma_S) + np.outer(A, A) / Sigma_t_given_S)
        Mu_S_given_t = Sigma_S_given_t.dot ( inv(Sigma_S).dot(mu_s) + np.transpose(A) * (t/Sigma_t_given_S) )

        S_sample = np.random.multivariate_normal(Mu_S_given_t, Sigma_S_given_t, 1)

        #Return numpy array with 2 elements, S1,S2
        return S_sample
    

    #Sample t from P(t | s1,s2,y)
    def t_sample_given_S_Y(mean): #Y truncates t, whilst S shifts the mode of t
        if outcome == 1:
            return truncnorm.rvs((0 - mean) / Sigma_t_given_S, (np.inf - mean) / Sigma_t_given_S, loc=mean, scale=np.sqrt(Sigma_t_given_S))
        elif outcome == -1:
            return truncnorm.rvs((-np.inf - mean) / Sigma_t_given_S, (0 - mean) / Sigma_t_given_S, loc=mean, scale=np.sqrt(Sigma_t_given_S))

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


if __name__=='__main__':

    n_draws= 10000

    all_mean_s1 = []
    all_mean_s2 = []
    all_var_s1 = []
    all_var_s2 = []

    mu_s1, var_s1, mu_s2, var_s2, covariance = 0, 1, 0, 1, 0

    games = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

    for i in range(len(games)):

        outcome = games[i]

        s1_samples, s2_samples = gibbs_sampling(outcome, n_draws, mu_s1, var_s1, mu_s2, var_s2, covariance)

        cov_matrix = np.cov(s1_samples, s2_samples)

        mu_s1 = np.mean(s1_samples)
        var_s1 = cov_matrix[0,0]
        mu_s2 = np.mean(s2_samples)
        var_s2 = cov_matrix[1,1]
        covariance = cov_matrix[0,1]

        all_mean_s1.append(mu_s1)
        all_var_s1.append(var_s1)
        all_mean_s2.append(mu_s2)
        all_var_s2.append(var_s2)
        

    fig, ax = plt.subplots(2, 2, figsize=(14, 12))

    ax[0, 0].plot(range(len(games)), all_mean_s1, color='blue')
    ax[0, 0].set_title("evolution of s1 mean")
    ax[0, 0].set_ylabel("$S_1$ mean")
    ax[0, 0].set_xlabel("games played")

    ax[1, 0].plot(range(len(games)), all_mean_s2, color='red')
    ax[1, 0].set_title("evolution of s2 mean")
    ax[1, 0].set_ylabel("$S_2$ mean")
    ax[1, 0].set_xlabel("games played")


    ax[0, 1].plot(range(len(games)), all_var_s1, color='blue')
    ax[0, 1].set_title("evolution of s1 variance")
    ax[0, 1].set_ylabel("$S_1$ variance")
    ax[0, 1].set_xlabel("games played")

    ax[1, 1].plot(range(len(games)), all_var_s2, color='red')
    ax[1, 1].set_title("evolution of s2 variance")
    ax[1, 1].set_ylabel("$S_2$ variance")
    ax[1, 1].set_xlabel("games played")


    plt.tight_layout()
    plt.show()
