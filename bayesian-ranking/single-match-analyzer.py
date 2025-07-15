from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time


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


def plot_traceplots(num_samples):
    s1_samples, s2_samples = gibbs_sampling(num_samples)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    #plot traceplots
    ax[0].plot(range(num_samples), s1_samples, color='blue')
    ax[0].set_title("Markov Chain of $S_1$")
    ax[0].set_ylabel("$S_1$ value")
    ax[0].set_xlabel("Iteration")

    ax[1].plot(range(num_samples), s2_samples, color='red')
    ax[1].set_title("Markov Chain of $S_2$")
    ax[1].set_ylabel("$S_2$ value")
    ax[1].set_xlabel("Iteration")
    plt.tight_layout()

    plt.savefig('output/traceplots.png',dpi=400)


def plot_histograms():

    num_samples = [100, 1000, 10000, 100000]

    fig, ax = plt.subplots(2, 4, figsize=(16, 10))

    for index, num_sample in enumerate(num_samples):
        start = time.time()
        

        s1_samples, s2_samples = gibbs_sampling(num_sample)
        end = time.time()
        print(f'time to draw {num_sample} samples: {end-start}')

        s1_mean = np.mean(s1_samples)
        s1_variance = np.var(s1_samples)
        s2_mean = np.mean(s2_samples)
        s2_variance = np.var(s2_samples)

        # Plot histogram for S1
        ax[0, index].hist(s1_samples, bins=30, color='brown', edgecolor='black', alpha=0.7, density=True)  
        x = np.linspace(s1_mean - 3*np.sqrt(s1_variance), s1_mean + 3*np.sqrt(s1_variance), 100)
        ax[0, index].plot(x, stats.norm.pdf(x, s1_mean, s1_variance), color='blue')  
        ax[0, index].set_title(f"Distribution of $S_1$: n={num_sample}")
        ax[0, index].set_ylabel("P($S_1$)")
        ax[0, index].set_xlabel(f"$\mu_{{S_1}} = {s1_mean:.2f}, \ \sigma_{{S_1}}^2 = {s1_variance:.2f}$")

        # Plot histogram for S2
        ax[1, index].hist(s2_samples, bins=30, color='brown', edgecolor='black', alpha=0.7, density=True)  
        x = np.linspace(s2_mean - 3*np.sqrt(s2_variance), s2_mean + 3*np.sqrt(s2_variance), 100)
        ax[1, index].plot(x, stats.norm.pdf(x, s2_mean, s2_variance), color='blue')  
        ax[1, index].set_title(f"Distribution of $S_2$: n={num_sample}")
        ax[1, index].set_ylabel("P($S_2$)")
        ax[1, index].set_xlabel(f"$\mu_{{S_2}} = {s2_mean:.2f}, \ \sigma_{{S_2}}^2 = {s2_variance:.2f}$")


    plt.tight_layout()
    plt.savefig('output/histograms.png',dpi=400)

def main():
    plot_traceplots(100000)
    plot_histograms()

if __name__=="__main__":
    main()
