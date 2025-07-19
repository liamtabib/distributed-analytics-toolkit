from numpy.linalg import inv
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import pandas as pd

def preprocess(df):
    df['outcome'] = df['score1'] - df['score2']
    df.drop(df[df.outcome == 0].index, inplace=True)

    for index,row in df.iterrows():
        if row['outcome'] < 0:
            df.at[index, 'outcome'] = -1
        else:
            df.at[index, 'outcome'] = 1
    return df


def init_priors(df: pd.DataFrame):

    all_teams = set(df['team1'])
    # set priors for each individual team
    priors = {}
    for team in all_teams:
        priors[team] = (0,1) #mean zero, variance one
    # set covariance prior between two teams
    covariances = {}
    for team1 in all_teams:
        for team2 in all_teams:
            if team1 != team2:
                covariances[team1+team2] = 0

    return priors, covariances


def gibbs_sampler(outcome, mu_s1, sigma_s1, mu_s2, sigma_s2, covariance, n_draws):


    #define variance of t_given_S to be fixed
    Sigma_t_given_S = sigma_s1 + sigma_s2 +  2 * covariance

    #Sample s1,s2 from the joint P(S1,S2 | t)
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
    for i in range(n_draws):
        S = S_sample_given_t(t)
        t_mean = S[0][0] - S[0][1]
        t = t_sample_given_S_Y(t_mean)
        s1_samples.append(S[0][0])
        s2_samples.append(S[0][1])
    
    return s1_samples, s2_samples


def run_season(df, shuffle: bool, n_draws, project_root: str):
    import os
    
    if shuffle:
        df = df.sample(frac=1)

    priors, covariances = init_priors(df)

    correct = 0
    incorrect = 0

    for index, row in df.iterrows():

        team1 = row['team1']
        team2 = row['team2']

        prior_team1 = priors[team1]
        prior_team2 = priors[team2]

        y = row['outcome']

        mean_difference = prior_team1[0] - prior_team2[0]
        if mean_difference > 0:
            prediction = 1
        else:
            prediction = -1

        if prediction == y:
            correct += 1
        else:
            incorrect += 1

        prior_covariance = covariances[team1+team2]

        s1_samples, s2_samples = gibbs_sampler(y, prior_team1[0], prior_team1[1], prior_team2[0], prior_team2[1], 
                                               prior_covariance, n_draws)

        cov_matrix = np.cov(s1_samples, s2_samples)

        mu_s1 = np.mean(s1_samples)
        var_s1 = cov_matrix[0,0]
        mu_s2 = np.mean(s2_samples)
        var_s2 = cov_matrix[1,1]
        covariance = cov_matrix[0,1]

        #update priors
        priors[team1] = (mu_s1, var_s1)
        priors[team2] = (mu_s2, var_s2)
        covariances[team1+team2] = covariance
        covariances[team2+team1] = covariance
        #print('\n')
        #print(f'Match between {team1} and {team2}: result {y}')
        #print(f'For {team1}: prior: {prior_team1}, posterior: {priors[team1]}')
        #print(f'For {team2}: prior: {prior_team2}, posterior: {priors[team2]}')
        #print(f'prior covariance {prior_covariance}, posterior: {covariance}')
    
    sorted_priors = dict(sorted(priors.items(), key=lambda item: item[1][0]))

    mean_skills = [skill for skill, _ in sorted_priors.values()]
    std_devs = [np.sqrt(variance) for _, variance in sorted_priors.values()]

    y_positions = range(len(sorted_priors))
    plt.clf()
    plt.hlines(y=y_positions, xmin=[mean - std for mean, std in zip(mean_skills, std_devs)], 
            xmax=[mean + std for mean, std in zip(mean_skills, std_devs)], color='blue', linewidth=8)

    plt.scatter(mean_skills, y_positions, color='red', s=40, zorder=5)
    plt.yticks(y_positions, sorted_priors.keys())

    # Set labels and title
    plt.xlabel("Skill with spread")
    plt.title("Final Skill")
    plt.grid(True, axis='x', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if shuffle:
        print(f'accuracy with shuffle: {correct/(correct+incorrect)}')
        output_path = os.path.join(project_root, 'outputs', 'plots', 'season_shuffle.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=400)
    else:
        print(f'accuracy: {correct/(correct+incorrect)}')
        output_path = os.path.join(project_root, 'outputs', 'plots', 'season.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=400)


def main():
    import os
    
    # Get the path to the data file relative to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..')
    data_file = os.path.join(project_root, 'data', 'serie_a.csv')
    
    df = pd.read_csv(data_file)
    df = preprocess(df)
    shuffle = False

    n_draws = 1000 #for generating the plot, we used n= 10 000
    #run correct season
    run_season(df, shuffle, n_draws, project_root)

    #run shuffle season
    shuffle = True
    run_season(df, shuffle, n_draws, project_root)    


if __name__=='__main__':
    main()