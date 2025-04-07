# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:27:01 2024

@author: carll
"""

#% Test RL models

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Symetrical learning rate model
def simulate_rw(df, alpha, beta, Qi0, Qi1):

    # Get outcomes for actions across trials
    rewards_action_a = df['Action_A'].to_numpy()
    rewards_action_b = df['Action_B'].to_numpy()

    Q = np.array([Qi0, Qi1], dtype="float64")  # array to track Q values
    simulated_actions = []  # list to store actions
    RPEs = []               # list to store RPEs
    q_values_a = []  # list to store Q values from action A
    q_values_b = []  # list to store Q values from action B
    EVs = []
    for t in range(len(rewards_action_a)):

        # Calculate action probabilities using softmax
        Q_ = Q * beta
        action_probs = np.exp(Q_) / np.sum(np.exp(Q_))

        # Choose an action based on the probabilities
        a = np.random.choice([0, 1], p=action_probs)
        simulated_actions.append(a)

        # Get the reward/outcome of the chosen action
        r = rewards_action_a[t] if a == 0 else rewards_action_b[t]

        # Get RPE
        RPE = r - Q[a]
        RPEs.append(RPE)

        # Get q_values
        q_values_a.append(Q[0])
        q_values_b.append(Q[1])

        # Get EV
        EVs.append(Q[a])

        # Update the Q values using Rescorla-Wagner rule
        Q[a] = Q[a] + alpha * (r - Q[a])

    return simulated_actions, RPEs, q_values_a, q_values_b, EVs


# Symetrical learning rate model
def simulate_rw_1arm(df, alpha, beta, Q0, Q1,
                     static_early_meta_cog_bias=0,
                     early_meta_cog_bias=0,
                     late_meta_cog_bias=0,
                     util_func_bias=0,
                     lr_asymmetry=False,
                     alpha_plus=0.5,
                     alpha_minus=0.9,
                     Qc_noise=False,
                     Qp_noise=False,
                     Q_reporting_noise=False,
                     perception_prop=False,
                     fb_condition='neut',
                     ):

    # Get outcomes for actions across trials
    rewards_action_a = df['Action_A'].to_numpy()
    #rewards_action_b = df['Action_B'].to_numpy()

    Q  = np.array([Q0], dtype="float64")  # array to track Q values
    Qc = np.array([Q0], dtype="float64")  # array to track Qc values
    Qp = np.array([Q0], dtype="float64")  # array to track Qp values

    # mean perceptual beleif
    #p_mean = 65

    simulated_actions = []  # list to store actions
    RPEs = []               # list to store RPEs
    q_values_a = []         # list to store Q values from action A
    EVs = []

    # Introduce early static (perceptual) meta cog bias - Before choice
    Q[0] = Q[0] + float(((Q[0]/100)*static_early_meta_cog_bias))

    for t in range(len(rewards_action_a)):

        # Introduce early dynamic (perceptual) meta cog bias - Before choice
        Qp[0] = Qp[0] + float(((Qp[0]/100)*early_meta_cog_bias))

        # Add iterative noise
        if Qc_noise != False:
            Qc[0] = np.random.normal(loc=Qc[0], scale=Qc_noise, size=1)[0]
        if Qp_noise != False:
            Qp[0] = np.random.normal(loc=Qp[0], scale=Qp_noise, size=1)[0]

        # Add static reporting noise
        if Q_reporting_noise != False:
            Q_w_noise = np.random.normal(loc=Qc[0], scale=Q_reporting_noise,
                                          size=1)[0]

        # Either integrate Qc and Qp beleifs
        Q[0] = (Qc[0]*(1-perception_prop)) + (Qp[0]*perception_prop)

        # Report noisy or raw internal representation
        if Q_reporting_noise != False:
            # Save q_values for the current trial
            q_values_a.append(Q_w_noise)
            # Save the current trial expected value (EV of current action)
            EVs.append(Q_w_noise)
            #print(f'report noise added: {Q_w_noise}')
        else:
            q_values_a.append(Q[0])
            # Save the current trial expected value (EV of current action)
            EVs.append(Q[0])

        # Action selection ---------------------------------------------------
        # Calculate action probabilities using softmax
      #  Q_ = Q * beta
      #  action_probs = np.exp(Q_) / np.sum(np.exp(Q_))

        # Choose an action based on the probabilities - only one action (0)
        #a = np.random.choice([0], p=action_probs)
        a = 0
        simulated_actions.append(a)
        # --------------------------------------------------------------------

        # Get the reward/outcome of the chosen action
        r = rewards_action_a[t]

        # Introduce late meta cog bias - after choice, before update
        Q[a] = Q[a] + float(((Q[a]/100)*late_meta_cog_bias))


        # Apply confidence update if feedback (r) is present
        if pd.isna(r):
            RPE = 0 - Q[a]
            RPEs.append(RPE)
            pass
        else:

            # apply condition
            if fb_condition=='neut':
                r=r
            if fb_condition=='neg':
                r=r-10
            if fb_condition=='pos':
                r=r+10

            # Introduce util_func_bias
            r = r + float(((r/100)*util_func_bias))
            #r = np.random.normal(loc=r, scale=1, size=1)

            # Get RPE
            RPE = r - Q[a]
            RPEs.append(RPE)

            # Update the Q values using Rescorla-Wagner rule
            if lr_asymmetry == False:
                Qc[a] = Q[a] + alpha * (r - Q[a])
            else:
                if RPE > 0:
                    Qc[a] = Q[a] + alpha_plus * (r - Q[a])
                else:
                    Qc[a] = Q[a] + alpha_minus * (r - Q[a])

    return simulated_actions, RPEs, q_values_a, EVs

# Build environment
df = pd.DataFrame()

n_trials = 20

# Parameters for the Gaussian distribution
mean_A = 20
std_A = 1  # Standard deviation
mean_B = 8
std_B = 1  # Standard deviation

num_samples = n_trials

# Generate the samples
samples_A = np.random.normal(loc=mean_A, scale=std_A, size=num_samples)
samples_B = np.random.normal(loc=mean_B, scale=std_B, size=num_samples)


df['Action_A'] = np.random.choice(samples_A,
                                  size=n_trials,
                                  replace=True)

df['Action_B'] = np.random.choice(samples_B,
                                  size=n_trials,
                                  replace=True)

# Add the lack of feedback during baseline
baseline_nan_feedback = np.array([np.nan]*10)
baseline_df = pd.DataFrame({'Action_A': baseline_nan_feedback,
                            'Action_B': baseline_nan_feedback})
df = pd.concat([baseline_df, df]).reset_index(drop=True)
#df = pd.concat([df, df]).reset_index(drop=True)
#df = pd.concat([df, df]).reset_index()

# Simulate agent
alpha = 0.9
beta = 0.1
Q0 = 50
Q1 = 0.0

results = simulate_rw_1arm(df, alpha, beta, Q0, Q1,
                     static_early_meta_cog_bias=0,
                     early_meta_cog_bias=0,
                     late_meta_cog_bias=0,
                     util_func_bias=0,
                     lr_asymmetry=False,
                     alpha_plus=0.5,
                     alpha_minus=0.9,
                     Qc_noise=False,
                     Qp_noise=False,
                     Q_reporting_noise=50,
                     perception_prop=False
                    )

simulated_actions, RPEs, q_values_a, EVs = results

df_model = pd.DataFrame({'simulated_actions': simulated_actions,
                         'RPEs': RPEs,
                         'q_values_a': q_values_a,
                        # 'q_values_b': q_values_b,
                         'EVs': EVs,
                         })


a_rewards_seen = df_model[df_model.simulated_actions==0].q_values_a
#b_rewards_seen = df_model[df_model.simulated_actions==1].q_values_b

# Plot results

# Set the number of timesteps based on the length of one of your result arrays
timesteps = range(len(simulated_actions))

# Plot distribution of option value
plt.figure(figsize=(6, 9))
ax0 = plt.subplot(5, 2, 1)  # 3 row, 3 columns, 3rd plot
sns.kdeplot(df['Action_A'], label='A', alpha=0.5, fill=True)
#sns.kdeplot(df['Action_B'], label='B', alpha=0.5, fill=True)
plt.axvline(df['Action_A'].mean(), ls='--', c='k', label='Mean')
#plt.axvline(df['Action_B'].mean(), ls='--', c='C1', label='Mean')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend(loc='upper right', bbox_to_anchor=(1,1), fontsize=11)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.set_xlim(ax0.get_xlim()[0], ax0.get_xlim()[1]+2)

# Plot Choices over Time
# =============================================================================
# ax1 = plt.subplot(4, 2, 2)  # 3 row, 3 columns, 1st plot
# plt.plot(timesteps, simulated_actions, 'o-', label='Choice')
# plt.xlabel('Trial')
# plt.ylabel('Action')
# #plt.title('Choices Over Time')
# plt.ylim(-0.1, 1.1)
# #plt.yticks([0, 1], ['A', 'B'])
# plt.yticks([0], ['A'])
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# =============================================================================

# Plot Value
ax2 = plt.subplot(5, 2, 3)  # 3 row, 3 columns, 2nd plot
plt.plot(timesteps, q_values_a[:], label='Learned value', alpha=1)
#plt.plot(timesteps, q_values_b[:], label='', alpha=1)
plt.axhline(df['Action_A'].mean(), ls='--', c='k', label='Mean A')
#plt.axhline(df['Action_B'].mean(), ls='--', c='C1', label='Mean B')
plt.xlabel('Trial')
plt.ylabel('Value')
plt.legend(loc='upper right', bbox_to_anchor=(1,1), fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Plot Reward Prediction Errors
ax3 = plt.subplot(5, 2, 5)  # 3 row, 3 columns, 3rd plot
plt.plot(timesteps, RPEs, label='RPE')
plt.xlabel('Trial')
plt.ylabel('RPE')
#plt.title('Reward Prediction Errors Over Time')
plt.axhline(0, color='gray', linestyle='--')  # Zero line for reference
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Plot that value align to mean of seen values
ax5 = plt.subplot(5, 2, 7)
accum_mean_PE_A = [abs(df['Action_A'].mean()-q_values_a[t])
                   for t in range(len(q_values_a))]
plt.plot(timesteps, accum_mean_PE_A, label='RPE')
#plt.plot(timesteps, accum_mean_PE_B, label='RPE')
plt.xlabel('Trial')
plt.ylabel('abs(Mean A - Pred.)')
#plt.title('Reward Prediction Errors Over Time')
plt.axhline(0, color='gray', linestyle='--')  # Zero line for reference
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# Plot that value align to mean of seen values
ax5 = plt.subplot(5, 2, 9)
accum_mean_PE_A = [(df['Action_A'].mean()-q_values_a[t])
                   for t in range(len(q_values_a))]
plt.plot(timesteps, accum_mean_PE_A, label='RPE')
#plt.plot(timesteps, accum_mean_PE_B, label='RPE')
plt.xlabel('Trial')
plt.ylabel('Rel(Mean A - Pred.)')
#plt.title('Reward Prediction Errors Over Time')
plt.axhline(0, color='gray', linestyle='--')  # Zero line for reference
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)


plt.tight_layout()  # Adjust subplots to fit in the figure area
plt.show()

print('\nAction A', '__'*30)
print(' True Mean', np.mean(df['Action_A'].values), '\n',
      'Mean of seen', np.mean(a_rewards_seen.values), '\n',
      'Mean EV', np.mean(q_values_a[-100:]), '\n',
      '__'*34)

#%%
# Simulate the original model
results_original = simulate_rw_1arm(df, alpha, beta, Q0, Q1,
                                    static_early_meta_cog_bias=0,
                                    early_meta_cog_bias=0,
                                    late_meta_cog_bias=0,
                                    util_func_bias=0,
                                    lr_asymmetry=False,
                                    alpha_plus=0.5,
                                    alpha_minus=0.5,
                                    Qc_noise=False,
                                    Qp_noise=False)

# Simulate the biased model
results_biased = simulate_rw_1arm(df, alpha, beta, Q0, Q1,
                                  static_early_meta_cog_bias=-50,
                                  early_meta_cog_bias=10,
                                  late_meta_cog_bias=0,
                                  util_func_bias=50,  # Applying bias here
                                  lr_asymmetry=True,
                                  alpha_plus=0.9,
                                  alpha_minus=0.4,
                                  Qc_noise=False,
                                  Qp_noise=False)

# Extract results for both models
actions_original, RPEs_original, q_values_a_original, EVs_original = results_original
actions_biased, RPEs_biased, q_values_a_biased, EVs_biased = results_biased

# Create DataFrames for both sets of results
df_model_original = pd.DataFrame({
    'actions': actions_original,
    'RPEs': RPEs_original,
    'q_values_a': q_values_a_original,
    'EVs': EVs_original
})

df_model_biased = pd.DataFrame({
    'actions': actions_biased,
    'RPEs': RPEs_biased,
    'q_values_a': q_values_a_biased,
    'EVs': EVs_biased
})

# Set the number of timesteps based on the length of one of the result arrays
timesteps = range(len(actions_original))

# Plotting the results
plt.figure(figsize=(6, 12))

# KDE plot of actions for both models
ax0 = plt.subplot(5, 1, 1)
sns.kdeplot(df['Action_A'], label='A', fill=True)
#sns.kdeplot(df['Action_A'], label='Biased Model', fill=True)
plt.title('Kernel Density Estimate of Actions')
plt.xlabel('Actions')
plt.ylabel('Density')
plt.legend()

# Plot q_values for both models
ax1 = plt.subplot(5, 1, 2)
plt.plot(timesteps, df_model_original['q_values_a'],
         label='Original Model', alpha=0.75)
plt.plot(timesteps, df_model_biased['q_values_a'],
         label='Biased Model', alpha=0.75)
plt.title('Q-values Over Time')
plt.xlabel('Trial')
plt.ylabel('Q-values')
plt.ylim(0,100)
plt.legend()

# Plot RPEs for both models
ax2 = plt.subplot(5, 1, 3)
plt.plot(timesteps, df_model_original['RPEs'],
         label='Original Model', alpha=0.75)
plt.plot(timesteps, df_model_biased['RPEs'],
         label='Biased Model', alpha=0.75)
plt.title('Reward Prediction Errors Over Time')
plt.xlabel('Trial')
plt.ylabel('RPE')
plt.legend()

# Plot value alignment to mean of seen values
ax3 = plt.subplot(5, 1, 4)
accum_mean_PE_A_original = [abs(np.mean(df['Action_A'].values[10:]) - q)
                            for q in q_values_a_original]
accum_mean_PE_A_biased = [abs(np.mean(df['Action_A'].values[10:]) - q)
                          for q in q_values_a_biased]
plt.plot(timesteps, accum_mean_PE_A_original, label='Original Model')
plt.plot(timesteps, accum_mean_PE_A_biased, label='Biased Model')
plt.title('Absolute Difference from Mean Action A')
plt.xlabel('Trial')
plt.ylabel('Abs(Mean - Prediction)')
plt.ylim(0,100)
plt.legend()

ax4 = plt.subplot(5, 1, 5)
accum_mean_PE_A_original = [(np.mean(df['Action_A'].values[10:]) - q)
                            for q in q_values_a_original]
accum_mean_PE_A_biased = [(np.mean(df['Action_A'].values[10:]) - q)
                          for q in q_values_a_biased]
plt.plot(timesteps, accum_mean_PE_A_original, label='Original Model')
plt.plot(timesteps, accum_mean_PE_A_biased, label='Biased Model')
plt.title('Relative Difference from Mean Action A')
plt.xlabel('Trial')
plt.ylabel('Rel(Mean - Prediction)')
plt.axhline(0, ls='--', c='k')
#plt.ylim(0,100)
#plt.legend()

# Remove top and right spines
for ax in [ax0, ax1, ax2, ax3, ax4]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
#%% Simulate over many agents

agents = [
            # neutral model
           [0,      # static_early_meta_cog_bias
            0,      # early_meta_cog_bias
            0,      # late_meta_cog_bias
            20,     # util_func_bias - the saliance of feedback
            True,  # lr_asymmetry
            0.5,    # alpha_plus
            0.5,   # alpha_minus

            False,  # Qc_noise
            False,  # Qp_noise
            20,     # Q_reporting_noise,
            0,      # perception_prop
            'neut'  # Condition
            ],

           # biased model
           [0,      # static_early_meta_cog_bias
            0,      # early_meta_cog_bias
            0,      # late_meta_cog_bias
            -20,    # util_func_bias - the saliance of feedback
            True,   # lr_asymmetry
            0.5,      # alpha_plus
            0.5,      # alpha_minus

            False,  # Qc_noise
            False,  # Qp_noise
            20,     # Q_reporting_noise,
            0,      # perception_prop
            'neut'  # Condition
            ]

        ]


plt.figure(figsize=(6, 12))
for agent, fb_bias in zip(agents, [0,0]):
    n_agents = 131
    RPE_list = []
    q_values_a_list = []
    EVs_list = []
    option_value = []
    Qs_bias_list = []
    Qs_neutral_list = []
    fb_conf_aligment_list = []
    abs_fb_conf_aligment_list = []
    for i in range(n_agents):

        # Build environment
        df = pd.DataFrame()

        n_trials = 20

        # Parameters for the Gaussian distribution
        mean_A = 20
        std_A = 10  # Standard deviation
        mean_B = 8
        std_B = 1  # Standard deviation

        num_samples = n_trials

        # Generate the samples
        samples_A = np.random.normal(loc=mean_A+fb_bias, scale=std_A, size=num_samples)
        samples_B = np.random.normal(loc=mean_B, scale=std_B, size=num_samples)

        df['Action_A'] = np.random.choice(samples_A,
                                          size=n_trials,
                                          replace=True)

        df['Action_B'] = np.random.choice(samples_B,
                                          size=n_trials,
                                          replace=True)

        # Add baseline with nan feedback
        baseline_nan_feedback = np.array([np.nan]*10)
        baseline_df = pd.DataFrame({'Action_A': baseline_nan_feedback,
                                    'Action_B': baseline_nan_feedback})

        df = pd.concat([baseline_df, df]).reset_index(drop=True)
        # Simulate agent
        alpha = 0.8
        beta = 0.1
        Qi0 = 45
        Qi1 = 0.0

        [agent]

        # Neutral agent
        results = simulate_rw_1arm(df, alpha, beta, Qi0, Qi1,
                                   *agent)

# =============================================================================
#         static_early_meta_cog_bias=-20,
#         early_meta_cog_bias=0,
#         late_meta_cog_bias=0,
#         util_func_bias=0,
#         lr_asymmetry=False,
#         alpha_plus=0.1,
#         alpha_minus=0.9,
#         Qc_noise=False,
#         Qp_noise=False
#
# =============================================================================
        simulated_actions, RPEs, q_values_a, EVs = results

        df_model = pd.DataFrame({'simulated_actions': simulated_actions,
                                 'RPEs': RPEs,
                                 'q_values_a': q_values_a,
                                # 'q_values_b': q_values_b,
                                 'EVs': EVs,
                                 })

        # Get aligment between feedback and confidence
        fb_conf_aligment = [np.mean(df.Action_A[10:])-q_values_a[10:][t]
                            for t in range(len(q_values_a[10:]))]
        abs_fb_conf_aligment = [abs(np.mean(df.Action_A[10:])-q_values_a[10:][t])
                            for t in range(len(q_values_a[10:]))]
        # baseline
        fb_conf_aligment_b = [np.mean(0)-q_values_a[t]
                            for t in range(len(q_values_a[:10]))]
        abs_fb_conf_aligment_b = [abs(np.mean(0)-q_values_a[t])
                            for t in range(len(q_values_a[:10]))]

        # concat
        fb_conf_aligment = np.concatenate([fb_conf_aligment_b,
                                           fb_conf_aligment])
        abs_fb_conf_aligment = np.concatenate([abs_fb_conf_aligment_b,
                                               abs_fb_conf_aligment])

        fb_conf_aligment_list.append(fb_conf_aligment)
        abs_fb_conf_aligment_list.append(abs_fb_conf_aligment)

        #plt.ylim(0,100)
        #plt.plot(Qs)

        a_rewards_seen = df_model[df_model.simulated_actions==0].q_values_a
        #b_rewards_seen = df_model[df_model.simulated_actions==1].q_values_b

        RPE_list.append(RPEs)
        q_values_a_list.append(q_values_a)
        EVs_list.append(EVs)
        option_value.append(df['Action_A'].values)


    # Transform lists to arrays
    EV_array = np.array(EVs_list)
    Q_array = np.array(q_values_a_list)
    RPE_array = np.array(RPE_list)
    option_value_array = np.array(option_value)

    # Calculate mean, std, and sem
    EV_mean = np.mean(EV_array,axis=0)
    Q_mean = np.mean(Q_array, axis=0)
    RPE_mean = np.mean(RPE_array,axis=0)
    option_value_array_mean = np.mean(option_value_array, axis=0)
    fb_conf_aligment_mean = np.mean(fb_conf_aligment_list, axis=0)
    abs_fb_conf_aligment_mean = np.mean(abs_fb_conf_aligment_list, axis=0)

    EV_std = np.std(EV_array,axis=0)
    Q_std = np.std(Q_array, axis=0)
    RPE_std = np.std(RPE_array,axis=0)
    option_value_array_std = np.std(option_value_array, axis=0)
    fb_conf_aligment_std = np.std(fb_conf_aligment_list, axis=0)
    abs_fb_conf_aligment_std = np.std(abs_fb_conf_aligment_list, axis=0)

    EV_sem = np.std(EV_array,axis=0)/np.sqrt(len(EV_array))
    Q_sem = np.std(Q_array, axis=0)/np.sqrt(len(Q_array))
    RPE_sem = np.std(RPE_array,axis=0)/np.sqrt(len(RPE_array))
    option_value_array_sem = (np.std(option_value_array, axis=0)
                              /np.sqrt(len(option_value_array)))
    fb_conf_aligment_sem = (np.std(fb_conf_aligment_list, axis=0)/
                            np.sqrt(len(fb_conf_aligment_list)))
    abs_fb_conf_aligment_sem = (np.std(abs_fb_conf_aligment_list, axis=0)/
                                np.sqrt(len(abs_fb_conf_aligment_list)))

    # Plot distribution of option value


    # Set the number of trials
    timesteps = range(len(simulated_actions))

    ax0 = plt.subplot(5, 2, 1)  # 3 row, 3 columns, 3rd plot
    sns.kdeplot(option_value_array_mean, label='A', alpha=0.5, fill=True)
    #sns.kdeplot(df['Action_B'], label='B', alpha=0.5, fill=True)
    plt.axvline(option_value_array_mean[10:].mean(), ls='--', c='k', label='Mean')
    #plt.axvline(df['Action_B'].mean(), ls='--', c='C1', label='Mean')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(loc='upper right', bbox_to_anchor=(1,1), fontsize=11)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.set_xlim(ax0.get_xlim()[0], ax0.get_xlim()[1]+2)

    null_vector = np.array([np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan, np.nan, np.nan, np.nan])
    # Plot Expected Value
    ax2 = plt.subplot(5, 2, 3)  # 3 row, 3 columns, 2nd plot
    #ax2.set_xlim(0,0)
    plt.plot(timesteps, Q_mean, label='Expected value', alpha=1)
    plt.fill_between(timesteps, Q_mean-EV_sem,  Q_mean+EV_sem, alpha=0.5)
    #plt.plot(timesteps, q_values_b[:], label='', alpha=1)
    plt.axhline(np.mean(option_value_array_mean[10:]), ls='--', c='k',
                label='Mean A')
    #plt.axhline(df['Action_B'].mean(), ls='--', c='C1', label='Mean B')
    plt.xlabel('Trial')
    plt.ylabel('EV')
    plt.legend(loc='upper right', bbox_to_anchor=(1,1), fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Plot Reward Prediction Errors
    ax3 = plt.subplot(5, 2, 5)  # 3 row, 3 columns, 3rd plot
    #ax3.set_xlim(0,30)
    plt.plot(timesteps, RPE_mean, label='RPE')
    plt.fill_between(timesteps, RPE_mean-RPE_sem, RPE_mean+RPE_sem, alpha=0.5)
    plt.xlabel('Trial')
    plt.ylabel('RPE')
    #plt.title('Reward Prediction Errors Over Time')
    plt.axhline(0, color='gray', linestyle='--')  # Zero line for reference
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Plot that value align to mean of seen values
    ax5 = plt.subplot(5, 2, 7)

    plt.plot(range(-10, len(fb_conf_aligment_mean)-10), fb_conf_aligment_mean,
             label='Relative aligment')
    plt.fill_between(range(-10, len(fb_conf_aligment_mean)-10),
                     fb_conf_aligment_mean-fb_conf_aligment_sem,
                     fb_conf_aligment_mean+fb_conf_aligment_sem,
                     alpha=0.5)
    plt.xlabel('Trial')
    plt.ylabel('Rel. aligment')
    plt.title('Avg feedback - confidence')
    plt.axhline(0, color='gray', linestyle='--')  # Zero line for reference
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # Plot that value align to mean of seen values
    ax6 = plt.subplot(5, 2, 9)
    plt.plot(range(-10, len(abs_fb_conf_aligment_mean)-10), abs_fb_conf_aligment_mean,
             label='Absolute aligment')
    plt.fill_between(range(-10, len(abs_fb_conf_aligment_mean)-10),
                     abs_fb_conf_aligment_mean-abs_fb_conf_aligment_sem,
                     abs_fb_conf_aligment_mean+abs_fb_conf_aligment_sem,
                     alpha=0.5)
    plt.xlabel('Trial')
    plt.ylabel('Abs. aligment')
    plt.title('abs (Avg feedback - confidence)')
    plt.axhline(0, color='gray', linestyle='--')  # Zero line for reference
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

plt.tight_layout()  # Adjust subplots to fit in the figure area
plt.show()

print('\nAction A', '__'*30)
print(' True Mean', np.mean(df['Action_A'].values), '\n',
      'Mean of seen', np.mean(a_rewards_seen.values), '\n',
      'Mean EV', np.mean(q_values_a[-100:]), '\n',
      '__'*34)




