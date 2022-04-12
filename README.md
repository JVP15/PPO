# Homework 3

This is a team assignment. As a team you can work in parallel on different components. 

In this home assignment, you will implement the PPO algorithm, and apply 
it to solve the task of swinging up an inverted pendulum, which is a 
continuous dynamical system with continuous state, s, and action, a, spaces. 

The actions are torque to the pivot point of the pole. The actions are one-dimensional
random variables, distributed with the Gaussian probability density function with the 
mean mu(s) and standard deviation std(s). For simplicity, in the basic implementation, 
you can implement a state independent standard deviation, which will be a 
hyperparameter (scalar) which you tune manually. In the basic implementation "a ~ N(mu(s), std)",
cf., "a ~ N(mu(s), std(s))" in the general case.

You are provided with the required utilities, which will help you to implement 
the PPO algorithm, given by the pseudocode in the PPO paper.
Different software-engineering design choices are possible.

Feel free to refer to the code from homework 1 as an inspiration. Don't copy
end-to-end implementations from the web, because that will not contribute to a 
thorough understanding of the algorithm (and it is easily detected..)

## Guidelines

* Start by implementing an environment interaction loop. You may 
refer to homework 1 for inspiration. 

* Create and test an experience replay buffer with a random policy, which is the 
Gaussian distribution with arbitrary (randomly initialized) weights of the policy feed-forward network,
receiving state, s, and returning the mean, mu(s) and the log_std, log_stg(s) 
(natural logarithm of the standard deviation) of actions.  As mentioned above, you can use 
a state-independent standard variance.

* Make an episode reward processing function to turn one-step rewards into discounted rewards-to-go:
R(s_1) = sum_{t=1} gamma^{t-1} r_t, which is the discounted reward, starting from the state, s_1.

* Start the model by implementing a vanilla policy gradient agent, where the gradient ascent steps
are done with the average of the gradient of log-likelihood over a trajectory weight by rewards-to-go   
from each state. Try different step sizes in the gradient ascent.  

* Pendulum is a continuous action space environment. 
Check out the example in `Modules.py` for torch implementation of the Gaussian module.

An additional (to the lecture slides) good resource for the terminology and the main components is  
[OpenAI Spinningup key concepts and terminology](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#key-concepts-and-terminology) 

* Add a feed-forward network for the critic, accepting the state, s=[sin(angle), cos(angle), angular velocity], and returning a scalar for the value of the state, s.

* Implement the generalized advantage, see Eq11-12 in the PPO, to be used instead of rewards-to-go.

* Implement the surrogate objective for the policy gradient, see Eq7, without and without clipping. 

* Implement the total loss, see Eq9 in the PPO.    

* Combine all together to Algorithm 1 in the PPO paper. (In your basic implementation, you can collect data with a single actor, N=1)

* You should see progress with default hyperparameters, but you can try tuning those to 
see how it will improve your results. 
 

## to extend HW3 to the final project

Do any four out of the following possible extensions:   
- Use an ensemble of B critics instead of a single critic: Vens(s) = 1/B sum_b V_b(s), where each V_b(s) parametrized by a separate network.  
- Collect data in parallel in a number of pendulum environments. Think about real pendulum which you run physically in parallel and collect data from all of them concurrently. If you chose this extension, then report the improvement in "wall clock" time of the training.
- Add recurrence to the policy and train in the environment with partial observability. E.g., use only the angular velocity, rather than the full state, s =[angle, angular velocity]. 
- Use images instead of state vectors. Note that since you won't have 
access to velocity, you will need either to stack a few last images or to use a recurrent policy.
- Train state-dependent standard deviation, log_std(s), instead of log_std, which you tune/set manually. 
- your ideas. (share please with me before implementing). 


## What To Submit
- Your runnable code.
- PDF with the full solution, (e.g., export of a notebook to PDF), including the code and the plots with "Learning curves" and "Loss curves":
   Learning curves should show the accumulated discounted reward by a policy at the k-th training episode, for each setting of the algorithm/parameters, e.g., i) w/, w/o clipping, ii) w/, w/o generalized advantage, iii) w, w/o stacking images (if you choose to extend HW3 to the final project), iv) etc. 
  Loss, Eq9, for each setting of the algorithm. e.g., i) w/, w/o clipping, ii) w/, w/o generalized advantage, etc.


please submit the files separately rather than in a single zip file. 
