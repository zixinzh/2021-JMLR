function [ X, mu_exp, alpha, beta, thompson_sample_part, regret,...
    regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
    = CTS_jmlr( K, L, T, w_opt, w_gap ) 

X = 0;  thompson_sample_part = 0; 
%cascading Thompson Sampling algorithm with Beta-Bernoulli Update 
%
% Input:
%     K --- no. of opt arms
%     L --- no. of all arms
%     w_opt --- w of opt arms
%     w_gap --- gap of w of opt and subopt arms
%     w_sub --- w_opt - w_gap; % w of subopt arms
%     T --- no. of trials
%     lambda --- parameter
% Output:
%     mu_exp --- the experimental mean of arms at each time t
%     N --- no. of observations of arms at each time t
%     regret --- true regret at each time t
%     regret_sum --- cumulative regret
%     regret_exp --- expected regret at each time t
%     regret_exp_sum --- expected cumulative regret
%     (NOT Output now)sigma --- variance of Gaussian distribution
%     arm_sel --- selected arms at each time t
%     W_sel --- the true outcome of selected arms at each time t
%     W_opt --- the true outcome of optimal arms at each time t
%     index_opt --- the random locations of optimal items

%% 1. initialize

if length(w_gap) == 1
    w_gap1 = w_gap;
    w_gap2 = w_gap*2;
end
if length(w_gap) == 2
    w_gap1 = w_gap(1);
    w_gap2 = w_gap1 + w_gap(2);
end
w = (w_opt - w_gap2 )*ones(1,L);
index_opt = K+1:2*K;
w(1:K) = w_opt - w_gap1;
w(index_opt) = w_opt; 
 
w_repeat = repmat(w,T,1); % all w repeat
w_repeat = reshape(w_repeat,1,T*L);


mu_exp = zeros(1,L); % experimental mu T 
N = zeros(1,L);% T 
regret = zeros(1,T);
regret_exp = zeros(1,T);

alpha = ones(1,L);
beta = ones(1,L);

arm_sel = zeros(1,K);
W_sel = zeros(1,K); 



%% 2. main phase
W_pull = binornd(1,w_repeat);
W_pull = reshape(W_pull,T,L);
W_opt = W_pull( :,index_opt ); % result of true opt arms (CAN BE OUT OF THE LOOP)   
reward_opt = prod(1-W_opt,2);
reward_exp_opt = (1-w_opt)^K;  %prod(1-w(1:K)); 
for t = 1:T
    % 1: generate priors and pull arms
    theta = betarnd(alpha, beta);
    % 2: select arms to pull
    [ ~, theta_ord ] = sort(theta, 'descend');
    arm_sel  = theta_ord(1:K); 
    
    % 3: pull selected and optimal arms
    W_sel = W_pull( t,arm_sel  ); % result of selected arms 
    
    % 4: calculate regret
    regret(t) = prod(1-W_sel ) - reward_opt(t,:); % regret
    regret_exp(t) = prod(1- w(arm_sel ) ) - reward_exp_opt; % expected regret 
    observ = find(W_sel ==1); 
    if isempty( observ ) == 1
        no_observe = K; % number of observations
    else
        no_observe = observ(1);
    end
    for i = 1:no_observe
        index = arm_sel( i);
        if W_sel( i) == 1 
            alpha(index) = alpha(index) + 1;
        else
            beta(index) = beta(index) + 1; % DO THE BETTER ARMS GET PULLED MORE OFTEN?
        end
    end 
end

regret_sum = cumsum(regret); % cumulative regret
regret_exp_sum = cumsum(regret_exp); % cumulative regret

end
 


