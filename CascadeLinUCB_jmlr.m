function [ X, mu_exp, N, B, U_t_part, regret, regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ] = ...
    CascadeLinUCB_jmlr( K, L, T, d, flag_train, tune, w_opt, w_gap  )
%Cascading Linear UCB algorithm 
%
% Input:
%     K --- no. of opt arms
%     L --- no. of all arms
%     d --- no. of features
%     w_opt --- w of opt arms
%     w_gap --- gap of w of opt and subopt arms
%     T --- no. of trials
%     flag_X --- 1: initial with data/else: trivial initialization
% Output:
%     X --- feature matrix
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

%% 1. initialization - part 1
mu_exp=0;

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

%% 2. generate feature matrix X (d*L)
% simulated train flag_X = 1
T_train = T/flag_train;
w_history = repmat(w,T_train,1); % all w repeat
w_history = reshape(w_history,1,T_train*L);
W_train = binornd(1,w_history);
W_train = reshape(W_train,T_train,L);
[~, S, V] = svds(W_train,d);
X = S*V';
scale = sqrt( max( diag(X'*X) ) )/ min( d/sqrt(K),1 );
X = X/scale;
clear w_history W_train S V T_train

%% 3.initialization - part 2
regret = zeros(1,T);
regret_exp = zeros(1,T); 
arm_sel = zeros(1,K);
W_sel = zeros(1,K);

W_pull = binornd(1,w_repeat);
W_pull = reshape(W_pull,T,L);
W_opt = W_pull( :,index_opt ); % result of true opt arms (CAN BE OUT OF THE LOOP)   
reward_opt = prod(1-W_opt,2);
reward_exp_opt = (1-w_opt)^K;  %prod(1-w(1:K)); 

%% 3. main phase
B = zeros(d,1);
N = eye(d);
sigma = tune;
c = sqrt( d*log(1+ T*K/d ) + 2*log(T*K) ) + 1; 
U_t_part = 0; 

for t = 1:T
    % 1: construct Thompson sample
    theta = N\B*sigma^(-2);
    u_tmp = sqrt( diag(X'*N^(-1)*X) ); 
    U_t = min( X'*theta + c * u_tmp  , 1 );
    
    % 2: select arms to pull
    [ U_sort, U_ord ] = sort(U_t, 'descend'); 
    arm_sel = U_ord(1:K); 

    % 3: pull selected and optimal arms
    W_sel = W_pull( t,arm_sel ); % result of selected arms 
    
    % 4: calculate regret
	regret(t) = prod(1-W_sel ) - reward_opt(t,:); % regret
    regret_exp(t) = prod(1- w( arm_sel ) ) - reward_exp_opt; % expected regret 

    % 5: update parameters
    observ = find(W_sel ==1); 
    
    if isempty( observ ) == 1
        no_observe = K; % number of observations
    else
        no_observe = observ(1);
    end
    for i = 1:no_observe
        index = arm_sel( i);
        N = N + sigma^(-2)*X(:,index)*X(:,index)';
        B = B + X(:,index)*W_sel( i);        
    end
end

regret_sum = cumsum(regret); % cumulative regret
regret_exp_sum = cumsum(regret_exp); % cumulative regret

end



