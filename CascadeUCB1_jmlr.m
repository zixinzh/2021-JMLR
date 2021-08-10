function [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
    regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
    = CascadeUCB1_jmlr( K, L, T, w_opt, w_gap  ) 

X = 0; B = 0; thompson_sample_part=0; 
%cascadingUCB1 
%
% Input:
%     K --- no. of opt arms
%     L --- no. of all arms
%     w_opt --- w of opt arms
%     w_gap --- gap of w of opt and subopt arms
%     w_sub --- w_opt - w_gap; % w of subopt arms
%     T --- no. of trials
% Output:
%     mu_exp --- the experimental mean of arms at each time t
%     T_exp --- no. of observations of arms at each time t
%     regret --- true regret at each time t
%     regret_sum --- cumulative regret
%     regret_exp --- expected regret at each time t
%     regret_exp_sum --- expected cumulative regret
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


regret = zeros(1,T);
regret_exp = zeros(1,T);
arm_sel = zeros(1,K);
W_sel = zeros(1,K); 

%% 2. main phase
% 2.1. generate random variables
w_repeat = repmat(w,T,1); % all w repeat
w_repeat = reshape(w_repeat,1,T*L);
W_pull = binornd(1,w_repeat);
W_pull = reshape(W_pull,T,L);
clear w_repeat
% 2.2. initialize parameters
T_exp = ones(1,L); %(T,L);
mu_exp = zeros(T,L); % experimental mu
mu_exp(1,:) = W_pull(1,:);
% U_exp = ones(1,L);
% 2.3. calculate for optimal items
W_opt = W_pull( 1:T,index_opt ); % result of true opt arms (CAN BE OUT OF THE LOOP)   
reward_opt = prod(1-W_opt,2);
reward_exp_opt = (1-w_opt)^K;   
% 2.4. run for t = 1
for t = 1
    % 2: select arms to pull 
    U_exp = mu_exp(t,:);
    [ ~, U_ord ] = sort(U_exp, 'descend');
    arm_sel  = U_ord(1:K); 
    
    % 3: pull selected and optimal arms
    W_sel = W_pull( t,arm_sel  ); % result of selected arms 
    
    % 4: calculate regret
    regret(t) = prod(1-W_sel ) - reward_opt(t,:); % regret
    regret_exp(t) = prod(1- w(arm_sel ) ) - reward_exp_opt; % expected regret  
    % 5: update parameters
    observ = find(W_sel ==1); 
    if isempty( observ ) == 1
        no_observe = K; % number of observations
    else
        no_observe = observ(1);
    end
    mu_exp(t+1,:) = mu_exp(t,:); 
    for i = 1:no_observe
        index = arm_sel( i);
        mu_exp(t+1,index) = ( T_exp(index)*mu_exp(t,index) + W_sel( i) )/( T_exp(index) + 1  ); 
        T_exp(index) = T_exp(index) + 1;  
    end
end
% 2.5. run for t>1
for t = 2:T
    % 1: calculate U_exp 
    U_exp = mu_exp(t,:) + sqrt( 1.5* log(t-1)./T_exp ); 
    
    % 2: select arms to pull    
    [ ~, U_ord ] = sort(U_exp, 'descend');
    arm_sel  = U_ord(1:K); 
    
    % 3: pull selected and optimal arms
    W_sel = W_pull( t,arm_sel  ); % result of selected arms 
    
    % 4: calculate regret
    regret(t) = prod(1-W_sel ) - reward_opt(t,:); % regret
    regret_exp(t) = prod(1- w(arm_sel ) ) - reward_exp_opt; % expected regret 
% 

    % 5: update parameters
    observ = find(W_sel ==1); 
    if isempty( observ ) == 1
        no_observe = K; % number of observations
    else
        no_observe = observ(1);
    end
    mu_exp(t+1,:) = mu_exp(t,:); 
    for i = 1:no_observe
        index = arm_sel( i);
        mu_exp(t+1,index) = ( T_exp(index)*mu_exp(t,index) + W_sel( i) )/( T_exp(index) + 1  ); 
        T_exp(index) = T_exp(index) + 1;  
    end
end

regret_sum = cumsum(regret); % cumulative regret
regret_exp_sum = cumsum(regret_exp); % cumulative regret

end



