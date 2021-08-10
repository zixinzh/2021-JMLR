clear
clc
close all

warning('off', 'all')



%% 1.initialize
K = 2;% no. of opt arms
L = 2048;% no. of all arms
d = 2; % dimension of feature vector


 
w_opt = 0.2;% w of opt arms
w_gap = [ 0.1 0.05 ]; % gap of w of opt and subopt arms


T0 = 5;
T = 10^T0;
no_seed = 20; 


d_range = 1:4;%[ 1 2 4 8 16 ]; 
tune_range = [ 1  ]; 
tune = 1;
flag_train_range = [ 500 1000 ];

timetable = zeros(1, no_seed);
algo_fun_names = {'TSCascade', 'CTS', 'CascadeUCB1',  'CascadeKL_UCB',...
    'LinTS_Cascade', 'CascadeLinUCB', 'CascadeLinTS', 'RankedLinTS'};
algo_index =  7; 

%% for nonlinear algorithm
disp('algo L K flag_train d tune w_opt w_gap mean std median run_time');
for K = [2 4]
    for d = 1
        for T0 = T0
            T = 10^T0;
            regret_exp_sum_all = zeros(no_seed,T);
            time_name = ['T=10e', num2str(T0)];
            for algo_index = 1:4            
                algo_fun_name = algo_fun_names{algo_index};
            
%                 disp([ algo_fun_name,  ' L=', num2str(L)  ' tune=', num2str(tune)]);
%                 disp('algo L K flag_train d tune w_opt w_gap mean std median run_time');
%                 if algo_index <= 4
%                     disp('L K w_opt w_gap mean std median run_time');
%                 else
%                     disp('L K flag_train d tune w_opt w_gap mean std median run_time');
%                 end
                for flag_train_ind = 1 
                    flag_train = flag_train_range(flag_train_ind);
                    
                    folder_name = [time_name, ' ', algo_fun_name, ' L=', num2str(L)  ];
                    if exist(folder_name, 'dir')==0
                        mkdir(folder_name);
                        mkdir([folder_name, '_', num2str(no_seed) ]);
                    end
                    
                    if algo_index <= 4
                        simulation_name = [ time_name, ' ', algo_fun_name, ' L=', num2str(L), ' K=', num2str(K), ...
                            ' w_opt', num2str(w_opt), ' w_gap=', num2str(w_gap)  ];
                    else
                        simulation_name = [ time_name, ' ', algo_fun_name, ' L=', num2str(L), ' K=', num2str(K), ...
                            ' flag_train=', num2str(flag_train), ' d=', num2str(d), ' tune=', num2str(tune), ...
                            ' w_opt', num2str(w_opt), ' w_gap=', num2str(w_gap) ];
                    end
                    
                    parfor seed_index = 1:no_seed
                        seed_val = seed_index;
                        rng(seed_val);
                        %% 2.run algorithm
                        tstart = tic;
                        if algo_index == 1
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = TSCascade_jmlr( K, L, T, w_opt, w_gap   );
                        end
                        if algo_index == 2
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = CTS_jmlr( K, L, T, w_opt, w_gap  );
                        end
                        if algo_index == 3
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = CascadeUCB1_jmlr( K, L, T, w_opt, w_gap   );
                        end
                        if algo_index == 4
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = CascadeKLUCB_jmlr( K, L, T, w_opt, w_gap   );
                        end
                        if algo_index == 6
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = CascadeLinUCB_jmlr( K, L, T, d, flag_train, tune, w_opt, w_gap  );
                        end
                        if algo_index == 7
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = CascadeLinTS_jmlr( K, L, T, d, flag_train, tune, w_opt, w_gap  );
                        end
                        if algo_index == 8
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = RankedLinTS_jmlr( K, L, T, d, flag_train, tune, w_opt, w_gap );
                        end
                        timetable_tmp = toc(tstart);
                        timetable(seed_index) = timetable_tmp;
                        
                        %% 3.plot
                        parsave([ folder_name, '_', num2str(no_seed), '/seed=', num2str(seed_val), ' ', simulation_name, '.mat'], regret_exp, regret, arm_sel, index_opt, thompson_sample_part); %, var_all );
                        %% 4.save expected cumulative regret in a matrix
                        regret_exp_sum_all(seed_index,:) = regret_exp_sum;
                    end
                    
                    %% 5. save 20 sets of data for each setting of K, L and w_gap
                    mean_regret = mean(regret_exp_sum_all);
                    std_regret = std(regret_exp_sum_all);
                    med_regret = median(regret_exp_sum_all);
                    ave_time = mean(timetable);
                    save([ folder_name, '/',simulation_name, '.mat'], 'regret_exp_sum_all', 'mean_regret', 'std_regret', 'med_regret', 'timetable', 'ave_time');
                    figure;
                    set(0,'DefaultFigureVisible', 'off')
                    upbd = mean_regret + std_regret;
                    lwbd = mean_regret - std_regret;
                    h = fill([ 1:T T:-1:1 ],[ upbd lwbd(end:-1:1)], 'b' ,'LineStyle','none' );
                    % Choose a number between 0 (invisible) and 1 (opaque) for facealpha.
                    set(h,'facealpha',.2);
                    hold on
                    plot(1:T, mean_regret, 'b');
                    saveas(gcf,[ folder_name, '/', simulation_name, '.jpg']);
                    hold off 
                    if algo_index <= 4
                        disp([ algo_fun_name, ' ',...
                            num2str(L), ' ', num2str(K), ' ',...
                        'NA', ' ', 'NA', ' ', 'NA', ' ', ...
                        num2str(w_opt), ' ', num2str(w_gap), ' ',...
                        num2str(mean_regret(T)), ' ', num2str(std_regret(T)), ' ', num2str( med_regret(T) ), ' ',...
                        num2str(ave_time) ]);
                    else
                        disp([ algo_fun_name, ' ',...
                            num2str(L), ' ', num2str(K), ' ',...
                        num2str(flag_train), ' ', num2str(d), ' ', num2str(tune), ' ', ...
                        num2str(w_opt), ' ', num2str(w_gap), ' ',...
                        num2str(mean_regret(T)), ' ', num2str(std_regret(T)), ' ', num2str( med_regret(T) ), ' ',...
                        num2str(ave_time) ]);
                    end
                    
                    
                end
            end
        end
    end
end

%% for linear algorithm
disp('algo L K flag_train d tune w_opt w_gap mean std median run_time');
for K = [2 4]
    for d = d_range
        for T0 = T0
            T = 10^T0;
            regret_exp_sum_all = zeros(no_seed,T);
            time_name = ['T=10e', num2str(T0)];
            for algo_index = 5:8            
                algo_fun_name = algo_fun_names{algo_index};
            
%                 disp([ algo_fun_name,  ' L=', num2str(L)  ' tune=', num2str(tune)]);
%                 disp('L K flag_train d tune w_opt w_gap mean std median run_time');
%                 if algo_index <= 4
%                     disp('L K w_opt w_gap mean std median run_time');
%                 else
%                     disp('L K flag_train d tune w_opt w_gap mean std median run_time');
%                 end
                for flag_train_ind = 1:length(flag_train_range)
                    flag_train = flag_train_range(flag_train_ind);
                    
                    folder_name = [time_name, ' ', algo_fun_name, ' L=', num2str(L)  ];
                    if exist(folder_name, 'dir')==0
                        mkdir(folder_name);
                        mkdir([folder_name, '_', num2str(no_seed) ]);
                    end
                    
                    if algo_index <= 4
                        simulation_name = [ time_name, ' ', algo_fun_name, ' L=', num2str(L), ' K=', num2str(K), ...
                            ' w_opt', num2str(w_opt), ' w_gap=', num2str(w_gap)  ];
                    else
                        simulation_name = [ time_name, ' ', algo_fun_name, ' L=', num2str(L), ' K=', num2str(K), ...
                            ' flag_train=', num2str(flag_train), ' d=', num2str(d), ' tune=', num2str(tune), ...
                            ' w_opt', num2str(w_opt), ' w_gap=', num2str(w_gap) ];
                    end
                    %                 disp([ algo_name{algo_index}, ' w_opt', num2str(w_opt),' w_gap=', num2str(w_gap), ' d=', num2str(d), ...
                    %                     ' corrup_bd=', num2str(corrup_bd)]);
                    %                 disp('L w_opt w_gap d corrup_bd mean std median run_time');
                    parfor seed_index = 1:no_seed
                        seed_val = seed_index;
                        rng(seed_val);
                        %% 2.run algorithm
                        tstart = tic;
                        if algo_index == 1
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = TSCascade_jmlr( K, L, T, w_opt, w_gap   );
                        end
                        if algo_index == 2
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = CTS_jmlr( K, L, T, w_opt, w_gap  );
                        end
                        if algo_index == 3
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = CascadeUCB1_jmlr( K, L, T, w_opt, w_gap   );
                        end
                        if algo_index == 4
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = CascadeKLUCB_jmlr( K, L, T, w_opt, w_gap   );
                        end
                        if algo_index == 6
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = CascadeLinUCB_jmlr( K, L, T, d, flag_train, tune, w_opt, w_gap  );
                        end
                        if algo_index == 7
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = CascadeLinTS_jmlr( K, L, T, d, flag_train, tune, w_opt, w_gap  );
                        end
                        if algo_index == 8
                            [ X, mu_exp, T_exp, B, thompson_sample_part, regret,...
                                regret_sum, regret_exp, regret_exp_sum, arm_sel, W_sel, W_opt, index_opt ]...
                                = RankedLinTS_jmlr( K, L, T, d, flag_train, tune, w_opt, w_gap );
                        end
                        timetable_tmp = toc(tstart);
                        timetable(seed_index) = timetable_tmp;
                        
                        %% 3.plot
                        parsave([ folder_name, '_', num2str(no_seed), '/seed=', num2str(seed_val), ' ', simulation_name, '.mat'], regret_exp, regret, arm_sel, index_opt, thompson_sample_part); %, var_all );
                        %% 4.save expected cumulative regret in a matrix
                        regret_exp_sum_all(seed_index,:) = regret_exp_sum;
                    end
                    
                    %% 5. save 20 sets of data for each setting of K, L and w_gap
                    mean_regret = mean(regret_exp_sum_all);
                    std_regret = std(regret_exp_sum_all);
                    med_regret = median(regret_exp_sum_all);
                    ave_time = mean(timetable);
                    save([ folder_name, '/',simulation_name, '.mat'], 'regret_exp_sum_all', 'mean_regret', 'std_regret', 'med_regret', 'timetable', 'ave_time');
                    figure;
                    set(0,'DefaultFigureVisible', 'off')
                    upbd = mean_regret + std_regret;
                    lwbd = mean_regret - std_regret;
                    h = fill([ 1:T T:-1:1 ],[ upbd lwbd(end:-1:1)], 'b' ,'LineStyle','none' );
                    % Choose a number between 0 (invisible) and 1 (opaque) for facealpha.
                    set(h,'facealpha',.2);
                    hold on
                    plot(1:T, mean_regret, 'b');
                    saveas(gcf,[ folder_name, '/', simulation_name, '.jpg']);
                    hold off 
                    if algo_index <= 4
                        disp([ algo_fun_name, ' ',...
                            num2str(L), ' ', num2str(K), ' ',...
                        'NA', ' ', 'NA', ' ', 'NA', ' ', ...
                        num2str(w_opt), ' ', num2str(w_gap), ' ',...
                        num2str(mean_regret(T)), ' ', num2str(std_regret(T)), ' ', num2str( med_regret(T) ), ' ',...
                        num2str(ave_time) ]);
                    else
                        disp([ algo_fun_name, ' ',...
                            num2str(L), ' ', num2str(K), ' ',...
                        num2str(flag_train), ' ', num2str(d), ' ', num2str(tune), ' ', ...
                        num2str(w_opt), ' ', num2str(w_gap), ' ',...
                        num2str(mean_regret(T)), ' ', num2str(std_regret(T)), ' ', num2str( med_regret(T) ), ' ',...
                        num2str(ave_time) ]);
                    end
                    
                    
                end
            end
        end
    end
end