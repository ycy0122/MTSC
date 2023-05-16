clc;clear;close all
%% 'SL Oscillators'
class_num = 5;
sample_num = 500;
time_point = 200;
exp_num = 10;                % set > 2
sptl_moment = 5;
Mterms_temporal = 10;
Nf = sample_num;                   %%% number of electrodes
% noise set to be 50,40,30,20,10
noise_level = []
global noise time

for i = 1:class_num
    Adj_base(:,:,i) = 0.05.*2.*(rand(Nf)-0.5);
end

x_init_all = rand(2.*sample_num,exp_num);

for i = 1:class_num 
    for j = 1:exp_num

    x_init_all = rand(2.*sample_num,exp_num);
    % t  = [0 50];               %%% Time span
    tstep = time_point;
    t = linspace(0,50,tstep+1); 
    time = t;

    sig = .01;                  %%% sigma 
    
    noise  = 45;
    a = 1;                    %%% Alpha  
    w = 2;                    %%% Omega
    Ton_off = 1;                 %%% Control ON/OFF time
    
    disp(' ================ Initial Parameters ===================== ')
    disp(['  # of Oscillators  : ', num2str(Nf)])
    disp(['  Coupling strength : ', num2str(sig)])
    disp(' ========================================================= ')
    
    param.a   = a;
    param.w   = w;
    param.sig = sig;
    param.Nf  = Nf;
    param.Adj_base = Adj_base(:,:,i);
    param.Adj = sprand(2.*Nf,2.*Nf,0.02.*i);

    
    % L = laplacian_matrix(param.Adj(1:2:end-1,1:2:end-1));
            
    pos = 1;                  %%% feedback postion        
    param.pos = pos;        
    
    disp(' ========================================================= ')
    %========
    % G = graph(param.Adj(1:2:end-1,1:2:end-1));
    % figure; plot(G)
    
    %% Initial conditions (x1 y1 x2 y2 x3 y3) 
    x0 = 3*randn(1.*Nf,1); % Random initial states
    x0 = x0./max(abs(x0));
    % x0 = x0 - max(x0)./2;
    
    disp(' ========================================================= ')

        x_init = x_init_all(:,j);
        [t,X] = SimulateAndPlot(t,param,sample_num,x0,Ton_off);
%         [t,X] = SimulateAndPlot(t,param,sample_num,x0,Ton_off);
        X_tmp(:,:,j+(i-1)*exp_num) = X.';

    end
end

X = X_tmp;

save(['Asys_syn_SNR',num2str(noise),'N',num2str(sample_num),'T',num2str(time_point),'.mat']);


return

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %% 
%%                Functions used in the main file                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Simulate the system function
function [t,x] = SimulateAndPlot(t,param,Nf,x0,Ton_off)
%% Setup Ode45
global noise
    options = odeset('refine',2,'RelTol',1e-6,'AbsTol',1e-8);

    disp(' ')
    disp('  Simulating the network...')

    Adj_base = param.Adj_base;
    Adj =  param.Adj(1:2:end,1:2:end);
    Adj_tmp = Adj;
    Adj_kron = zeros(size(Adj_tmp));
    Adj_kron(:,1:1*Nf) = Adj_base.*Adj_tmp;
    param.Adj_kron = Adj_kron;
    % Coupling in both x and y variables 
    [t,x] = ode45(@(t,X) Asys_gen1(t,X,param,Ton_off),t,x0,options); 
     x = awgn(x,noise,'measured');
end

function dx = Asys_gen1(t,x,param,Ton_off) 
% This file has couplig in both x and y variables

    sig  = param.sig;        % Sigma (coupling strength)
    Nf   = param.Nf;
    Adj_kron = param.Adj_kron;
    % Initialization
    N = 1*Nf;
    dx = zeros(N,1);
    dx = Adj_kron*x;
end