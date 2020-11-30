% e3 h2

clear all  ; close all  ; clc  ;

%%

data = load('shuttle.dat','-ascii');
addpath('../LSSVMlab')

% 9 dim, 58 000 data points

% last column : class  (5 classes)


%% train / test sets
rng(0681349,'twister') ; % reproducability
idx_tr = randperm(size(data,1),43500) ;
X = data(idx_tr,1:end-1);
Y = data(idx_tr,end);
testX = data(:,1:end-1);
testY = data(:,end);
testX(idx_tr,:) = [] ; % pas mal comme technique
testY(idx_tr) = [] ;

% 0.75 for training, 0.25 test

k = 3;%15 ; 6 12 24
function_type = 'c';
kernel_type = 'lin_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'ds'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);


