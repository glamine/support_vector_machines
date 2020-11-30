%e3h2 bis

clear all  ; close all  ; clc  ;

data = load('breast_cancer_wisconsin_data.mat','-ascii');
addpath('../LSSVMlab')

% data = load('shuttle.dat','-ascii');
% addpath('../LSSVMlab')

% data = load('california.dat','-ascii');
% addpath('../LSSVMlab')

%%

tr_set_prop = 4/5 ;

% X = data(:,1:end-1);
% Y = data(:,end);
% testX = [];
% testY = [];

rng(0681349,'twister') ; % reproducability
idx_tr = randperm(size(data,1),round(size(data,1)*tr_set_prop)) ;
%idx_tr = randperm(size(data,1),43500) ;
X = data(idx_tr,1:end-1);
Y = data(idx_tr,end);
testX = data(:,1:end-1);
testY = data(:,end);
testX(idx_tr,:) = [] ;
testY(idx_tr) = [] ;

%%
%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 6;
function_type = 'f'; %'c' - classification, 'f' - regression  
kernel_type = 'lin_kernel';% 'RBF_kernel'; % or , 'poly_kernel'
global_opt = 'ds';% 'csa'; % 'csa' or 

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);
