% e2q2

clear all ; close all ; clc ;

%% ASSIGNMENT part1

% params
optFun = 'gridsearch';
globalOptFun = 'csa';

X = (-3:0.01:3)' ;
Y = sinc(X) + 0.1.*randn(length(X),1) ;

Xtrain = X(1:2:length(X)) ;
Ytrain = Y(1:2:length(Y)) ;
Xtest = X(2:2:length(X)) ;
Ytest = Y(2:2:length(Y)) ;

[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', globalOptFun},optFun,'crossvalidatelssvm',{10,'mse'}) ;

% gam = 4.3758;
% sig2 = 0.6708;

% sig2 = 0.04037; % good !
% gam = 16.3;

sig2 = 0.0231;%0.2154;
gam = 2.812;%278.3;

[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}) ;

figure(1)  ;
plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b}) ;
hold on; 
plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');
title('Training set results')  ;
legend('Regression','Training points') ;
hold off

%% ASSIGNMENT part2

% params
optFun = 'gridsearch';% 'simplex';%
globalOptFun = 'csa';%'ds';% 
n = 1e+3 ;
m = 15;%5e+1 ;

Time = zeros(1,m) ;
Gam = zeros(1,m) ;
Sig2 = zeros(1,m) ;
Cost = zeros(1,m) ;

hw = waitbar(0) ;

for idx = 1:m
    X = linspace(-3,3,n)' ;
    Y = sinc(X) + 0.1.* randn( length(X), 1);
    
    tic ;
    [gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', globalOptFun},optFun,'crossvalidatelssvm',{10,'mse'}) ;
    Time(idx) = toc ;
    Gam(idx) = gam ;
    Sig2(idx) = sig2 ;
    Cost(idx) = cost ;
    waitbar(idx/m,hw) ;
end

delete(hw) ;

%%

disp('time') ;
disp(mean(Time)) ;
disp(std(Time)) ;

disp('gamma') ;
disp(mean(Gam)) ;
disp(std(Gam)) ;

disp('sig2') ;
disp(mean(Sig2)) ;
disp(std(Sig2)) ;

disp('cost') ;
disp(mean(Cost)) ;
disp(std(Cost)) ;

%GRID + DS

% time
%     8.1475
% 
%     0.2938
% 
% gamma
%    2.7291e+04
% 
%    1.0368e+05
% 
% sig2
%     1.3613
% 
%     0.8947
% 
% cost
%     0.0102
% 
%    4.9223e-04

% GRID + CSA

% time
%     7.7428
% 
%     0.2724
% 
% gamma
%   156.2268
% 
%   308.5060
% 
% sig2
%     0.9463
% 
%     0.5262
% 
% cost
%     0.0100
% 
%    4.3060e-04

% SIMP + DS

% time
%     5.1711
% 
%     0.3674
% 
% gamma
%    2.4780e+03
% 
%    7.7286e+03
% 
% sig2
%     1.0560
% 
%     0.4506
% 
% cost
%     0.0100
% 
%    5.4511e-04

% SIMP + CSA

% time
%     5.3011
% 
%     0.4265
% 
% gamma
%    1.1122e+03
% 
%    3.1218e+03
% 
% sig2
%     0.9797
% 
%     0.5993
% 
% cost
%     0.0103
% 
%    3.7016e-04

%%
%% ASSIGNMENT

X = (-3:0.01:3)' ;
Y = sinc(X) + 0.1.*randn(length(X),1) ;

Xtrain = X(1:2:length(X)) ;
Ytrain = Y(1:2:length(Y)) ;
Xtest = X(2:2:length(X)) ;
Ytest = Y(2:2:length(Y)) ;

X = Xtrain ;
Y = Ytrain ;
Xt = Xtest ;
Yt = Ytest ;

% %% standardize
% X_mean = mean(X,1) ;
% X_std = std(X,1) ;
% X_norm = X-repmat(X_mean,size(X,1),1) ;
% X_norm = X_norm./repmat(X_std,size(X,1),1) ;
% 
% Xt_norm = Xt-repmat(X_mean,size(Xt,1),1) ;
% Xt_norm = Xt_norm./repmat(X_std,size(Xt,1),1) ;

% params
type = 'f' ;
n = 50 ;%50;%

min_gam = -2 ;
max_gam = 3 ;%2;%
n_gam = n ;
gam_span = linspace(min_gam,max_gam,n_gam) ;
gam_span = 10.^gam_span ;

min_sig = -2 ;
max_sig = 4 ;%2;%
n_sig = n ;
sig_span = linspace(min_sig,max_sig,n_sig) ;
sig_span = 10.^sig_span ;

performance = zeros(length(gam_span),length(sig_span)) ;

for idx1 = 1:length(gam_span)
    gam_loc = gam_span(idx1) ;
    for idx2 = 1:length(sig_span)
        % train lssvm model
        performance(idx1,idx2) = crossvalidate({X,Y,'f',gam_span(idx1),sig_span(idx2),'RBF_kernel'}, 10,'mse');
    end
    disp(idx1);
end

[C,h] = contourf(gam_span,sig_span,performance'*100) ;



set(h,'LineColor','none') ;
colormap(flipud(gray)) ;
caxis([0 50]) ; %50
colorbar('eastoutside');

hold on ;
ax = gca ;
set(ax,'xscale','log','yscale','log');
lin = findobj(gca, 'Type', 'Line') ;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
set(0,'DefaultLineColor','k');
set(gca,'box','off') ;
set(gca, 'FontName', 'Baskervald ADF Std')
set(gca, 'FontSize', 18) ;
set(gca,'LineWidth',1.2) ;
set(lin,'LineWidth',2) ;
%set(lin,'MarkerFaceColor','k') ;
%set(gca,'XTickLabel',[]) ; set(gca,'YTickLabel',[])

xlabel('\gamma') ; ylabel('\sigma^2') ;
zlabel('Test error [%]') ;

% gam = 0.04037
% sig2 = 16.3


