%e2q3

clear all ; close all ; clc ;

%%

X = (-3:0.01:3)' ;
Y = sinc(X) + 0.1.*randn(length(X),1) ;

Xtrain = X(1:2:length(X)) ;
Ytrain = Y(1:2:length(Y)) ;
Xtest = X(2:2:length(X)) ;
Ytest = Y(2:2:length(Y)) ;

sig2 = 0.4;
gam = 10;
crit_L1 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1);
crit_L2 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2);
crit_L3 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3);

%model = bay_initlssvm({X,Y,type,[],[]});

[~, alpha ,b] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1);
[~, gam] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2);
[~, sig2 ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3);

sig2e = bay_errorbar({ Xtrain , Ytrain , 'f', gam , sig2 }, 'figure');
hold on; 
plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'b-.');
%set(gca,'LineWidth',5)  ;
hold off;
%legend();

disp(crit_L1);
disp(crit_L2);
disp(crit_L3);

disp(gam);
disp(sig2);

%     0.4579
% 
%     0.2700

%%

%% ASSIGNMENT

% params
% gam = 1e+4 ;
% sig2 = .15 ;

gam = 0.4579;
sig2 = 0.2700;

n = 100 ;
gam_span = 10.^linspace(-3,8,n) ;
sig2_span = linspace(0,0.6,n) ;

%prealloc
gam_post = zeros(size(gam_span)) ;
sig2_post = zeros(size(sig2_span)) ;

% Bayesian framework
% L1
criterion_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1) ;

% L2
for idx=1:length(gam_post)
    gam_post(idx) = bay_lssvm({Xtrain,Ytrain,'f',gam_span(idx),sig2},2) ;
end

[~,idx_gam] = min(gam_post) ;
gam = gam_span(idx_gam) ;

figure(1) ;
semilogx(gam_span,gam_post,'-k') ;
xlabel('\gamma') ;
ylabel('-log(P(\gamma|D,H))') ;
set(gca, 'FontSize', 18)  ;

% L3
for idx = 1:length(sig2_span)
    sig2_post(idx) = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2_span(idx)},3) ;
end

[~,sig2_idx] = min(sig2_post) ;
sig2 = sig2_span(sig2_idx) ;

figure(2) ;
plot(sig2_span,sig2_post,'-k') ;
xlabel('\sigma^2') ;
ylabel('-log(P(\sigma^2|D,H))') ;
set(gca, 'FontSize', 18)  ;