% e2 h1

clear all  ; close all  ; clc  ;

%% ASSIGNMENT

load logmap ;

%%
x = 1:200;

%%

figure(1) ;
plot(Z)

figure(2)
plot(Ztest) ;

figure(3)
plot(Ztest_withoutnoise) ;

figure(4)
plot(x(1:150),Z,'b') ;
hold on;
plot(x(150:200),[Z(end,1);Ztest],'r')
legend('Train set','Test set')
hold off
%%

order = 23;%10;
X = windowize (Z, 1:( order + 1));
Y = X(:, end);
X = X(:, 1: order );

%%

gam = 10 ;
sig2 = 10 ;

% sig = 4.635891622428067e+03
% 


% gam = 1.69e+5; % NO NO NO
% sig2 = 9176;

% sig2 = 4.0424e+03;
% gam = 3.3222e+06;

gam = 8330071.4386;      
sig2 = 9360.2001164;


%[gam,sig2] = tunelssvm({X,Y,'f',gam,sig2,'RBF_kernel'},'gridsearch','crossvalidatelssvm',{10,'mse'}) ; % 50 'simplex' 'gridsearch' 
model = {X,Y,'f',gam,sig2,'RBF_kernel'} ;
[alpha,b] = trainlssvm(model);

Xs = Z(end - order +1: end , 1);
nb = 50;
prediction = predict({X, Y, 'f', gam , sig2 }, Xs , nb);

figure ;
hold on;
plot (Ztest , 'k');
plot ( prediction , 'r');
hold off;

figure(10)
plot(x(1:150),Z,'b') ;
hold on;
plot(x(150:200),[Z(end,1);prediction],'r')
legend('Train set','prediction')
hold off

%%


