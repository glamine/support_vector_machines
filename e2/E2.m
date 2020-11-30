
clear all ; close all ; clc ;

%% SVM exercice 2
X = ( -3:0.01:3)';
Y = sinc (X) + 0.1.* randn ( length (X), 1);

Xtrain = X (1:2: end);
Ytrain = Y (1:2: end);
Xtest = X (2:2: end);
Ytest = Y (2:2: end);

% gam = 10;
% sig2 = 0.3;

% Xt = 3.*randn(10,1);     

mse = zeros(9,1);

    i = 1;   
for gam = [10 1000 1000000]
   for sig2 = [0.1 1 100]
       figure(i)
       type = 'function estimation';
       [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
       Yt = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
       plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
       hold on; 
       plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');
       plot(Xtest,Ytest,'*k')
       legend('estimation','train data','sinc','test data')
       hold off;
       
       mse(i,1) = mean((Yt - Ytest).*(Yt - Ytest));
       
       i = i + 1;
    
   end 
end

% Xt = 3.*randn(10,1);
% Yt = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xt);
% plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
% hold on; plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');

