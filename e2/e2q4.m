% e2 q4

gam = 0.4579;
sig2 = 0.2700;

X = 6.*rand(100,3) - 3;
Y = sinc(X(:,1)) + 0.1.*randn(100,1);

[ selected , ranking ] = bay_lssvmARD({X, Y, 'f', gam , sig2 });

selected = 2;

% inputs = bay_lssvmARD({X,Y,type, 10,3});
[alpha,b] = trainlssvm({X(:,selected),Y,'f', gam,sig2});

figure(1)  ;
plotlssvm({X(:,selected),Y,'f',gam,sig2,'RBF_kernel'},{alpha,b}) ;
hold on; 
plot(min(X(:,selected)):.1:max(X(:,selected)),sinc(min(X(:,selected)):.1:max(X(:,selected))),'r-.');
title('Training set results')  ;
legend('Regression','Training points') ;
hold off