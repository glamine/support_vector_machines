% Exercice 1 - SVM

X1 = randn (50 ,2) + 1;
X2 = randn (51 ,2) - 1;

Y1 = ones (50 ,1);
Y2 = -ones (51 ,1);

figure ;
hold on;
plot (X1 (: ,1) , X1 (: ,2) , 'ro');
plot (X2 (: ,1) , X2 (: ,2) , 'bo');
hold off;

%% 1.3 
close all
clear all

load iris.mat

type = 'classification';
kernel_type = 'RBF_kernel'; %'poly_kernel'; %  'lin_kernel';
process = 'preprocess'; % 'original'; %

gam = 0.1;%10;
sig2 = 1; %0.2

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,kernel_type,process});

Y_approx = simlssvm({Xtrain,Ytrain,type,gam,sig2,kernel_type,process},{alpha,b},Xtest);
Y_train_approx = simlssvm({Xtrain,Ytrain,type,gam,sig2,kernel_type,process},{alpha,b},Xtrain);

figure
plotlssvm({Xtrain,Ytrain,type,gam,sig2,kernel_type,process},{alpha,b});

e_rms_test = sqrt(mean((Y_approx - Ytest).^2))
e_rms_train = sqrt(mean((Y_train_approx - Ytrain).^2))

figure 
roc(Ytest,Y_approx)
figure
roc(Ytrain,Y_train_approx)

%%

gam = 0.1;%10;
sig2 = 1; %0.2
t = 1;
degree = 4;
parameters = sig2; %[t;degree] ;%  [];
type = 'classification';
kernel_type = 'RBF_kernel'; %'poly_kernel'; %  'lin_kernel';
process = 'preprocess'; % 'original'; %

gam = 1:2:20;%ones(1,n); % 
n = length(gam);
parameters = ones(1,n)*0.5;% 0.01:1:10;%
  
e_rms_test = zeros(n,1);
e_rms_train = zeros(n,1);

%%

%%

for i = 1:n
    

    %[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}); 
    %[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,parameters,kernel_type,'original'});
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam(i),parameters(i),kernel_type,process});

    Y_approx = simlssvm({Xtrain,Ytrain,type,gam(i),parameters(i),kernel_type,process},{alpha,b},Xtest);
    Y_train_approx = simlssvm({Xtrain,Ytrain,type,gam(i),parameters(i),kernel_type,process},{alpha,b},Xtrain);

    figure(i)
    plotlssvm({Xtrain,Ytrain,type,gam,parameters(i),kernel_type,process},{alpha,b});

    e_rms_test(i) = sqrt(mean((Y_approx - Ytest).^2));
    e_rms_train(i) = sqrt(mean((Y_train_approx - Ytrain).^2));

end

    figure(10)
    plot(parameters,e_rms_test)
    title('Test RMS error')
    figure(11)
    plot(parameters,e_rms_train)
    title('Train RMS error')
    
    figure(12)
    plot(gam,e_rms_test)
    title('Test RMS error')
    figure(13)
    plot(gam,e_rms_train)
    title('Train RMS error')
    
    figure(14)
    plot(gam,e_rms_test)
    title('Test RMS error')
    figure(15)
    plot(gam,e_rms_train)
    title('Train RMS error')
    
%% 1.3.2 automated tuning

gam = [0.01 0.1 1 10 100 1000];
sig2 = [0.01 0.1 1 10 100 1000];

n1 = length(gam);
n2 = length(sig2);
perf1 = zeros(1,n2+(n1-1)*n2);
perf2 = zeros(1,n2+(n1-1)*n2);
perf3 = zeros(1,n2+(n1-1)*n2);

for i = 1:n1
    
    for j = 1:n2

        % random split
        perf1(j + (i-1)*(n2)) = rsplitvalidate ({ Xtrain , Ytrain , 'c', gam(i) , sig2(j) ,'RBF_kernel'}, 0.80 , 'misclass');

        % k-fold crossvalidate
        perf2(j + (i-1)*(n2)) = crossvalidate ({ Xtrain , Ytrain , 'c', gam(i) , sig2(j),'RBF_kernel'}, 10, 'misclass');

        % leave-one-out
        perf3(j + (i-1)*(n2)) = leaveoneout ({ Xtrain , Ytrain , 'c', gam(i) , sig2(j),'RBF_kernel'}, 'misclass');
    
    end
end

figure
plot(log(gam),perf1([1,7,13,19,25,31]),'r')
hold on 
plot(log(gam),perf2([1,7,13,19,25,31]),'g')
plot(log(gam),perf3([1,7,13,19,25,31]),'b')
hold off

figure
plot(log(sig2),perf1(1:6),'r')
hold on 
plot(log(sig2),perf2(1:6),'g')
plot(log(sig2),perf3(1:6),'b')
hold off

%%

algorithm = 'simplex';%'gridsearch'; %  'linesearch'; %
validation = 'crossvalidatelssvm';%'leaveoneoutlssvm';%'rsplitvalidate'; %  'rcrossvalidatelssvm';% 
costargs = {10, 'misclass'}; % {'misclass'}; %{0.8, 'misclass'}; %

[gam ,sig2 , cost] = tunelssvm({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, algorithm, validation,costargs);

cost

%% 1.3.4

% gam = 2.2895;
% sig2 = 0.20374;

gam = 1;
sig2 = 10;

[alpha , b] = trainlssvm({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'});

Ylatent = simlssvm({Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'}, {alpha , b}, Xtest );

[tpr,fpr,thresholds] = roc((Ytest' + 1)/2,(Ylatent' + 1)/2)
figure
plotroc((Ytest' + 1)/2,(Ylatent' + 1)/2)
figure
plotlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel','preprocess'},{alpha,b});


%% 1.3.5

% gam = 1;
% sig2 = 10;

% gam = 2.2895;
% sig2 = 0.20374;

gam = 20;
sig2 = 0.5;

Ymodout = bay_modoutClass ({ Xtrain , Ytrain , 'c', gam , sig2,'RBF_kernel' }, 'figure');


