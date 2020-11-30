% Homework1

clear all ; close all ; clc ;

%% ASSIGNMENT
load ripley.mat ;

Xt = Xtest;
Yt = Ytest;
X = Xtrain;
Y = Ytrain;

X1 = X(Y==1,:) ;
X2 = X(Y==-1,:) ;
Xt1 = Xt(Yt==1,:) ;
Xt2 = Xt(Yt==-1,:) ;

% visualise
figure(1) ;
hold on ;
plot(X1(:,1),X1(:,2),'*r') ;
plot(X2(:,1),X2(:,2),'ob') ;

%% lin
figure(2) ;
gam = 5e+0 ;
[alpha,b] = trainlssvm({X,Y,'c',gam,[],'lin_kernel'});
plotlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b});

figure(3) ;
plotlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b});
hold on ;
plot(Xt1(:,1),Xt1(:,2),'^y') ;
plot(Xt2(:,1),Xt2(:,2),'vb') ;
legend({'Classifier','Training set (class 1)','Training set (class 2)','Test set (class 1)','Test set (class 2)'}) ;
[Yht, Yl] = simlssvm({X,Y,'c',gam,[],'lin_kernel'}, {alpha,b}, Xt);
err = sum(Yht~=Yt);
pc_err = err/length(Yt) ;

disp("lin error")
disp(pc_err) ;

roc(Yl,Yt);
%lin = findobj(gca, 'Type', 'Line') ;
%set(lin,'Color','k') ;
%set(lin,'LineWidth',4) ;

%% rbf
model = {X,Y,'c',[],[],'RBF_kernel','csa'};
[gam,sig2,cost] = tunelssvm(model,'simplex', 'crossvalidatelssvm',{10,'misclass'}); %simplex

gam = 0.4132;%0.231;%0.2595;%443.1;
sig2 = 0.4329;%1.322;%2.009;%4977;
%cost

figure(5) ;
[alpha,b] = trainlssvm({X,Y,'c',gam,sig2,'RBF_kernel'});
plotlssvm({X,Y,'c',gam,sig2,'RBF_kernel'},{alpha,b});

figure(6) ;
plotlssvm({X,Y,'c',gam,sig2,'RBF_kernel'},{alpha,b});
hold on ;
plot(Xt1(:,1),Xt1(:,2),'^y') ;
plot(Xt2(:,1),Xt2(:,2),'vb') ;
legend({'Classifier','Training set (class 1)','Training set (class 2)','Test set (class 1)','Test set (class 2)'}) ;

[Yht, Yl] = simlssvm({X,Y,'c',gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
err = sum(Yht~=Yt);
pc_err = err/length(Yt) ;

disp('perf rbf') ;
disp(pc_err) ;

roc(Yl,Yt);
% lin = findobj(gca, 'Type', 'Line') ;
% set(lin,'Color','k') ;
% set(lin,'LineWidth',4) ;

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ASSIGNMENT
load ripley.mat ;

Xt = Xtest;
Yt = Ytest;
X = Xtrain;
Y = Ytrain;

% params
type = 'c' ;
n = 100;%0 ;

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
        performance(idx1,idx2) = crossvalidate({X,Y,'c',gam_span(idx1),sig_span(idx2),'RBF_kernel'}, 10,'misclass');
    end
    disp(idx1)
end

[C,h] = contourf(gam_span,sig_span,performance'*100) ;
set(h,'LineColor','none') ;
colormap(flipud(gray)) ;
caxis([0 50]) ;
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

