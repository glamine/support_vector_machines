% Ex 1 Homework 3

clear all ; close all ; clc ;

%% ASSIGNMENT
load diabetes.mat ;

X = trainset ;
Y = labels_train ;
Xt = testset ;
Yt = labels_test ;

% normalise
X_mean = mean(X,1) ;
X_std = std(X,1) ;
X = X-repmat(X_mean,size(X,1),1) ;
X = X./repmat(X_std,size(X,1),1) ;

Xt = Xt-repmat(X_mean,size(Xt,1),1) ;
Xt = Xt./repmat(X_std,size(Xt,1),1) ;

X1 = X(Y==1,:) ;
X2 = X(Y==-1,:) ;
Xt1 = Xt(Yt==1,:) ;
Xt2 = Xt(Yt==-1,:) ;

%% visualise

titles = {'Number of times pregnant ';'Plasma glucose concentration ';'Diastolic blood pressure ';'Triceps skin fold thickness ';'2-Hour serum insulin ';'Body mass index ';'Diabetes pedigree function ';'Age '};

figure(1) ;
for idx = 1:8
    subplot(2,4,idx) ; hold on ;
    edges = linspace(min(X(:,idx)),max(X(:,idx)),20) ;
    histogram(X1(:,idx),edges,'FaceColor','b','Normalization','probability') ;
    histogram(X2(:,idx),edges,'FaceColor','r','Normalization','probability') ;
    legend('Class 1','Class 2') ;
    title(titles{idx}) ;
    
    ax = gca ;
    lin = findobj(gca, 'Type', 'Line') ;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
    set(0,'DefaultLineColor','k');
    set(gca,'box','off') ;
    set(gca, 'FontName', 'Baskervald ADF Std')
    set(gca, 'FontSize', 10) ;
    set(gca,'LineWidth',1.2) ;
end

%%

figure(2) ;
plot(mean(X1,2),std(X1,1,2),'ob') ;
hold on ;
plot(mean(X2,2),std(X2,1,2),'or') ;
legend('Class 1','Class 2') ;
xlabel('\mu') ; ylabel('\sigma') ;

%% lin
gam = 4e+0 ;
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'});

[Yht, Yl] = simlssvm({X,Y,'c',gam,[],'lin_kernel'}, {alpha,b}, Xt);
err = sum(Yht~=Yt);
pc_err = err/length(Yt) ;
disp(pc_err) ;
% 0.2857

roc(Yl,Yt);
% lin = findobj(gca, 'Type', 'Line') ;
% set(lin,'Color','k') ;
% set(lin,'LineWidth',4) ;

%% rbf
model = {X,Y,'c',[],[],'RBF_kernel','csa'};
[gam,sig2,cost] = tunelssvm(model,'simplex', 'crossvalidatelssvm',{10,'misclass'});

% gam = 3.765;
% sig2 = 0.3765;

sig2 = 21.54;%3.765;
gam = 0.5214;%0.3765;

[alpha,b] = trainlssvm({X,Y,'c',gam,sig2,'RBF_kernel'});

[Yht, Yl] = simlssvm({X,Y,'c',gam,sig2,'RBF_kernel'}, {alpha,b}, Xt);
err = sum(Yht~=Yt);
pc_err = err/length(Yt) ;
disp('perf') ;
disp(pc_err) ;
% perf
%     0.2262
% 0.3869

% [gamma sig2]: 0.511701      26.6496 => 0.2262

roc(Yl,Yt);
% lin = findobj(gca, 'Type', 'Line') ;
% set(lin,'Color','k') ;
% set(lin,'LineWidth',4) ;

%% ASSIGNMENT

load diabetes.mat ;

X = trainset ;
Y = labels_train ;
Xt = testset ;
Yt = labels_test ;

% standardize
X_mean = mean(X,1) ;
X_std = std(X,1) ;
X_norm = X-repmat(X_mean,size(X,1),1) ;
X_norm = X_norm./repmat(X_std,size(X,1),1) ;

Xt_norm = Xt-repmat(X_mean,size(Xt,1),1) ;
Xt_norm = Xt_norm./repmat(X_std,size(Xt,1),1) ;

% params
type = 'c' ;
n = 100;%100 ;%50;%

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
        performance(idx1,idx2) = crossvalidate({X_norm,Y,'c',gam_span(idx1),sig_span(idx2),'RBF_kernel'}, 10,'misclass');
    end
    disp(idx1);
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

% gam = 9.103
% sig2 = 26.83
