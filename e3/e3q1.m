% e3 q3

clear all  ; close all  ; clc  ;

%% ASSIGNMENT
%rng(0681349) ; % reproducability
X = 3.*randn(100,2);

sig2_span = 10.^linspace(-2,2,70) ;
en = zeros(size(sig2_span)) ;
en1 = zeros(size(sig2_span)) ;
en2 = zeros(size(sig2_span)) ;

hw = waitbar(0) ;
for idx=1:length(sig2_span)
    ssize = 10;
    sig2 = sig2_span(idx) ;
    subset = zeros(ssize,2);
    
    idx_perm = randperm(size(X,1)) ;
    X = X(idx_perm,:) ;
    for t = 1:400
        
        %
        % new candidate subset
        %
        r = ceil(rand*ssize);
        candidate = [subset([1:r-1 r+1:end],:); X(mod(t,100)+1,:)]; % add point X(t,:) to subset in random place r
        
        %
        % is this candidate better than the previous?
        %
        if kentropy(candidate, 'RBF_kernel',sig2) > kentropy(subset, 'RBF_kernel',sig2)
            subset = candidate;
        end
        if t==10
            en1(idx) = kentropy(subset, 'RBF_kernel',sig2) ;
        elseif t==50
            en2(idx) = kentropy(subset, 'RBF_kernel',sig2) ;
        end
    end
    
    en(idx) = kentropy(subset, 'RBF_kernel',sig2) ;
   waitbar(idx/length(sig2_span),hw) ;
end
delete(hw) ;

%%
figure(1)  ;
semilogx(sig2_span,en);%,'-k','LineWidth',1.7) ; %hold on ;
% semilogx(sig2_span,en1,':k','LineWidth',1.7) ;
% semilogx(sig2_span,en2,'--k','LineWidth',1.7) ;
xlabel('\sigma^2') ; ylabel('Entropy') ;


% ax = gca   ;
% lin = findobj(gca, 'Type', 'Line')   ;
% %ax.XAxisLocation = 'origin'  ;
% %ax.YAxisLocation = 'origin'  ;
% set(0,'DefaultLineColor','k')  ;
% set(gca,'box','off')   ;
% set(gca, 'FontName', 'Baskervald ADF Std')
% set(gca, 'FontSize', 18)   ;
% set(gca,'LineWidth',1.2)   ;
% %set(lin,'LineWidth',1)   ;