function [Xhat,T,V,H,Z,cost,SDRs] = MNMF(X,x,src,N,K,it,draw,refMic,shiftSize,sigScale,window,len,H,T,V,Z,SDR_ILRMA)
%
% coded by D. Kitamura (daichi_kitamura@ipc.i.u-tokyo.ac.jp) on 29 Jun. 2017
%
% This function decomposes input 4th-order tensor by multichannel
% nonnegative matrix factorization (MNMF).
%
% see also
% H. Sawada, H. Kameoka, S. Araki, and N. Ueda,
% "Multichannel extensions of non-negative matrix factorization with
% complex-valued data," IEEE Trans. ASLP, vol.21, no.5, pp.971-982, 2013.
%
% see also
% http://d-kitamura.net
%
% [syntax]
%   [T,V,H,Z,cost] = MNMF(XX,ns)
%   [T,V,H,Z,cost] = MNMF(XX,ns,nb)
%   [T,V,H,Z,cost] = MNMF(XX,ns,nb,it)
%   [T,V,H,Z,cost] = MNMF(XX,ns,nb,it,draw)
%   [T,V,H,Z,cost] = MNMF(XX,ns,nb,it,draw,T)
%   [T,V,H,Z,cost] = MNMF(XX,ns,nb,it,draw,T,V)
%   [T,V,H,Z,cost] = MNMF(XX,ns,nb,it,draw,T,V,H)
%   [T,V,H,Z,cost] = MNMF(XX,ns,nb,it,draw,T,V,H,Z)
%
% [inputs]
%       XX: input 4th-order tensor (time-frequency-wise covariance matrices) (I x J x M x M)
%        N: number of sources
%        K: number of bases (default: ceil(J/10))
%       it: number of iteration (default: 300)
%     draw: plot cost function values or not (true or false, default: false)
%        T: initial basis matrix (I x K)
%        V: initial activation matrix (K x J)
%        H: initial spatial covariance tensor (I x N x M x M)
%        Z: initial partitioning matrix (K x N)
%
% [outputs]
%     Xhat: output 4th-order tensor reconstructed by T, V, H, and Z (I x J x M x M)
%        T: basis matrix (I x K)
%        V: activation matrix (K x J)
%        H: spatial covariance tensor (I x N x M x M)
%        Z: partitioning matrix (K x N)
%     cost: Convergence of cost function (it+1 x 1)
%

eps=1e-13;

% Check errors and set default values
[I,J,M,M] = size(X);
if size(X,3) ~= size(X,4)
    error('The size of input tensor is wrong.\n');
end
if (nargin < 5)
    K = ceil(J/10);
end
if (nargin < 6)
    it = 300;
end
if (nargin < 7)
    draw = false;
end
if (nargin < 13)
    H = repmat(sqrt(eye(M)/M),[1,1,I,N]);
    H = permute(H,[3,4,1,2]); % I x N x M x M
end
if (nargin < 14)
    T = max(rand(I,K),eps);
end
%MNMF(X,src,N,K,it,draw,refMic,H,sigScale,window,len,T,V,Z)
if (nargin < 15)
    V = max(rand(K,J),eps);
end
if (nargin < 16)
    varZ = 0.01;
    Z = varZ*rand(K,N) + 1/N;
    Z = bsxfun(@rdivide, Z, sum(Z,2)); % ensure "sum_n Z_kn = 1" (This can be rewritten as "Z = Z./sum(Z,2);" using implicit expansion for later R2016b)
end

if sum(size(T) ~= [I,K]) 
    size(T)
    [I,K]
    error('The size of input initial variable is incorrect.\n');
end
if sum(size(V) ~= [K,J])
    error('size of V');
end
if sum(size(H) ~= [I,N,M,M])
    error('size of H');
end
if sum(size(Z) ~= [K,N])
    error('size of Z');
end

Xhat = local_Xhat( T, V, H, Z, I, J, M ); % initial model tensor

% Iterative update
%fprintf('MNMF iteration: ');
if ( draw == true )
    cost = zeros( it+1, 1 );
    cost(1) = local_cost( X, Xhat, I, J, M ); % initial cost value
    SDRs = zeros( it+1, 1 );
    [SDR,SIR,SAR] = local_eval( Xhat,I,J,M,N,K,Z,T,V,H,x,shiftSize,window,len,src,refMic )
    SDRs(1) = SDR(1);
    for itr = 1:it
        [ Xhat, T, V, H, Z ] = local_iterativeUpdate( X, Xhat, T, V, H, Z, I, J, K, N, M );
        m=0;
        for i=1:I
            for n=1:N
                if m<rcond(H(i,n,:,:))
                    m = rcond(H(i,n,:,:));
                end
            end
        end
        fprintf('%f',m);
        cost(itr+1) = local_cost( X, Xhat, I, J, M );
        fprintf('%d ',itr);
        if itr>-1
            [SDR,SIR,SAR] = local_eval( Xhat,I,J,M,N,K,Z,T,V,H,x,shiftSize,window,len,src,refMic )
            SDRs(itr+1) = SDR(1);
        end
    end
%     figure;
%     semilogy( (0:it), cost );
%     set(gca,'FontName','Times','FontSize',16);
%     xlabel('Number of iterations','FontName','Arial','FontSize',16);
%     ylabel('Value of cost function','FontName','Arial','FontSize',16);
%     figure;
%     if nargin<17
%         plot((1:it+1),SDRs);
%     else
%         plot((1:it+2),[SDR_ILRMA(1) SDRs.']);
%         label = {'ILRMA'}
%         for itr = 2:it+2
%             label(itr) = {sprintf('%d',itr-2)};
%         end
%         xticks([1:it+2]);
%         xticklabels(label);
%     end
%     ylabel('SDR');
else
    cost = 0;
    SDRs = zeros( it+1, 1 );
    [SDR,SIR,SAR] = local_eval( Xhat,I,J,M,N,K,Z,T,V,H,x,shiftSize,window,len,src,refMic );
    SDRs(1) = SDR(1);
    for itr = 1:it
        [ Xhat, T, V, H, Z ] = local_iterativeUpdate( X, Xhat, T, V, H, Z, I, J, K, N, M );
        m=0;
        for i=1:I
            for n=1:N
                if m<rcond(squeeze(H(i,n,:,:)))
                    m = rcond(squeeze(H(i,n,:,:)));
                end
            end
        end
        fprintf('%f\n',m);
        [SDR,SIR,SAR] = local_eval( Xhat,I,J,M,N,K,Z,T,V,H,x,shiftSize,window,len,src,refMic );
        SDRs(itr+1) = SDR(1);
        SDR(1)
        fprintf('%d ',itr);
    end
end
%fprintf('\n');
end

%%% Cost function %%%
function [ cost ] = local_cost( X, Xhat, I, J, M )
invXhat = local_inverse( Xhat, I, J, M );
XinvXhat = local_multiplication( X, invXhat, I, J, M );
trXinvXhat = local_trace( XinvXhat, I, J, M );
detXinvXhat = local_det( XinvXhat, I, J, M );
cost = real(trXinvXhat) - log(real(detXinvXhat)) - M;
cost = sum(cost(:));
end

%%% Iterative update %%%
function [ Xhat, T, V, H, Z ] = local_iterativeUpdate( X, Xhat, T, V, H, Z, I, J, K, N, M )
%%%%% Update T %%%%%
invXhat = local_inverse( Xhat, I, J, M );
invXhatXinvXhat = local_multiplicationXYX( invXhat, X, I, J, M );
Tnume = local_Tfrac( invXhatXinvXhat, V, Z, H, I, K, M ); % I x K
Tdeno = local_Tfrac( invXhat, V, Z, H, I, K, M ); % I x K
T = T.*max(sqrt(Tnume./Tdeno),eps);
Xhat = local_Xhat( T, V, H, Z, I, J, M );

%%%%% Update V %%%%%
invXhat = local_inverse( Xhat, I, J, M );
invXhatXinvXhat = local_multiplicationXYX( invXhat, X, I, J, M );
Vnume = local_Vfrac( invXhatXinvXhat, T, Z, H, J, K, M ); % K x J
Vdeno = local_Vfrac( invXhat, T, Z, H, J, K, M ); % K x J
V = V.*max(sqrt(Vnume./Vdeno),eps);
Xhat = local_Xhat( T, V, H, Z, I, J, M );

%%%%% Update Z %%%%%
invXhat = local_inverse( Xhat, I, J, M );
invXhatXinvXhat = local_multiplicationXYX( invXhat, X, I, J, M );
Znume = local_Zfrac( invXhatXinvXhat, T, V, H, K, N, M ); % K x N
Zdeno = local_Zfrac( invXhat, T, V, H, K, N, M ); % K x N
Z = Z.*sqrt(Znume./Zdeno);
Z = max(bsxfun(@rdivide, Z, sum(Z,2)),eps); % ensure "sum_n Z_kn = 1" (This can be rewritten as "Z = Z./sum(Z,2);" using implicit expansion for later R2016b)
Xhat = local_Xhat( T, V, H, Z, I, J, M );

%%%%% Update H %%%%%
invXhat = local_inverse( Xhat, I, J, M );
invXhatXinvXhat = local_multiplicationXYX( invXhat, X, I, J, M );
[T,H] = local_RiccatiSolver( invXhatXinvXhat, invXhat, T, V, H, Z, I, J, N, M );
Xhat = local_Xhat( T, V, H, Z, I, J, M );
end

%%% Xhat %%%
function [ Xhat ] = local_Xhat( T, V, H, Z, I, J, M )
Xhat = zeros(I,J,M,M);
for mm = 1:M*M
    Hmm = H(:,:,mm); % I x N
    Xhat(:,:,mm) = ((Hmm*Z').*T)*V;
end
end

%%% Tfrac %%%
function [ Tfrac ] = local_Tfrac( X, V, Z, H, I, K, M )
Tfrac = zeros(I,K);
for mm = 1:M*M
    Tfrac = Tfrac + real( (X(:,:,mm)*V.').*(conj(H(:,:,mm))*Z') ); % using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end
end

%%% Vfrac %%%
function [ Vfrac ] = local_Vfrac( X, T, Z, H, J, K, M )
Vfrac = zeros(K,J);
for mm = 1:M*M
    Vfrac = Vfrac + real( ((H(:,:,mm)*Z').*T)'*X(:,:,mm) ); % using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end
end

%%% Zfrac %%%
function [ Zfrac ] = local_Zfrac( X, T, V, H, K, N, M)
Zfrac = zeros(K,N);
for mm = 1:M*M
    Zfrac = Zfrac + real( ((X(:,:,mm)*V.').*T)'*H(:,:,mm) ); % using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end
end

%%% Riccati solver %%%
function [ TT, H ] = local_RiccatiSolver(X, Y, T, V, H, Z, I, J, N, M)
X = reshape(permute(X, [3 4 2 1]), [M*M, J, I]); % invXhatXinvXhat, MM x J x I
Y = reshape(permute(Y, [3 4 2 1]), [M*M, J, I]); % invXhat, MM x J x I
deltaEyeM = eye(M)*eps; % to avoid numerical unstability
TT = T;
for n = 1:N % Riccati equation solver described in [Sawada+, 2013]
    for i = 1:I
        ZTV = (T(i,:).*Z(:,n)')*V;
        A = reshape(Y(:,:,i)*ZTV', [M, M]);
        B = reshape(X(:,:,i)*ZTV', [M, M]);
        Hin = reshape(H(i,n,:,:), [M, M]);
        C = Hin*B*Hin;
        AC = [zeros(M), -1*A; -1*C, zeros(M)];
        [eigVec, eigVal] = eig(AC);
        ind = find(diag(eigVal)<0);
        F = eigVec(1:M,ind);
        G = eigVec(M+1:end,ind);
        Hin = G/F;
        %[eigVec, eigVal] = eig(A*C);
        %eigVal = max(eigVal,0);
        %sqrtAC = eigVec*sqrt(eigVal)/eigVec;
        %Hin = A\sqrtAC;
        Hin = (Hin+Hin')/2;
        [eigVec, eigVal] = eig(Hin);
        eigVal = diag(max(diag(eigVal),eps));
        Hin = eigVec*eigVal/eigVec;
        Hin = (Hin+Hin')/2+deltaEyeM;
        scale = trace(Hin);
        H(i,n,:,:) = Hin/scale;
    end
end
% for n = 1:N % Another solution of Riccati equation, which is slower than the above one. The result coincides with that of the above calculation.
%     for i = 1:I
%         ZTV = (T(i,:).*Z(:,n).')*V; % 1 x J
%         A = reshape( Y(:,:,i)*ZTV.', [M, M] ); % M x M
%         B = sqrtm(reshape( X(:,:,i)*ZTV.', [M, M] )); % M x M
%         Hin = reshape( H(i,n,:,:), [M, M] );
%         Hin = Hin*B/sqrtm((B*Hin*A*Hin*B))*B*Hin; % solution of Riccati equation
%         Hin = (Hin+Hin')/2; % "+eye(M)*delta" should be added here for avoiding rank deficient in such a case
%         H(i,n,:,:) = Hin/trace(Hin);
%     end
% end
end

%%% MultiplicationXXX %%%
function [ XYX ] = local_multiplicationXYX( X, Y, I, J, M )
if M == 2
    XYX = zeros( I, J, M, M );
    x2 = real(X(:,:,1,2).*conj(X(:,:,1,2)));
    xy = X(:,:,1,2).*conj(Y(:,:,1,2));
    ac = X(:,:,1,1).*Y(:,:,1,1);
    bd = X(:,:,2,2).*Y(:,:,2,2);
    XYX(:,:,1,1) = Y(:,:,2,2).*x2 + X(:,:,1,1).*(2*real(xy)+ac);
    XYX(:,:,2,2) = Y(:,:,1,1).*x2 + X(:,:,2,2).*(2*real(xy)+bd);
    XYX(:,:,1,2) = X(:,:,1,2).*(ac+bd+xy) + X(:,:,1,1).*X(:,:,2,2).*Y(:,:,1,2);
    XYX(:,:,2,1) = conj(XYX(:,:,1,2));
else % slow
    XY = local_multiplication( X, Y, I, J, M );
    XYX = local_multiplication( XY, X, I, J, M );
end
end

%%% Multiplication %%%
function [ XY ] = local_multiplication( X, Y, I, J, M )
if M == 2
    XY = zeros( I, J, M, M );
    XY(:,:,1,1) = X(:,:,1,1).*Y(:,:,1,1) + X(:,:,1,2).*Y(:,:,2,1);
    XY(:,:,1,2) = X(:,:,1,1).*Y(:,:,1,2) + X(:,:,1,2).*Y(:,:,2,2);
    XY(:,:,2,1) = X(:,:,2,1).*Y(:,:,1,1) + X(:,:,2,2).*Y(:,:,2,1);
    XY(:,:,2,2) = X(:,:,2,1).*Y(:,:,1,2) + X(:,:,2,2).*Y(:,:,2,2);
elseif M == 3
    XY = zeros( I, J, M, M );
    XY(:,:,1,1) = X(:,:,1,1).*Y(:,:,1,1) + X(:,:,1,2).*Y(:,:,2,1) + X(:,:,1,3).*Y(:,:,3,1);
    XY(:,:,1,2) = X(:,:,1,1).*Y(:,:,1,2) + X(:,:,1,2).*Y(:,:,2,2) + X(:,:,1,3).*Y(:,:,3,2);
    XY(:,:,1,3) = X(:,:,1,1).*Y(:,:,1,3) + X(:,:,1,2).*Y(:,:,2,3) + X(:,:,1,3).*Y(:,:,3,3);
    XY(:,:,2,1) = X(:,:,2,1).*Y(:,:,1,1) + X(:,:,2,2).*Y(:,:,2,1) + X(:,:,2,3).*Y(:,:,3,1);
    XY(:,:,2,2) = X(:,:,2,1).*Y(:,:,1,2) + X(:,:,2,2).*Y(:,:,2,2) + X(:,:,2,3).*Y(:,:,3,2);
    XY(:,:,2,3) = X(:,:,2,1).*Y(:,:,1,3) + X(:,:,2,2).*Y(:,:,2,3) + X(:,:,2,3).*Y(:,:,3,3);
    XY(:,:,3,1) = X(:,:,3,1).*Y(:,:,1,1) + X(:,:,3,2).*Y(:,:,2,1) + X(:,:,3,3).*Y(:,:,3,1);
    XY(:,:,3,2) = X(:,:,3,1).*Y(:,:,1,2) + X(:,:,3,2).*Y(:,:,2,2) + X(:,:,3,3).*Y(:,:,3,2);
    XY(:,:,3,3) = X(:,:,3,1).*Y(:,:,1,3) + X(:,:,3,2).*Y(:,:,2,3) + X(:,:,3,3).*Y(:,:,3,3);
elseif M == 4
    XY = zeros( I, J, M, M );
    XY(:,:,1,1) = X(:,:,1,1).*Y(:,:,1,1) + X(:,:,1,2).*Y(:,:,2,1) + X(:,:,1,3).*Y(:,:,3,1) + X(:,:,1,4).*Y(:,:,4,1);
    XY(:,:,1,2) = X(:,:,1,1).*Y(:,:,1,2) + X(:,:,1,2).*Y(:,:,2,2) + X(:,:,1,3).*Y(:,:,3,2) + X(:,:,1,4).*Y(:,:,4,2);
    XY(:,:,1,3) = X(:,:,1,1).*Y(:,:,1,3) + X(:,:,1,2).*Y(:,:,2,3) + X(:,:,1,3).*Y(:,:,3,3) + X(:,:,1,4).*Y(:,:,4,3);
    XY(:,:,1,4) = X(:,:,1,1).*Y(:,:,1,4) + X(:,:,1,2).*Y(:,:,2,4) + X(:,:,1,3).*Y(:,:,3,4) + X(:,:,1,4).*Y(:,:,4,4);
    XY(:,:,2,1) = X(:,:,2,1).*Y(:,:,1,1) + X(:,:,2,2).*Y(:,:,2,1) + X(:,:,2,3).*Y(:,:,3,1) + X(:,:,2,4).*Y(:,:,4,1);
    XY(:,:,2,2) = X(:,:,2,1).*Y(:,:,1,2) + X(:,:,2,2).*Y(:,:,2,2) + X(:,:,2,3).*Y(:,:,3,2) + X(:,:,2,4).*Y(:,:,4,2);
    XY(:,:,2,3) = X(:,:,2,1).*Y(:,:,1,3) + X(:,:,2,2).*Y(:,:,2,3) + X(:,:,2,3).*Y(:,:,3,3) + X(:,:,2,4).*Y(:,:,4,3);
    XY(:,:,2,4) = X(:,:,2,1).*Y(:,:,1,4) + X(:,:,2,2).*Y(:,:,2,4) + X(:,:,2,3).*Y(:,:,3,4) + X(:,:,2,4).*Y(:,:,4,4);
    XY(:,:,3,1) = X(:,:,3,1).*Y(:,:,1,1) + X(:,:,3,2).*Y(:,:,2,1) + X(:,:,3,3).*Y(:,:,3,1) + X(:,:,3,4).*Y(:,:,4,1);
    XY(:,:,3,2) = X(:,:,3,1).*Y(:,:,1,2) + X(:,:,3,2).*Y(:,:,2,2) + X(:,:,3,3).*Y(:,:,3,2) + X(:,:,3,4).*Y(:,:,4,2);
    XY(:,:,3,3) = X(:,:,3,1).*Y(:,:,1,3) + X(:,:,3,2).*Y(:,:,2,3) + X(:,:,3,3).*Y(:,:,3,3) + X(:,:,3,4).*Y(:,:,4,3);
    XY(:,:,3,4) = X(:,:,3,1).*Y(:,:,1,4) + X(:,:,3,2).*Y(:,:,2,4) + X(:,:,3,3).*Y(:,:,3,4) + X(:,:,3,4).*Y(:,:,4,4);
    XY(:,:,4,1) = X(:,:,4,1).*Y(:,:,1,1) + X(:,:,4,2).*Y(:,:,2,1) + X(:,:,4,3).*Y(:,:,3,1) + X(:,:,4,4).*Y(:,:,4,1);
    XY(:,:,4,2) = X(:,:,4,1).*Y(:,:,1,2) + X(:,:,4,2).*Y(:,:,2,2) + X(:,:,4,3).*Y(:,:,3,2) + X(:,:,4,4).*Y(:,:,4,2);
    XY(:,:,4,3) = X(:,:,4,1).*Y(:,:,1,3) + X(:,:,4,2).*Y(:,:,2,3) + X(:,:,4,3).*Y(:,:,3,3) + X(:,:,4,4).*Y(:,:,4,3);
    XY(:,:,4,4) = X(:,:,4,1).*Y(:,:,1,4) + X(:,:,4,2).*Y(:,:,2,4) + X(:,:,4,3).*Y(:,:,3,4) + X(:,:,4,4).*Y(:,:,4,4);
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    Y = reshape(permute(Y, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    XY = zeros( M, M, I*J );
    parfor ij = 1:I*J
        XY(:,:,ij) = X(:,:,ij)*Y(:,:,ij);
    end
    XY = permute(reshape(XY, [M,M,I,J]), [3,4,1,2]); % I x J x M x M
end
end

%%% Inverse %%%
function [ invX ] = local_inverse( X, I, J, M )
if M == 2
    invX = zeros(I,J,M,M);
    detX = X(:,:,1,1).*X(:,:,2,2) - X(:,:,1,2).*X(:,:,2,1);
    invX(:,:,1,1) = X(:,:,2,2);
    invX(:,:,1,2) = -1*X(:,:,1,2);
    invX(:,:,2,1) = conj(invX(:,:,1,2));
    invX(:,:,2,2) = X(:,:,1,1);
    invX = bsxfun(@rdivide, invX, detX); % This can be rewritten as "invX = invX./detX;" using implicit expansion for later R2016b
elseif M == 3
    invX = zeros(I,J,M,M);
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3) + X(:,:,2,1).*X(:,:,3,2).*X(:,:,1,3) + X(:,:,3,1).*X(:,:,1,2).*X(:,:,2,3) - X(:,:,1,1).*X(:,:,3,2).*X(:,:,2,3) - X(:,:,3,1).*X(:,:,2,2).*X(:,:,1,3) - X(:,:,2,1).*X(:,:,1,2).*X(:,:,3,3);
    invX(:,:,1,1) = X(:,:,2,2).*X(:,:,3,3) - X(:,:,2,3).*X(:,:,3,2);
    invX(:,:,1,2) = X(:,:,1,3).*X(:,:,3,2) - X(:,:,1,2).*X(:,:,3,3);
    invX(:,:,1,3) = X(:,:,1,2).*X(:,:,2,3) - X(:,:,1,3).*X(:,:,2,2);
    invX(:,:,2,1) = X(:,:,2,3).*X(:,:,3,1) - X(:,:,2,1).*X(:,:,3,3);
    invX(:,:,2,2) = X(:,:,1,1).*X(:,:,3,3) - X(:,:,1,3).*X(:,:,3,1);
    invX(:,:,2,3) = X(:,:,1,3).*X(:,:,2,1) - X(:,:,1,1).*X(:,:,2,3);
    invX(:,:,3,1) = X(:,:,2,1).*X(:,:,3,2) - X(:,:,2,2).*X(:,:,3,1);
    invX(:,:,3,2) = X(:,:,1,2).*X(:,:,3,1) - X(:,:,1,1).*X(:,:,3,2);
    invX(:,:,3,3) = X(:,:,1,1).*X(:,:,2,2) - X(:,:,1,2).*X(:,:,2,1);
    invX = bsxfun(@rdivide, invX, detX); % This can be rewritten as "invX = invX./detX;" using implicit expansion for later R2016b
elseif M == 4
    invX = zeros(I,J,M,M);
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,3) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,2);
    invX(:,:,1,1) = X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,2);
    invX(:,:,1,2) = X(:,:,1,2).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,1,3).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,1,4).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,1,4).*X(:,:,3,2).*X(:,:,4,3);
    invX(:,:,1,3) = X(:,:,1,2).*X(:,:,2,3).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,4,3) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,4,4) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,4,2);
    invX(:,:,1,4) = X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,3) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,4) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,2) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,2) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,3);
    invX(:,:,2,1) = X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,3);
    invX(:,:,2,2) = X(:,:,1,1).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,1,4).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,1).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,1,4).*X(:,:,3,3).*X(:,:,4,1);
    invX(:,:,2,3) = X(:,:,1,1).*X(:,:,2,4).*X(:,:,4,3) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,4,4) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,4,3);
    invX(:,:,2,4) = X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,4) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,1) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,3) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,3) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,4) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,1);
    invX(:,:,3,1) = X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,2) - X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,1);
    invX(:,:,3,2) = X(:,:,1,1).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,1,2).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,1,4).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,1,2).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,3,1).*X(:,:,4,2);
    invX(:,:,3,3) = X(:,:,1,1).*X(:,:,2,2).*X(:,:,4,4) + X(:,:,1,2).*X(:,:,2,4).*X(:,:,4,1) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,4,2) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,4,4) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,4,1);
    invX(:,:,3,4) = X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,2) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,4) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,4) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,2);
    invX(:,:,4,1) = X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,2) + X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,3) + X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,2);
    invX(:,:,4,2) = X(:,:,1,1).*X(:,:,3,2).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,3,3).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,3,1).*X(:,:,4,2) - X(:,:,1,1).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,3,2).*X(:,:,4,1);
    invX(:,:,4,3) = X(:,:,1,1).*X(:,:,2,3).*X(:,:,4,2) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,4,3) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,4,3) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,4,1) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,4,2);
    invX(:,:,4,4) = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3) + X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,1) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,2) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,3) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,1);
    invX = bsxfun(@rdivide, invX, detX); % This can be rewritten as "invX = invX./detX;" using implicit expansion for later R2016b
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    eyeM = eye(M);
    invX = zeros(M,M,I*J);
    parfor ij = 1:I*J
            invX(:,:,ij) = X(:,:,ij)\eyeM;
    end
    invX = permute(reshape(invX, [M,M,I,J]), [3,4,1,2]); % I x J x M x M
end
end

%%% Trace %%%
function [ trX ] = local_trace( X, I, J, M )
if M == 2
    trX = X(:,:,1,1) + X(:,:,2,2);
elseif M == 3
    trX = X(:,:,1,1) + X(:,:,2,2) + X(:,:,3,3);
elseif M == 4
    trX = X(:,:,1,1) + X(:,:,2,2) + X(:,:,3,3) + X(:,:,4,4);
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    trX = zeros(I*J,1);
    parfor ij = 1:I*J
            trX(ij) = trace(X(:,:,ij));
    end
    trX = reshape(trX, [I,J]); % I x J
end
end

%%% Determinant %%%
function [ detX ] = local_det( X, I, J, M )
if M == 2
    detX = X(:,:,1,1).*X(:,:,2,2) - X(:,:,1,2).*X(:,:,2,1);
elseif M == 3
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3) + X(:,:,2,1).*X(:,:,3,2).*X(:,:,1,3) + X(:,:,3,1).*X(:,:,1,2).*X(:,:,2,3) - X(:,:,1,1).*X(:,:,3,2).*X(:,:,2,3) - X(:,:,3,1).*X(:,:,2,2).*X(:,:,1,3) - X(:,:,2,1).*X(:,:,1,2).*X(:,:,3,3);
elseif M == 4
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,3) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,2);
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    detX = zeros(I*J,1);
    parfor ij = 1:I*J
            detX(ij) = det(X(:,:,ij));
    end
    detX = reshape(detX, [I,J]); % I x J
end
end

function [SDR,SIR,SAR] = local_eval( Xhat,I,J,M,N,K,Z,T,V,H,x,shiftSize,window,len,src,refMic )
Y = zeros(I,J,M,N);
XXhat = permute(Xhat,[3,4,1,2]);
for i = 1:I
    for j = 1:J
        for sc = 1:N
            ys = 0;
            for k=1:K
                ys = ys + Z(k,sc)*T(i,k)*V(k,j);
            end
            Y(i,j,:,sc) = ys * squeeze(H(i,sc,:,:))/XXhat(:,:,i,j)*x(:,i,j);
        end
    end
end
for sc = 1:N
    sep(:,:,sc) = ISTFT(Y(:,:,:,sc),shiftSize,window,len);
end
[SDR,SIR,SAR] = bss_eval_sources(squeeze(sep(:,refMic,:)).',src.');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
