% This is the main code for "Intra-cavity lossless phase retrieval for nondegenerate on-demand laser via an extra-cavity-like scheme".
% The design example here is a 'X' shaped beam.
% A Nvidia graphic card with more than 6GB graphic RAM is strongly prefered. ELse, please comment out all codes about "gpuArray".
% Any questions please contact Hengyang Li via d202180830@hust.edu.cn
% Coded and tested on Matlab R2024b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clear;
close all;
gpuDevice(1);
reset(gpuDevice);
%% Basic physical quantity definition
lambda = 1064e-6;
k = 2 * pi / lambda;
dx = 12.5e-3;
pm = 500;
pn = 500;
lx = pm * dx;
ly = pn * dx;
z1 = 570;
z2 = 510;

x = linspace(-lx / 2 + dx / 2,lx / 2 - dx / 2, pm);
[x,y] = meshgrid(x,x);
[theta,r] = cart2pol(x,y);
theta = theta + pi;
ap = ones(pm);
ap(r > 3) = 0;
%%  Read target beam intensity
B1k = double(rgb2gray(imread('Xuse2.jpg')))/255*(-1)+1;
B1k = imresize(B1k,[120,120],"nearest");
B1 = zeros(pm,pn);
B1(191:310,191:310) = B1k;
B1 = smoothdata(B1,1,"gaussian",8);
Q_RA = smoothdata(B1,2,"gaussian",8);
Q_RA(Q_RA > 0) = 1;
Q_RA = smoothdata(Q_RA,1,"gaussian",30);
Q_RA = smoothdata(Q_RA,2,"gaussian",30);

IntenLoss = 0.999;   
%% Initial guess of the holograms
PHI1with = zeros(pm);
PHI1 = zeros(pm);
%% Preparing MTP propagation environment & transfer matrices to gRAM
pa1 = Fx_MTP_env(pm, lambda, z1, lx, lx, 1);
pa2 = Fx_MTP_env(pm, lambda, z2, lx, lx, 1);
veloc1 = zeros(pm,pn);
veloc1 = gpuArray(single(veloc1));
veloc2 = zeros(pm,pn);
veloc2 = gpuArray(single(veloc2));
Q_RA = gpuArray(single(Q_RA));
PHI1 = gpuArray(single(PHI1));
PHI1with = gpuArray(single(PHI1with));
%% Inverse Fox-Li design (optimization section)
LR1 = 20e-2;
LR2 = 20e-2;
cutoffRatio = 0;    %suppression weight
momentum = 0.90;    % momentum in SGDM
TVratio = 0.001;    % weight of TV regularization
Loss_RA = zeros(10000000,1);
tic
for ii = 1:2000
    %%%%%%%%%%%%%%%%%%%%%%%%%% desired mode LG2,2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U_depart = Q_RA .* exp(1i * PHI1with);
    U_SLM1_pre = Fx_Fresnel_MTP(U_depart, pa1);
    U_SLM1_post = U_SLM1_pre .* exp(1i * PHI1);
    U_OC_dejavu = Fx_Fresnel_MTP(U_SLM1_post, pa2) .* ap;
    DIFF_RA = abs(U_OC_dejavu - IntenLoss * U_depart).^2;
    Loss_RA(ii) = sum(DIFF_RA,"all");

    U_OC_dejavu_ = 2 * (U_OC_dejavu - IntenLoss * U_depart);
    U_depart_1_ = IntenLoss * -2 * (U_OC_dejavu - IntenLoss * U_depart);
    U_SLM1_post_ = Fx_Fresnel_MTP_bp(U_OC_dejavu_ .* ap, pa2);
    U_SLM1_pre_ = U_SLM1_post_ .* exp(-1i * PHI1);
    PHI1_RA_ = imag(U_SLM1_post_ .* conj(U_SLM1_post));
    U_depart_2_ = Fx_Fresnel_MTP_bp(U_SLM1_pre_, pa1);
    U_depart_ = U_depart_1_ + U_depart_2_;
    PHI1with_ = imag(U_depart_ .* conj(U_depart));
    %%%%%%%%%%%%%%%%%%%%%%%%% TV regularization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    TV1x = PHI1 - circshift(PHI1,1,1);
    TV1y = PHI1 - circshift(PHI1,1,2);
    TV1xs = abs(TV1x).^2;
    TV1ys = abs(TV1y).^2;
    TVnorm(ii) = TVratio * sum((TV1xs + TV1ys),"all");

    TV1x_ = 2 * TVratio * TV1x;
    TV1y_ = 2 * TVratio * TV1y;
    TVPHI1_ = (TV1x_ + TV1y_ - circshift(TV1x_,-1,1) - circshift(TV1y_,-1,2));
    %%%%%%%%%%%%%%%%%%%%%%%%%% SGDM Optimizer  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fang1 = - PHI1_RA_ - TVPHI1_ +  momentum * veloc1;
    fang2 = - PHI1with_ +  momentum * veloc2;
    PHI1 = PHI1 + LR1 * fang1;
    PHI1with = PHI1with + LR2 * fang2;
    if ii > 10
        veloc1 = fang1;
        veloc2 = fang2;
    end
    if ii == 500
        LR1 = 80e-2;
        LR2 = 80e-2;
        momentum = 0.98;
    end
    % ii
    Loss_RA = gather(Loss_RA);
    ii
end
toc

PHI1 = gather(PHI1);
PHI1(1,1) = pi;
PHI1(1,2) = -pi;
figure
imagesc(angle(exp(1i * (PHI1))))
colormap(othercolor('BuOr_12'))
title('Hologram1')
%% Cascaded holograms
PHI1 = gather(PHI1);
PHI1(1,1) = pi;
PHI1(1,2) = -pi;
figure
imagesc(angle(exp(1i * PHI1)))
colormap(othercolor('BuOr_12'))
title('Hologram1')

PHI1with = gather(PHI1with);
PHI1with(1,1) = pi;
PHI1with(1,2) = -pi;
figure
imagesc(angle(exp(1i * PHI1with)))
colormap(othercolor('BuOr_12'))
title('Hologram1')



%% Single round trip tomography
sliceZ = 10;
slicenum1 = z1 / sliceZ;
slicenum2 = z2 / sliceZ;
for ii = 1:slicenum1
    UUU1(:,:,ii) = Fx_fresnelDFFT(Q_RA .* exp(1i * PHI1with),pm,(ii-1) * sliceZ,lambda,ly);
end
UUUUUU1 = Fx_fresnelDFFT(Q_RA .* exp(1i * PHI1with),pm,z1,lambda,ly);
UUUUUU1 = UUUUUU1 .* exp(1i * PHI1);
for ii = 1:slicenum2
    UUU2(:,:,ii) = Fx_fresnelDFFT(UUUUUU1,pm,(ii-1) * sliceZ,lambda,ly);
end
UUU = cat(3,UUU1,UUU2);
UUU = gather(UUU);
figure;
sliceViewer(abs(UUU).^2)
colormap("turbo")
title('Single round trip tomography____Intensity')
figure;
sliceViewer(angle(UUU))
colormap(othercolor('BuOr_12'))
title('Single round trip tomography____Phase')
%% tomography
OCfield_RA = zeros(pm,pn,200);
for ii = 1:200
    if ii == 1
        U_SLM1_pre = Fx_Fresnel_MTP(Q_RA .* exp(1i * PHI1with), pa1);
    else
        U_SLM1_pre = Fx_Fresnel_MTP(U_OC_dejavu, pa1);
    end
    U_SLM1_post = U_SLM1_pre .* exp(1i * PHI1);
    U_OC_dejavu = Fx_Fresnel_MTP(U_SLM1_post, pa2) .* ap; 
    OCfield_RA(:,:,ii) = gather(U_OC_dejavu);
    ii
end
for ii = 1:200
    OCfield_RA(:,:,ii) = OCfield_RA(:,:,ii) ./ abs(max(max(OCfield_RA(:,:,ii))));
end
figure;
sliceViewer(abs(OCfield_RA(:,:,1:end)).^2);
colormap("turbo")
title('Chiral mode suprression tomography____Intensity')
figure;
sliceViewer(angle(OCfield_RA(:,:,1:end)));
colormap(othercolor('BuOr_12'))
title('Chiral mode suprression tomography____Phase')



%% Fox-Li tomography
Q_CO = Fx_gaussianbeam(pm,pn,30,dx);
for ii = 1:200
    if ii == 1
        U_SLM1_pre = Fx_Fresnel_MTP(Q_CO, pa1);
    else
        U_SLM1_pre = Fx_Fresnel_MTP(U_OC_dejavu, pa1) .* ap;
    end
    U_SLM1_post = U_SLM1_pre .* exp(1i * PHI1);
    U_SLM2_pre = Fx_Fresnel_MTP(U_SLM1_post, pa2) .* ap;
    U_OC_dejavu = U_SLM2_pre;
    OCfield_CO(:,:,ii) = U_OC_dejavu;
    ii
end
OCfield_CO = gather(OCfield_CO);
for ii = 1:200
    OCfield_CO(:,:,ii) = OCfield_CO(:,:,ii) ./ abs(max(max(OCfield_CO(:,:,ii))));
end
figure;
sliceViewer(abs(OCfield_CO).^2);
colormap("turbo")
title('LG55 suprression tomography____Intensity')
figure;
sliceViewer(angle(OCfield_CO));
colormap(othercolor('BuOr_12'))
title('LG55 suprression tomography____Phase')
