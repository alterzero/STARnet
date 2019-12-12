clc;
clear;

addpath(genpath(fullfile(pwd,'niqe_release')));
%% Loading model
load niqe_release/modelparameters.mat
blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

%% set parameters
file_list = 'tri_testlist.txt';
target_dir = 'vimeo_triplet/sequences/';
%file_list = 'sep_testlist_evaluate.txt';
%target_dir = 'vimeo_septuplet/sequences/';
%file_list = 'test.txt';
%target_dir = 'ucf101_interp_ours/';
%file_list = 'test.txt';
%target_dir = 'Middlebury/other-data/';


%FOR SPACE-TIME SR
result_dir = 'Results/vimeo_triplet/sequences/';
%result_dir = 'Results_FBPNSR_RBPN_V1_NEW_V4_REF_REAL/other-data/'
%result_dir = 'Results/vimeo_interp_test_spacetime/input/';
%result_dir = 'Results/vimeo_dbpntoflow/';
%result_dir = 'Results/vimeo_interp_test_flowdbpn/input/';
%result_dir = 'Results/DAIN_Baseline/sequences/';
%result_dir = 'Results/vimeo_interp_test_dbpn/input/';

%FOR FRAME INTERPOLATION
%result_dir = 'Results_interpolation_FBPNSR_RBPN_V1_NEW_V4_REF_MSE/vimeo_triplet/sequences/';
%result_dir = 'Results_interpolation_FBPNSR_RBPN_V1_REF_VGG_multi4x/vimeo_septuplet_4xr/sequences/';
%result_dir = 'Results_interpolation_FBPNSR_RBPN_V1_REF_JOINT/vimeo_interp_test_spacetime/input/';
%result_dir = 'Results_interpolation/vimeo_toflow/';
%result_dir = 'Results_interpolation_FBPNSR_RBPN_V1_NEW_V4_REF_MSE/ucf101_interp_ours/';
%result_dir = 'Results_interpolation/Middlebury_other_DAIN/';

%FRAME INTERPOLATION LR
%result_dir = 'Results_LR_2x/vimeo_interp_test_spacetime/input/';
%result_dir = 'Results_LR/toflow/';
%result_dir = 'Results_LR/DAIN/input/';

im_file = '/im2.png';
%im_file = '/frame_01_gt.png';

i=1;
fid = fopen([target_dir file_list]);
while ~feof(fid)
    tline = fgetl(fid);
    filelist{i} = tline;
    i=i+1;
end
fclose(fid);

psnr_drbpsr = zeros(length(filelist),1);
ssim_drbpsr = zeros(length(filelist),1);
fsim_drbpsr = zeros(length(filelist),1);
niqe_drbpsr = zeros(length(filelist),1);
ie_drbpsr = zeros(length(filelist),1);

parfor j=1:length(filelist)-1
    im_gnd = imread([target_dir filelist{j} im_file]);
    %im_gnd = imresize(imread([target_dir filelist{j} '/im2.png']), 1/4, 'bicubic');
    im_sr = imread([result_dir filelist{j} im_file]);
    
    %% compute PSNR
    psnr_drbpsr(j) = psnr(im_gnd,im_sr);
    if isinf(psnr_drbpsr(j))
        fprintf('%s;', [target_dir filelist{j} im_file])
    end
     
     %% compute SSIM
    ssim_drbpsr(j) = ssim(im_gnd,im_sr);

    ie_drbpsr(j) = inter_error(im_gnd,im_sr); 
    %% compute NIQE
    %niqe_drbpsr(j) = 0
    niqe_drbpsr(j) = computequality(im_sr,blocksizerow,blocksizecol,...
        blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
end

fprintf('%s, %f , %f , %f, %f\n', im_file, mean(psnr_drbpsr), mean(ssim_drbpsr), mean(niqe_drbpsr), mean(ie_drbpsr));
%fprintf('%s, %f , %f\n', im_file, mean(psnr_drbpsr), mean(ssim_drbpsr));

