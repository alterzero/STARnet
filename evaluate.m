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
result_dir = 'Result_T_SR_HR_MSE/vimeo_triplet/sequences/';

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
    niqe_drbpsr(j) = computequality(im_sr,blocksizerow,blocksizecol,...
        blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
end

fprintf('%s, %f , %f , %f, %f\n', im_file, mean(psnr_drbpsr), mean(ssim_drbpsr), mean(niqe_drbpsr), mean(ie_drbpsr));

