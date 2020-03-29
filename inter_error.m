function ie=inter_error(im1,im2)
diff_rgb = 128.0 + im2 - im1;
ie = mean(mean(mean(abs(diff_rgb - 128.0))));

