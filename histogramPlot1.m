function histogramPlot(folderName, numBins)
%folderName = '12Modes';
%numBins = 1000;

figName = sprintf('%sBins%d',folderName, numBins);

input1 = load([folderName, '/1gen/input.txt']);
input1 = input1(:,2);
out1 = load([folderName, '/1gen/out.txt']);
out1 = out1(:,2);

input3 = load([folderName, '/3gen/input.txt']);
input3 = input3(:,2);
out3 = load([folderName, '/3gen/out.txt']);
out3 = out3(:,2);


[f1in, x1in] = hist(input1, numBins);
[f1out, x1out] = hist(out1, numBins);

[f3in, x3in] = hist(input3, numBins);
[f3out, x3out] = hist(out3, numBins);


figure(1),
bar(x1in,f1in/trapz(x1in,f1in), 'r'), hold on,
bar(x1out,f1out/trapz(x1out,f1out)),
legend('Train Data', 'GAN-BN');
saveas(gcf, [figName, '-GAN.eps']);

figure(2),
bar(x3in,f3in/trapz(x3in,f3in), 'r'), hold on,
bar(x3out,f3out/trapz(x3out,f3out))
legend('Train Data', 'MAD-GAN');
saveas(gcf, [figName, '-MAD-GAN.eps'])

close all; clear all;
end