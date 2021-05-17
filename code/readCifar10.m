function [imgsTr, labelsTr, imgsTe, labelsTe] = readCifar10(directoryPath)
%READCIFAR10 Download and read CIFAR-10 data.

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
unpackedData = fullfile(directoryPath, 'cifar-10-batches-mat');
if ~exist(unpackedData, 'dir')
  fprintf('Downloading CIFAR-10 dataset...');     
  untar(url, directoryPath); 
  fprintf('done.\n\n');
end

location = fullfile(directoryPath, 'cifar-10-batches-mat');

[imgsTr1, labelsTr1] = loadBatchAsFourDimensionalArray(location, 'data_batch_1.mat');
[imgsTr2, labelsTr2] = loadBatchAsFourDimensionalArray(location, 'data_batch_2.mat');
[imgsTr3, labelsTr3] = loadBatchAsFourDimensionalArray(location, 'data_batch_3.mat');
[imgsTr4, labelsTr4] = loadBatchAsFourDimensionalArray(location, 'data_batch_4.mat');
[imgsTr5, labelsTr5] = loadBatchAsFourDimensionalArray(location, 'data_batch_5.mat');

imgsTr = cat(4, imgsTr1, imgsTr2, imgsTr3, imgsTr4, imgsTr5);
labelsTr = [labelsTr1; labelsTr2; labelsTr3; labelsTr4; labelsTr5];

[imgsTe, labelsTe] = loadBatchAsFourDimensionalArray(location, 'test_batch.mat');

function [imgs, labels] = loadBatchAsFourDimensionalArray(location, batchFileName)
  tmp = load(fullfile(location,batchFileName));
  imgs = tmp.data';
  imgs = reshape(imgs, 32,32,3,[]);
  imgs = permute(imgs, [2 1 3 4]);
  labels = tmp.labels; % Class IDs: integers in 0:9
end

end
