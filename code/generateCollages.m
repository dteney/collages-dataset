function generateCollages(inputPath, outputPath, enableDisplay)
%GENERATEDATACOLLAGESIMAGES Stand-alone function to generate dataset of collages.

%   Author: Damien Teney

if ~nargin % No given parameters: set default values
  inputPath = 'C:\Data\data-collages'; % Where to find MNIST, CIFAR-10, etc.
  outputPath = '.'; % Where to save the collages
  enableDisplay = true;
else
  assert(nargin == 3);
end

%------------------------------------------------------------------------------------------------------
% Dataset options: uncomment the desired options
p.randomBlockOrder = false; % Random order in the assembly of the images from the source datasets
%p.randomBlockOrder = true; % Random order in the assembly of the images from the source datasets

%p.nBlocks = 2; % MNIST + CIFAR-10
p.nBlocks = 4; % MNIST + CIFAR-10 + Fashing-MNIST + SVHN

% Pairs of classes to keep from each source dataset
p.labelsToKeep{1} = [0 1]; % MNIST: digits 0/1
p.labelsToKeep{2} = [1 9]; % CIFAR: 1=automobile / 9=truck; 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck
p.labelsToKeep{3} = [2 4]; % Fashion-MNIST: see https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/fashion-mnist-sprite.png
p.labelsToKeep{4} = [10 1]; % SVHN: digits 0/1

%p.downsamplingLevel = 0; % Full-size images
%p.downsamplingLevel = 1; % 1/2-Size images
p.downsamplingLevel = 2; % 1/4-Size images
%p.downsamplingLevel = 3; % 1/8-Size images
%p.downsamplingLevel = 4; % 1/16-Size images

switch p.nBlocks
  case 2, nRows = 2; nCols = 1; blockNames = {'mnist', 'cifar'};
  case 4, nRows = 2; nCols = 2; blockNames = {'mnist', 'cifar', 'fashion', 'svhn'};
  otherwise, assert(false);
end
p.setNames = cat(2, ...
  'train-all', ...
  strcat('train-', blockNames, '-forUpperBoundsOnly'), ...
  'val-all', ...
  strcat('test-', blockNames) );
p.nInstancesPerSet = [repelem(1024*50, 1+p.nBlocks), repelem(1024*10, 1+p.nBlocks)]; % Size of sets (training/test)
assert(numel(p.setNames) == numel(p.nInstancesPerSet));

switch abs(p.downsamplingLevel)
  case 0, p.dimImage = [nRows*32 nCols*32 1]; % Original images, converted to black and white
  case 1, p.dimImage = [nRows*16 nCols*16 1]; % Downsampled black-and-white images
  case 2, p.dimImage = [nRows*8 nCols*8 1]; % Downsampled black-and-white images
  case 3, p.dimImage = [nRows*4 nCols*4 1]; % Downsampled black-and-white images
  case 4, p.dimImage = [nRows*2 nCols*2 1]; % Downsampled black-and-white images
  otherwise, assert(false);
end

p.datasetName = sprintf('collages-%gblocks-randomOrder%g-downsampling%g', p.nBlocks, p.randomBlockOrder, p.downsamplingLevel);

%------------------------------------------------------------------------------------------------------
% Load the original data
imgsLoaded = cell(p.nBlocks, 2);
labelsLoaded = cell(p.nBlocks, 2);

fprintf('Loading MNIST...\n');
fileNames = fullfile(inputPath, {'mnist-train-images.idx3-ubyte', 'mnist-train-labels.idx1-ubyte'; 'mnist-t10k-images.idx3-ubyte', 'mnist-t10k-labels.idx1-ubyte'});
nTr = 50000; nTe = 10000;
[imgsLoaded{1, 1}, labelsLoaded{1, 1}] = readMnist(fileNames{1, 1}, fileNames{1, 2}, nTr, 0, false); % Load training data
[imgsLoaded{1, 2}, labelsLoaded{1, 2}] = readMnist(fileNames{2, 1}, fileNames{2, 2}, nTe, 0, false); % Load test data
for trTe = 1:2 % Pad images to 32x32x3 (CIFAR dimensions)
  imgsLoaded{1, trTe} = repmat(permute(imgsLoaded{1, trTe}, [1 2 4 3]), 1, 1, 3, 1);
  imgsLoaded{1, trTe} = padarray(imgsLoaded{1, trTe}, [2 2], 0, 'both');
end

fprintf('Loading CIFAR-10...\n');
[imgsLoaded{2, 1}, labelsLoaded{2, 1}, imgsLoaded{2, 2}, labelsLoaded{2, 2}] = readCifar10(inputPath);
for trTe = 1:2
  labelsLoaded{2, trTe} = double(labelsLoaded{2, trTe}); % Turn CIFAR category names to class indices
end

if p.nBlocks == 4
  fprintf('Loading Fashion-MNIST...\n');
  fileNames = fullfile(inputPath, {'fashion-mnist-train-images.idx3-ubyte', 'fashion-mnist-train-labels.idx1-ubyte'; 'fashion-mnist-t10k-images.idx3-ubyte', 'fashion-mnist-t10k-labels.idx1-ubyte'});
  nTr = 60000; nTe = 10000;
  [imgsLoaded{3, 1}, labelsLoaded{3, 1}] = readMnist(fileNames{1, 1}, fileNames{1, 2}, nTr, 0, false); % Load training data
  [imgsLoaded{3, 2}, labelsLoaded{3, 2}] = readMnist(fileNames{2, 1}, fileNames{2, 2}, nTe, 0, false); % Load test data
  for trTe = 1:2 % Pad images to 32x32x3 (CIFAR dimensions)
    imgsLoaded{3, trTe} = repmat(permute(imgsLoaded{3, trTe}, [1 2 4 3]), 1, 1, 3, 1);
    imgsLoaded{3, trTe} = padarray(imgsLoaded{3, trTe}, [2 2], 0, 'both');
  end

  fprintf('Loading SVHN...\n');
  fileNames = fullfile(inputPath, {'svhn_train_32x32.mat', 'svhn_test_32x32.mat'});
  tmp = load(fileNames{1}); imgsLoaded{4, 1} = tmp.X; labelsLoaded{4, 1} = tmp.y; % Load training data
  tmp = load(fileNames{2}); imgsLoaded{4, 2} = tmp.X; labelsLoaded{4, 2} = tmp.y; % Load test data
  clearvars tmp;
end

% Normalize all loaded images and keep only the chosen classes
for c = 1:p.nBlocks
  for trTe = 1:2 % For training/test data
    if max(imgsLoaded{c, trTe}(:)) == 255, imgsLoaded{c, trTe} = double(imgsLoaded{c, trTe}) / 255; end % Convert CIFAR images to values in [0,1]
    if ndims(imgsLoaded{c, trTe}) > 3, imgsLoaded{c, trTe} = squeeze(mean(imgsLoaded{c, trTe}, 3)); end % Remove color channel
    assert(ndims(imgsLoaded{c, trTe}) == 3);
    assert(max(imgsLoaded{c, trTe}(:)) <= 1); % Make sure all images have values in [0,1]
    assert(min(imgsLoaded{c, trTe}(:)) >= 0); % Make sure all images have values in [0,1]
    assert(size(imgsLoaded{c, trTe}, 1) == 32);
    assert(size(imgsLoaded{c, trTe}, 2) == 32);

    assert(numel(p.labelsToKeep{c}) == 2);
    toKeep = (labelsLoaded{c, trTe} == p.labelsToKeep{c}(1)) ... % Only keep the two chosen classes
           | (labelsLoaded{c, trTe} == p.labelsToKeep{c}(2)) ;
    assert(any(labelsLoaded{c, trTe} == p.labelsToKeep{c}(1))); % Make sure there are instances with the chosen labels
    assert(any(labelsLoaded{c, trTe} == p.labelsToKeep{c}(2))); % Make sure there are instances with the chosen labels
    imgsLoaded{c, trTe} = imgsLoaded{c, trTe}(:, :, toKeep);
    labelsLoaded{c, trTe} = labelsLoaded{c, trTe}(toKeep);
    labelsLoaded{c, trTe} = (labelsLoaded{c, trTe} == p.labelsToKeep{c}(end)); % Make labels logical values 0/1
  end
end

%------------------------------------------------------------------------------------------------------
% Generate the collages
rng(0, 'threefry'); % Reset the random number generator for reproducibility

for s = 1 : numel(p.setNames) % For each set
  if startsWith(p.setNames{s}, 'train-') % Training set
    trTe = 1; % Use the original training data
  else % Validation/test sets
    trTe = 2; % Use the original test data
  end

  % Decide which images to combine, depending on the set
  idsClass0 = cell(1, p.nBlocks);
  idsClass1 = cell(1, p.nBlocks);
  for c = 1 : p.nBlocks
    idsClass0{c} = find(labelsLoaded{c, trTe} == 0); % By default: use predicive images
    idsClass1{c} = find(labelsLoaded{c, trTe} == 1);
  end
  if isequal(p.setNames{s}, 'train-all') || isequal(p.setNames{s}, 'val-all') % Training / validation (only useful to monitor overfitting) data: all blocks are predictive
    % Nothing to do
  elseif startsWith(p.setNames{s}, 'train-') || startsWith(p.setNames{s}, 'test-') % Training data for baselines (force learning only on one part of the images) / test data: only 1 block is predictive
    blockNotToRandomize = find(cellfun(@(a)(contains(p.setNames{s}, a)), blockNames)); % ID of the block to be predictive
    assert((numel(blockNotToRandomize) == 1) && any(blockNotToRandomize == 1 : p.nBlocks));
    for c = setdiff(1:p.nBlocks, blockNotToRandomize) % For every block but one
      idsClass0{c} = 1 : numel(labelsLoaded{c, trTe}); % Can use *any* image
      idsClass1{c} = 1 : numel(labelsLoaded{c, trTe});
    end
  else
    assert(false);
  end

  nPerClass = p.nInstancesPerSet(s)/2;
  labels = cat(1, zeros(nPerClass, 1), ones(nPerClass, 1)); % Use half/half of each class: every set is balanced
  imgsSelected = cell(1, p.nBlocks);
  for c = 1 : p.nBlocks % For each source
    ids = cat(1, getRandomElements(idsClass0{c}, nPerClass), getRandomElements(idsClass1{c}, nPerClass));
    imgsSelected{c} = imgsLoaded{c, trTe}(:, :, ids);
  end

  if p.randomBlockOrder
    assert(numel(imgsSelected) == p.nBlocks);
    assert(ndims(imgsSelected{1}) == 3);
    tmp = imgsSelected;
    for i = 1 : size(imgsSelected{1}, 3)
      k2s = randperm(p.nBlocks); % Generate a different order of the sources for every example
      for k = 1 : p.nBlocks
        imgsSelected{k}(:, :, i) = tmp{k2s(k)}(:, :, i);
      end
    end
  end

  switch p.nBlocks
    case 2 % Assemble images vertically
      imgsCollages = cat(1, imgsSelected{:});
    case 4 % Assemble 4 images as a square
      column1 = cat(1, imgsSelected{1}, imgsSelected{2});
      column2 = cat(1, imgsSelected{3}, imgsSelected{4});
      imgsCollages = cat(2, column1, column2);
    otherwise, assert(false);
  end

  switch p.downsamplingLevel
    case 0 % Nothing to do
    % Resize by nearest neighbour
    case 1, imgsCollages = imgsCollages(1:2:end, 1:2:end, :, :);
    case 2, imgsCollages = imgsCollages(1:4:end, 1:4:end, :, :);
    case 3, imgsCollages = imgsCollages(1:8:end, 1:8:end, :, :);
    case 4, imgsCollages = imgsCollages(1:16:end, 1:16:end, :, :);
     % Resize by bilinear interpolation: not recommended, it blurs the images to the point of making learning very difficult)
     %{
    case 1, newImages = imresize(newImages, 1/2, 'method', 'bilinear');
    case 2, newImages = imresize(newImages, 1/4, 'method', 'bilinear');
    case 3, newImages = imresize(newImages, 1/8, 'method', 'bilinear');
    case 4, newImages = imresize(newImages, 1/16, 'method', 'bilinear');
     %}
    otherwise, assert(false);
  end
  imgsCollages = permute(imgsCollages, [1 2 4 3]); % Insert 3rd (color) channel (not used)
  assert(isequal(p.dimImage, size(imgsCollages, 1:3)));

  % Display: mean image + one random image per class in the current set
  if enableDisplay
    h = figure;
    tmp = imgsCollages(:, :, :, labels == 0); subplot(2, 2, 1); imshow(mean(tmp, 4), [0 1]); subplot(2, 2, 3); imshow(tmp(:, :, :, randi(size(tmp, 4))), [0 1]); xlabel('Class 0');
    tmp = imgsCollages(:, :, :, labels == 1); subplot(2, 2, 2); imshow(mean(tmp, 4), [0 1]); subplot(2, 2, 4); imshow(tmp(:, :, :, randi(size(tmp, 4))), [0 1]); xlabel('Class 1');
    setFigureTitle(h, ['Set ' num2str(s) ' (' p.setNames{s} ')']); pause(0.5)
  end

  % Save images of the current set
  for classId = [0 1]
    outputDir = fullfile(outputPath, p.datasetName, p.setNames{s}, num2str(classId));
    if ~exist(outputDir, 'dir'), mkdir(outputDir); end
    fprintf('Set %g (%s), class %g: %s\n', s, p.setNames{s}, classId, outputDir);

    imgs = imgsCollages(:, :, :, labels == classId);
    for i = 1 : size(imgs, 4)
      fileName = fullfile(outputDir, sprintf('%06d.png', i));
      imwrite(uint8(255*imgs(:, :, i)), fileName);
    end
  end
end % For each set

% Save the dataset options in a JSON file
fileName = fullfile(outputPath, p.datasetName, 'options.json');
fid = fopen(fileName, 'w');
fprintf(fid, jsonencode(p, 'PrettyPrint', true));
fclose(fid); 
