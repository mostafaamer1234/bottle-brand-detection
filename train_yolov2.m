%% Train YOLOv2 Bottle Detector (Branded vs Generic) – Datastore Pipeline
% Run from project root, e.g.:
% /Users/mostafa/Documents/MATLAB/finalProject/
%
% Expected structure:
% finalProject/
%   train_yolov2.m      <-- this file
%   labeled_dataset/
%       images/
%       bottleGroundTruth.mat   (exported from Image Labeler)
%   test_images/        (for later testing – not used here)
%   models/             (script will create if missing)

clc; clear; close all;

%% 1. Load groundTruth from Image Labeler

gtFile = fullfile(pwd, "labeled_dataset", "bottleGroundTruth.mat");

if ~isfile(gtFile)
    error("Could not find bottleGroundTruth.mat at: %s", gtFile);
end

load(gtFile, "gTruth");   % loads groundTruth object

disp("Loaded groundTruth:");
disp(gTruth)

numImages = height(gTruth.LabelData);
fprintf("Total labeled images: %d\n", numImages);

%% 2. Define YOLOv2 input size and classes

inputSize = [416 416 3];   % YOLOv2 input size

% Get class names from label definitions
classNames = string(gTruth.LabelDefinitions.Name);  % e.g. "branded_bottle", "generic_bottle"
numClasses = numel(classNames);

fprintf("Number of classes: %d\n", numClasses);
disp("Class names:");
disp(classNames);

%% 3. Create datastores from groundTruth

% This variant returns an imageDatastore and boxLabelDatastore
[imds, blds] = objectDetectorTrainingData(gTruth);

% Combine into one datastore: each read gives {image, bboxes, labels}
trainingData = combine(imds, blds);

%% 4. Preprocess data for anchor estimation (resize to inputSize)

% Preprocess copy of trainingData to match network input size
trainingDataForEstimation = transform(trainingData, ...
    @(data) preprocessData(data, inputSize));

%% 5. Estimate anchor boxes on preprocessed data

numAnchors = 5;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

fprintf("Estimated anchor boxes (already in input scale %dx%d):\n", ...
    inputSize(1), inputSize(2));
disp(anchorBoxes);
fprintf("Mean IoU of anchors: %.3f\n", meanIoU);

%% 6. Load base network (ResNet-50) and create YOLOv2 detector

% New API: imagePretrainedNetwork (recommended in 2025a)
if exist("imagePretrainedNetwork", "file") == 2
    fprintf("Loading ResNet-50 via imagePretrainedNetwork...\n");
    [baseNet, ~] = imagePretrainedNetwork("resnet50");
else
    % Fallback if needed
    fprintf("imagePretrainedNetwork not found. Trying resnet50()...\n");
    if exist("resnet50","file") ~= 2
        error(["Could not find ResNet-50. Install ", ...
            "'Deep Learning Toolbox Model for ResNet-50 Network'."]);
    end
    baseNet = resnet50;
end

% Detection head source layer – same as MathWorks YOLOv2 example
detectionSource = "activation_40_relu";

% Create untrained YOLOv2 detector object (no yolov2Layers here)
detector = yolov2ObjectDetector(baseNet, classNames, anchorBoxes, ...
    DetectionNetworkSource=detectionSource);

%% 7. Final training data pipeline (with resize only, no fancy augmentation)

% We reuse trainingData (imds+blds combined) and just resize images + boxes.
preprocessedTrainingData = transform(trainingData, ...
    @(data) preprocessData(data, inputSize));

%% 8. Training options (no validation data – keep it simple)

options = trainingOptions("adam", ...
    MiniBatchSize=8, ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=20, ...
    Shuffle="every-epoch", ...
    VerboseFrequency=20, ...
    Plots="training-progress");   % opens live training plot

%% 9. Train YOLOv2 detector

fprintf("\n********************************************************\n");
fprintf("Training YOLOv2 bottle detector (this can take a while)...\n");
fprintf("********************************************************\n\n");

[detector, info] = trainYOLOv2ObjectDetector(preprocessedTrainingData, detector, options);

%% 10. Save trained model and training info

modelsDir = fullfile(pwd, "models");
if ~exist(modelsDir, "dir")
    mkdir(modelsDir);
end

save(fullfile(modelsDir, "bottleDetector.mat"), "detector");
save(fullfile(modelsDir, "trainingInfo.mat"), "info");

disp("------------------------------------------------------");
disp("Training complete.");
fprintf("Detector saved to:      %s\n", fullfile(modelsDir, "bottleDetector.mat"));
fprintf("Training info saved to: %s\n", fullfile(modelsDir, "trainingInfo.mat"));
disp("------------------------------------------------------");

%% === Local helper function(s) ===

function dataOut = preprocessData(dataIn, targetSize)
% Resize image and bounding boxes to targetSize [H W 3].
%
% dataIn is a 1x3 cell:
%   {1} = image (H x W x 3)
%   {2} = N-by-4 bounding boxes [x y w h]
%   {3} = N-by-1 labels (categorical or string)

    I      = dataIn{1};
    bboxes = dataIn{2};
    labels = dataIn{3};

    % Original spatial size
    origSize = size(I,[1 2]);          % [H W]
    scale    = targetSize(1:2) ./ origSize;

    % Resize image
    Iresized = imresize(I, targetSize(1:2));

    % Resize boxes
    if ~isempty(bboxes)
        bboxes = bboxresize(bboxes, scale);
    end

    % Return in same cell format
    dataOut = {Iresized, bboxes, labels};
end
