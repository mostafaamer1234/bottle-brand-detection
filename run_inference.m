%% Run inference on test images with trained bottle detector
% Run this from the project root (same place as train_yolov2.m)

clc; clear; close all;

%% 1. Paths

projectRoot = pwd;
modelFile   = fullfile(projectRoot, "models", "bottleDetector.mat");
testDir     = fullfile(projectRoot, "test_images");
outDir      = fullfile(projectRoot, "test_results");

if ~isfile(modelFile)
    error("Model file not found at: %s", modelFile);
end

if ~isfolder(testDir)
    error("Test images folder not found at: %s", testDir);
end

if ~isfolder(outDir)
    mkdir(outDir);
end

%% 2. Load trained detector

load(modelFile, "detector");
disp("Loaded detector:");
disp(detector)

%% 3. Create imageDatastore for test images
% (let MATLAB pick image files automatically)

imdsTest = imageDatastore(testDir);
numTest  = numel(imdsTest.Files);
fprintf("Found %d test images in %s\n", numTest, testDir);

if numTest == 0
    error("No images found in test_images folder. Add some JPG/PNG/etc and try again.");
end

%% 4. Loop over test images, run detect(), save annotated results

resultsTable = table('Size', [numTest 4], ...
    'VariableTypes', ["string","double","double","double"], ...
    'VariableNames', ["filename","numDetections","numBranded","numGeneric"]);

brandedCat = categorical("branded_bottle");
genericCat = categorical("generic_bottle");

for i = 1:numTest
    I = readimage(imdsTest, i);
    [bboxes, scores, labels] = detect(detector, I);

    % Count detections by class (labels are categorical)
    numDet    = size(bboxes,1);
    numBrand  = sum(labels == brandedCat);
    numGener  = sum(labels == genericCat);

    % Annotate image if any detections
    if ~isempty(bboxes)
        labelStrings = strcat(string(labels), " (", string(round(scores,2)), ")");
        Iout = insertObjectAnnotation(I, "rectangle", bboxes, cellstr(labelStrings), ...
            "LineWidth", 3, "FontSize", 14);
    else
        Iout = I;
    end

    % Save annotated image
    [~, baseName, ~] = fileparts(imdsTest.Files{i});
    outFile = fullfile(outDir, baseName + "_detected.png");
    imwrite(Iout, outFile);

    % Store stats
    resultsTable.filename(i)      = string(imdsTest.Files{i});
    resultsTable.numDetections(i) = numDet;
    resultsTable.numBranded(i)    = numBrand;
    resultsTable.numGeneric(i)    = numGener;

    fprintf("Processed %d/%d: %s  (detections: %d, branded: %d, generic: %d)\n", ...
        i, numTest, baseName, numDet, numBrand, numGener);
end

%% 5. Save results summary

summaryFile = fullfile(outDir, "detection_summary.csv");
writetable(resultsTable, summaryFile);
fprintf("\nAnnotated images saved in: %s\n", outDir);
fprintf("Summary CSV saved to:      %s\n", summaryFile);
