<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html><head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"><meta name="generator" content="MATLAB R2016a"><meta http-equiv="X-UA-Compatible" content="IE=edge,IE=9,chrome=1"><title>Read the Input Image Sequence</title><style type="text/css">
* {margin: 0; padding: 0;}
body {text-align: start; line-height: 17.2339992523193px; min-height: 0px; white-space: normal; color: rgb(0, 0, 0); font-family: Consolas, Inconsolata, Menlo, monospace; font-style: normal; font-size: 14px; font-weight: normal; text-decoration: none; white-space: normal; }
h1, h2 {font-weight: normal;}
.content { padding: 30px; }

.S0 { margin-left: 0px; margin-top: 0px; margin-bottom: 0px; margin-right: 0px;  }
.S1 { line-height: 26.3999996185303px; min-height: 24px; white-space: pre-wrap; color: rgb(213, 80, 0); font-family: Helvetica, Arial, sans-serif; font-size: 22px; white-space: pre-wrap; margin-left: 4px; margin-top: 3px; margin-bottom: 15px; margin-right: 10px;  }
.S2 { min-height: 0px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px; margin-right: 0px;  }
.S3 { margin-left: 3px; margin-top: 10px; margin-bottom: 4px; margin-right: 3px;  }
.S4 { line-height: 15.5926666259766px; min-height: 18px; white-space: nowrap; font-size: 12.6666669845581px; white-space: nowrap; margin-left: 0px; margin-top: 0px; margin-bottom: 0px; margin-right: 0px;  }
.S5 { line-height: 15.5926675796509px; min-height: 0px; white-space: pre; color: rgb(34, 139, 34); font-size: 12.6666679382324px; white-space: pre; margin-left: 0px; margin-top: 0px; margin-bottom: 0px; margin-right: 45px;  }
.S6 { line-height: 15.5926675796509px; min-height: 0px; white-space: pre; font-size: 12.6666679382324px; white-space: pre; margin-left: 0px; margin-top: 0px; margin-bottom: 0px; margin-right: 0px;  }
.S7 { line-height: 15.5926675796509px; min-height: 0px; white-space: pre; color: rgb(160, 32, 240); font-size: 12.6666679382324px; white-space: pre; margin-left: 0px; margin-top: 0px; margin-bottom: 0px; margin-right: 0px;  }
.S8 { line-height: 15.5926675796509px; min-height: 0px; white-space: pre; font-size: 12.6666679382324px; white-space: pre; margin-left: 0px; margin-top: 0px; margin-bottom: 0px; margin-right: 45px;  }
.S9 { color: rgb(64, 64, 64); margin-left: 0px; margin-top: 0px; margin-bottom: 0px; margin-right: 0px;  }
.S10 { line-height: 15.5926675796509px; min-height: 0px; white-space: pre; color: rgb(0, 0, 255); font-size: 12.6666679382324px; white-space: pre; margin-left: 0px; margin-top: 0px; margin-bottom: 0px; margin-right: 0px;  }
.S11 { line-height: 15.5926675796509px; min-height: 0px; white-space: pre; color: rgb(0, 0, 255); font-size: 12.6666679382324px; white-space: pre; margin-left: 0px; margin-top: 0px; margin-bottom: 0px; margin-right: 45px;  }
.S12 { line-height: 20.576000213623px; min-height: 20px; white-space: pre-wrap; color: rgb(60, 60, 60); font-family: Helvetica, Arial, sans-serif; font-size: 16px; font-weight: bold; white-space: pre-wrap; margin-left: 4px; margin-top: 3px; margin-bottom: 9px; margin-right: 10px;  }
.S13 { line-height: 15.5926675796509px; min-height: 0px; white-space: pre; color: rgb(160, 32, 240); font-size: 12.6666679382324px; white-space: pre; margin-left: 0px; margin-top: 0px; margin-bottom: 0px; margin-right: 45px;  }

.LineNodeBlock {margin: 10px 0 10px 0; background-color: #F7F7F7;}
.LineNodeBlock+.paragraphNode {margin-top: 10px;}
.lineNode {padding-left: 10px; border-left: 1px solid #E9E9E9; border-right: 1px solid #E9E9E9;}
.inlineWrapper:first-child .lineNode,.inlineWrapper.outputs+.inlineWrapper .lineNode {padding-top: 5px; border-top: 1px solid #E9E9E9;}
.inlineWrapper:last-child .lineNode,.inlineWrapper.outputs .lineNode {padding-bottom: 5px; border-bottom: 1px solid #E9E9E9;}
.lineNode .textBox {white-space: pre;}
.outputGroup {margin: 2px 0 2px 0; padding: 2px 2px 2px 4px;}
.outputRegion {}
.outputParagraph {color: rgba(64, 64, 64, 1); padding: 10px 0 6px 17px; background: white; overflow-x: hidden;}
.inlineWrapper:last-child .outputParagraph {border-bottom-left-radius: 4px; border-bottom-right-radius: 4px;}
.outputParagraph:empty {margin: 0;}
.inlineElement .symbolicElement {margin-top: 1px; margin-bottom: 1px;}
.embeddedOutputsSymbolicElement .MathEquation {margin-top: 4px; margin-bottom: 4px;}
.embeddedOutputsSymbolicElement .MathEquation.displaySymbolicElement {margin-left: 15px;}
.embeddedOutputsSymbolicElement .MathEquation.inlineSymbolicElement {}
.embeddedOutputsSymbolicElement {overflow-x: auto; overflow-y: hidden;}
.embeddedOutputsSymbolicElement { overflow: initial !important;}
.embeddedOutputsTextElement,.embeddedOutputsVariableStringElement {font-family: Consolas, Inconsolata, Menlo, monospace; font-size: 12px; white-space: pre; word-wrap: initial; min-height: 18px; max-height: 250px; overflow: auto;}
.textElement,.rtcDataTipElement .textElement {padding-top: 3px;}
.embeddedOutputsTextElement.inlineElement,.embeddedOutputsVariableStringElement.inlineElement {}
.inlineElement .textElement {}
.embeddedOutputsTextElement,.embeddedOutputsVariableStringElement { max-height: none !important; overflow: initial !important;}
.veSpecifier {}
.veContainer {}
.veSpecifierBox {height: 400px; width: 500px;}
.veSpecifier .veTable {padding-top: 3px; padding-bottom: 4px;}
.veSpecifierBox .veSpecifier .veContainer {position: relative; width: 100%; height: 370px;}
.veSpecifierBox .dijitDialogPaneContent {width: 97% !important; height: 88% !important;}
.veSpecifier .veTable .rowHeadersWrapper {padding-bottom: 0;}
.veSpecifier .veTable .scroller .variableEditorRenderers {padding-right: 3px; -webkit-user-select: none; -moz-user-select: none; -ms-user-select: none;}
.veSpecifier .veTable .topHeaderWrapper,.veSpecifier .veTable .bottomRowHeaderWrapper {visibility: hidden; z-index: 0;}
.veMetaSummary {font-style: italic;}
.veSpecifier .veTable .scroller {overflow: hidden;}
.veSpecifier .veTable:hover .scroller {overflow: auto;}
.veSpecifier .veVariableName,.veSpecifier .veDimensionFont {font-family: Consolas, Inconsolata, Menlo, monospace; font-size: 12px;}
.veSpecifier .veVariableName {padding-top: 2px;}
.veSpecifier .veDimensionFont {font-style: italic; color: #9A9A9A;}
.veSpecifier .scroller::-webkit-scrollbar-track {background-color: white;}
.veSpecifier .scroller::-webkit-scrollbar-corner {background-color: white;}
.veSpecifier .veTable .topRowHeaderWrapper {border: none; background-color: #F8F9FA;}
.veSpecifier .mw_type_ListBase.showCellBorders,.veSpecifier .veTable .topHeaderWrapper,.veSpecifier .veTable .bottomRowHeaderWrapper,.veSpecifier .veTable .verticalScrollSpacer,.veSpecifier .veTable .horizontalScrollSpacer {border: none;}
.veSpecifier .veTable .dataScrollerNode {border: 1px solid #BFBFBF;}
.veSpecifier .veTable .columnHeaderNode,.veSpecifier .veTable .rowHeaderNode,.veSpecifier .veTable .dataBody {font-family: Arial; font-size: 13px;}
.veSpecifier .veTable .columnHeaderNode,.veSpecifier .veTable .rowHeaderNode {color: #7F7F7F;}
.veSpecifier .veTable .dataBody {color: #000000;}
.veSpecifier .veTable .columnHeaderNode .cell .drag,.veSpecifier .veTable .columnHeaderNode .cell .header,.veSpecifier .veTable .topHeaderWrapper,.veSpecifier .veTable .bottomRowHeaderWrapper {background-color: #F8F9FA;}
.veSpecifier .veTable .columnHeaderNode .cell .dragBorder {border-right: 1px solid #F8F9FA;}
.veSpecifier .veTable .rowHeaderNode .cell {padding-top: 3px; text-align: center; border-bottom: 1px solid #F8F9FA;}
.veSpecifier .veTable .dataBody .cell .plainText {text-align: right;}
.veSpecifier .veTable .dataBody .row .cell {border-bottom: 1px solid #E9E9E9; border-right: 1px solid #E9E9E9;}
.embeddedOutputsVariableElement {font-family: Consolas, Inconsolata, Menlo, monospace; font-size: 12px; white-space: pre-wrap; word-wrap: break-word; min-height: 18px; max-height: 250px; overflow: auto;}
.variableElement {}
.embeddedOutputsVariableElement.inlineElement {}
.inlineElement .variableElement {}
.variableNameElement {margin-bottom: 3px; display: inline-block;}
.variableValue { width: 100% !important; }
.embeddedOutputsMatrixElement {min-height: 18px; box-sizing: border-box; font-family: Consolas, Inconsolata, Menlo, monospace; font-size: 12px;}
.embeddedOutputsMatrixElement .matrixElement,.rtcDataTipElement .matrixElement {position: relative;}
.matrixElement .variableValue,.rtcDataTipElement .matrixElement .variableValue {white-space: pre; display: inline-block; vertical-align: top; overflow: hidden;}
.embeddedOutputsMatrixElement.inlineElement {}
.embeddedOutputsMatrixElement.inlineElement .topHeaderWrapper {display: none;}
.embeddedOutputsMatrixElement.inlineElement .veTable .body {padding-top: 0 !important; max-height: 100px;}
.inlineElement .matrixElement {max-height: 300px;}
.embeddedOutputsMatrixElement .matrixElement .valueContainer,.rtcDataTipElement .matrixElement .valueContainer {white-space: nowrap; margin-bottom: 3px;}
.embeddedOutputsMatrixElement .matrixElement .valueContainer .horizontalEllipsis.hide,.embeddedOutputsMatrixElement .matrixElement .verticalEllipsis.hide,.rtcDataTipElement .matrixElement .valueContainer .horizontalEllipsis.hide,.rtcDataTipElement .matrixElement .verticalEllipsis.hide {display: none;}
.embeddedOutputsMatrixElement .matrixElement .valueContainer .horizontalEllipsis {margin-bottom: -3px;}
.matrixElement { max-height: none !important; }
.dijitTooltipDialog .dijitTooltipContainer .dijitTooltipContents .alertPlugin-alertMessage {min-width: 12px; max-width: 400px; max-height: 300px; overflow: auto;}
.dijitTooltipDialog .alertPlugin-alertMessage::-webkit-scrollbar {width: 11px; height: 11px;}
.dijitTooltipDialog .alertPlugin-alertMessage::-webkit-scrollbar-track {background-color: rgba(0, 0, 0, 0);}
.dijitTooltipDialog .alertPlugin-alertMessage::-webkit-scrollbar-corner {background-color: rgba(0, 0, 0, 0);}
.dijitTooltipDialog .alertPlugin-alertMessage::-webkit-scrollbar-thumb {border-radius: 7px; background-color: rgba(0, 0, 0, 0.1); border: 2px solid rgba(0, 0, 0, 0); background-clip: padding-box;}
.dijitTooltipDialog .alertPlugin-alertMessage::-webkit-scrollbar-thumb:hover {background-color: rgba(0, 0, 0, 0.2);}
.dijitTooltipDialog .alertPlugin-alertMessage::-webkit-scrollbar-thumb {background-color: rgba(0, 0, 0, 0);}
.dijitTooltipDialog .alertPlugin-alertMessage:hover::-webkit-scrollbar-thumb {background-color: rgba(0, 0, 0, 0.1);}
.dijitTooltipDialog .alertPlugin-alertMessage:hover::-webkit-scrollbar-thumb:hover {background-color: rgba(0, 0, 0, 0.2);}
.alertPlugin-alertLine {position: absolute; display: initial; width: 40px; text-align: right; cursor: text;}
.alertPlugin-onTextLine {visibility: hidden;}
.alertPlugin-hasTooltip .alertPlugin-warningImg,.alertPlugin-hasTooltip .alertPlugin-errorImg {cursor: pointer;}
.alertPlugin-alertLine .alertPlugin-warningElement,.alertPlugin-alertLine .alertPlugin-errorElement {display: inline-block; margin-right: 4px;}
.alertPlugin-isStale {-webkit-filter: opacity(0.4) grayscale(80%); filter: opacity(0.4) grayscale(80%);}
.diagnosticMessage-wrapper {font-family: Consolas, Inconsolata, Menlo, monospace; font-size: 12px;}
.diagnosticMessage-wrapper.diagnosticMessage-warningType {color: rgb(255,100,0);}
.diagnosticMessage-wrapper.diagnosticMessage-warningType a {color: rgb(255,100,0); text-decoration: underline;}
.diagnosticMessage-wrapper.diagnosticMessage-errorType {color: rgb(230,0,0);}
.diagnosticMessage-wrapper.diagnosticMessage-errorType a {color: rgb(230,0,0); text-decoration: underline;}
.diagnosticMessage-wrapper .diagnosticMessage-messagePart {white-space: pre-wrap;}
.diagnosticMessage-wrapper .diagnosticMessage-stackPart {white-space: pre;}
.embeddedOutputsWarningElement{min-height: 18px; max-height: 250px; overflow: auto;}
.embeddedOutputsWarningElement.inlineElement {}
.embeddedOutputsErrorElement {min-height: 18px; max-height: 250px; overflow: auto;}
.embeddedOutputsErrorElement.inlineElement {}
.variableNameElement .headerElement {color: rgb(179, 179, 179); font-style: italic;}
.variableNameElement .headerElement .headerDataType {color: rgb(147, 176, 230);}
<!-- 
##### SOURCE BEGIN #####
%% Read the Input Image Sequence

% acquire images from folder
imds = imageDatastore('./photos');

% Display the images.
figure
montage(imds.Files, 'Size', [3, 2]);

% Convert the images to grayscale.
images = cell(1, numel(imds.Files));
for i = 1:numel(imds.Files)
    I = readimage(imds, i);
    images{i} = rgb2gray(I);
end

title('Input Image Sequence');

%% Load Camera Parameters
%%
load(fullfile('./', 'cameraParams.mat'));
save('step1.mat');
%% Create a View Set Containing the First View
%%
load('step1.mat');
% Undistort the first image.
I = undistortImage(images{1}, cameraParams);

% Detect features. Increasing 'NumOctaves' helps detect large-scale
% features in high-resolution images. Use an ROI to eliminate spurious
% features around the edges of the image.
border = 50;
roi = [border, border, size(I, 2)- 2*border, size(I, 1)- 2*border];
prevPoints   = detectSURFFeatures(I, 'NumOctaves', 8, 'ROI', roi);

% Extract features. Using 'Upright' features improves matching, as long as
% the camera motion involves little or no in-plane rotation.
prevFeatures = extractFeatures(I, prevPoints, 'Upright', true);

% Create an empty viewSet object to manage the data associated with each
% view.
vSet = viewSet;

% Add the first view. Place the camera associated with the first view
% and the origin, oriented along the Z-axis.
viewId = 1;
vSet = addView(vSet, viewId, 'Points', prevPoints, 'Orientation', ...
    eye(3, 'like', prevPoints.Location), 'Location', ...
    zeros(1, 3, 'like', prevPoints.Location));
save('step2.mat');
%% Add the Rest of the Views
%%
load('step2.mat');
for i = 2:numel(images)
    % Undistort the current image.
    I = undistortImage(images{i}, cameraParams);

    % Detect, extract and match features.
    currPoints   = detectSURFFeatures(I, 'NumOctaves', 8, 'ROI', roi);
    currFeatures = extractFeatures(I, currPoints, 'Upright', true);
    indexPairs = matchFeatures(prevFeatures, currFeatures, ...
        'MaxRatio', .25, 'Unique',  true);

    % Select matched points.
    matchedPoints1 = prevPoints(indexPairs(:, 1));
    matchedPoints2 = currPoints(indexPairs(:, 2));

    % Estimate the camera pose of current view relative to the previous view.
    % The pose is computed up to scale, meaning that the distance between
    % the cameras in the previous view and the current view is set to 1.
    % This will be corrected by the bundle adjustment.
    [relativeOrient, relativeLoc, inlierIdx] = helperEstimateRelativePose(...
        matchedPoints1, matchedPoints2, cameraParams);

    % Add the current view to the view set.
    vSet = addView(vSet, i, 'Points', currPoints);

    % Store the point matches between the previous and the current views.
    vSet = addConnection(vSet, i-1, i, 'Matches', indexPairs(inlierIdx,:));

    % Get the table containing the previous camera pose.
    prevPose = poses(vSet, i-1);
    prevOrientation = prevPose.Orientation{1};
    prevLocation    = prevPose.Location{1};

    % Compute the current camera pose in the global coordinate system
    % relative to the first view.
    orientation = relativeOrient * prevOrientation;
    location    = prevLocation + relativeLoc * prevOrientation;
    vSet = updateView(vSet, i, 'Orientation', orientation, ...
        'Location', location);

    % Find point tracks across all views.
    tracks = findTracks(vSet);

    % Get the table containing camera poses for all views.
    camPoses = poses(vSet);

    % Triangulate initial locations for the 3-D world points.
    xyzPoints = triangulateMultiview(tracks, camPoses, cameraParams);

    % Refine the 3-D world points and camera poses.
    [xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(xyzPoints, ...
        tracks, camPoses, cameraParams, 'FixedViewId', 1, ...
        'PointsUndistorted', true);

    % Store the refined camera poses.
    vSet = updateView(vSet, camPoses);

    prevFeatures = currFeatures;
    prevPoints   = currPoints;
end
save('step3.mat');
%% Display Camera Poses
%%
load('step3.mat');
% Display camera poses.
camPoses = poses(vSet);
figure;
plotCamera(camPoses, 'Size', 0.2);
hold on

% Exclude noisy 3-D points.
goodIdx = (reprojectionErrors < 5);
xyzPoints = xyzPoints(goodIdx, :);

% Display the 3-D points.
pcshow(xyzPoints, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);
grid on
hold off

% Specify the viewing volume.
loc1 = camPoses.Location{1};
xlim([loc1(1)-5, loc1(1)+4]);
ylim([loc1(2)-5, loc1(2)+4]);
zlim([loc1(3)-1, loc1(3)+20]);
camorbit(0, -30);

title('Refined Camera Poses');
save('step4.mat');
%% Compute Dense Reconstruction
%%
load('step4.mat');
% Read and undistort the first image
I = undistortImage(images{1}, cameraParams);

% Detect corners in the first image.
prevPoints = detectMinEigenFeatures(I, 'MinQuality', 0.001);

% Create the point tracker object to track the points across views.
tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 6);

% Initialize the point tracker.
prevPoints = prevPoints.Location;
initialize(tracker, prevPoints, I);

% Store the dense points in the view set.
vSet = updateConnection(vSet, 1, 2, 'Matches', zeros(0, 2));
vSet = updateView(vSet, 1, 'Points', prevPoints);

% Track the points across all views.
for i = 2:numel(images)
    % Read and undistort the current image.
    I = undistortImage(images{i}, cameraParams);

    % Track the points.
    [currPoints, validIdx] = step(tracker, I);

    % Clear the old matches between the points.
    if i < numel(images)
        vSet = updateConnection(vSet, i, i+1, 'Matches', zeros(0, 2));
    end
    vSet = updateView(vSet, i, 'Points', currPoints);

    % Store the point matches in the view set.
    matches = repmat((1:size(prevPoints, 1))', [1, 2]);
    matches = matches(validIdx, :);
    vSet = updateConnection(vSet, i-1, i, 'Matches', matches);
end

% Find point tracks across all views.
tracks = findTracks(vSet);

% Find point tracks across all views.
camPoses = poses(vSet);

% Triangulate initial locations for the 3-D world points.
xyzPoints = triangulateMultiview(tracks, camPoses,...
    cameraParams);

% Refine the 3-D world points and camera poses.
[xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(...
    xyzPoints, tracks, camPoses, cameraParams, 'FixedViewId', 1, ...
    'PointsUndistorted', true);
save('step5.mat');
%% Display Dense Reconstruction
%%
load('step5.mat');
% Display the refined camera poses.
figure;
plotCamera(camPoses, 'Size', 0.2);
hold on

% Exclude noisy 3-D world points.
goodIdx = (reprojectionErrors < 5);

% Display the dense 3-D world points.
pcshow(xyzPoints(goodIdx, :), 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);
grid on
hold off

% Specify the viewing volume.
loc1 = camPoses.Location{1};
xlim([loc1(1)-5, loc1(1)+4]);
ylim([loc1(2)-5, loc1(2)+4]);
zlim([loc1(3)-1, loc1(3)+20]);
camorbit(0, -30);

title('Dense Reconstruction');
##### SOURCE END #####
--></body></html>