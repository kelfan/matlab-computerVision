% read imgs 
negImgs =  imageDatastore('neg');
posImgs = imageDatastore('pos');

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();
% noseDetector = vision.CascadeObjectDetector('Nose', 'UseROI', true);

bboxImgs=[];

for i = 1:60
    img = readimage(posImgs, i);
    bbox = step(faceDetector, img);
    % noseBBox     = step(noseDetector, img, bbox(1,:));
    
    videoOut = insertObjectAnnotation(img,'rectangle',bbox,'Face');
    figure, imshow(videoOut), title('Detected face');
    bboxImgs = [bboxImgs; bbox];
end
save('step1.mat');
%% add address and bbox into variable
%%
load('step1.mat');
posInstance = [];
pos = [];
negInstance = [];
for i = 1:59
    % bboxImgs(i,:)
    % posImgs.Files(i)
    pos = [posImgs.Files(i), bboxImgs(i,:)];
    posInstance = [posInstance; pos];
    %negInstance = [negInstance; negImgs.Files(i)];
end
save('step2.mat');
%% Training
%%
load('step2.mat');
filenames = {'imageFilename', 'objectBoundingBoxes'};
posInstance = cell2struct(posInstance, filenames, 2);
negFolder='C:\Users\chaofanz\Desktop\utasLearning\412ComputerVision\week10 face detection\neg';
%%

% Train a cascade object detector called 'stopSignDetector.xml' using HOG features.
trainCascadeObjectDetector('kelDetector.xml',posInstance, ...
    negFolder,'FalseAlarmRate',0.1,'NumCascadeStages',5,'FeatureType', 'Haar');
save('step3.mat');
%% Testing
%%
% load('step3.mat');
% Use the newly trained classifier to detect a stop sign in an image.
detector = vision.CascadeObjectDetector('kelDetector.xml');

% load img 
testImgs = imageDatastore('test');

for i=1:10
    img = readimage(testImgs, i);
    
    % Detect a stop sign.
    bbox = step(detector,img);
    
    % Insert bounding box rectangles and return the marked image.
    detectedImg = insertObjectAnnotation(img,'rectangle',bbox,'stop sign');
    
    % Display the detected stop sign.
    figure; imshow(detectedImg);
end 
% the small box can be filter by setting the range of bbox easily 
% the most important is to get the big box to recognise the right person rather than the wrong person;
% some small box can be filter by the distance of the target