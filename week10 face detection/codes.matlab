% read imgs 
negImgs =  imageDatastore('neg');
posImgs = imageDatastore('pos');

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();
noseDetector = vision.CascadeObjectDetector('Nose', 'UseROI', true);

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