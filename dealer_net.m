%% Nnet for dealer chip

dealer_Path = fullfile('C:','Users','marti_000','Documents','MATLAB','octave',...
    'deal_or_no_deal');
dealer_data = imageDatastore(dealer_Path, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

trainingNumFiles = 0.8;
rng(1) % For reproducibility
[trainDealer,testDealer] = splitEachLabel(dealer_data, ...
    trainingNumFiles,'randomize');
trainDealer.countEachLabel % for info
testDealer.countEachLabel % for info
classWeights = 1./countcats(trainDealer.Labels);
classWeights = classWeights'/mean(classWeights);

% %% Uneven classes so above method of decimal trainingNumFiles.
% % Could have been this otherwise, but 50:50 split on trainign & test
% dealer_data = shuffle(dealer_data)
% trainDealer = partition(dealer_data,2,1);
% testDealer = partition(dealer_data,2,2);
%
% dealer_data.countEachLabel % for info
% trainDealer.countEachLabel % for info
% testDealer.countEachLabel % for info
% classWeights = 1./countcats(dealer_data.Labels);
% classWeights = classWeights'/mean(classWeights);

layers = [ ...
    imageInputLayer([42 45 3])
    convolution2dLayer(3,16)
    batchNormalizationLayer()
    reluLayer()
    convolution2dLayer(3,32)
    batchNormalizationLayer()
    reluLayer()
    % tansigLayer(32)
    % fullyConnectedLayer(64)
    dropoutLayer(0.4)
    fullyConnectedLayer(2)
    softmaxLayer()
    classificationLayer()];
% weightedClassificationLayer(classWeights, 'weighted')];


%     'MiniBatchSize',64,...
%     'ValidationData',testDealer, ...
%     'ValidationFrequency',30, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.2, ...
%     'LearnRateDropPeriod',5, ...

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ValidationData',testDealer, ...
    'ValidationFrequency',20, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5);

dealer_finder = trainNetwork(dealer_data,layers,options);
% save('dealer_find.mat','dealer_finder')
%% Ok lets build a RCNN
load('dealer_find.mat');
% load('fullscreen.mat') % octave folder.. use datastore instead
% Need a table for RCNN, can use datastore for file names and positions
% should correspond to:
% 1-9 you
% 10-11 you, dark
% 12 blank
% 13-50 left of you dealer
% 51-55 same,dark
% 56-100 top left dealer
you_deal=[758,562,42,45];
next_deal=[370,490,42,45];
third_deal=[323,223,42,45];

RCNN_Path = fullfile('C:','Users','marti_000','Documents','MATLAB','octave',...
    'fullscreen');
RCNN_data = imageDatastore(RCNN_Path, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% filerefs=string(RCNN_data.Files);
% filerefs([10,11,12 51:55])=[];% its not sorted that way
filerefs=cell(92,1);
locations=cell(92,1);
for i=1:9
    filerefs(i)=cellstr(strcat('C:\Users\marti_000\Documents\MATLAB\octave\fullscreen\',int2str(i),'fullscreen.png'));
    locations(i)={you_deal};
end
for j=10:47
    i=j+3;
    filerefs(j)=cellstr(strcat('C:\Users\marti_000\Documents\MATLAB\octave\fullscreen\',int2str(i),'fullscreen.png'));
    locations(j)={next_deal};
end
for j=48:92
    i=j+8;
    filerefs(j)=cellstr(strcat('C:\Users\marti_000\Documents\MATLAB\octave\fullscreen\',int2str(i),'fullscreen.png'));
    locations(j)={next_deal};
end

RCNN_input=table(filerefs,locations);

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-6, ...
    'MaxEpochs', 10);

detector = trainRCNNObjectDetector(RCNN_input,dealer_finder,options);

%%
% save('fullscreen_test','images_test')
img = images_test(:,:,:,35);

[bbox, score, label] = detect(detector, I, 'MiniBatchSize', 32);
% Display strongest detection result.

[score, idx] = max(score);

bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(I, 'rectangle', bbox, annotation);

figure
imshow(detectedImg)

%%
% Perform OCR.
results = ocr(img);

% Display one of the recognized words.
word = results.Words

%% Let's try subsets of image. Go for pot size
%
r=randperm(80,10)+20; % a few at the beginning with obscured bits - we'll filter for no $ symbol in words
figure;
for i=1:10
    img = images_test(:,:,:,r(i));
    subplot(5,2,i)
    img_sub = img(100:220,560:860,:);
    results = ocr(img_sub);
    word = results.Words;
    
    try
        wordBBox = results.WordBoundingBoxes(1,:); % location of first word
        img_sub = insertObjectAnnotation(img_sub,'rectangle',wordBBox,word);
    catch
    end
    imshow(img_sub)
    title(word);
end

% Good. We should drop anything more than 1 word or not preceded by $
%% Next, let's work out everyones personal stack
stack_yours = img(580:604,635:784,:); % 162 not 120, 27 height not 20
stack_opponent1 = img(515:534,240:359,:);
stack_opponent2 = img(225:244,190:309,:);
stack_opponent3 = img(225:244,1100:1219,:);
stack_opponent4 = img(515:534,1055:1174,:);
stacks=cell(5,1);results=cell(5,1);
stacks(1,1)={stack_yours};
stacks(2,1)={stack_opponent1};stacks(3,1)={stack_opponent1};stacks(4,1)={stack_opponent3};stacks(5,1)={stack_opponent4};
figure;
for i=1:5
    BW = imbinarize(stacks{i,1},'adaptive');
    results = ocr(BW,'TextLayout','Block');
    word = results.Words{1};
    subplot(5,1,i)
    imshowpair(stacks{i,1},BW,'montage');
%     imshow(stacks{i,1});
    title(strcat('Player',{' '},int2str(i),{' '},'stack:',{' '},string(word)));
end


% imshow(img)

%% what suit?

card1 = img(232:270,468:508,:);
card2 = img(232:270,593:633,:);
card3 = img(232:270,718:758,:);
card4 = img(232:270,848:888,:);
card5 = img(232:270,973:1013,:);

% cardsonthetable=cell(5,1);
cardsonthetable(1)={card1};cardsonthetable(2)={card2};cardsonthetable(3)={card3};cardsonthetable(4)={card4};cardsonthetable(5)={card5};

figure;
for i=1:5
subplot(5,1,i)
imshow(cardsonthetable{i});
end

% repeat above 4 times to acquire 20 images
labels={'blank','blank','blank','blank','blank','club','diam','heart','diam','diam','heart','diam','diam','spade','heart','heart','spade','diam','club','club','diam','spade','spade','club','club'};
% save('cardsontable.mat','cardsonthetable');
labels=categorical(labels);
%%
layers = [ ...
    imageInputLayer([39 41 3])
    convolution2dLayer(3,16)
    batchNormalizationLayer()
    reluLayer()
    convolution2dLayer(3,32)
    batchNormalizationLayer()
    reluLayer()
    % tansigLayer(32)
    % fullyConnectedLayer(64)
    dropoutLayer(0.4)
    fullyConnectedLayer(5)
    softmaxLayer()
    classificationLayer()];


options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.005, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress')
for i=1:25
net_4din(:,:,:,i)=(cardsonthetable{i});
end
suit_tablecards=trainNetwork(net_4din,labels,layers,options);

%%
word = {char(classify(suit_tablecards,card1))};
wordBBox = [468 232 41 39]; % location of first card suit
img = insertObjectAnnotation(img,'rectangle',wordBBox,word);

word = {char(classify(suit_tablecards,card2))};
wordBBox = [593 232 41 39]; % location of second card suit
img = insertObjectAnnotation(img,'rectangle',wordBBox,word);

word = {char(classify(suit_tablecards,card3))};
wordBBox = [718 232 41 39]; % location of second card suit
img = insertObjectAnnotation(img,'rectangle',wordBBox,word);

word = {char(classify(suit_tablecards,card4))};
wordBBox = [848 232 41 39]; % location of second card suit
img = insertObjectAnnotation(img,'rectangle',wordBBox,word);

word = {char(classify(suit_tablecards,card5))};
wordBBox = [973 232 41 39]; % location of second card suit
img = insertObjectAnnotation(img,'rectangle',wordBBox,word);


figure;imshow(img)
%this is good. layering the image with annotations
        
%% Lets try OCR against the card value
card1 = img(232:255,410:430,:);
card2 = img(232:255,535:555,:);
card3 = img(232:255,662:682,:);
card4 = img(232:255,788:808,:);
card5 = img(232:255,913:933,:);
% cardsonthetable=cell(5,1);
cardsonthetable(1)={card1};cardsonthetable(2)={card2};cardsonthetable(3)={card3};cardsonthetable(4)={card4};cardsonthetable(5)={card5};

figure;
for i=1:5
     BW = imbinarize(cardsonthetable{i});
    results = ocr(BW,'TextLayout','Block','CharacterSet','23456789JQKA');
    word = results.Words{1};
    img = insertObjectAnnotation(cardsonthetable{i},'rectangle',wordBBox,word);
subplot(1,5,i)
imshow(img);
    title(strcat('Card',{' '},int2str(i),{' '},'Value:',{' '},string(word)));
end

%% Ok so now we're looking at betting/folding options presented
img = images_test(:,:,:,50);
% options_sub = img(700:800,20:1400,:); % for the RCNN lets consider screen size might adjust. not used at yet
%  We will need images specified for a net to be wrapped in an RCNN
% figure;imshow(img);
% [x,y] = ginput(4)
options1 = img(755:800,377:557); % with respect to options sub 45:100,337:517
options2 = img(755:800,621:801);
options3 = img(755:800,867:1047);
figure;
subplot(1,3,1)
imshow(options1);
subplot(1,3,2)
imshow(options2);
subplot(1,3,3)
imshow(options3);
% for rcnn..
options1=[377,755,180,45];options2=[621,755,180,45];options3=[867,755,180,45];
options_raisebar=[1196,756,60,45]; % ok symbol we're actually doing
figure;word={'test'};
img = insertObjectAnnotation(img,'rectangle',options1,word);
imshow(img)
%% for RCNN
% 212-213 options fold check raise
% 219-222 options fold check raise
% 229-230 options fold check raise
% 231-233 raise bar
% 250-254 options fold call raise
% 264-268 options fold check raise
% 274-275 options fold call raise
% 276-278 raise bar
% 284-285 options fold call raise
% 290 options fold check raise
% 291-292 raise bar

filerefs=cell(11,1);
option_fold={options1;options1;options1;options1;options1;options1;options1;options1;[];[];[];options1;options1;options1;options1;options1;options1;options1;options1;options1;options1;options1;options1;[];[];[];options1;options1;options1;[];[]};
option_check={options2;options2;options2;options2;options2;options2;options2;options2;[];[];[];[];[];[];[];[];options2;options2;options2;options2;options2;[];[];[];[];[];[];[];options2;[];[]};
option_call={options2;options2;options2;options2;options2;options2;options2;options2;[];[];[];options2;options2;options2;options2;options2;[];[];[];[];[];options2;options2;[];[];[];options2;options2;[];[];[]};
option_raise = {options3;options3;options3;options3;options3;options3;options3;options3;[];[];[];options3;options3;options3;options3;options3;options3;options3;options3;options3;options3;options3;options3;[];[];[];options3;options3;options3;[];[]};
option_ok={[];[];[];[];[];[];[];[];options_raisebar;options_raisebar;options_raisebar;[];[];[];[];[];[];[];[];[];[];[];[];options_raisebar;options_raisebar;options_raisebar;[];[];[];options_raisebar;options_raisebar};
idx=[212:213 219:222 229:233 250:254 264:268 274:278 284:285 290:292];

for i=1:31
    filerefs(i)=cellstr(strcat('C:\Users\marti_000\Documents\MATLAB\octave\fullscreen\test_fullscreen',int2str(idx(i)),'.png'));
%     locations(i)={you_deal}; %syntax we used earlier
end

RCNN_input=table(filerefs,option_fold,option_check,option_call,option_raise,option_ok);

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-5, ...
    'MaxEpochs', 30,'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.8, ...
    'LearnRateDropPeriod',5);

%% REUSE LAYERS BUT ASK FOR 6 CLASSES
% detector = trainRCNNObjectDetector(RCNN_input,layers,options); % suggests
% good score but seems to struggle a little
fold_t=RCNN_input(:,1:2);
detector_fold = trainRCNNObjectDetector(fold_t,layers,options); % needs fc layer with 2 neurons
check_t=RCNN_input(:,[1 3]);
detector_check = trainRCNNObjectDetector(check_t,layers,options);
img=options_sub;
[bbox, score, label] = detect(detector_check, img, 'MiniBatchSize', 32);
% Display strongest detection result.

[score, idx] = max(score);

bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

figure
imshow(detectedImg)

%% Just try OCR 
results = ocr(img,'TextLayout','Block');
for i=1:3
wordBBox = results.WordBoundingBoxes(i,:); % location of first word
    word = results.Words{i};
    img = insertObjectAnnotation(img,'rectangle',wordBBox,word);
end
    figure
imshow(img)
