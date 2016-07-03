require 'nn'
require 'dp'
require 'torch'
require 'paths'
require 'optim'

local function ColorTransformTrain(set) --convert data images from RGB to YUV color space
	local trainset = torch.Tensor(50000, 3, 32, 32);
	for i=1,((#set)[1]) do	--do it on training set
		local rgb = set[i];
		local yuv = image.rgb2yuv(rgb);
		trainset[{{i}}] = yuv;
	end
	return trainset;
end

function ColorTransformTest(set) --convert data images from RGB to YUV color space
	local testset = torch.Tensor(50000, 3, 32, 32);
	for i=1,((#set)[1]) do  --do it on test set
		local rgb = set[i];
		local yuv = image.rgb2yuv(rgb);
		testset[{{i}}] = yuv;
	end
	return testset;
end

function CreateNet()
	net = nn.Sequential()
		V1 = nn.SpatialConvolution(3,32,5,5)
		--BN1= nn.SpatialBatchNormalization(32)
		V2 = nn.ReLU()
		V3 = nn.SpatialMaxPooling(2,2,2,2)
		V4 = nn.SpatialConvolution(32,64,5,5)
		--BN2= nn.SpatialBatchNormalization(64)
		V5 = nn.ReLU()
		V6 = nn.SpatialMaxPooling(2,2,2,2)
		V7 = nn.View(64*5*5)
		V8 = nn.Dropout()
		V9 = nn.Linear(64*5*5, 10)
		BN3= nn.BatchNormalization(10)
		V10 = nn.LogSoftMax()
	net:add(V1)
	--net:add(BN1)
	net:add(V2)
	net:add(V3)
	net:add(V4)
	--net:add(BN2)
	net:add(V5)
	net:add(V6)
	net:add(V7)
	net:add(V8)
	net:add(V9)
	net:add(BN3)
	net:add(V10)
	return net;
end

function LoadData()
	classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

	-- load training data
	trainset = { data = torch.Tensor(50000, 3, 32, 32), label = torch.Tensor(50000)}

	for i=0,4 do
		local batch = torch.load('gitTorch/Dataset/cifar-10-batches-t7/data_batch_'..(i+1)..'.t7','ascii')
		trainset.data[{ {i*10000+1, (i+1)*10000} }]=batch.data:t()
		trainset.label[{ {i*10000+1, (i+1)*10000} }]=batch.labels
	end

	trainset.label = trainset.label + 1;

	--load test data
	testset = {data = torch.Tensor(10000, 3, 32, 32), label = torch.Tensor(10000)}

	local batch = torch.load('gitTorch/Dataset/cifar-10-batches-t7/test_batch.t7', 'ascii')
		testset.data[{ {1, 10000} }]=batch.data:t()
		testset.label[{ {1, 10000} }]=batch.labels + 1

	trainset.data = trainset.data:double()	--convert data from a ByteTensor to a DoubleTensor
	testset.data = testset.data:double()	--convert data from a ByteTensor to a DoubleTensor
end

function UVNormalization()
	mean = {}	--store the mean, to normalize the test set
	stdv = {}	--store the standard-deviation
	for i=1,3 do --only do it for U,V channel
		--[{image index},{channel},{vertical pixel},{horizontal pixel}]
		mean[i] = trainset.data[{ {}, {i}, {}, {} }]:mean()	--mean of each channel
		stdv[i] = trainset.data[{ {}, {i}, {}, {} }]:std()	--std for each channel
		trainset.data[{ {}, {i}, {}, {} }]:add(-mean[i])	--zero out the mean
		trainset.data[{ {}, {i}, {}, {} }]:div(stdv[i])	--scale for std
		--Normalize test data
		testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])	--zero out the mean
		testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])	--scale for std
	end
end

function BatchCreate(batchSize)
	totalBatch = (#trainset.data)[1]/batchSize;
	batchData = torch.Tensor(totalBatch ,batchSize, 3, 32, 32);
	batchLabel = torch.Tensor(totalBatch,batchSize);
	--[[
	for i=1,(#trainset.data)[1],batchSize do
		local Ymean = trainset.data[{ {i,i+batchSize-1}, {1}, {}, {} }]:mean() --get batch Y mean
		local Yvar = trainset.data[{ {i,i+batchSize-1}, {1}, {}, {} }]:std()
		trainset.data[{ {i,i+batchSize-1}, {1}, {}, {} }]:add(-Ymean) --normalize batch Y values
		trainset.data[{ {i,i+batchSize-1}, {1}, {}, {} }]:div(Yvar)
	end
	]]
	for i=1,totalBatch do
		batchData[{{i}}]=trainset.data[{{i,i+batchSize-1}}];
		batchLabel[{{i}}]=trainset.label[{{i,i+batchSize-1}}];
	end
	
end

function TestBatchCreate(batchSize)
	totalTestBatch = (#testset.data)[1]/batchSize;
	batchTestData = torch.Tensor(totalTestBatch, batchSize, 3, 32, 32);
	batchTestLabel = torch.Tensor(totalTestBatch, batchSize);
	--[[
	for i=1,(#testset.data)[1],batchSize do
		local testMean = testset.data[{ {i,i+batchSize-1}, {1}, {}, {} }]:mean()
		local testVar = testset.data[{ {i,i+batchSize-1}, {1}, {}, {} }]:std()
		testset.data[{ {i,i+batchSize-1}, {1}, {}, {} }]:add(-testMean)
		testset.data[{ {i,i+batchSize-1}, {1}, {}, {} }]:div(testVar)
	end
	]]
	for i=1,totalTestBatch do 
		batchTestData[{{i}}]=testset.data[{{i,i+batchSize-1}}];
		batchTestLabel[{{i}}]=testset.label[{{i,i+batchSize-1}}];	
	end
end

function trainBatch(batchIndex)
	input = batchData[batchIndex];
	target = batchLabel[batchIndex];
end

local function func(x)  --optimization function
	if x ~= parameters then  -- get new parameters
 	  parameters:copy(x)
   end
   gradParameters:zero()  -- reset gradients
   local output = net:forward(input)
   local f = criterion:forward(output, target)
   local df_do = criterion:backward(output, target)  -- estimate df/dW
   net:backward(input, df_do)
   if optim_params.coefL1 ~= 0 or optim_params.coefL2 ~= 0 then  -- penalties (L1 and L2)
      local norm,sign= torch.norm,torch.sign  -- locals
      f = f + optim_params.coefL1 * norm(parameters,1)  -- L1 Loss
      f = f + optim_params.coefL2 * norm(parameters,2)^2/2  -- L2 Loss     
	   gradParameters:add( sign(parameters):mul(optim_params.coefL1) + parameters:clone():mul(optim_params.coefL2))  -- gradients
   end
   confusion:batchAdd(output, target)  -- update confusion
	return f,gradParameters
end

function preProcessing()
	net:training();	--enable training mode
	time = sys.clock()  -- initialize time
end

function postProcessing()
	print(confusion);  --output confusion matrix
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero();
	
   time = sys.clock() - time;  -- time taken
   time = time / ((#trainset.data)[1]);
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   local filename = paths.concat('/home/lakechen/gitTorch/Deep06', 'trained.net')  -- save/log current net
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
	torch.save(filename, net)
	print('Complete saving the network...')
end

function test()
	net:evaluate();
	for i=1,totalTestBatch do
		local groundtruth = batchTestLabel[i]
		local confidences, prediction = torch.sort(net:forward(batchTestData[i]),true);
		confusion:batchAdd(prediction[{{},{1}}], groundtruth)  -- update confusion
	end
	print(confusion)
	print('Network accuracy: '..(confusion.totalValid * 100)..' %');
	testLogger:add{['% mean class accuracy (test set)'] = (confusion.totalValid * 100)}  --Output confusion matrix
	confusion:zero();
end

local function visualize()  --Plot progress
	trainLogger:style{['% mean class accuracy (train set)'] = '-'}
	testLogger:style{['% mean class accuracy (test set)'] = '-'}
	trainLogger:plot();
	testLogger:plot();
end

CreateNet();
--net = torch.load("gitTorch/Deep05/Run1/trained.net")  --Load Trained Network
print('Network Structure: \n' .. net:__tostring());  --Visualize the Network
LoadData();
	print('Complete loading data...');
trainset.data = ColorTransformTrain(trainset.data); --Convert Training Data into YUV Space
testset.data = ColorTransformTest(testset.data);  --Convert Test Data into YUV Space
UVNormalization();  --Normalize all UV channel
	print('Complete data transformation...');
batchSize = 10;  --Define Batch Size
BatchCreate(batchSize);  --Y Mini Batch Normalization and Data Regroupping
TestBatchCreate(batchSize);
	print('Complete packing batch data...');
criterion = nn.ClassNLLCriterion();  -- a negative log-likelihood criterion for multi-class classification
optim_params = { learningRate = 0.01, momentum = 0.5, coefL1=0, coefL2=0, maxIteration=200}
	parameters,gradParameters = net:getParameters() -- retrieve parameters and gradients
	print('Network size:'..(#parameters)[1])
	confusion = optim.ConfusionMatrix(classes)  -- this matrix records the current confusion across classes
	-- log results to files
	trainLogger = optim.Logger(paths.concat('/home/lakechen/gitTorch/Deep06', 'train.log'))
	testLogger = optim.Logger(paths.concat('/home/lakechen/gitTorch/Deep06', 'test.log'))
lrDecayStep = 10;
print('training start...');
for i=1,optim_params.maxIteration do
	if i%lrDecayStep==0 then optim_params.learningRate = optim_params.learningRate/2 end--update learning rate
	preProcessing();
	for j=1,totalBatch do	
		xlua.progress((i-1)*totalBatch+j,optim_params.maxIteration*totalBatch);  --display the progress
		trainBatch(j);
		optim.sgd(func, parameters, optim_params);
	end
	postProcessing();
		print('Start test...');
	test();
	visualize();
end
