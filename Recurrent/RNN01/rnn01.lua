require 'rnn'
require 'gnuplot'
local nninit = require 'nninit'

-- hyper-parameters 
batchSize = 1
rho = 10 -- sequence length
inputSize = 1
outputSize = 10
lr = 1
decayPeriod = 20

-- build simple recurrent neural network
local r = nn.Recurrent(
		outputSize, 						--size of output
		nn.Linear(inputSize, outputSize),	--input
		nn.Linear(outputSize, outputSize),	--feedback
		nn.Sigmoid(),							--transfer
		rho									--rho
		)

local rnn = nn.Sequential()
	:add(r)
   :add(nn.Linear(outputSize, inputSize))
   :add(nn.Sigmoid())
rnn = nn.Recursor(rnn, rho)


print(rnn)

-- build criterion
criterion = nn.MSECriterion()

-- load signal data
signal = torch.load('gitTorch/Recurrent/Data/Signal.t7')
shift = signal[{ {}, {2} }]:add(-signal[{ {}, {2} }]:min())
range = signal[{ {}, {2} }]:max()-signal[{ {}, {2} }]:min()
signal[{ {}, {2} }] = shift:div(range)


prediction = torch.Tensor(#signal):fill(0)

--training
local index = 1
local iteration = 1
while iteration < 30 do
	local total_err = 0
	for index = 1, ((#signal)[1]-rho-1) do
		outputs = {}
		local err = 0
		rnn:zeroGradParameters() 
   	rnn:forget() --forget all past time-steps
	
		--forward
		for step = 1, rho do
			outputs[step] = rnn:forward(signal[{ {index+step}, {2} }])
			--print(outputs[step])
			prediction[{ {index+step+1}, {} }] = torch.cat(torch.DoubleTensor({index+step+1}), torch.DoubleTensor(outputs[step]))	--record the prediction
			err = err + criterion:forward(outputs[step], signal[{ {index+step+1}, {2} }])
		end
		
		total_err = total_err + err
		
		local gradOutputs, gradInputs = {}, {}
   	for step=rho,1,-1 do -- reverse order of forward calls
     		gradOutputs[step] = criterion:backward(outputs[step], signal[{ {index+step+1}, {2} }])
      	gradInputs[step] = rnn:backward(signal[{ {index+step}, {2} }], gradOutputs[step])
   	end
		rnn:updateParameters(lr)  -- update
	end

	print(string.format("Iteration %d ; Mean MSE err = %f ", iteration, math.sqrt(total_err/(#signal)[1]) ))
	gnuplot.plot({signal}, {prediction})
	iteration = iteration + 1
		
	if iteration%decayPeriod==0 then
		lr = lr/2
		print('Learning Rate: ', lr)
	end
end
