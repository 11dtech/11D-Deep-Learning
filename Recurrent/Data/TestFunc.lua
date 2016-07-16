require 'gnuplot'

function sine(t,amp, period, phase)
	local value = amp*math.sin(phase/180*math.pi+t*2*math.pi/period);
	return value
end

function dc(shift)
	return shift
end

dt = 1
period = 50
amplitude = 1
seriesLength = 10*period

data = torch.Tensor(seriesLength,2)

for t=1,seriesLength do
	local value = sine(t, 0.5, 50, 0) + sine(t, 0.3, 100, 0) + sine(t, 0.4, 125, 45) + sine(t, 0.5, 125, 75)
	data[{ {t} }] = torch.Tensor({{t, value}})
end

gnuplot.plot(data)

torch.save('gitTorch/Recurrent/Data/Signal.t7', data);
