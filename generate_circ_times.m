function out = generate_circ_times(mean,n,kappa)

% one extra point because we're going from -pi to pi (i.e. repeating midnight)
alpha = linspace(-pi,pi,25);
theta = alpha(mean+1);  % mean will be at midday

VMdata = circ_vmrnd(theta,kappa,n);
VMdata = 24*(VMdata+pi)/(2*pi);     % scale to range [0 24]
out = floor(VMdata);
out(out==24) = 0;