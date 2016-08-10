clear
close all
clc

%% Generate some fake data for now
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% using the von-Mises distribution (circular normal)

Npoints = 400;              % no of data points
mean = randi([0 23]);      % choose a mean (hour of the day)
kappa = 1;                  % concentration parameter for dist (don't change)
% a monomodal example distribution
SzTimesMono = generate_circ_times(mean,Npoints,kappa);

% make a bimodal example dist by generating two different monomodal
% populations
mean1 = randi([0 11]);
p = round(10*rand)/10;
mean2 = randi([12 23]);
kappa = 1.5;
Times1 = generate_circ_times(mean1,round(p*Npoints),kappa);
Times2 = generate_circ_times(mean2,round((1-p)*Npoints),kappa);
SzTimesBi = [Times1;Times2];
Shuffle = randperm(Npoints);
SzTimesBi = SzTimesBi(Shuffle);  % mix up the data because we want to estimate sequentially

%% Now we run the PDF estimator
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Kn = 24;         % number of kernels
Kmean = 0:23;    % kernel centers
Kbw = 1;         % this is the "bandwidth" or std of the Gaussian kernels

% initial estimate of density from gaussian kernels

times = 0:24/Kn:23;  % evaluate the kernels over the entire day

% Create Kn gaussians spanning the day
Kgauss = normpdf(repmat(times',1,Kn),...   
    repmat(Kmean,Kn,1),...  % set the mean of the gaussians and the extras
    Kbw*ones(Kn));          % the std is the same for everyone
Kextra = normpdf(times,24,1);
Kextra2 = [Kgauss(1,2:end) 0];
% the density is given by the normalized sum of the kernels
Kdensity = sum(Kgauss,2); % keep this as raw value so we can iterate the sum

Kwrap = Kextra2 + fliplr(Kextra2(1,:));
% create the pdf & normalize
Kdensity = Kdensity + Kwrap';
time_pdf = Kdensity;
time_pdf = 1/Kn*time_pdf;  % this is to normalize the area to 1 (ish)

%Data_to_run = [10 11 12 11 10 1 3 4];
%Data_to_run = SzTimesBi;
Data_to_run = SzTimesMono;

% true vals
SzHist = histc(Data_to_run,0:23)/length(Data_to_run);


%% plot stuff
PDF_fig = figure; 
figure(PDF_fig);
round_val = 0.02; % round to nearest xxx
yran = [0 ceil(max(SzHist/round_val))*round_val];
Scatter_ax = subplot(211);
title(Scatter_ax,'Data Points');
PDF_ax = subplot(212);
title(PDF_ax,'PDF Estimate');
xlabel(PDF_ax,'Time (hours)','fontsize',8);
set(PDF_ax,'box','off','xlim',[-1 24],'ylim',yran,'ytick',yran, ...
    'position',[0.1 0.1 0.8 0.6],'fontsize',8);
hold(PDF_ax,'on');
set(Scatter_ax,'box','off','xlim',[-1 24],'xtick',[],'ylim',[0 1.5],'ytick',[], ...
    'position',[0.1 0.75 0.8 0.2],'fontsize',8);
hold(Scatter_ax,'on');
%%

for n = 1:length(Data_to_run)
   % plot the changing estimate
   if n>1
   set(p,'color',[128 128 128]/255,'linewidth',1);
   end
   p = plot(PDF_ax,0:23,time_pdf,'m','linewidth',2); drawnow;
%    pause;
   % the index is the time plus 1
   time_ind = Data_to_run(n)+1;
   % plot the data points
   plot(Scatter_ax,time_ind-1+normrnd(0,0.2),0.5+normrnd(0,0.2),'kx');
   % update the density
   Kdensity = Kdensity+Kgauss(:,time_ind);
   if time_ind == 1
       Kdensity = Kdensity + Kwrap';
   end
   % renormalize the pdf
   time_pdf = Kdensity;
   time_pdf = time_pdf/trapz(0:23,time_pdf);
end

% plot more stuff
set(p,'color',[128 128 128]/255);
bar(PDF_ax,0:23,SzHist,'facecolor','w','edgecolor','k');
hold(PDF_ax,'on');
plot(PDF_ax,0:23,time_pdf,'m','linewidth',2);
set(get(PDF_ax,'title'),'string','PDF Estimate and Actual Histogram');