%RUN_THRESHOLD_PLANE  Is the wideband threshold gain a property of the method,
%or of one lucky operating point?
%
% The 9.5 dB threshold gain reported so far was measured at a SINGLE point,
% theta = 0.6, r = 12 m. A first low-power sweep (40 trials, ~4 dB ladder)
% suggested it is NOT representative - the gap looked like 3-4 dB over much of
% the plane, near zero at theta = 0.75, and occasionally negative - but those
% numbers were dominated by estimator noise, not structure. At 40 trials the
% binomial standard deviation at a 10% outlier rate is 4.7%, which on a 4 dB
% ladder translates to several dB of scatter in the threshold itself.
%
% This script does it properly. It is the run that decides whether the
% headline number survives, so it is worth real compute.
%
%   NR = 300 trials per SNR point, 16-point ladder (~2.5 dB steps)
%   5 angles x 4 ranges x 2 schemes
%
% Runtime is roughly 5 angles * 4 ranges * 16 SNR * 300 trials * 2 schemes
% = 192k estimates. Expect several hours. Reduce NR to 150 or drop a range if
% that is too long; do NOT reduce below 150, which is where the scatter starts
% swamping the effect being measured.
%
% What to report back: the whole `gap` matrix, not just its best entry.
% A method that helps by 3-4 dB almost everywhere is a good result. A method
% that helps by 9.5 dB at one angle and 0 dB elsewhere is a different, much
% weaker claim, and we need to know which one this is.

clear; clc;
L = nf_lib();
a = L.array(256, 40e9);
beta = 0.05; M = 9; dec = 4;

NR     = 300;
A0list = [3 4 5.5 7.5 10 13 17 22 29 38 50 66 87 115 150 200];
thetas = [0.15 0.30 0.45 0.60 0.75];
ranges = [8 12 18 25];
outTol = 0.25;
rMin = 5; rMax = 60;

thGrid = linspace(-0.9, 0.9, 121);
uGrid  = 1 ./ linspace(rMin, rMax, 45);

phis = a.grid(1:dec:end);
schemes = { {'NB', a.fc,               1}, ...
            {'WB', L.subc(a,beta,M),   M} };
for s = 1:2
    schemes{s}{4} = L.dict(a, schemes{s}{2}, phis, thGrid, uGrid);
end

TH = nan(numel(ranges), numel(thetas), 2);
for ir = 1:numel(ranges)
    for it = 1:numel(thetas)
        for s = 1:2
            fm = schemes{s}{2}; Ms = schemes{s}{3}; D = schemes{s}{4};
            orate = zeros(size(A0list)); pk = orate;
            for q = 1:numel(A0list)
                A = A0list(q)/sqrt(Ms);
                y0 = A*steer_complex(a, thetas(it), ranges(ir), fm, phis);
                pk(q) = 10*log10(max(abs(y0(:)).^2));
                rng(23); nOut = 0;
                for t = 1:NR
                    n = sqrt(0.5)*(randn(size(y0)) + 1i*randn(size(y0)));
                    [~, rHat] = L.estimate(a, abs(y0+n), fm, phis, ...
                                           thGrid, uGrid, D, rMin, rMax);
                    nOut = nOut + (abs(rHat-ranges(ir))/ranges(ir) > outTol);
                end
                orate(q) = nOut/NR;
            end
            TH(ir,it,s) = crossing(pk, orate, 0.10);
        end
        fprintf('r=%4.0f m  theta=%.2f   NB %6.1f dB   WB %6.1f dB   gap %6.1f dB\n', ...
                ranges(ir), thetas(it), TH(ir,it,1), TH(ir,it,2), ...
                TH(ir,it,1)-TH(ir,it,2));
    end
end
save('threshold_plane.mat','TH','thetas','ranges','NR','beta','M','dec');

gap = TH(:,:,1) - TH(:,:,2);
fprintf('\ngap matrix (rows = ranges %s m, cols = angles %s)\n', ...
        mat2str(ranges), mat2str(thetas));
disp(round(gap,1));
fprintf('median gap %.1f dB, min %.1f dB, max %.1f dB, NaN entries %d\n', ...
        median(gap(~isnan(gap))), min(gap(:)), max(gap(:)), sum(isnan(gap(:))));

figure; imagesc(thetas, ranges, gap); colorbar
xlabel('\theta'); ylabel('r (m)'); title('threshold gain of wideband over narrowband (dB)');

% ---------------------------------------------------------------- helpers

function y = steer_complex(a, theta, r, fm, phis)
fm = fm(:); phis = phis(:);
rn = sqrt(r^2 + (a.dn*a.d).^2 - 2*r*theta*a.dn*a.d);
ph = exp(1i*2*pi*(fm*(rn - r).')/a.c);
A  = exp(-1i*pi*(phis*a.dn.'));
y  = (ph*A')/a.N;
end

function th = crossing(pk, orate, level)
th = NaN;
i = find(orate <= level, 1, 'first');
if isempty(i), return; end
if i == 1, th = pk(1); return; end
th = interp1(orate([i i-1]), pk([i i-1]), level);
end
