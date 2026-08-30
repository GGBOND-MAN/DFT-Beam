%RUN_THRESHOLD_MC  The heavy Monte Carlo: where the estimator leaves the CRB.
%
% This is the run worth spending real compute on. It produces the paper's main
% figure and the number the whole direction rests on: how much lower the
% outlier threshold is when the OFDM subcarriers are kept separate instead of
% summed, at an identical pilot budget and identical total energy.
%
% Fairness convention: per-pilot energy is FIXED, so pilot count IS total
% energy. The wideband schemes additionally split each pilot's energy over M
% subcarriers (amplitude A0/sqrt(M)), which cancels any pure averaging gain.
% Any difference that survives is structural.
%
% Reference values from the Python run (150 trials, theta=0.6, r=12 m):
%     narrowband, full codebook   256 pilots   threshold <= 19.4 dB
%     narrowband, every 4th beam   64 pilots   threshold  = 23.4 dB
%     wideband, every 4th beam     64 pilots   threshold  = 14.0 dB
% The full-codebook entry was never bracketed - the sweep bottomed out at
% 19.4 dB with zero outliers. Pinning it is the point of the wider SNR range
% below; please report where it actually breaks.
%
% Runtime scales as numel(A0list)*NR*numel(schemes). With the defaults below
% (150 trials, 12 SNR points, 3 schemes) expect tens of minutes. Raise NR to
% 1000 for publication-quality curves if you can leave it running.

clear; clc;
L = nf_lib();
a = L.array(256, 40e9);
beta = 0.05; M = 9;

theta0 = 0.6; r0 = 12;              % test point
NR     = 150;                       % trials per SNR point  <-- raise me
A0list = [4 5 7 10 14 20 28 40 56 80 115 160 230];
outTol = 0.25;                      % outlier := |rHat-r0|/r0 > outTol
rMin   = 5; rMax = 60;              % range search bracket

thGrid = linspace(-0.9, 0.9, 121);
uGrid  = 1 ./ linspace(rMin, rMax, 45);

schemes = { {'narrowband, full codebook',      1, []}, ...
            {'narrowband, every 4th beam',     4, []}, ...
            {'wideband B/fc=0.05, every 4th',  4, beta} };

R = struct();
for s = 1:numel(schemes)
    nm = schemes{s}{1}; dec = schemes{s}{2}; b = schemes{s}{3};
    phis = a.grid(1:dec:end);
    if isempty(b), fm = a.fc; Ms = 1; else, fm = L.subc(a,b,M); Ms = M; end
    D = L.dict(a, fm, phis, thGrid, uGrid);
    G0unit = L.gain(a, theta0, r0, fm, phis);

    pk = zeros(size(A0list)); orate = pk; rmse = pk; crb = pk;
    for q = 1:numel(A0list)
        A = A0list(q)/sqrt(Ms);                       % fixed per-pilot energy
        y0 = A*steer_complex(a, theta0, r0, fm, phis);
        pk(q) = 10*log10(max(abs(y0(:)).^2));
        rng(11);                                      % reproducible
        err = zeros(NR,1);
        for t = 1:NR
            n = sqrt(0.5)*(randn(size(y0)) + 1i*randn(size(y0)));
            z = abs(y0 + n);
            [~, rHat] = L.estimate(a, z, fm, phis, thGrid, uGrid, D, rMin, rMax);
            err(t) = rHat - r0;
        end
        orate(q) = mean(abs(err)/r0 > outTol);
        rmse(q)  = sqrt(mean(err.^2));
        crb(q)   = crb_range(a, theta0, r0, fm, phis, A);
    end

    R(s).name = nm; R(s).K = numel(phis); R(s).peakSNR = pk;
    R(s).outlier = orate; R(s).rmse = rmse; R(s).crb = crb;
    R(s).Eeff = L.eeff(a, theta0, r0, fm, phis);

    fprintf('%-31s %4d pilots   E_eff = %.2f\n', nm, numel(phis), R(s).Eeff);
    fprintf('   peakSNR dB : %s\n', sprintf('%7.1f', pk));
    fprintf('   outlier    : %s\n', sprintf('%6.0f%%', 100*orate));
    fprintf('   RMSE_r  m  : %s\n', sprintf('%7.3f', rmse));
    fprintf('   CRB_r   m  : %s\n', sprintf('%7.3f', crb));
    if min(orate) <= 0.10 && max(orate) >= 0.10
        th = interp1(orate(end:-1:1), pk(end:-1:1), 0.10);
        fprintf('   -> 10%%-outlier threshold = %.1f dB\n\n', th);
    else
        fprintf('   -> NOT bracketed (min outlier %.0f%%) - widen A0list\n\n', 100*min(orate));
    end
end
save('threshold_mc.mat','R','theta0','r0','NR','beta','M');

figure; hold on; grid on
for s = 1:numel(R)
    plot(R(s).peakSNR, R(s).rmse, '-o', 'DisplayName', ...
         sprintf('%s (%d)', R(s).name, R(s).K));
    plot(R(s).peakSNR, R(s).crb, '--', 'HandleVisibility','off');
end
set(gca,'YScale','log'); xlabel('peak SNR (dB)'); ylabel('range RMSE (m)');
legend('Location','southwest'); title('solid: estimator, dashed: CRB');

% ---------------------------------------------------------------- helpers

function y = steer_complex(a, theta, r, fm, phis)
fm = fm(:); phis = phis(:);
rn = sqrt(r^2 + (a.dn*a.d).^2 - 2*r*theta*a.dn*a.d);
ph = exp(1i*2*pi*(fm*(rn - r).')/a.c);
A  = exp(-1i*pi*(phis*a.dn.'));
y  = (ph*A')/a.N;
end

function s = crb_range(a, theta, r, fm, phis, A)
%CRB_RANGE  high-SNR (Gaussian) approximation to the amplitude-only Rician CRB
%           for r, with one nuisance amplitude. Valid above threshold, which is
%           the only regime where the CRB is meaningful anyway.
L = nf_lib(); ht = 1e-6; hr = 1e-4;
G   = L.gain(a, theta, r, fm, phis);
dGt = (L.gain(a, theta+ht, r, fm, phis) - L.gain(a, theta-ht, r, fm, phis))/(2*ht);
dGr = (L.gain(a, theta, r+hr, fm, phis) - L.gain(a, theta, r-hr, fm, phis))/(2*hr);
J = [A*dGt(:), A*dGr(:), G(:)];
F = (2/1.0) * (J.'*J);              % sigma^2 = 1
C = pinv(F);
s = sqrt(C(2,2));
end
